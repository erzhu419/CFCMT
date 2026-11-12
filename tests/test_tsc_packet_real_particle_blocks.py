"""Pure cached-payload arithmetic: no sampler, physical resolver or server."""
import gzip
import json
from math import exp, log, sqrt

import numpy as np
import pytest

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_cache_20261110.verify import parameter_probes
from experiments.tsc_packet_real_particle_pilot_20261110.blocks import compare_query, run


def payload(block_ones):
    phi = [1.]+[0.]*37
    rows = []
    for count in block_ones:
        for index in range(256):
            bit = int(index < count)
            rows.append(dict(factor_ids=[bit], focal_group=0,
                focal_targets={'left': 'b', 'stay': 'a'},
                outcome=dict(survived=True, physical_speed_mps=2., started=None,
                    entry_event=bit, lane_change='stay', lane_id='a', position_m=10.)))
    return dict(focal_id='focal', features=[phi],
        factors=[dict(kind='bernoulli', actor='focal', feature_id=0, block=0, bit=bit)
                 for bit in (0, 1)],
        focal_groups=[dict(feature_id=0, allowed=[0, 1])], particles=rows,
        old_stopped=False, eligibility=dict(motion=True, start=False,
            yield_entry=True, lane_change=True), stats=dict(physical_calls=0))


def pools(data):
    return ParticleLaw(data), [ParticleLaw(dict(data,
        particles=data['particles'][index*256:(index+1)*256])) for index in range(4)]


def test_disjoint_blocks_keep_source_indices_and_all190_gradient_coordinates():
    data = payload((64, 128, 192, 128))
    full, blocks = pools(data)
    row = compare_query(full, blocks, parameter_probes()[0],
                        dict(head='yield_entry', y=1, manifest_ordinal=7))
    assert row['all_five_readouts_usable']
    assert row['manifest_ordinal'] == 7
    assert exp(-row['full_1024']['nll']) == pytest.approx(.5)
    assert [exp(-item['nll']) for item in row['blocks_256']] == pytest.approx((.25, .5, .75, .5))
    assert row['full_1024']['gradient_dimension'] == 190
    assert all(item['gradient_dimension'] == 190 for item in row['blocks_256'])
    assert row['full_1024']['gradient_norm_l2'] == pytest.approx(.5)
    assert row['gradient_mean_shift_l2'] == pytest.approx(0., abs=1e-12)
    assert row['gradient_block_rms_dispersion_l2'] == pytest.approx(sqrt(.03125))
    assert row['normalized_gradient_block_rms_dispersion'] == pytest.approx(sqrt(.03125)/1.5)
    assert row['cosine_to_full'] == pytest.approx([1.]*4)
    nonzero = compare_query(full, blocks, parameter_probes()[1], dict(head='yield_entry', y=1))
    fractions = np.array((.25, .5, .75, .5))
    ratio = exp(.1)
    block_gradients = ratio*fractions/(1.-fractions+ratio*fractions)-1.
    full_gradient = ratio/(1.+ratio)-1.
    shift = abs(block_gradients.mean()-full_gradient)
    dispersion = float(np.sqrt(np.mean((block_gradients-block_gradients.mean())**2)))
    assert shift > 0.
    assert nonzero['normalized_gradient_mean_shift'] == pytest.approx(shift/(1.+abs(full_gradient)))
    assert nonzero['normalized_gradient_block_rms_dispersion'] == pytest.approx(dispersion/(1.+abs(full_gradient)))


def test_zero_numerator_retains_failed_blocks_and_blocks_all_finite_subset_averages():
    full, blocks = pools(payload((256, 256, 0, 0)))
    row = compare_query(full, blocks, parameter_probes()[0], dict(head='yield_entry', y=1))
    assert not row['all_five_readouts_usable']
    assert len(row['blocks_256']) == 4
    assert [item['failure'] for item in row['blocks_256']] == [None, None, 'ZERO_NUMERATOR', 'ZERO_NUMERATOR']
    assert row['mean_block_nll_infinite']
    assert row['normalized_gradient_mean_shift'] is None
    assert row['normalized_gradient_block_rms_dispersion'] is None
    assert row['mean_block_nll_minus_full'] is None


def test_exactly_zero_gradient_signal_has_undefined_cosines():
    full, blocks = pools(payload((256,)*4))
    row = compare_query(full, blocks, parameter_probes()[0], dict(head='yield_entry', y=1))
    assert row['all_five_readouts_usable'] and row['full_zero_signal']
    assert row['zero_signal_blocks'] == [0, 1, 2, 3]
    assert row['cosine_to_full'] == [None]*4


class FakeReadout:
    def __init__(self, **changes):
        self.row = dict(defined=True, supported=True, nll=.5,
            gradient=np.ones((5, 38)), global_ess=1., survivor_ess=1.,
            posterior_ess=1., max_normalized_weight=1.)
        self.row.update(changes)
    def marginal(self, *args):
        return self.row


@pytest.mark.parametrize('bad,expected', [
    ({'defined': False, 'supported': False, 'nll': None, 'gradient': None}, 'NO_SURVIVOR_SAMPLES'),
    ({'nll': float('inf')}, 'NONFINITE_NLL'),
    ({'gradient': np.full((5, 38), np.nan)}, 'NONFINITE_GRADIENT'),
])
def test_undefined_and_nonfinite_blocks_are_preserved_as_unusable(bad, expected):
    full = FakeReadout()
    blocks = [FakeReadout(), FakeReadout(**bad), FakeReadout(), FakeReadout()]
    row = compare_query(full, blocks, parameter_probes()[0], dict(head='motion', y=1.))
    assert len(row['blocks_256']) == 4 and row['blocks_256'][1]['failure'] == expected
    assert not row['all_five_readouts_usable']
    assert row['gradient_mean_shift_l2'] is None


def test_cache_only_run_uses_both_frozen_points_and_primary_observation_labels(tmp_path):
    evaluation = tmp_path/'evaluation'
    evaluation.mkdir()
    ref = dict(seed=76116901, time_sec=1, packet_id='focal')
    receipt = dict(reference=ref, case_index=0, status='PASS_REAL_PARTICLE_CASE', origin='original9',
        readouts={'1024': {str(point): dict(observed_heads=[
            dict(head='yield_entry', y=1, manifest_ordinal=7)]) for point in (0, 1)}})
    (evaluation/'summary.json').write_text(json.dumps(dict(status='PASS_BOUNDED_REAL_PARTICLE_PILOT',
        requested_states=1, cases=[receipt])))
    directory = evaluation/'case00_seed76116901_t1'
    directory.mkdir()
    (directory/'summary.json').write_text(json.dumps(receipt))
    with gzip.open(directory/'particles_server_only.json.gz', 'wt') as stream:
        json.dump(payload((64, 128, 192, 128)), stream)
    result = run(evaluation, tmp_path/'secondary.json')
    assert result['exploratory_secondary_not_preregistered_gate']
    assert result['aggregate']['query_point_records'] == 2
    assert result['new_particle_full_forward_calls'] == result['new_exact_full_forward_calls'] == 0
    assert set(result['cases'][0]['points']) == {'0', '1'}
    for point in result['cases'][0]['points'].values():
        assert point[0]['head'] == 'yield_entry' and point[0]['y'] == 1
        assert point[0]['manifest_ordinal'] == 7
    assert result['cases'][0]['points']['0'][0]['full_1024']['nll'] == pytest.approx(log(2.))
    assert result['cases'][0]['points']['1'][0]['full_1024']['nll'] == pytest.approx(log(1.+exp(-.1)))
    assert 'not exact truth' in result['definitions']['full_reference']
