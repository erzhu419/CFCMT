"""Pure fixed-payload arithmetic and receipt checks; no physics or optimizer."""
from copy import deepcopy
from math import exp, log

import numpy as np
import pytest

from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_fitted_cache_stability_20261111 import readout


HEADS = ('motion', 'start', 'yield_entry', 'lane_change')


def payload(counts, block_size=64):
    """Nonzero factor/group IDs ensure slicing cannot silently renumber them."""
    phi = [1., -.25]+[0.]*36
    factors = [dict(kind='bernoulli', actor='focal', feature_id=feature,
                    block=0, bit=bit) for feature in (0, 1) for bit in (0, 1)]
    particles = []
    for count in counts:
        for index in range(block_size):
            bit = int(index < count)
            particles.append(dict(sample_id=len(particles), factor_ids=[2+bit],
                focal_group=1, focal_targets={'left': 'b', 'stay': 'a', 'right': 'c'},
                outcome=dict(survived=True, physical_speed_mps=1.+bit,
                    started=bit, entry_event=bit, lane_change='stay', lane_id='a', position_m=10.)))
    return dict(focal_id='focal', features=[[1.]+[0.]*37, phi], factors=factors,
        focal_groups=[dict(feature_id=0, allowed=[1]), dict(feature_id=1, allowed=[0, 1, 2])],
        particles=particles, old_stopped=True,
        eligibility={head: True for head in HEADS}, stats=dict(physical_calls=0))


def candidate(value=.8):
    result = np.zeros((5, 38))
    result[0, 0] = value
    return result


def test_extended_differences_preserve_infinite_and_undefined_states():
    scalar = readout.extended
    assert readout.paired_difference(scalar(1.), scalar(3.)) == {'state': 'finite', 'value': -2.}
    assert readout.paired_difference(scalar(float('inf')), scalar(3.))['state'] == 'positive_infinity'
    assert readout.paired_difference(scalar(3.), scalar(float('inf')))['state'] == 'negative_infinity'
    assert readout.paired_difference(scalar(float('inf')), scalar(float('inf')))['state'] == 'undefined'
    assert scalar(float('nan'))['state'] == scalar(None)['state'] == 'undefined'


def test_strict_mean_keeps_all_infinite_or_missing_contributions():
    scalar = readout.extended
    assert readout.strict_mean([scalar(1.), scalar(3.)]) == {'state': 'finite', 'value': 2.}
    assert readout.strict_mean([scalar(1.), scalar(float('inf'))])['state'] == 'positive_infinity'
    assert readout.strict_mean([scalar(float('inf')), scalar(-float('inf'))])['state'] == 'undefined'
    assert readout.strict_mean([scalar(1.), scalar(None)])['state'] == 'undefined'


def test_256_pool_is_four_disjoint_consecutive64_blocks_without_table_renumbering():
    data = payload((16, 32, 48, 32))
    original = deepcopy(data)
    with physical_forbidden():
        full = readout.slice_pool(data, 0, 256)
        blocks = [readout.slice_pool(data, index*64, (index+1)*64) for index in range(4)]
        zero, zero_g = readout.read_query(full, np.zeros((5, 38)), 'yield_entry', 1)
        block_rows = [readout.read_query(pool, np.zeros((5, 38)), 'yield_entry', 1) for pool in blocks]
        final, final_g = readout.read_query(full, candidate(), 'yield_entry', 1)
    assert data == original
    assert exp(-zero['nll']) == pytest.approx(.5)
    assert [exp(-row['nll']) for row, unused in block_rows] == pytest.approx((.25, .5, .75, .5))
    assert [row['sample_id'] for block in blocks for row in block.rows] == list(range(256))
    assert all(pool.data['features'] == data['features'] and pool.data['factors'] == data['factors'] for pool in blocks)
    assert all(row['factor_ids'][0] in (2, 3) and row['focal_group'] == 1 for block in blocks for row in block.rows)
    expected = exp(.8)/(1.+exp(.8))
    assert final['nll'] == pytest.approx(-log(expected))
    assert zero_g.shape == final_g.shape == (190,)
    change = readout.paired_difference(final['loss'], zero['loss'])
    assert change['state'] == 'finite'
    assert change['value'] == pytest.approx(-log(expected)+log(.5))


def test_auxiliary_first256_and_all1024_keep_the_same_frozen_factors():
    data = payload((64, 192, 128, 192), block_size=256)
    with physical_forbidden():
        prefix = readout.slice_pool(data, 0, 256)
        full = readout.slice_pool(data, 0, 1024)
        a, unused = readout.read_query(prefix, np.zeros((5, 38)), 'yield_entry', 1)
        b, unused = readout.read_query(full, np.zeros((5, 38)), 'yield_entry', 1)
        ac, unused = readout.read_query(prefix, candidate(), 'yield_entry', 1)
        bc, unused = readout.read_query(full, candidate(), 'yield_entry', 1)
    assert a['nll'] == pytest.approx(-log(.25))
    assert b['nll'] == pytest.approx(-log(.5625))
    for fraction, row in ((.25, ac), (.5625, bc)):
        probability = exp(.8)*fraction/(1.-fraction+exp(.8)*fraction)
        assert row['nll'] == pytest.approx(-log(probability))
    assert prefix.data['factors'] == full.data['factors'] == data['factors']
    assert [row['sample_id'] for row in prefix.rows] == list(range(256))


def test_zero_observation_block_remains_infinite_at_both_points():
    data = payload((0, 32, 32, 32))
    with physical_forbidden():
        full = readout.slice_pool(data, 0, 256)
        blocks = [readout.slice_pool(data, index*64, (index+1)*64) for index in range(4)]
        old = [readout.read_query(pool, np.zeros((5, 38)), 'yield_entry', 1) for pool in blocks]
        final = [readout.read_query(pool, candidate(), 'yield_entry', 1) for pool in blocks]
        full_row = readout.read_query(full, candidate(), 'yield_entry', 1)
        comparison = readout.gradient_comparison(full_row, final)
    assert old[0][0]['loss']['state'] == final[0][0]['loss']['state'] == 'positive_infinity'
    assert old[0][1] is None and final[0][1] is None
    assert readout.paired_difference(final[0][0]['loss'], old[0][0]['loss'])['state'] == 'undefined'
    assert readout.strict_mean([row['loss'] for row, unused in old])['state'] == 'positive_infinity'
    assert comparison['all_gradients_usable'] is False


def test_no_survivor_is_undefined_rather_than_a_zero_numerator():
    data = payload((16, 32, 48, 32))
    for row in data['particles']:
        row['outcome']['survived'] = False
    with physical_forbidden():
        pool = readout.slice_pool(data, 0, 256)
        row, gradient = readout.read_query(pool, np.zeros((5, 38)), 'motion', 1.)
    assert row['loss']['state'] == 'undefined'
    assert gradient is None and row['failure'] is not None


def test_concentrated_but_genuinely_constant_query_keeps_zero_signal():
    data = payload((1, 0, 0, 0))
    for row in data['particles']:
        row['outcome']['entry_event'] = 1
    with physical_forbidden():
        full = readout.slice_pool(data, 0, 256)
        blocks = [readout.slice_pool(data, index*64, (index+1)*64) for index in range(4)]
        full_row = readout.read_query(full, candidate(10.), 'yield_entry', 1)
        block_rows = [readout.read_query(pool, candidate(10.), 'yield_entry', 1) for pool in blocks]
        comparison = readout.gradient_comparison(full_row, block_rows)
    row, gradient = full_row
    assert row['zero_signal'] and row['nll'] == 0.
    assert row['global_ess'] < 2. and row['max_normalized_weight'] > .98
    assert np.array_equal(gradient, np.zeros(190))
    assert all(item['zero_signal'] for item, unused in block_rows)
    assert comparison['all_gradients_usable'] is True


def test_receipt_check_catches_a_wrong_coefficient_or_reference_readout():
    with physical_forbidden():
        pool = readout.slice_pool(payload((16, 32, 48, 32)), 0, 256)
        row, unused = readout.read_query(pool, candidate(), 'yield_entry', 1)
    assert readout.receipt_check(row, row['nll'])['matched']
    assert not readout.receipt_check(row, row['nll']+.01)['matched']
    missing = readout.missing_readout('MISSING_CACHED_POOL')
    assert not readout.receipt_check(missing, row['nll'])['matched']


def record(head, ordinal, loss, fitted=None):
    fitted = loss/2. if fitted is None else fitted
    def scalar(value):
        return dict(loss=readout.extended(value), nll=value, failure=None,
            zero_signal=False, gradient_norm_l2=1., global_ess=100., survivor_ess=100.,
            posterior_ess=50., max_normalized_weight=.01)
    zero, final = scalar(loss), scalar(fitted)
    change = readout.paired_difference(final['loss'], zero['loss'])
    return dict(case_index=ordinal, head=head, y=1, manifest_ordinal=ordinal,
        points={'zero': dict(full_256=zero, blocks_64=[deepcopy(zero) for unused in range(4)]),
                'candidate': dict(full_256=final, blocks_64=[deepcopy(final) for unused in range(4)])},
        paired_delta=dict(full_256=change, blocks_64=[dict(change) for unused in range(4)]))


def test_aggregate_uses_each_head_mean_and_preserves_duplicate_reference_weights():
    values = {'motion': (1., 1., 4.), 'start': (2., 6.), 'yield_entry': (8.,), 'lane_change': (3., 9.)}
    rows = [record(head, ordinal, loss) for head in HEADS for ordinal, loss in enumerate(values[head])]
    counts = {head: len(values[head]) for head in HEADS}
    summary = readout.aggregate_records(rows, expected_counts=counts)
    assert summary['points']['zero']['head_mean_sum']['value'] == pytest.approx(20.)
    assert summary['points']['candidate']['head_mean_sum']['value'] == pytest.approx(10.)
    assert summary['paired_change']['head_mean_sum']['value'] == pytest.approx(-10.)
    assert summary['points']['zero']['head_means']['motion']['value'] == pytest.approx(2.)


def test_aggregate_cannot_drop_missing_or_infinite_reference():
    rows = [record(head, ordinal, float(ordinal+1)) for head in HEADS for ordinal in range(32)]
    summary = readout.aggregate_records(rows)
    assert summary['points']['zero']['head_mean_sum']['state'] == 'finite'
    missing = readout.aggregate_records(rows[:-1])
    assert missing['points']['zero']['head_mean_sum']['state'] == 'undefined'
    rows[-1]['points']['candidate']['blocks_64'][0]['loss'] = readout.extended(float('inf'))
    rows[-1]['paired_delta']['blocks_64'][0] = readout.extended(float('inf'))
    failed = readout.aggregate_records(rows, view='block_0')
    assert failed['points']['candidate']['head_mean_sum']['state'] == 'positive_infinity'
    assert failed['paired_change']['head_mean_sum']['state'] == 'positive_infinity'
    assert len(rows) == 128 and rows[-1]['manifest_ordinal'] == 31


def test_primary_reference_identity_gate_rejects_duplicates_before_opening_pools(tmp_path, monkeypatch):
    import json
    cases, expected = [], []
    for head in HEADS:
        for ordinal in range(32):
            index = len(cases)
            obs = dict(head=head, y=1, manifest_ordinal=ordinal)
            cases.append(dict(case_index=index, reference=dict(seed=76116904,
                time_sec=index, packet_id='focal'+str(index)), observations=[obs],
                particle_path='must_not_be_opened'))
            expected.append(dict(obs, case_index=index, nll=1.))
    # Counts are still32 per head, but one whole reference is duplicated.
    cases[1] = deepcopy(cases[0])
    particles = tmp_path/'particles'
    particles.mkdir()
    (particles/'summary.json').write_text(json.dumps(dict(
        status='PASS_FROZEN_PARTICLE_TRAINING_POOLS', cases=cases)))
    fit = {point:dict(development=dict(references=expected)) for point in ('zero', 'final')}
    def forbidden_payload(*args):
        raise AssertionError('Invalid reference cohort reached cache decoding')
    monkeypatch.setattr(readout, '_load_payload', forbidden_payload)
    records = []
    with physical_forbidden(), pytest.raises(ValueError, match='identities differ'):
        readout.primary_readout(tmp_path, candidate(), fit, records)
    assert records == []


def test_auxiliary_sensitivity_retains_nonfinite_rows_and_sign_reversals():
    def make(index, first, full):
        changes = {'256': readout.extended(first), '1024': readout.extended(full)}
        return dict(reference=dict(seed=76116904, time_sec=index, packet_id='focal'),
            case_index=index, head='yield_entry', y=1, paired_delta=changes,
            size_sensitivity={}, paired_change_sensitivity=readout.paired_difference(
                changes['1024'], changes['256']))
    rows = [make(0, -.2, .1), make(1, .3, .7)]
    finite = readout.sensitivity_summary(rows)
    assert finite['maximum_absolute_paired_change_sensitivity'] == pytest.approx(.4)
    assert len(finite['finite_sign_reversals']) == 1
    assert finite['finite_sign_reversals'][0]['reference']['time_sec'] == 0
    rows.append(make(2, float('inf'), float('inf')))
    with_failure = readout.sensitivity_summary(rows)
    assert with_failure['query_count'] == 3
    assert with_failure['mean_signed_paired_change_sensitivity']['state'] == 'undefined'
    assert with_failure['maximum_absolute_paired_change_sensitivity'] is None
    assert with_failure['worst_five_retaining_nonfinite_first'][0]['case_index'] == 2
