"""Pure cache/cohort tests: no physical responses, optimizer or server calls."""
from copy import deepcopy
from random import Random

import numpy as np
import pytest

from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import physical_forbidden
from experiments.tsc_packet_particle_first_fit_20261110.model import CompositeObjective
from experiments.tsc_packet_particle_first_fit_20261110 import fit
from experiments.tsc_packet_particle_first_fit_20261110.evaluate import (
    collect_receipts, observations_supported, reference_dispositions,
)
from experiments.tsc_packet_particle_first_fit_20261110.register import (
    group_cases, reference_key, select_references,
)


HEADS = ('motion', 'start', 'yield_entry', 'lane_change')
SEEDS = (76116901, 76116902, 76116903, 76116904)


def cached_pool(constant=False):
    """A hand-written finite law; building it does not sample any physics."""
    features = [[1.]+[(-1.)**(j+k)/(j+1) for j in range(1, 38)] for k in range(3)]
    factors = [dict(kind='bernoulli', actor='focal', feature_id=block,
                    block=block, bit=bit) for block in range(3) for bit in (0, 1)]
    rows = []
    for index in range(8):
        bits = (1, 1, 1) if constant else tuple((index >> block) & 1 for block in range(3))
        b0, b1, b2 = bits
        rows.append(dict(factor_ids=[2*block+bit for block, bit in enumerate(bits)],
            focal_group=0 if constant else index % 2,
            focal_targets={'stay': 'a'} if constant else {'left': 'b', 'stay': 'a', 'right': 'c'},
            outcome=dict(survived=True if constant else index != 0,
                physical_speed_mps=1. if constant else (1.+b1+b2 if b0 else 0.),
                started=b0, entry_event=b2, lane_change='stay', lane_id='a', position_m=10.)))
    groups = [dict(feature_id=0, allowed=[1])] if constant else [
        dict(feature_id=0, allowed=[0, 1, 2]), dict(feature_id=2, allowed=[1, 2])]
    return ParticleLaw(dict(focal_id='focal', features=features, factors=factors,
        focal_groups=groups, particles=rows, old_stopped=True,
        eligibility={head: True for head in HEADS}, stats=dict(physical_calls=0)))


def references():
    labels = {'motion': (1.3, 1.3, 2.4), 'start': (0, 1),
              'yield_entry': (1,), 'lane_change': (0, 2)}
    return [dict(case_index=0, seed=SEEDS[0], head=head, y=y,
                 manifest_ordinal=ordinal, split='train')
            for head in HEADS for ordinal, y in enumerate(labels[head])]


def test_objective_preserves_head_means_multiplicity_and_nonbias_penalty():
    pool, refs = cached_pool(), references()
    theta = np.arange(190, dtype=float).reshape(5, 38)/1900.
    objective = CompositeObjective({0: pool}, refs)
    with physical_forbidden():
        result = objective.evaluate(theta)
        manual = {head: np.mean([pool.marginal(theta, head, ref['y'])['nll']
                                 for ref in refs if ref['head'] == head]) for head in HEADS}
    assert result['head_counts'] == {'motion': 3, 'start': 2, 'yield_entry': 1, 'lane_change': 2}
    assert result['head_means'] == pytest.approx(manual)
    expected = sum(manual.values())
    assert result['unpenalized_sum'] == pytest.approx(expected)
    penalty = .001*np.sum(theta[:, 1:]**2)
    assert result['regularization'] == pytest.approx(penalty)
    assert result['objective'] == pytest.approx(expected+penalty)
    all_reference_mean = np.average(list(manual.values()), weights=[3, 2, 1, 2])
    assert abs(result['unpenalized_sum']-4*all_reference_mean) > 1e-3
    deduplicated_motion = np.mean([pool.marginal(theta, 'motion', y)['nll'] for y in (1.3, 2.4)])
    assert abs(result['head_means']['motion']-deduplicated_motion) > 1e-3


def test_composite_finite_snis_gradient_matches_fixed_directions_without_optimizer():
    objective = CompositeObjective({0: cached_pool()}, references())
    theta = .04*np.cos(np.arange(190)).reshape(5, 38)
    rng = np.random.default_rng(20261112)
    directions = rng.normal(size=(6, 5, 38))
    # Exercise every block's bias and one non-bias direction explicitly too.
    axes = np.eye(190)[[block*38+offset for block in range(5) for offset in (0, 17)]]
    directions = np.concatenate((directions, axes.reshape(10, 5, 38)))
    with physical_forbidden():
        actual = objective.evaluate(theta)['gradient']
        for direction in directions:
            direction = direction/np.linalg.norm(direction)
            h = 1e-5
            finite_difference = (objective.evaluate(theta+h*direction)['objective']-
                                 objective.evaluate(theta-h*direction)['objective'])/(2*h)
            assert finite_difference == pytest.approx(float(np.sum(actual*direction)), abs=2e-8)
    assert actual.shape == (5, 38)


def test_true_no_signal_law_has_zero_data_gradient_and_unpenalized_biases():
    pool = cached_pool(constant=True)
    refs = [dict(case_index=0, seed=SEEDS[0], head=head,
                 y=1. if head == 'motion' else 1,
                 manifest_ordinal=0, split='train') for head in HEADS]
    objective = CompositeObjective({0: pool}, refs)
    theta = np.zeros((5, 38))
    theta[:, 0] = np.linspace(-.5, .5, 5)
    with physical_forbidden():
        result = objective.evaluate(theta)
    assert result['regularization'] == 0.
    assert result['gradient'] == pytest.approx(np.zeros((5, 38)), abs=1e-14)
    assert result['zero_signal_references'] == 4
    theta[2, 7] = .4
    with physical_forbidden():
        penalized = objective.evaluate(theta)
    assert penalized['regularization'] == pytest.approx(.001*.4**2)
    expected = np.zeros((5, 38))
    expected[2, 7] = .002*.4
    assert penalized['gradient'] == pytest.approx(expected, abs=1e-14)


def test_development_labels_do_not_enter_the_training_objective():
    refs = references()
    development = [dict(ref, seed=SEEDS[3], split='development') for ref in refs]
    alternate = deepcopy(development)
    for ref in alternate:
        if ref['head'] == 'lane_change':
            ref['y'] = 1
        elif ref['head'] in ('start', 'yield_entry'):
            ref['y'] = 1-ref['y']
        else:
            ref['y'] = 1.8
    first = CompositeObjective({0: cached_pool()}, refs+development)
    second = CompositeObjective({0: cached_pool()}, refs+alternate)
    theta = np.zeros((5, 38))
    with physical_forbidden():
        a, b = first.evaluate(theta), second.evaluate(theta)
        old_dev = first.evaluate(theta, split='development')
        new_dev = second.evaluate(theta, split='development')
    assert a['objective'] == b['objective']
    assert np.array_equal(a['gradient'], b['gradient'])
    assert old_dev['unpenalized_sum'] != new_dev['unpenalized_sum']


def test_unsupported_selected_label_is_not_dropped_or_probability_floored():
    pool = cached_pool(constant=True)
    refs = [dict(case_index=0, seed=SEEDS[0], head=head,
                 y=1. if head == 'motion' else 1,
                 manifest_ordinal=0, split='train') for head in HEADS]
    refs.append(dict(refs[1], y=0, manifest_ordinal=1))
    objective = CompositeObjective({0: pool}, refs)
    with physical_forbidden(), pytest.raises(ValueError, match='Unsupported or nonfinite selected query'):
        objective.fun(np.zeros(190))
    assert len(objective.references) == 5


def test_nonfinite_coefficients_terminate_instead_of_returning_a_finite_penalty():
    objective = CompositeObjective({0: cached_pool()}, references())
    value = np.zeros(190)
    value[73] = np.nan
    with physical_forbidden(), pytest.raises(ValueError, match='Nonfinite coefficients'):
        objective.fun(value)


def test_partial_pool_gate_blocks_optimizer_before_loading_or_reusing_any_case(tmp_path, monkeypatch):
    import json
    particles = tmp_path/'particles'
    particles.mkdir()
    (particles/'summary.json').write_text(json.dumps(dict(status='HOLD_FIRST_FIT_PARTICLE_SUPPORT',
        cases=[dict(particle_path='missing_file_should_never_be_opened')], selected_references=512)))
    def forbidden_optimizer(*args, **kwargs):
        raise AssertionError('Optimizer was invoked without complete512-reference support')
    monkeypatch.setattr(fit, 'minimize', forbidden_optimizer)
    with physical_forbidden():
        result = fit.run(particles, tmp_path/'fit')
    assert result['status'] == 'HOLD_FIRST_FIT_PRECHECK'
    assert result['new_fits'] == result['optimizer_invocations'] == 0
    assert result['new_particle_full_forward_calls'] == result['new_native_starts'] == 0
    assert 'complete frozen cohort support gate' in result['error']


class CurrentOnlyReference(dict):
    def __getitem__(self, key):
        if key == 'y':
            raise AssertionError('Label read before cohort freeze')
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == 'y':
            raise AssertionError('Label read before cohort freeze')
        return super().get(key, default)


def test_cohort_sampling_is_fixed_current_only_and_retains_all16_strata():
    manifest = {}
    for seed in SEEDS:
        current = [CurrentOnlyReference(seed=seed, time_sec=index//10,
            packet_id='vehicle'+str(index), y='unread') for index in range(10000)]
        manifest[str(seed)] = {head: current for head in HEADS}
    selected = select_references(manifest)
    assert len(selected) == 512
    expected_rng = Random(20261112)
    for seed in SEEDS:
        for head in HEADS:
            rows = [ref for ref in selected if ref['seed'] == seed and ref['head'] == head]
            assert [ref['manifest_ordinal'] for ref in rows] == sorted(expected_rng.sample(range(10000), 32))
            assert len(rows) == 32 and len({ref['manifest_ordinal'] for ref in rows}) == 32
            assert all('y' not in ref for ref in rows)
            assert all(ref['split'] == ('development' if seed == SEEDS[3] else 'train') for ref in rows)
    assert sum(row['split'] == 'train' for row in selected) == 384
    assert select_references(manifest) == selected


def test_case_grouping_preserves_focal_identity_and_crosshead_references():
    refs = [dict(seed=SEEDS[0], time_sec=7, packet_id=packet,
                 head=head, manifest_ordinal=ordinal, split='train')
            for ordinal, (packet, head) in enumerate((('b', 'start'), ('a', 'motion'), ('a', 'lane_change')))]
    assert reference_key(refs[0]) != reference_key(refs[1])
    grouped = group_cases(refs)
    assert [reference_key(row['reference']) for row in grouped] == [
        (SEEDS[0], 7, 'a'), (SEEDS[0], 7, 'b')]
    assert [row['case_index'] for row in grouped] == [0, 1]
    assert sorted(ref['head'] for row in grouped for ref in row['references']) == sorted(ref['head'] for ref in refs)
    assert sum(len(row['references']) for row in grouped) == len(refs)


def test_zero_support_or_nonfinite_observation_prevents_whole_case_support():
    good = dict(defined=True, supported=True, gradient_finite=True, nll=.7)
    assert observations_supported([good, good])
    assert not observations_supported([])
    for change in ({'defined': False}, {'supported': False}, {'gradient_finite': False},
                   {'nll': None}, {'nll': float('inf')}):
        assert not observations_supported([good, dict(good, **change)])


def test_failed_partial_and_unstarted_cases_keep_every_reference_disposition():
    ref = dict(seed=SEEDS[0], time_sec=1, packet_id='a', head='motion',
               manifest_ordinal=0, split='train')
    registered = [dict(case_index=index, references=[dict(ref, manifest_ordinal=index)]) for index in range(4)]
    observed = dict(ref, defined=True, supported=False, gradient_finite=None, nll=None)
    completed = [dict(case_index=0, status='HOLD_FIRST_FIT_LABEL_SUPPORT', observations=[observed]),
                 dict(case_index=1, status='HOLD_FIRST_FIT_PARTICLE_CASE', observations=[])]
    unknown = [dict(case_index=2, physical_calls_lower_bound=128)]
    rows = reference_dispositions(registered, completed, unknown)
    assert len(rows) == 4
    assert [row['disposition'] for row in rows] == [
        'UNSUPPORTED_OR_NONFINITE', 'HOLD_FIRST_FIT_PARTICLE_CASE',
        'STARTED_COUNTERS_UNKNOWN', 'NOT_STARTED']
    assert [row['manifest_ordinal'] for row in rows] == list(range(4))


def test_receipt_recovery_retains_finished_workers_and_unknown_consumed_lower_bounds(tmp_path):
    import json
    ref = dict(seed=SEEDS[0], time_sec=1, packet_id='a')
    complete = dict(case_index=0, reference=ref, physical_calls=256, canonical_feature_evaluations=16,
                    status='PASS_FIRST_FIT_PARTICLE_CASE')
    directory = tmp_path/'case000_seed76116901_t1'
    directory.mkdir()
    (directory/'summary.json').write_text(json.dumps(complete))
    incomplete = tmp_path/'case001_seed76116901_t2'
    incomplete.mkdir()
    (incomplete/'started.json').write_text(json.dumps(dict(case_index=1, reference=dict(ref, time_sec=2))))
    (incomplete/'progress.json').write_text(json.dumps(dict(physical_calls=128, canonical_feature_evaluations=75)))
    returned = dict(complete, case_index=2, physical_calls=64)
    completed, unknown = collect_receipts(tmp_path, [returned])
    assert [row['case_index'] for row in completed] == [0, 2]
    assert sum(row['physical_calls'] for row in completed) == 320
    assert len(unknown) == 1 and unknown[0]['case_index'] == 1
    assert unknown[0]['physical_calls_lower_bound'] == 128
    assert unknown[0]['canonical_feature_evaluations_lower_bound'] == 75
    assert unknown[0]['consumed_counters_exact'] is False
