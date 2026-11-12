"""Small cached-law checks of the fixed, naturally weighted four-point readout."""

import gzip
import json
from math import log

import numpy as np
import pytest

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw
from experiments.tsc_packet_conditional_cache_20261110.verify import physical_evaluation_forbidden
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS
from experiments.tsc_packet_lc_negative_cache_20261111.evaluate import inspect_case
from experiments.tsc_packet_lc_negative_cache_20261111.prepare import task_from_row
from experiments.tsc_packet_lc_negative_cache_20261111.readout import (
    aggregate_records, matched_summary, read_case,
)
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import extended
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


WIDTH = 38


def frozen_points():
    zero = np.zeros((5, WIDTH))
    original = zero.copy()
    original[0, 0] = .2
    original[3, :2], original[4, :2] = [.1, .05], [-.1, -.02]
    frequency = original.copy()
    frequency[3:] = 0.
    frequency[3, 0], frequency[4, 0] = log(165/29669), log(166/29669)
    refitted = original.copy()
    refitted[3, :2], refitted[4, :2] = [4., .03], [.6, -.04]
    return dict(zero=zero, original=original, frequency_bias=frequency, refitted=refitted)


def outcome(label, survived=True):
    return dict(survived=survived, lane_change=label if survived else None,
                physical_speed_mps=1. if survived else None, started=None, entry_event=0)


def exact_law(*, stay_only=False):
    allowed = [1] if stay_only else [0, 1, 2]
    factors = [dict(kind='bernoulli', actor='ego', feature_id=0, block=0, bit=bit) for bit in (0, 1)]
    prefixes = [dict(parent=-1, factor_id=-1, log_prior=0.),
                dict(parent=0, factor_id=0, log_prior=0.), dict(parent=0, factor_id=1, log_prior=0.)]
    leaves = [dict(prefix_id=2, outcome_id=3)]
    for choice in allowed:
        factors.append(dict(kind='categorical', actor='ego', feature_id=0, allowed=allowed, choice=choice))
        leaves.append(dict(prefix_id=len(prefixes), outcome_id=choice))
        prefixes.append(dict(parent=1, factor_id=len(factors)-1, log_prior=0.))
    return CachedLaw(dict(focal_id='ego', features=[[1., .2]+[0.]*(WIDTH-2)],
        factors=factors, prefixes=prefixes, leaves=leaves,
        outcomes=[outcome('left'), outcome('stay'), outcome('right'), outcome(None, False)],
        stats=dict(physical_calls=2, leaf_visits=len(leaves)), old_stopped=False,
        stochastic_participants=['ego'], frame_actor_count=1,
        parameter_names=list(PARAMETERS), feature_names=list(FEATURE_NAMES),
        eligibility=dict(lane_change=True, motion=True, start=False, yield_entry=False)))


def mixed_law(anchor):
    data = dict(focal_id='ego', features=[[1., .2]+[0.]*(WIDTH-2)], stats={}, old_stopped=False,
        stochastic_participants=['other0', 'other1', 'other2', 'ego'], frame_actor_count=4,
        parameter_names=list(PARAMETERS), feature_names=list(FEATURE_NAMES),
        factors=[dict(kind='categorical', actor=f'other{i}', feature_id=0, allowed=[0, 1], choice=choice)
                 for i in range(3) for choice in (0, 1)],
        focal_groups=[dict(feature_id=0, allowed=[0, 1, 2]), dict(feature_id=0, allowed=[1, 2])],
        particles=[], proposal=dict(components={'0':'q0', '1':'qf'},
            seeds={'0':123, '1':456}, component_counts={'0':128, '1':128},
            design_weights={'0':.5, '1':.5}, component_order=[0, 1]),
        eligibility=dict(lane_change=True, motion=True, start=False, yield_entry=False))
    patterns = (([0, 2, 4], 0, True), ([1, 3, 5], 0, True),
                ([0, 2, 5], 1, True), ([], -1, False))
    for i in range(256):
        factors, group, survived = patterns[i % 4]
        data['particles'].append(dict(factor_ids=factors, focal_group=group,
            outcome=outcome('stay', survived), proposal_component=i % 2))
    ratios = ParticleLaw(data)._evaluate(anchor)[0]
    for i, row in enumerate(data['particles']):
        row['proposal_log_ratio'] = float(ratios[i]) if i % 2 else 0.
    return MixtureParticleLaw(data, anchor)


def task(index, y, weight, method='exact'):
    return dict(case_index=index, seed=76116901, time_sec=10+index, packet_id='ego', y=y,
        reference=dict(seed=76116901, time_sec=10+index, packet_id='ego'),
        raw_weight=weight*30000, objective_weight=weight, proposed_method=method,
        directions={direction:dict(matched_positive_keys=[]) for direction in ('left', 'right')})


def reused_task(tmp_path, law, method):
    original = task(0, 1, 1., method)
    original.update(head='lane_change', split='train')
    path = tmp_path/'paid_cache_server_only.json.gz'
    with gzip.open(path, 'wt') as stream:
        json.dump(law.data, stream)
    anchor = tmp_path/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    anchor.parent.mkdir(parents=True)
    anchor.write_text('{}')
    actors = law.data['stochastic_participants']
    physical = 2 if method == 'exact' else 512
    reuse = dict(kind='exact' if method == 'exact' else 'mixture', particle_path=str(path),
        particle_count=None if method == 'exact' else 256, reference=original['reference'],
        stochastic_participants=actors, frame_actor_count=law.data['frame_actor_count'],
        anchor_file=str(anchor), seed_zero=123, seed_fitted=456)
    row = dict(original, required_participants=actors, required_actor_count=len(actors),
        frame_actor_count=law.data['frame_actor_count'], eligibility=law.data['eligibility'],
        symbolic_status='REGISTERED_SYMBOLIC_UPPER_BOUND',
        symbolic_budget=dict(status='REGISTERED_SYMBOLIC_UPPER_BOUND',
            per_actor={actor:dict(current_positive_goals=1) for actor in actors},
            physical_resolver_calls_upper_bound=physical, lateral_leaf_branches_upper_bound=4),
        current_goal_combinations=1, all_exact_full_forward_ceiling=physical,
        all_exact_feature_ceiling=len(actors), all_exact_leaf_ceiling=4,
        all_mixture_full_forward_ceiling=256, all_mixture_feature_ceiling=len(actors),
        proposed_reused_cache=reuse, proposed_new_full_forward_ceiling=0,
        proposed_new_feature_ceiling=0, proposed_new_exact_leaf_ceiling=0)
    return task_from_row(row, original, 0, tmp_path), path


@pytest.mark.parametrize('method', ['exact', 'mixture256'])
def test_paid_exact_and_mixture_caches_are_read_with_zero_new_construction(tmp_path, method):
    points = frozen_points()
    law = exact_law() if method == 'exact' else mixed_law(points['original'])
    prepared, path = reused_task(tmp_path, law, method)
    assert prepared['physical_limit'] == prepared['feature_limit'] == prepared['leaf_limit'] == 0
    assert prepared['reused_exact_path'] == (str(path) if method == 'exact' else None)
    if method == 'mixture256':
        assert (prepared['seed_zero'], prepared['seed_fitted']) == (123, 456)
    # No worker kernel exists in this process. A misplaced construction path fails here.
    with physical_evaluation_forbidden():
        row = inspect_case(prepared, points, tmp_path)
    assert row['status'] == 'COMPLETE_STRATIFIED_CACHE_CASE'
    assert row['reused_cache']
    assert row['cache_path'] == str(path)
    assert row['physical_calls'] == row['canonical_feature_evaluations'] == row['leaf_visits'] == 0
    assert not (tmp_path/prepared['stem']/'exact_cache_server_only.json.gz').exists()
    assert not (tmp_path/prepared['stem']/'mixture_cache_server_only.json.gz').exists()


def test_four_point_natural_weight_aggregation_matches_direct_laws_and_keeps_large_raw_ratios():
    points = frozen_points()
    laws = [exact_law(), mixed_law(points['original']), exact_law()]
    tasks = [task(0, 0, 1/30000), task(1, 1, 29998/30000, 'mixture256'), task(2, 2, 1/30000)]
    records, gradients = [], {}
    with physical_evaluation_forbidden():
        for law, reference in zip(laws, tasks, strict=True):
            row, private = read_case(law, points, reference)
            records.append(row)
            gradients[reference['case_index']] = private
        result = aggregate_records(records, gradients, points, expected_references=3)
        assert result['native_support_passed']
        assert result['native']['reference_count'] == 3
        assert result['native']['objective_weight_sum'] == pytest.approx(1.)
        assert set(result['native']['points']) == set(points)
        for point, theta in points.items():
            native = [law.marginal(theta, 'lane_change', reference['y'])
                      for law, reference in zip(laws, tasks, strict=True)]
            loss = sum(reference['objective_weight']*value['nll']
                       for reference, value in zip(tasks, native, strict=True))
            gradient = sum((reference['objective_weight']*value['gradient']
                            for reference, value in zip(tasks, native, strict=True)), np.zeros((5, WIDTH)))
            summary = result['native']['points'][point]
            penalty = .001*np.sum(theta[3:, 1:]**2)
            assert summary['lc_regularization'] == pytest.approx(penalty, abs=1e-14)
            for view in ('primary', '128'):
                value = summary[view]
                assert value['all_references_supported']
                assert value['weighted_nll_contribution']['value'] == pytest.approx(loss, abs=1e-13)
                assert value['penalized_objective']['value'] == pytest.approx(loss+penalty, abs=1e-13)
                assert value['weighted_gradient']['weighted_gradient_l2'] == pytest.approx(np.linalg.norm(gradient), abs=1e-12)
                for reference, expected in zip(tasks, native, strict=True):
                    group = value['by_class'][str(reference['y'])]
                    assert group['objective_weight'] == pytest.approx(reference['objective_weight'])
                    assert group['conditional_weighted_mean_nll']['value'] == pytest.approx(expected['nll'], abs=1e-13)
            assert result['native']['mixed_size_sensitivity'][point]['value'] == pytest.approx(0., abs=1e-13)
        for point in ('frequency_bias', 'refitted'):
            assert result['native']['points'][point]['primary']['mixture_diagnostics']['maximum_raw_weight'] > 2.
        assert result['native']['points']['original']['primary']['mixture_diagnostics']['maximum_raw_weight'] <= 2.
        with pytest.raises(ValueError):
            aggregate_records(records[:-1], gradients, points, expected_references=3)


def test_native_support_failure_remains_in_full_loss_while_zero_direction_probes_are_allowed():
    points = frozen_points()
    with physical_evaluation_forbidden():
        unsupported, private = read_case(exact_law(stay_only=True), points, task(0, 0, 1.))
        result = aggregate_records([unsupported], {0:private}, points, expected_references=1)
        supported, _ = read_case(exact_law(stay_only=True), points, task(1, 1, 1.))
    assert not unsupported['native_support_passed']
    assert result['native']['reference_count'] == 1
    assert not result['native_support_passed']
    assert result['unpaired'] is result['matched'] is None
    for point in points:
        value = result['native']['points'][point]['primary']
        assert value['unsupported_references'] == 1
        assert value['weighted_nll_contribution']['state'] == 'positive_infinity'
        assert not value['weighted_gradient']['defined']
        assert supported['points'][point]['0']['direction_probability'] == 0.
        assert supported['points'][point]['2']['direction_probability'] == 0.
    assert supported['native_support_passed']


def probability_probe(probability):
    logp = log(probability) if probability else -float('inf')
    return dict(direction_log_probability=extended(logp), allowed_log_mass=extended(0. if probability else -float('inf')),
        conditional_log_probability=extended(logp if probability else None), mean_category_score=logp if probability else -.4,
        mean_allowed_category_score=logp if probability else None)


def test_matched_means_and_pair_ratio_keep_distinct_denominators_and_explicit_missing_cases():
    records = []
    # A and B share one stay; C has no selected neighbour. W is a zero-permission stay.
    for index, (label, weight, probability) in enumerate(
            ((0, 1., .6), (0, 1., .1), (0, 1., .4),
             (1, 1., .2), (1, 3., .8), (1, 6., .05), (1, 2., 0.))):
        row = task(index, label, weight/30000)
        row['points'] = {'frozen':{'0':probability_probe(probability)}}
        records.append(row)
    keys = [[row['seed'], row['time_sec'], row['packet_id']] for row in records[:3]]
    for row, neighbors in zip(records[3:], ([keys[0]], [keys[0]], keys[:2], [keys[1]]), strict=True):
        row['directions']['left']['matched_positive_keys'] = neighbors
    group = matched_summary(records, 'frozen')['left']['all']
    assert group['positive_count'] == 3
    assert group['positive_with_selected_neighbors'] == 2
    assert group['zero_neighbor_positive_count'] == 1
    assert group['registered_positive_stay_edges'] == 5
    ordinary = group['metrics']['direction_log_probability']
    assert ordinary['equal_positive_mean_of_weighted_neighbor_concordance'] is None
    assert ordinary['covered_positive_mean_of_weighted_neighbor_concordance'] == pytest.approx(.85)
    assert ordinary['explicitly_defined_positive_count'] == 2
    assert ordinary['pair_weighted_concordance_ratio'] == pytest.approx(5/6)
    assert ordinary['pair_weight_denominator'] == pytest.approx(18/30000)
    assert ordinary['pair_comparison_counts'] == dict(wins=4, ties=0, losses=1, undefined=0)
    conditional = group['metrics']['conditional_log_probability']
    assert conditional['covered_positive_mean_of_weighted_neighbor_concordance'] is None
    assert conditional['explicitly_defined_positive_count'] == 1
    assert conditional['explicitly_defined_positive_mean_of_weighted_neighbor_concordance'] == pytest.approx(.7)
    assert conditional['pair_weighted_concordance_ratio'] is None
    assert conditional['pair_weight_denominator'] == pytest.approx(18/30000)
    assert conditional['defined_pair_weight_denominator'] == pytest.approx(16/30000)
    assert conditional['explicitly_defined_pair_weighted_concordance_ratio'] == pytest.approx(13/16)
    assert conditional['pair_comparison_counts']['undefined'] == 1
