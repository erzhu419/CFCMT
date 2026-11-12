"""Pure differences for 904 calibration, prefix views and frozen cache tasks."""

from copy import deepcopy
from math import log

import pytest

from experiments.tsc_packet_lc_904_evaluation_20261111.prepare import task_from_row
from experiments.tsc_packet_lc_904_evaluation_20261111.readout import aggregate_probability


POINTS = ('old_lc', 'new_lc', 'frequency_bias')


def extended_log(probability):
    return dict(state='finite', value=log(probability)) if probability else dict(state='negative_infinity', value=None)


def probability(values, label, *, zero_left_permission=False):
    return dict(class_probabilities=list(values), class_log_probabilities=[extended_log(p) for p in values],
        allowed_mass={'0':0. if zero_left_permission else 1., '2':1.},
        conditional_log_probability={'0':dict(state='undefined', value=None) if zero_left_permission else extended_log(values[0]),
                                     '2':extended_log(values[2])},
        brier=sum((p-float(k == label))**2 for k,p in enumerate(values)),
        normalization_error=abs(sum(values)-1.), normalization_passed=True)


def saved_records():
    records = []
    training_frequency = (165/30000, 29669/30000, 166/30000)
    for label, weight, values in ((0, 56., (.6, .3, .1)), (1, 9883., (.1, .8, .1)), (2, 61., (.2, .3, .5))):
        new_values = (.7, .2, .1) if label == 0 else values
        row = dict(y=label, raw_weight=weight, objective_weight=weight/10000,
            proposed_method='mixture256' if label == 1 else 'exact',
            readouts={'primary':{'points':{
                'old_lc':dict(probability=probability(values, label)),
                'new_lc':dict(probability=probability(new_values, label)),
                'frequency_bias':dict(probability=probability(training_frequency, label))}}})
        if label == 1:
            row['readouts']['128'] = deepcopy(row['readouts']['primary'])
            for point in ('old_lc', 'new_lc'):
                row['readouts']['128']['points'][point]['probability'] = probability((.12, .76, .12), label)
        records.append(row)
    return records


def test_multiclass_brier_natural_class_totals_and_exact_retaining_prefix_sensitivity():
    records = saved_records()
    old = aggregate_probability(records, 'old_lc')
    new = aggregate_probability(records, 'new_lc')
    bias = aggregate_probability(records, 'frequency_bias')
    assert old['brier'] == pytest.approx(.063072)
    assert new['brier'] == pytest.approx(.0624)
    assert old['predicted_class_totals'] == pytest.approx({'0':1034.1, '1':7941.5, '2':1024.4})
    assert old['observed_class_totals'] == pytest.approx({'0':56., '1':9883., '2':61.})
    assert sum(old['predicted_class_totals'].values()) == pytest.approx(10000.)
    # The frozen training-frequency bias is evaluated on10000 rows, not re-estimated from904 labels.
    assert bias['predicted_class_totals'] == pytest.approx({'0':55., '1':10000*29669/30000, '2':10000*166/30000})
    prefix = aggregate_probability(records, 'old_lc', view='128')
    assert prefix['brier']-old['brier'] == pytest.approx(.02609112)
    assert prefix['observed_class_totals'] == old['observed_class_totals']
    assert {key:prefix['predicted_class_totals'][key]-old['predicted_class_totals'][key] for key in ('0','1','2')} == pytest.approx(
        {'0':197.66, '1':-395.32, '2':197.66})


def test_zero_permission_stay_keeps_marginal_auc_and_explicit_conditional_missing_weight():
    records = saved_records()
    zero = records[1]
    zero['raw_weight'], zero['objective_weight'] = 9882., 9882/10000
    ordinary = deepcopy(zero)
    ordinary['raw_weight'], ordinary['objective_weight'] = 1., 1/10000
    for view in zero['readouts'].values():
        for point in POINTS:
            view['points'][point]['probability'] = probability((0., .8, .2), 1, zero_left_permission=True)
    records.append(ordinary)
    result = aggregate_probability(records, 'old_lc')['discrimination']['left']
    assert result['weighted_auc'] == 1.
    assert result['conditional_weighted_auc'] is None
    subset = result['explicitly_defined_conditional_auc']
    assert subset['auc'] == 1.
    assert subset['excluded_stay_count'] == 1
    assert subset['excluded_stay_weight'] == pytest.approx(9882/10000)


def registered_task(tmp_path, method, *, physical=512, reuse=True, index=5):
    original = dict(case_index=index, seed=76116904, time_sec=10, packet_id='ego', y=1,
        head='lane_change', split='regression', raw_weight=1., objective_weight=1/10000,
        inclusion_probability=1., directions={'left':{}, 'right':{}})
    actors = ['a', 'b', 'c', 'ego']
    anchor = tmp_path/'target_packet_particle_first_fit/fit/coefficients_server_only.json'
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text('{}')
    paid = tmp_path/f'paid_{method}_server_only.json.gz'
    paid.write_bytes(b'')
    receipt = dict(kind='exact' if method == 'exact' else 'mixture', particle_path=str(paid),
        particle_count=None if method == 'exact' else 256,
        reference={name:original[name] for name in ('seed', 'time_sec', 'packet_id')},
        stochastic_participants=actors, frame_actor_count=4,
        anchor_file=str(anchor), seed_zero=123, seed_fitted=456) if reuse else None
    features = len(actors)
    row = dict(original, required_participants=actors, required_actor_count=len(actors), frame_actor_count=4,
        eligibility=dict(lane_change=True), symbolic_status='REGISTERED_SYMBOLIC_UPPER_BOUND',
        symbolic_budget=dict(status='REGISTERED_SYMBOLIC_UPPER_BOUND',
            per_actor={actor:dict(current_positive_goals=1) for actor in actors},
            physical_resolver_calls_upper_bound=physical, lateral_leaf_branches_upper_bound=900),
        current_goal_combinations=1, all_exact_full_forward_ceiling=physical,
        all_exact_feature_ceiling=features, all_exact_leaf_ceiling=900,
        all_mixture_full_forward_ceiling=256, all_mixture_feature_ceiling=features,
        available_caches=dict(exact=[receipt] if reuse and method == 'exact' else [], q0=[],
                              mixture=[receipt] if reuse and method == 'mixture256' else []),
        proposed_method=method, proposed_reused_cache=receipt,
        proposed_new_full_forward_ceiling=0 if reuse else physical if method == 'exact' else 256,
        proposed_new_feature_ceiling=0 if reuse else features,
        proposed_new_exact_leaf_ceiling=0 if reuse or method != 'exact' else 900)
    return task_from_row(row, original, index, tmp_path)


def test_904_tasks_keep_paid_preference_original_proposal_seeds_and_10000_weights(tmp_path):
    exact = registered_task(tmp_path, 'exact', physical=512)
    assert exact['proposed_method'] == 'exact'
    assert exact['physical_limit'] == exact['feature_limit'] == exact['leaf_limit'] == 0
    assert exact['objective_weight'] == 1/10000
    mixed = registered_task(tmp_path, 'mixture256', physical=128)
    assert mixed['physical_limit'] == 0
    assert (mixed['seed_zero'], mixed['seed_fitted']) == (123, 456)
    missing = registered_task(tmp_path, 'mixture256', reuse=False, index=6)
    assert missing['physical_limit'] == 256
    assert (missing['seed_zero'], missing['seed_fitted']) == (20261210+600000, 20261211+600000)
