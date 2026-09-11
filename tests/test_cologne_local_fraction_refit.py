from copy import deepcopy

import numpy as np
import pytest

from scripts.data import run_cologne_local_fraction_refit as local
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    COUNTERFACTUAL_SAFETY_PROTOCOL_V3, STATE_SNAPSHOT_PROTOCOL_V3,
    SUMO_EXECUTION_PROTOCOL, STRICT_SAFETY_MONITORING_V3, WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    merge_counterfactual_datasets_v3,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL


def group(seed, interval):
    return f'cologne1:seed{seed}:{interval}:{local.TLS}'


def test_temporal_requests_exclude_all_previous_requests_even_when_censored():
    previous = [group(2027, 6), group(2027, 11), group(2027, 12)]
    actual = local.new_group_requests(2027, previous)
    assert len(actual) == 69
    assert not set(previous) & set(actual)
    assert actual == [group(2027, i) for i in range(16, 360, 5)]


def test_local_b100_is_seed_balanced_time_covering_and_independent_of_row_order():
    pool = [group(seed, i) for seed in local.TRAIN_SEEDS for i in range(6, 360, 5)]
    selected, audit = local.select_local_b100(pool)
    assert len(selected) == len(set(selected)) == 100
    assert local.select_local_b100(list(reversed(pool))) == (selected, audit)
    for seed, quota in local.QUOTAS.items():
        expected_positions = np.linspace(0, 70, quota).round().astype(int)
        expected = [group(seed, 6 + 5 * int(i)) for i in expected_positions]
        assert audit[str(seed)]['selected_groups'] == expected
        assert expected[0] == group(seed, 6)
        assert expected[-1] == group(seed, 356)
    with pytest.raises(ValueError, match='below fixed quota'):
        local.select_local_b100([g for g in pool if not g.startswith('cologne1:seed2027:')]
                               + [group(2027, i) for i in range(33)])


def dataset_fixture():
    retained = [group(2027, 6), group(2027, 11)]
    censored, unavailable = group(2027, 16), group(2027, 21)
    requested = [*retained, censored, unavailable]
    safety = {'protocol': COUNTERFACTUAL_SAFETY_PROTOCOL_V3, 'candidate_groups': 3,
        'retained_groups': 2, 'censored_groups': 1, 'candidate_branches': 6,
        'attempted_branches': 5, 'retained_branches': 4, 'censored_branches': 2,
        'unevaluated_censored_branches': 1, 'observed_unsafe_branches': 1,
        'raw_collision_events_observed': 2, 'collision_event_steps_observed': 2,
        'unique_collision_incidents_observed': 1, 'collision_incident_signatures': ['retained-incident'],
        'collision_samples': [], 'censored_group_fraction': 1/3, 'censored_group_ids': [censored],
        'starting_teleports': 0, 'ending_teleports': 0, 'no_teleport_passed': True,
        'symmetric_group_censoring_passed': True}
    targets = {'interval_cost': np.arange(4, dtype=float), 'prefix_mean_cost_450s': np.arange(4, dtype=float)}
    meta = {**deepcopy(local.SETTINGS), 'scenario': 'cologne1', 'seed': 2027, 'sumocfg': 'fixed.sumocfg',
        'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'counterfactual_horizon_sec': 450, 'local_mechanism_horizon_sec': 10, 'rollout_value_horizon_sec': 450,
        'counterfactual_estimand': WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        'counterfactual_cost_population': 'sumo_last_step_halting_number_on_controlled_lanes',
        'state_snapshot_protocol': STATE_SNAPSHOT_PROTOCOL_V3, 'sumo_execution_protocol': SUMO_EXECUTION_PROTOCOL,
        'strict_safety_monitoring': STRICT_SAFETY_MONITORING_V3,
        'action_group_ids': [retained[0]] * 2 + [retained[1]] * 2,
        'selected_action_group_ids': requested, 'row_tls': [local.TLS] * 4,
        'row_times': [25260, 25260, 25310, 25310], 'candidate_states': ['Gr', 'rG'] * 2,
        'mechanism_output_names': list(targets), 'counterfactual_safety_audit': safety,
        'counterfactual_replay_audit': {'passed': True, 'checks': 1, 'max_abs_difference': 0.},
        'behavior_safety_audit': {'raw_collision_events': 0, 'collision_event_steps': 0,
            'unique_collision_incidents': 0, 'starting_teleports': 0, 'ending_teleports': 0,
            'collision_samples': [], 'no_teleport_passed': True}}
    data = MechanismDataset(feature_names=('green_down_occ',), features=np.zeros((4, 1)),
        context_names=('context',), context=np.ones((4, 1)), priors=deepcopy(targets), targets=targets,
        domains=np.asarray(['cologne1'] * 4), metadata=meta)
    return data, requested


def test_complete_collection_can_retain_symmetric_censor_and_unavailable_groups():
    data, requested = dataset_fixture()
    audit = local.audit_dataset(data, 2027, requested, 'fixed.sumocfg')
    assert set(audit['retained_groups']) == {group(2027, 6), group(2027, 11)}
    assert audit['censored_groups'] == [group(2027, 16)]
    assert audit['unavailable_groups'] == [group(2027, 21)]
    assert 'counterfactual_cache_identity' not in data.metadata
    with pytest.raises(ValueError, match='collide across datasets'):
        merge_counterfactual_datasets_v3([data, data])


@pytest.mark.parametrize('change', ['units', 'target', 'horizon', 'network', 'teleport', 'asymmetric', 'retained_censored'])
def test_real_metadata_target_and_safety_failures_cannot_enter_pool(change):
    data, requested = dataset_fixture()
    if change == 'units': data.metadata['occupancy_equation_protocol'] = 'old-equations'
    elif change == 'target': data.targets['prefix_mean_cost_450s'][0] += .1
    elif change == 'horizon': data.metadata['rollout_value_horizon_sec'] = 30
    elif change == 'network': data.metadata['sumocfg'] = 'other.sumocfg'
    elif change == 'teleport': data.metadata['behavior_safety_audit']['starting_teleports'] = 1
    elif change == 'asymmetric': data.metadata['counterfactual_safety_audit']['symmetric_group_censoring_passed'] = False
    else: data.metadata['counterfactual_safety_audit']['censored_group_ids'] = [group(2027, 6)]
    with pytest.raises(ValueError):
        local.audit_dataset(data, 2027, requested, 'fixed.sumocfg')


def test_shared_evaluator_uses_new_protocol_without_changing_old_driver():
    from scripts.data import run_cologne_fraction_refit as old
    assert old.PROTOCOL == local.PREVIOUS_PROTOCOL
    assert local.shared.evaluate.__globals__['PROTOCOL'] == local.PROTOCOL
