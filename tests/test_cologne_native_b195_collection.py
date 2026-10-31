from copy import deepcopy
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


@pytest.fixture
def collector(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    return importlib.import_module('collect_cologne_native_b195_addition')


def test_audit_outcome_keeps_original450_costs_aggregates_and_continues_collisions(collector, monkeypatch):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    api = SimpleNamespace(now=25210.)
    api.simulation = SimpleNamespace(getTime=lambda: api.now)
    lanes = tuple(f'lane{index}' for index in range(8))
    calls, collide = [], [False]
    monkeypatch.setattr(v3, '_candidate_for_state', lambda info, phase: SimpleNamespace(state=phase))
    monkeypatch.setattr(v3, '_rule_candidate', lambda *args: SimpleNamespace(state='PP'))
    monkeypatch.setattr(v3, '_read_local_state', lambda *args: api.now)
    monkeypatch.setattr(v3, '_candidate_aggregates', lambda state, candidate: {
        key: state + (1 if candidate.state == 'chosen' else 0)
        for key in ('total_q', 'green_q', 'red_q', 'green_down_occ', 'mean_speed')})

    def advance(**kwargs):
        calls.append((kwargs['actions'][collector.TLS], kwargs['collision_mode']))
        assert kwargs['cost_lanes'] == lanes
        start = api.now
        api.now += 10
        if collide[0] and len(calls) == 1:
            kwargs['safety_ledger']['raw_collision_events'] += 1
            if kwargs['collision_mode'] == 'fail':
                raise AssertionError('Collision must not truncate the new labels')
        return [8.*(time-start) for time in np.arange(start+1, start+11)]

    monkeypatch.setattr(v3, '_advance_with_actions', advance)
    infos, executors, states = {collector.TLS: None}, {collector.TLS: None}, {collector.TLS: 25210.}
    old = v3._counterfactual_branch_outcome_v3(sumo_api=api, infos=infos, states=states,
        executors=executors, focal_id=collector.TLS, candidate=SimpleNamespace(state='chosen'),
        context=np.asarray([8.]), control_interval_sec=10, counterfactual_horizon_intervals=45,
        controlled_lane_count=8, controlled_lanes=lanes, counterfactual_cost_mode='halted_queue')
    api.now, calls[:], collide[0] = 25210., [], True
    traces = {}
    result = collector.audit_native_outcome(api, v3, infos, np.asarray([8.]), executors, states,
                                            'chosen', lanes, traces, 'action')
    assert result['status'] == 'COMPLETE' and result['sample_count'] == 450
    assert result['end_time_sec'] == 25660.
    assert result['safety']['raw_collision_events'] == 1
    assert traces['action'] == old[0]
    assert result['one_step'] == old[1] and result['terminal'] == old[2]
    assert result['one_step']['total_q'] == 25221.
    assert result['terminal']['total_q'] == 25661.
    assert calls == [('chosen', 'audit')] + [('PP', 'audit')]*44


def test_all_eight_fresh_prefixes_remain_attempted_when_one_branch_is_invalid(collector, monkeypatch):
    phases = [f'phase{index}' for index in range(8)]
    group = f'cologne1:seed2027:1:{collector.TLS}'
    original = MechanismDataset(feature_names=('queue',), features=np.arange(8.)[:, None],
        context_names=('lanes',), context=np.full((8, 1), 8.), targets={}, priors={},
        domains=np.full(8, 'cologne1'), metadata={'action_group_ids': [group]*8,
            'candidate_states': phases, 'row_times': [25210]*8})
    spec = {'group_id': group, 'seed': 2027, 'interval_index': 1, 'time_sec': 25210}
    api = SimpleNamespace(now=25210., closed=0)
    api.simulation = SimpleNamespace(getTime=lambda: api.now, getPendingVehicles=lambda: [])
    def close():
        api.closed += 1
    api.close = close
    prefixes, actions = [], []
    safety = dict(raw_collision_events=1, no_teleport_passed=True)
    executor = SimpleNamespace(feasible_states_now=lambda: phases)
    def prefix(api, v3, cfg, seed, interval):
        api.now = 25210.
        prefixes.append((seed, interval))
        return {collector.TLS: None}, np.asarray([8.]), {collector.TLS: executor}, {collector.TLS: None}, safety
    monkeypatch.setattr(collector, 'native_prefix', prefix)
    monkeypatch.setattr(collector, 'physical_snapshot', lambda *args: {
        'time_sec': 25210., 'vehicles': {}, 'executors': {}, 'pending_raw': 0})
    v3 = SimpleNamespace(FEATURE_NAMES_V3=('queue',), _controlled_lane_set=lambda infos: range(8),
        _candidate_for_state=lambda info, phase: SimpleNamespace(state=phase),
        candidate_features_v3=lambda state, candidate, executor, **kwargs: [float(candidate.state[-1])])
    def outcome(api, v3, infos, context, executors, states, phase, lanes, traces, key):
        actions.append(phase)
        api.now += 450
        return {'status': 'INVALID' if phase == 'phase2' else 'COMPLETE', 'sample_count': 449 if phase == 'phase2' else 450,
                'end_time_sec': api.now, 'safety': safety, 'cost': 1.}
    monkeypatch.setattr(collector, 'audit_native_outcome', outcome)
    result = collector.collect_group(spec, original, {'sumocfg': 'fixed.sumocfg'}, api, v3, {})
    assert prefixes == [(2027, 1)]*8 and actions == phases and api.closed == 8
    assert result['status'] == 'INVALID' and len(result['native_branches']) == 8
    assert sum(branch['status'] == 'COMPLETE' for branch in result['native_branches']) == 7
    assert all(branch['prefix_safety']['raw_collision_events'] == 1 for branch in result['native_branches'])
    assert collector.compact_group(result)['prefix_replay_sec'] == 80.


def test_six_seed_shards_cover_the_fixed95_additions_once_and_never_replace_ids(collector):
    specs = [{'group_id': f'cologne1:seed{seed}:{index}:{collector.TLS}', 'seed': seed,
              'interval_index': index, 'time_sec': 25200+10*index, 'bank_path': 'original.npz'}
             for seed, count in ((2027, 32), (3037, 32), (4047, 31)) for index in range(count)]
    protocol = {'groups': specs, 'additional_group_ids': [spec['group_id'] for spec in specs],
                'original_group_ids': [f'original_{index}' for index in range(100)]}
    shards = [collector.shard_groups(protocol, seed, shard) for seed in (2027, 3037, 4047) for shard in (0, 1)]
    combined = [spec['group_id'] for shard in shards for spec in shard]
    assert len(combined) == len(set(combined)) == 95
    assert set(combined) == set(protocol['additional_group_ids'])
    assert [len(shard) for shard in shards] == [16, 16, 16, 16, 16, 15]
    changed = deepcopy(protocol)
    changed['groups'][0]['time_sec'] += 10
    with pytest.raises(ValueError, match='capture time'):
        collector.shard_groups(changed, 2027, 0)
