"""The new cohort ends source arms at ten seconds and counts discovery once."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor
from experiments.tsc_response_cost_readout_20261109.advance_native import advance_horizon
from test_repaired_action_cost_advance import setup


@pytest.fixture
def collector(monkeypatch):
    directory = Path(__file__).resolve().parents[1]/'experiments'/'tsc_response_cost_readout_20261109'
    monkeypatch.syspath_prepend(str(directory))
    return importlib.import_module('experiments.tsc_response_cost_readout_20261109.collect_native')


def test_source_stops_at_ten_before_a_second_request_and_has_no_450_cost(monkeypatch):
    api, producer, observer, args = setup()

    def no_continuation(*args):
        raise AssertionError('A source arm must not request PP continuation at t+10')

    monkeypatch.setattr(producer, '_read_graph_states_v3', no_continuation)
    result = advance_horizon(**args, horizon_sec=10)
    assert all(result['checks'].values())
    assert api.time == result['completed_steps'] == 10
    assert producer.calls == list(range(10))
    assert all(ex.requests == [(0, args['initial_actions'][tls])]
               for tls, ex in api.executors.items())
    assert len(observer.samples) == 11
    assert result['cost'] is None
    assert set(result['endpoint_states']) == {'0', '10'}
    assert result['integrals'] == {'0': 0, '10': sum(result['halted_counts'])}
    assert result['segments'] == {'0_10': result['integrals']['10']}
    assert len(result['actual_signals']) == len(result['population_ledger']['per_second']) == 10
    assert result['endpoint_states']['10']['role'] == 'future_measurement_label'


def test_nominal_clone_preserves_live_executor_and_never_reads_future_occupancy(collector):
    writes = []

    def forbidden(*args, **kwargs):
        raise AssertionError('Nominal input must not query or advance the native simulation')

    api = SimpleNamespace(trafficlight=SimpleNamespace(
        setRedYellowGreenState=lambda tls, state: writes.append((tls, state))),
        lane=SimpleNamespace(getLastStepVehicleNumber=forbidden), simulationStep=forbidden)
    live = SafePhaseExecutor(sumo_api=api, tls_id='signal',
        timings=[PhaseTiming(state, 30., 3., 3.) for state in ('Gr', 'rG')],
        initial_state='Gr', initial_green_elapsed_sec=40., clearance_lane_ids=('internal',))
    before = deepcopy(live.snapshot())
    nominal = collector.nominal_plan(live, 'rG')
    assert writes == [] and live.snapshot() == before
    assert nominal['request_accepted'] and len(nominal['states']) == 10
    assert nominal['executor10']['snapshot']['current_green_state'] == 'rG'
    assert nominal['executor10']['snapshot']['green_elapsed_sec'] == 4.
    assert nominal['executor10']['feasible_states'] == ['rG']


def test_discovery_is_one_of_eight_source_arms_each_closed_and_roster_is_224(
        collector, tmp_path, monkeypatch):
    protocol = json.loads((collector.HERE/'protocol.json').read_text())
    roster = collector.cases(protocol, tmp_path)
    assert len(roster) == 28 and len(roster)*8 == 224
    assert sum(row['horizon_sec'] == 10 for row in roster) == 8
    assert sum(row['horizon_sec'] == 450 for row in roster) == 20
    monkeypatch.setitem(sys.modules, 'cologne_action_branch_checks', SimpleNamespace(
        compare_physical=lambda a, b, tolerance: {'passed': a == b}))
    monkeypatch.setitem(sys.modules, 'recover_tsc_joint_response_trial', SimpleNamespace(
        revised_audit=lambda *args: ({'prefix_no_collision': False,
            'branch_no_collision': False, 'execution_conforms_to_clearance_rule': True}, {})))
    calls, closes = [], []
    api = SimpleNamespace(simulation=SimpleNamespace(getTime=lambda: 630),
                          close=lambda: closes.append(1))

    def branch(api, v3, case, candidate, output, lane_static):
        assert case['horizon_sec'] == 10
        assert ('resolved_checkpoint_sec' not in case if candidate is None
                else case['resolved_checkpoint_sec'] == 620)
        calls.append(candidate)
        index = 5 if candidate is None else candidate
        return dict(candidate_index=index, reference_index=5, candidate_states=list('abcdefgh'),
            resolved_checkpoint_sec=620, start={'pending_ids': [], 'vehicles': ['same']},
            checks={'complete': True}, observation={'frames': [{'offset_sec': 0}]},
            preaction={'movement_ids': ['m']}, network_preaction={'state': 'same'},
            inputs={'predecision_only': True}, targets={'y': [[index, 0, 0, 0]]},
            cost=None, integrals={'0': 0, '10': index},
            prefix_safety={'raw_collision_events': 0}, safety={'raw_collision_events': 1},
            future={'error': None})

    monkeypatch.setattr(collector, 'collect_branch', branch)
    result = collector.collect_case(api, None, dict(roster[0], checkpoint_sec=600),
                                    tmp_path/'group', {}, {}, [])
    assert result['status'] == 'COMPLETE'
    assert calls == [None, 0, 1, 2, 3, 4, 6, 7]
    assert len(closes) == result['new_branches_attempted'] == 8
    group = json.loads((tmp_path/'group'/'group_data_server_only.json').read_text())
    assert [row['candidate_index'] for row in group['branches']] == list(range(8))
    assert group['pp_index'] == 5 and group['complete']
    assert all(row['cost'] is None for row in group['branches'])
    assert all(row['safety']['raw_collision_events'] == 1 for row in result['branches'])
