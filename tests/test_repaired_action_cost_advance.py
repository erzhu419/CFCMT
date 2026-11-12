"""Native step order, passive getters, complete ledgers and partial preservation."""
import json
from types import SimpleNamespace

import pytest

from experiments.tsc_repaired_action_cost_pilot_20261109.advance_measurement import advance_horizon


class Executor:
    def __init__(self, api, tls):
        self.api, self.tls = api, tls
        self.current_state = self.current_green_state = 'G'
        self.target, self.mode, self.requests = 'G', 'green', []
        self.timings = {state: SimpleNamespace(min_green_sec=6) for state in ('G', 'r')}

    def snapshot(self):
        return dict(current_state=self.current_state, current_green_state=self.current_green_state, mode=self.mode)

    def feasible_states_now(self):
        return ('G', 'r')

    def request(self, state):
        self.requests.append((self.api.time, state))
        if state == self.current_green_state:
            return False  # The unchanged same-phase request is valid measurement data.
        self.target, self.current_state, self.mode = state, 'y', 'yellow'
        return True

    def advance(self, elapsed):
        if self.mode == 'yellow':
            self.current_state = self.current_green_state = self.target
            self.mode = 'green'


class API:
    def __init__(self):
        self.time, self.last_applied = 0, {'A': 'G', 'B': 'G'}
        self.executors = {tls: Executor(self, tls) for tls in ('A', 'B')}
        self.simulation = SimpleNamespace(getTime=lambda: self.time,
            getPendingVehicles=lambda: ['b'] if self.time < 5 else [],
            getDepartedIDList=lambda: ['b'] if self.time == 5 else [],
            getArrivedIDList=lambda: ['a'] if self.time == 5 else [])
        self.vehicle = SimpleNamespace(getIDList=lambda: ['a'] if self.time < 5 else ['b'])
        self.trafficlight = SimpleNamespace(getRedYellowGreenState=lambda tls: self.executors[tls].current_state)
        self.lane = SimpleNamespace(getLastStepVehicleNumber=lambda lane: 4,
            getLastStepHaltingNumber=lambda lane: self.time % 3 + int(self.last_applied['A'] == 'r'),
            getLastStepOccupancy=lambda lane: 10., getLastStepMeanSpeed=lambda lane: 2.)

    def simulationStep(self):
        self.last_applied = {tls: ex.current_state for tls, ex in self.executors.items()}
        self.time += 1


class Producer:
    def __init__(self, fail_step=None):
        self.fail_step, self.calls = fail_step, []

    def _read_graph_states_v3(self, api, infos, context, graph):
        return {tls: api.time for tls in infos}

    def _rule_candidate(self, state, executor, policy):
        assert policy == 'phase_pressure'
        return SimpleNamespace(state=executor.current_green_state)

    def _advance_with_actions(self, *, sumo_api, executors, actions, control_interval_sec,
                              cost_lanes, collision_mode, safety_ledger):
        assert actions == {} and control_interval_sec == 1 and collision_mode == 'audit'
        self.calls.append(sumo_api.time)
        sumo_api.simulationStep()
        for executor in executors.values():
            executor.advance(1)
        if sumo_api.time == self.fail_step:
            raise RuntimeError('native step invalid')
        return [sum(sumo_api.lane.getLastStepHaltingNumber(lane) for lane in cost_lanes)]


class Observer:
    def __init__(self, api):
        self.api, self.lane_geometry, self.samples = api, {'receiving': {}}, []

    def sample(self, now, applied_states=None):
        self.api.vehicle.getIDList()  # Extra observation is read-only.
        self.samples.append((now, dict(applied_states or {})))


def setup(fail_step=None):
    api, producer = API(), Producer(fail_step)
    observer = Observer(api); observer.sample(0)
    args = dict(api=api, v3=producer, infos={tls: None for tls in api.executors}, graph=None, context=None,
                executors=api.executors, initial_actions={'A': 'r', 'B': 'G'}, lanes=('L0', 'L1'), ledger={}, observer=observer)
    return api, producer, observer, args


def test_actual_first_step_color_precedes_executor_advance_and_each_epoch_requests_once():
    api, producer, observer, args = setup()
    result = advance_horizon(**args)
    assert all(result['checks'].values())
    assert len(result['execution_trace']) == 45
    assert all([time for time, state in ex.requests] == list(range(0, 450, 10)) for ex in api.executors.values())
    assert result['actual_signals450'][0]['applied_states']['A'] == 'y'
    assert result['actual_signals450'][1]['applied_states']['A'] == 'r'
    assert observer.samples[1] == (1, {'A': 'y', 'B': 'G'})
    assert len(observer.samples) == 11
    assert result['execution_trace'][0]['request_returns']['B'] is False
    assert result['checks']['request_epochs_complete']
    assert producer.calls == list(range(450))


def test_step_five_population_events_integrals_and_future_roles_are_complete():
    api, producer, observer, args = setup()
    result = advance_horizon(**args)
    ledger = result['population_ledger']
    assert ledger['per_second'][4]['departed_ids'] == ['b']
    assert ledger['per_second'][4]['arrived_ids'] == ['a']
    assert ledger['departed_count'] == ledger['arrived_count'] == 1
    assert ledger['initial_active_ids'] == ['a'] and ledger['final_active_ids'] == ['b']
    assert ledger['count_closure'] and ledger['identity_closure']
    assert sum(result['segments'].values()) == result['integrals']['450']
    assert result['integrals']['10'] == sum(result['halted_counts450'][:10])
    assert result['cost'] == pytest.approx(result['integrals']['450']/(450*2))
    assert result['endpoint_states']['0']['role'] == 'preaction_input'
    assert result['endpoint_states']['10']['role'] == 'future_measurement_label'
    assert set(result['endpoint_states']['450']['lanes']) == {'L0', 'L1', 'receiving'}
    json.dumps(result, allow_nan=False)


def test_extra_measurement_getters_do_not_change_native_cost_or_execution():
    api, producer, observer, args = setup()
    measured = advance_horizon(**args)
    plain, plain_producer, unused, baseline = setup()
    costs = []
    for step in range(450):
        if step % 10 == 0:
            actions = baseline['initial_actions'] if step == 0 else {
                tls: plain_producer._rule_candidate(step, ex, 'phase_pressure').state for tls, ex in plain.executors.items()}
            for tls, ex in plain.executors.items():
                ex.request(actions[tls])
        costs.extend(plain_producer._advance_with_actions(sumo_api=plain, executors=plain.executors, actions={},
            control_interval_sec=1, cost_lanes=baseline['lanes'], collision_mode='audit', safety_ledger={}))
    assert measured['halted_counts450'] == costs
    assert {tls: ex.snapshot() for tls, ex in api.executors.items()} == {tls: ex.snapshot() for tls, ex in plain.executors.items()}
    assert {tls: ex.requests for tls, ex in api.executors.items()} == {tls: ex.requests for tls, ex in plain.executors.items()}


def test_invalid_native_step_preserves_partial_trace_and_never_returns_partial_cost_as_label():
    api, producer, observer, args = setup(fail_step=7)
    result = advance_horizon(**args)
    assert result['completed_steps'] == 6
    assert len(result['actual_signals450']) == 7
    assert result['cost'] is None
    assert result['integrals'] == {'0': 0, '10': None, '30': None, '60': None, '450': None}
    assert all(value is None for value in result['segments'].values())
    assert result['error'] == 'RuntimeError: native step invalid'
    assert not result['checks']['full_cost_horizon']
    assert not result['checks']['no_execution_error']
