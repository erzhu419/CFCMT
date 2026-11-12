"""Toy admission tests for passive dense-network measurement, without SUMO."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor
from experiments.tsc_network_dense_measurement_20261109.measurement_audit import MeasurementAuditor


PHASES = ('GGrr', 'rrGg')


def executor_rows(executors):
    return {tls: dict(snapshot=ex.snapshot(), feasible_states=list(ex.feasible_states_now()),
                     minimum_green_sec=ex.timings[ex.current_green_state].min_green_sec)
            for tls, ex in executors.items()}


class Producer:
    """Same PP queue/green-lane/tie semantics, consuming only the cached API."""
    def _read_graph_states_v3(self, api, infos, context, graph):
        return {tls: SimpleNamespace(info=info, q={lane:
            api.lane.getLastStepHaltingNumber(lane)+.35*api.lane.getLastStepVehicleNumber(lane)+
            .025*api.lane.getLastStepOccupancy(lane) for lane in info.incoming_lanes})
            for tls, info in infos.items()}

    def _rule_candidate(self, state, executor, policy):
        assert policy == 'phase_pressure'

        def score(phase):
            permitted = {state.info.controlled_links[index][0][0]
                         for index, char in enumerate(phase) if char in 'Gg'}
            return sum(state.q[lane] for lane in state.info.incoming_lanes if lane in permitted)

        return SimpleNamespace(state=max(executor.feasible_states_now(), key=score))


def frame_at(t):
    auxiliary = ['c' if t < 5 else 'd']+(['e'] if t >= 8 else [])
    internal = ['c'] if t in (3, 4) else []
    if internal:
        auxiliary = []
    ids = dict(a=['a'], b=['b'], aux=auxiliary, internal=internal)
    lanes = {lane: dict(ids=vehicles, N=len(vehicles), H=int(lane in ('a', 'b')),
        O=.5 if lane in ('a', 'b') else .01*len(vehicles),
        mean_speed_mps=0. if lane in ('a', 'b') else 2. if vehicles else 13.)
        for lane, vehicles in ids.items()}
    rebuilt = deepcopy(lanes)
    # This admitted floating error would reverse a symmetric PP tie if the
    # retrospective policy used reconstructed O instead of cached native O.
    rebuilt['b']['O'] += 1e-9
    return dict(time_s=600+t, offset_sec=t, role='preaction_observation' if not t else 'future_measurement_label',
        packets={vid: {} for values in ids.values() for vid in values},
        native_lanes=lanes, reconstructed_lanes=rebuilt,
        native_active_ids=sorted(vid for values in ids.values() for vid in values),
        native_pending_ids=['e'] if t in (6, 7) else [],
        departed_ids=['d'] if t == 5 else ['e'] if t == 8 else [],
        arrived_ids=['c'] if t == 5 else [], starting_teleport_ids=[], ending_teleport_ids=[],
        collision_ids=['a', 'b'] if t == 7 else [],
        checks=dict(ids_exact=True, N_exact=True, H_exact=True, O_close=True,
                    mean_speed_mps_close=True, body_paths_complete=True, native_time_exact=True),
        unresolved=[], events=[])


def run_toy(mutate=None):
    holder = {'frame': frame_at(0)}
    api = SimpleNamespace(trafficlight=SimpleNamespace(setRedYellowGreenState=lambda *_: None),
        lane=SimpleNamespace(getLastStepVehicleNumber=lambda lane: holder['frame']['native_lanes'][lane]['N']))
    ex = SafePhaseExecutor(sumo_api=api, tls_id='A',
        timings=[PhaseTiming(state, 5., 2., 1.) for state in PHASES],
        initial_state=PHASES[0], initial_green_elapsed_sec=30.,
        clearance_lane_ids=('internal',), clearance_poll_sec=1.)
    executors = {'A': ex}
    info = SimpleNamespace(incoming_lanes=('a', 'b'), controlled_links=(
        (('a', 'out_a', ''),), (('a', 'out_a2', ''),),
        (('b', 'out_b', ''),), (('b', 'out_b2', ''),)))
    producer = Producer()
    auditor = MeasurementAuditor(producer, {'A': info}, None, None, executors, ('a', 'b'), focal_tls='A')
    initial = ex.snapshot()
    holder['frame']['executors'] = executor_rows(executors)
    assert auditor.observe(holder['frame'], None)
    assert ex.snapshot() == initial  # Audit construction and frame0 never mutate live state.
    transitions = []
    for t in range(1, 451):
        offset = t-1
        before = executor_rows(executors)
        requests, returns = {}, {}
        if offset % 10 == 0:
            # Native O preserves exact ties, with canonical PHASES[0] first.
            states = producer._read_graph_states_v3(auditor.policy_api, {'A': info}, None, None)
            requests = {'A': producer._rule_candidate(states['A'], ex, 'phase_pressure').state}
            if offset == 0:
                requests['A'] = PHASES[1]
            returns = {'A': bool(ex.request(requests['A']))}
        after = executor_rows(executors)
        applied = {'A': ex.current_state}
        next_frame = frame_at(t)
        holder['frame'] = next_frame
        ex.advance(1.)
        next_frame['executors'] = executor_rows(executors)
        transition = dict(offset_sec=offset, time_sec=600+offset, requests=requests,
            request_returns=returns, before=before, after=after, applied_states=applied,
            after_executors=executor_rows(executors), halted_int=2,
            due_ids=['d'] if t == 5 else ['e'] if t == 6 else [])
        if mutate is not None:
            mutate(t, next_frame, transition)
        transitions.append(deepcopy(transition))
        live_snapshot = ex.snapshot()
        if not auditor.observe(next_frame, transition):
            assert ex.snapshot() == live_snapshot
            return auditor, auditor.summary(next_frame), transitions, ex
        assert ex.snapshot() == live_snapshot
    return auditor, auditor.summary(next_frame), transitions, ex


def test_native_replay_preserves_ties_request_once_clearance_extensions_and_poststep_cost():
    auditor, result, transitions, live = run_toy()
    assert result['status'] == 'PASS_DENSE_MEASUREMENT'
    assert all(result['checks'].values())
    assert len(auditor.halted) == 450 and result['frames_seen'] == 451
    assert result['I450'] == 900 and result['C450'] == 1
    assert result['integrals']['10'] == 20
    assert sum(result['segments'].values()) == 900
    assert result['request_epochs'] == 45 and live.audit.requests == 45
    assert live.audit.occupancy_clearance_extensions == 2
    assert transitions[0]['applied_states']['A'] == 'YYrr'
    assert transitions[9]['requests'] == {} and transitions[10]['requests'] == {'A': PHASES[0]}
    assert transitions[20]['request_returns'] == {'A': False}  # Same-phase is valid.
    assert result['population'] == dict(initial_active_count=3, initial_pending_count=0,
        final_active_count=4, final_pending_count=0, newly_due_count=2, departed_count=2, arrived_count=1)
    assert result['collision_metadata']['vehicle_occurrences'] == 2  # Metadata does not fail admission.


@pytest.mark.parametrize('kind,expected', [
    ('due', 'pending_due_ledger'), ('arrival', 'population_identity'), ('teleport', 'no_teleport')])
def test_due_pending_population_and_teleport_failures_stop_with_partial_records(kind, expected):
    def mutate(t, frame, transition):
        if t == 5:
            if kind == 'due':
                transition['due_ids'] = []
            elif kind == 'arrival':
                frame['arrived_ids'] = []
            else:
                frame['starting_teleport_ids'] = ['c']
    auditor, result, unused, live = run_toy(mutate)
    assert result['status'] == 'HOLD_DENSE_MEASUREMENT'
    assert result['first_failure']['check'] == expected
    assert result['checked_poststeps'] == 4 and result['C450'] is None and result['I450'] is None
    assert not auditor.observe(frame_at(6), {})
    assert not auditor.summary(frame_at(6))['checks'][expected]


def test_reconstruction_error_is_not_hidden_by_later_good_frame_or_final_summary():
    def mutate(t, frame, transition):
        if t == 3:
            frame['reconstructed_lanes']['a']['O'] += 2e-8
    auditor, result, unused, live = run_toy(mutate)
    assert result['first_failure']['check'] == 'lane_sensor_reconstruction'
    assert result['first_failure']['detail']['field'] == 'O'
    assert result['checked_poststeps'] == 2 and result['C450'] is None
    assert auditor.summary(frame_at(450))['status'] == 'HOLD_DENSE_MEASUREMENT'


@pytest.mark.parametrize('kind,expected', [
    ('signal', 'signals_before_step'), ('return', 'request_acceptance'),
    ('clock', 'native_clock_exact'), ('pp', 'original_pp_requests')])
def test_applied_signal_acceptance_clock_and_original_pp_drifts_are_rejected(kind, expected):
    def mutate(t, frame, transition):
        if t == (11 if kind == 'pp' else 1):
            if kind == 'signal':
                transition['applied_states']['A'] = PHASES[1]
            elif kind == 'return':
                transition['request_returns']['A'] = False
            elif kind == 'clock':
                frame['time_s'] += 1
            else:
                transition['requests']['A'] = PHASES[1]
    unused, result, unused_rows, live = run_toy(mutate)
    assert result['first_failure']['check'] == expected
    assert result['status'] == 'HOLD_DENSE_MEASUREMENT' and result['C450'] is None
