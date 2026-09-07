from dataclasses import dataclass

import pytest

from cf_h2o.traffic_signal.safe_phase_controller import (
    PhaseTiming,
    SafePhaseExecutor,
    _junction_clearance_lane_ids,
)


class _TrafficLight:
    def __init__(self):
        self.writes = []

    def setRedYellowGreenState(self, tls_id, state):
        self.writes.append((tls_id, state))


class _Sumo:
    def __init__(self):
        self.trafficlight = _TrafficLight()
        self.lane = _Lane()


class _Lane:
    def __init__(self):
        self.vehicles = {}

    def getIDList(self):
        return tuple(self.vehicles)

    def getLastStepVehicleNumber(self, lane_id):
        return self.vehicles.get(lane_id, 0)


def _executor(*, initial_elapsed=5.0):
    api = _Sumo()
    executor = SafePhaseExecutor(
        sumo_api=api,
        tls_id="J0",
        timings=(
            PhaseTiming("GGrr", min_green_sec=5.0, yellow_sec=3.0, all_red_sec=1.0),
            PhaseTiming("rrGG", min_green_sec=5.0, yellow_sec=3.0, all_red_sec=1.0),
        ),
        initial_state="GGrr",
        initial_green_elapsed_sec=initial_elapsed,
    )
    return api, executor


def test_switch_executes_yellow_all_red_then_target_green():
    api, executor = _executor()

    assert executor.request("rrGG")
    assert api.trafficlight.writes == [("J0", "yyrr")]
    executor.advance(3.0)
    assert api.trafficlight.writes[-1] == ("J0", "rrrr")
    executor.advance(1.0)
    assert api.trafficlight.writes[-1] == ("J0", "rrGG")
    assert executor.mode == "green"
    assert executor.audit.switches == 1
    assert executor.audit.yellow_seconds == pytest.approx(3.0)
    assert executor.audit.all_red_seconds == pytest.approx(1.0)


def test_minimum_green_and_busy_requests_are_rejected():
    _, executor = _executor(initial_elapsed=2.0)

    assert executor.feasible_states_now() == ("GGrr",)
    assert not executor.request("rrGG")
    assert executor.audit.rejected_min_green == 1
    executor.advance(3.0)
    assert executor.request("rrGG")
    assert executor.feasible_states_now() == ("rrGG",)
    assert not executor.request("GGrr")
    assert executor.audit.rejected_busy == 1


def test_same_phase_request_does_not_reset_minimum_green_clock():
    api, executor = _executor(initial_elapsed=1.0)

    assert not executor.request("GGrr")
    assert executor.green_elapsed_sec == pytest.approx(1.0)
    assert not api.trafficlight.writes
    assert executor.audit.same_phase_requests == 1


def test_projected_clearance_is_zero_for_stay_and_explicit_for_switch():
    _, executor = _executor()

    assert executor.projected_clearance_sec("GGrr") == pytest.approx(0.0)
    assert executor.projected_clearance_sec("rrGG") == pytest.approx(4.0)


def test_unknown_or_malformed_phases_fail_fast():
    _, executor = _executor()
    with pytest.raises(KeyError):
        executor.request("GrrG")
    with pytest.raises(ValueError):
        SafePhaseExecutor(
            sumo_api=_Sumo(),
            tls_id="J0",
            timings=(PhaseTiming("GG", 5.0, 3.0, 1.0),),
            initial_state="GGrr",
        )


def test_runtime_snapshot_restore_rewinds_clearance_and_audit():
    api, executor = _executor()
    saved = executor.snapshot()
    assert executor.request("rrGG")
    executor.advance(3.0)
    assert executor.mode == "all_red"
    assert executor.audit.switches == 1

    executor.restore(saved)

    assert executor.mode == "green"
    assert executor.current_state == "GGrr"
    assert executor.current_green_state == "GGrr"
    assert executor.target_green_state == "GGrr"
    assert executor.green_elapsed_sec == pytest.approx(5.0)
    assert executor.audit.switches == 0
    assert api.trafficlight.writes[-1] == ("J0", "GGrr")


def test_all_red_extends_until_internal_clearance_lane_is_empty():
    api = _Sumo()
    executor = SafePhaseExecutor(
        sumo_api=api,
        tls_id="J0",
        timings=(
            PhaseTiming("GGrr", min_green_sec=5.0, yellow_sec=3.0, all_red_sec=1.0),
            PhaseTiming("rrGG", min_green_sec=5.0, yellow_sec=3.0, all_red_sec=1.0),
        ),
        initial_state="GGrr",
        initial_green_elapsed_sec=5.0,
        clearance_lane_ids=(":J0_0",),
    )
    api.lane.vehicles[":J0_0"] = 1

    assert executor.request("rrGG")
    executor.advance(3.0)
    executor.advance(1.0)

    assert executor.mode == "all_red"
    assert api.trafficlight.writes[-1] == ("J0", "rrrr")
    assert executor.audit.occupancy_clearance_extensions == 1
    assert executor.audit.occupancy_clearance_extension_seconds == pytest.approx(1.0)

    api.lane.vehicles[":J0_0"] = 0
    executor.advance(1.0)

    assert executor.mode == "green"
    assert api.trafficlight.writes[-1] == ("J0", "rrGG")


def test_clearance_lanes_expand_to_all_internal_segments_of_junction():
    api = _Sumo()
    api.lane.vehicles = {
        ":J0_4_0": 0,
        ":J0_21_0": 1,
        ":J1_4_0": 1,
        "external": 1,
    }

    lanes = _junction_clearance_lane_ids(
        api,
        ((('in', 'out', ':J0_4_0'),),),
    )

    assert lanes == (":J0_4_0", ":J0_21_0")
