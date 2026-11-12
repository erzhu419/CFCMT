from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor
from scripts.cluster.tsc_physical_response_data import (
    FEATURE_NAMES, analytic_baseline, build_preaction_rows,
    extract_response_labels, nominal_signal_plan,
)


class ObservationAPI:
    def __init__(self):
        self.writes = []
        self.rows = {
            "in_0": dict(length=100., limit=10., vehicles=10, halted=8, occupancy=.5, speed=2.),
            "out_0": dict(length=90., limit=10., vehicles=1, halted=0, occupancy=.05, speed=9.),
            "side_0": dict(length=75., limit=5., vehicles=0, halted=0, occupancy=0., speed=-1.),
            ":a_0": dict(length=5., limit=5., vehicles=2, halted=1, occupancy=.8, speed=1.),
            ":b_0": dict(length=10., limit=5., vehicles=0, halted=0, occupancy=0., speed=-1.),
        }
        mapping = {
            "getLength": "length", "getMaxSpeed": "limit", "getLastStepVehicleNumber": "vehicles",
            "getLastStepHaltingNumber": "halted", "getLastStepOccupancy": "occupancy",
            "getLastStepMeanSpeed": "speed",
        }
        self.lane = SimpleNamespace(**{name: (lambda lane, key=key: self.rows[lane][key])
                                      for name, key in mapping.items()})
        self.trafficlight = SimpleNamespace(setRedYellowGreenState=lambda *args: self.writes.append(args))
        # No vehicle route, next-TLS, step or time-advance interface is provided.


def executor(api, elapsed=10):
    return SafePhaseExecutor(sumo_api=api, tls_id="tls",
        timings=[PhaseTiming(state, 5, 3, 1) for state in ("Gr", "rG")],
        initial_state="Gr", initial_green_elapsed_sec=elapsed,
        clearance_lane_ids=(":a_0", ":b_0"))


def movements():
    return {
        "tls|in|out": dict(tls_id="tls", incoming_edge="in", outgoing_edge="out",
            incoming_lanes=["in_0"], outgoing_lanes=["out_0"], internal_lanes=[":a_0"], link_indices=[0]),
        "tls|in|side": dict(tls_id="tls", incoming_edge="in", outgoing_edge="side",
            incoming_lanes=["in_0"], outgoing_lanes=["side_0"], internal_lanes=[":b_0"], link_indices=[1]),
    }


def test_nominal_copy_preserves_live_executor_and_never_writes_sumo():
    api = ObservationAPI()
    original = executor(api)
    before = deepcopy(original.snapshot())
    plan = nominal_signal_plan(original, "rG")
    assert plan["states"] == ["Yr"] * 3 + ["rr"] + ["rG"] * 6
    assert plan["request_accepted"]
    assert "No future occupancy" in plan["assumption"]
    assert original.snapshot() == before
    assert api.writes == []
    # Live occupancy is nonzero; no future occupancy oracle was queried to
    # make this nominal first green occur after four seconds.
    assert api.rows[":a_0"]["vehicles"] == 2


def test_minimum_green_is_respected_in_pre_action_nominal_schedule():
    api = ObservationAPI()
    plan = nominal_signal_plan(executor(api, elapsed=0), "rG")
    assert not plan["request_accepted"]
    assert plan["states"] == ["Gr"] * 10
    assert not api.writes


def test_shared_incoming_lane_allocates_demand_once_without_routes():
    api = ObservationAPI()
    ex = executor(api)
    before = deepcopy(ex.snapshot())
    result = build_preaction_rows(api, {}, {"tls": ex}, {"tls": "rG"}, movements(), focal_tls="tls")
    x = np.asarray(result["features"])
    columns = {name: i for i, name in enumerate(result["feature_names"])}
    assert tuple(result["feature_names"]) == FEATURE_NAMES
    assert x[:, columns["turn_allocated_vehicle_count"]].sum() == 10
    assert x[:, columns["turn_allocated_halted_count"]].sum() == 8
    assert x[:, columns["movement_lane_equivalent"]].sum() == 1
    assert x[:, columns["clearance_vehicle_count"]].tolist() == [2, 2]
    assert x[:, columns["internal_vehicle_count"]].tolist() == [2, 0]
    assert x[:, columns["nominal_major_green_sec"]].tolist() == [0, 6]
    assert np.isfinite(x).all()
    assert ex.snapshot() == before and not api.writes


def test_analytic_rule_obeys_known_signal_demand_and_receiving_space():
    api = ObservationAPI()
    data = build_preaction_rows(api, {}, {"tls": executor(api)}, {"tls": "Gr"}, movements())
    prediction = analytic_baseline(data["features"], data["feature_names"])
    assert prediction[0, 0] == 2.5
    assert prediction[1, 0] == 0  # Red movement gets no nominal new entry.
    assert prediction[0, 1] > 0  # Receipts can include initial internal vehicles.
    modified = np.asarray(data["features"])
    modified[:, FEATURE_NAMES.index("outgoing_free_vehicle_spaces")] = 0
    assert np.array_equal(analytic_baseline(modified), np.zeros((2, 2)))


def test_labels_keep_initial_receipts_and_reject_unresolved_measurement():
    record = dict(stopline_entries=2, receiving_entries=4, initial_internal=2,
                  pending_internal=0, unresolved_exits=0, unobserved_entries=0,
                  accounting_balanced=True)
    measurement = {"complete": True, "windows": {"10": {"movements": {"m": record}}}}
    assert extract_response_labels(measurement, ["m"]) == [[2., 4.]]
    measurement["complete"] = False
    with pytest.raises(ValueError, match="complete"):
        extract_response_labels(measurement, ["m"])
    measurement["complete"] = True
    with pytest.raises(KeyError):
        extract_response_labels(measurement, ["missing"])


def test_unavailable_focal_movements_are_not_empty_training_rows():
    api = ObservationAPI()
    with pytest.raises(ValueError, match="No movements"):
        build_preaction_rows(api, {}, {"tls": executor(api)}, {"tls": "Gr"}, movements(), focal_tls="other")
