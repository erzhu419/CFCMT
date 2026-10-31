from pathlib import Path
from types import SimpleNamespace

import pytest

from cf_h2o.eval.traffic_signal_cologne_collision_replay import (
    END_SEC, STAGES, exact_prefix, passenger_polygon, polygon_axis_penetration,
    vehicle_pose,
)


def test_native_passenger_cut_corners_and_lane_following_rear():
    # A front-corner box overlap is not sufficient for SUMO's passenger polygon.
    a = passenger_polygon((0, 0), (-4, 0), 2)
    assert max(p[0] for p in a) == 0
    assert sorted({abs(p[1]) for p in a}) == [.6, 1.0]
    b = passenger_polygon((3.95, 1.8), (-.05, 1.8), 2)
    assert polygon_axis_penetration(a, b) < 0
    assert polygon_axis_penetration(a, passenger_polygon((0, .2), (-4, .2), 2)) > 0

    calls = []
    v = SimpleNamespace(
        getLaneID=lambda _: "lane", getRoadID=lambda _: "edge", getLanePosition=lambda _: 8.659,
        getLength=lambda _: 4.3, getWidth=lambda _: 1.8, getLateralLanePosition=lambda _: 0.0,
        getPosition=lambda _: (10., 5.), getTypeID=lambda _: "pkw", getLaneIndex=lambda _: 0,
        getAngle=lambda _: 20., getSpeed=lambda _: 0., getAcceleration=lambda _: 0.,
        getWaitingTime=lambda _: 4., getRoute=lambda _: ("in", "out"), getRouteIndex=lambda _: 0,
        getNextTLS=lambda _: (),
    )
    api = SimpleNamespace(vehicle=v, vehicletype=SimpleNamespace(getShapeClass=lambda _: "passenger"),
                          simulation=SimpleNamespace(convert2D=lambda *args: calls.append(args) or (6., 3.)))
    pose = vehicle_pose(api, "v")
    assert calls == [("edge", 8.659 - 4.3, 0, False)]
    assert pose["back_position_xy_m"] == [6., 3.]
    assert pose["passenger_polygon_xy_m"] == passenger_polygon((10., 5.), (6., 3.), 1.8)
    v.getLanePosition = lambda _: 2.
    assert vehicle_pose(api, "v")["polygon_unavailable_reason"] == "rear_on_preceding_lane"
    assert len(calls) == 1


def test_exact_prefix_rejects_changed_action_score_or_missing_row():
    row = {"time_sec": END_SEC - 6, "selected_phase_state": "Gr", "score": .125}
    reference = {"metrics": {"accepted_intervention_trace": [row, {"time_sec": END_SEC + 4}]}}
    assert exact_prefix(reference, {"accepted_intervention_trace": [dict(row)]})["passed"]
    assert not exact_prefix(reference, {"accepted_intervention_trace": [{**row, "score": .126}]})["passed"]
    assert not exact_prefix(reference, {"accepted_intervention_trace": []})["passed"]


def test_real_runner_observer_order_preserves_original_executor_actions(monkeypatch):
    import cf_h2o.eval.traffic_signal_resco_cfcmt_v3 as runner
    from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor

    for name, value in {
        "_start_sumo": lambda *a, **k: None,
        "_tls_phase_infos": lambda *a: {"tls": None},
        "build_signal_routing_graph": lambda *a: SimpleNamespace(adjacency={}),
        "_controlled_lane_set": lambda *a: ("in",),
        "build_tls_intervention_graph": lambda *a: None,
        "add_routing_conflicts": lambda *a: None,
        "_scenario_context_v3": lambda **k: None,
        "_read_graph_states_v3": lambda *a: {"tls": None},
        "_pressure_candidate_v3": lambda *a, **k: SimpleNamespace(state="rG"),
        "_lane_queue": lambda *a: 0.,
        "_parse_tripinfo_metrics": lambda *a: {},
    }.items():
        monkeypatch.setattr(runner, name, value)

    def run(observe):
        clock = {"time": 0., "signal": "Gr"}
        writes, records = [], []
        def write(tls, state):
            writes.append((clock["time"], state))
            clock["signal"] = state
        api = SimpleNamespace(
            simulationStep=lambda: clock.update(time=clock["time"] + 1), close=lambda: None,
            trafficlight=SimpleNamespace(getIDList=lambda: ["tls"], setRedYellowGreenState=write,
                                         getRedYellowGreenState=lambda _: clock["signal"]),
            lane=SimpleNamespace(getLastStepHaltingNumber=lambda _: 0),
            vehicle=SimpleNamespace(getIDCount=lambda: 0, getIDList=lambda: []),
            simulation=SimpleNamespace(getTime=lambda: clock["time"], getDepartedNumber=lambda: 0,
                getArrivedNumber=lambda: 0, getLoadedNumber=lambda: 0, getStartingTeleportNumber=lambda: 0,
                getEndingTeleportNumber=lambda: 0, getCollisions=lambda: (),
                getEmergencyStoppingVehiclesNumber=lambda: 0, getPendingVehicles=lambda: ()),
        )
        executor = SafePhaseExecutor(sumo_api=api, tls_id="tls", initial_state="Gr",
            timings=[PhaseTiming(state, 0., 1., 1.) for state in ("Gr", "rG")])
        monkeypatch.setattr(runner, "build_safe_phase_executors", lambda *a: {"tls": executor})
        def observer(**kw):
            records.append((kw["time_sec"], kw["stage"], clock["signal"], kw["executors"]["tls"].snapshot()["mode"]))
        metrics = runner.evaluate_policy_v3(sumo_api=api, sumocfg=Path("unused"), scenario="fake",
            policy="phase_pressure", models=None, duration_sec=2, control_interval_sec=1,
            warmup_sec=0, seed=41242, step_observer=observer if observe else None)
        assert metrics["ok"], metrics
        return metrics, writes, records

    plain, plain_writes, _ = run(False)
    observed, observed_writes, records = run(True)
    assert plain == observed
    assert plain_writes == observed_writes
    assert [(row[0], row[1]) for row in records] == [
        (0., STAGES[0]), (1., STAGES[1]), (1., STAGES[2]),
        (1., STAGES[0]), (2., STAGES[1]), (2., STAGES[2]),
    ]
    assert records[0][2:] == records[1][2:] == ("yr", "yellow")
    assert records[2][2:] == ("rr", "all_red")
    assert records[-1][2:] == ("rG", "green")
