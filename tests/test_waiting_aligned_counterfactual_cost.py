from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3_suite as suite
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
    _advance_with_actions,
    counterfactual_cost_contract_v5,
)


def test_halted_queue_contract_matches_waiting_estimand():
    contract = counterfactual_cost_contract_v5("halted_queue")
    assert contract == {
        "mode": "halted_queue",
        "estimand_protocol": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
        "scope": "global_halted_vehicles_per_controlled_lane",
        "population": "sumo_last_step_halting_number_on_controlled_lanes",
        "normalization": "controlled_lane_count",
    }
    with pytest.raises(ValueError):
        counterfactual_cost_contract_v5("future_waiting_time")


def test_halted_queue_cache_identity_cannot_alias_legacy_mode(monkeypatch):
    monkeypatch.setattr(suite, "libsumo_version", lambda: "test-sumo")
    monkeypatch.setattr(suite, "_scenario_input_fingerprint", lambda _: "inputs")
    common = {
        "sumocfg": Path("network.sumocfg"),
        "scenario": "city",
        "duration_sec": 3600.0,
        "control_interval_sec": 10,
        "warmup_sec": 60.0,
        "seed": 1,
        "max_focal_tls": 4,
        "counterfactual_horizon_intervals": 90,
        "behavior_policy": "phase_pressure",
        "collection_shard_index": 0,
        "collection_shard_count": 16,
    }
    legacy = suite._counterfactual_cache_identity(**common)
    waiting = suite._counterfactual_cache_identity(
        **common, counterfactual_cost_mode="halted_queue"
    )
    assert "counterfactual_cost_mode" not in legacy
    assert waiting["counterfactual_cost_mode"] == "halted_queue"
    assert waiting["counterfactual_cost_contract"]["estimand_protocol"] == (
        WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6
    )
    assert legacy != waiting


class _Simulation:
    def __init__(self) -> None:
        self.time = 0.0

    def getTime(self) -> float:
        return self.time

    def simulationStep(self) -> None:
        self.time += 1.0

    def getStartingTeleportNumber(self) -> int:
        return 0

    def getEndingTeleportNumber(self) -> int:
        return 0

    def getCollisions(self) -> tuple[()]:
        return ()


def test_halted_queue_cost_does_not_mix_vehicle_count_or_occupancy() -> None:
    simulation = _Simulation()
    lane = SimpleNamespace(
        getLastStepHaltingNumber=lambda _: 2,
        getLastStepVehicleNumber=lambda _: 100,
        getLastStepOccupancy=lambda _: 80.0,
    )
    api = SimpleNamespace(
        simulation=simulation,
        simulationStep=simulation.simulationStep,
        lane=lane,
        vehicle=SimpleNamespace(getIDList=lambda: ()),
        trafficlight=SimpleNamespace(getIDList=lambda: ()),
    )

    costs = _advance_with_actions(
        sumo_api=api,
        executors={},
        actions={},
        control_interval_sec=1,
        collision_mode="fail",
        cost_lanes=("lane_0",),
    )

    assert costs == [2.0]
