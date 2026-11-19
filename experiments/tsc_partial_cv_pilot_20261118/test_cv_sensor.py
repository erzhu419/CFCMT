"""Unit tests for strict non-leakage and physical behavior of PartialCvObserver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import (
    PartialCvObserver,
    is_equipped_cv,
    select_observed_pressure_phase,
)


@dataclass
class _MockCandidate:
    phase_index: int
    state: str
    duration: float = 30.0
    green_count: int = 1


@dataclass
class _MockInfo:
    tls_id: str
    controlled_links: Any
    candidates: tuple[_MockCandidate, ...]
    incoming_lanes: tuple[str, ...]


class _MockExecutor:
    def __init__(self, states: tuple[str, ...]) -> None:
        self._states = states

    def feasible_states_now(self) -> tuple[str, ...]:
        return self._states


class _StrictNonLeakSumoApi:
    """Raises AssertionError if any unequipped vehicle is queried."""

    def __init__(self, seed: int, p_cv: float) -> None:
        self.seed = seed
        self.p_cv = p_cv
        self.tls_state = "Gr"
        self.lane_vehicles: dict[str, list[str]] = {
            "in_0": [],
            "in_1": [],
            "out_0": [],
            "out_1": [],
        }
        self.veh_speed: dict[str, float] = {}
        self.veh_pos: dict[str, float] = {}

        api = self

        class _Lane:
            def getLength(self, lane_id: str) -> float:
                return 300.0

            def getMaxSpeed(self, lane_id: str) -> float:
                return 11.11

            def getLastStepVehicleIDs(self, lane_id: str) -> tuple[str, ...]:
                return tuple(api.lane_vehicles[lane_id])

        class _Vehicle:
            def getSpeed(self, veh_id: str) -> float:
                if not is_equipped_cv(api.seed, veh_id, api.p_cv):
                    raise AssertionError(f"Leakage! Queried unequipped vehicle {veh_id}")
                return api.veh_speed[veh_id]

            def getLanePosition(self, veh_id: str) -> float:
                if not is_equipped_cv(api.seed, veh_id, api.p_cv):
                    raise AssertionError(f"Leakage! Queried unequipped vehicle {veh_id}")
                return api.veh_pos[veh_id]

        class _TrafficLight:
            def getRedYellowGreenState(self, tls_id: str) -> str:
                return api.tls_state

        self.lane = _Lane()
        self.vehicle = _Vehicle()
        self.trafficlight = _TrafficLight()


def test_strict_no_leakage_and_shockwave_tail() -> None:
    seed = 20261118
    p_cv = 0.10
    api = _StrictNonLeakSumoApi(seed=seed, p_cv=p_cv)
    info = _MockInfo(
        tls_id="J1",
        controlled_links=(
            (("in_0", "out_0", ":J1_0"),),
            (("in_1", "out_1", ":J1_1"),),
        ),
        candidates=(_MockCandidate(0, "Gr"), _MockCandidate(1, "rG")),
        incoming_lanes=("in_0", "in_1"),
    )
    observer = PartialCvObserver(
        sumo_api=api, infos={"J1": info}, seed=seed, p_cv=p_cv
    )

    # Find 1 equipped CV and 20 unequipped vehicles
    equipped_ids = [f"v_{i}" for i in range(500) if is_equipped_cv(seed, f"v_{i}", p_cv)]
    unequipped_ids = [f"v_{i}" for i in range(500) if not is_equipped_cv(seed, f"v_{i}", p_cv)]
    assert len(equipped_ids) > 10 and len(unequipped_ids) > 100

    # Place 15 unequipped vehicles on in_0 (green) and 8 unequipped + 1 halted CV at 60m from stopline on in_1 (red)
    cv_tail = equipped_ids[0]
    api.lane_vehicles["in_0"] = unequipped_ids[:15]
    api.lane_vehicles["in_1"] = unequipped_ids[15:23] + [cv_tail]
    api.veh_speed[cv_tail] = 0.0
    api.veh_pos[cv_tail] = 300.0 - 60.0  # 60m upstream of stop-bar -> ~8 vehicles

    for _ in range(10):
        observer.observe_step(api, dt_sec=1.0)

    obs0 = observer.observations["in_0"]
    obs1 = observer.observations["in_1"]
    assert obs0.cv_veh_count == 0.0
    assert obs1.cv_halted_count == 1.0
    assert abs(obs1.cv_max_halt_dist_m - 60.0) < 1e-9
    # Shockwave estimate: max(1/0.10=10, 60/7.5=8) = 10.0
    assert obs1.shockwave_queue_est >= 8.0

    executor = _MockExecutor(("Gr", "rG"))
    chosen = select_observed_pressure_phase(
        info=info, executor=executor, observer=observer, mode="observed_gp"
    )
    assert chosen == "rG"


def test_deterministic_persistence() -> None:
    hits = [is_equipped_cv(42, f"veh_{i}", 0.15) for i in range(2000)]
    rate = sum(hits) / len(hits)
    assert 0.12 <= rate <= 0.18
    # Calling again produces exact identical assignment
    assert hits == [is_equipped_cv(42, f"veh_{i}", 0.15) for i in range(2000)]


if __name__ == "__main__":
    test_strict_no_leakage_and_shockwave_tail()
    test_deterministic_persistence()
    print("ALL_CV_SENSOR_TESTS_PASSED")
