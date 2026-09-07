from __future__ import annotations

from pathlib import Path

import numpy as np

from cf_h2o.traffic_signal.demand_schedule_context import (
    DEMAND_SCHEDULE_FEATURE_NAMES,
    augment_with_demand_profile,
    build_demand_schedule_profile,
)


def _write_fixture(root: Path, departures: list[int]) -> Path:
    (root / "network.net.xml").write_text(
        "<net><tlLogic id='tls0' type='static' programID='0' offset='0'>"
        "<phase duration='30' state='G'/></tlLogic></net>",
        encoding="utf-8",
    )
    vehicles = "".join(
        f"<vehicle id='v{index}' depart='{depart}'><route edges='a b'/></vehicle>"
        for index, depart in enumerate(departures)
    )
    (root / "demand.rou.xml").write_text(
        f"<routes>{vehicles}</routes>", encoding="utf-8"
    )
    sumocfg = root / "scenario.sumocfg"
    sumocfg.write_text(
        "<configuration><input><net-file value='network.net.xml'/>"
        "<route-files value='demand.rou.xml'/></input>"
        "<time><begin value='0'/><end value='3600'/></time></configuration>",
        encoding="utf-8",
    )
    return sumocfg


def test_uniform_schedule_has_zero_bin_variation(tmp_path: Path) -> None:
    sumocfg = _write_fixture(tmp_path, [100, 700, 1300, 1900, 2500, 3100])
    profile = build_demand_schedule_profile(sumocfg)
    values = dict(zip(profile.feature_names, profile.features))

    assert profile.feature_names == DEMAND_SCHEDULE_FEATURE_NAMES
    assert profile.scheduled_vehicle_count == 6.0
    assert profile.tls_count == 1
    assert values["departure_600s_cv"] == 0.0
    assert values["departure_600s_trough_ratio"] == 1.0
    assert values["departure_600s_peak_ratio"] == 1.0
    assert values["origin_hhi"] == 1.0


def test_burst_schedule_and_augmentation_are_deterministic(tmp_path: Path) -> None:
    sumocfg = _write_fixture(tmp_path, [10, 20, 30, 40, 50, 60])
    first = build_demand_schedule_profile(sumocfg)
    second = build_demand_schedule_profile(sumocfg)
    values = dict(zip(first.feature_names, first.features))

    assert first == second
    assert values["departure_600s_cv"] > 1.0
    augmented = augment_with_demand_profile(np.ones((2, 3)), first)
    assert augmented.shape == (2, 3 + len(DEMAND_SCHEDULE_FEATURE_NAMES))
    np.testing.assert_allclose(augmented[0, 3:], augmented[1, 3:])
