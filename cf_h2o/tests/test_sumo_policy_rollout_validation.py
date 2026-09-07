import shutil

import pandas as pd
import pytest

from cf_h2o.eval.sumo_apc_avl_sumo_generation import build_city_sumo
from cf_h2o.eval.sumo_policy_rollout_validation import (
    _load_libsumo,
    _run_city_policy,
    _stop_index,
    _target_headways,
)


def _write_line(line_dir) -> None:
    line_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "stop_name": ["S1", "S2", "S3", "S4"],
            "latitude": [42.0, 42.001, 42.002, 42.003],
            "longitude": [-71.0, -71.001, -71.002, -71.003],
        }
    ).to_excel(line_dir / "stop_news.xlsx", index=False)
    pd.DataFrame(
        {
            "start_stop": ["S1", "S2", "S3"],
            "end_stop": ["S2", "S3", "S4"],
            "distance": [100.0, 120.0, 140.0],
            "V_max": [8.0, 8.0, 8.0],
            **{f"{hour:02d}:00:00": [8.0, 8.0, 8.0] for hour in range(24)},
        }
    ).to_excel(line_dir / "route_news.xlsx", index=False)
    pd.DataFrame({"launch_time": [0, 120, 240], "direction": [1, 1, 1]}).to_excel(
        line_dir / "time_table.xlsx",
        index=False,
    )


def test_stop_index_parses_generated_bus_stop_id():
    assert _stop_index("bs_city_line_0012") == 12


def test_target_headways_from_line_map():
    assert _target_headways([{"line_uid": "L", "departures": [0, 120, 300]}])["L"] == 150.0


def test_libsumo_policy_rollout_changes_dwell(tmp_path):
    if not shutil.which("netconvert") or not shutil.which("sumo"):
        pytest.skip("SUMO binaries are not installed")
    sumo_api = _load_libsumo()
    assert sumo_api.__name__ == "libsumo"
    line_dir = tmp_path / "env" / "data" / "L1"
    _write_line(line_dir)
    manifest = [
        {
            "city_key": "toy",
            "city": "Toy",
            "line_key": "L1",
            "line_dir": str(line_dir),
            "ok": True,
        }
    ]
    city = build_city_sumo(
        city_key="toy",
        line_manifest=manifest,
        out_dir=tmp_path / "sumo",
        max_lines=0,
        max_departures_per_line=3,
        duration_sec=600.0,
        include_bus_stops=True,
        base_dwell_sec=8.0,
    )

    no_hold = _run_city_policy(
        sumo_api=sumo_api,
        city_key="toy",
        city=city,
        policy="no_hold",
        model=None,
        actions=[0.0, 30.0],
        base_dwell_sec=8.0,
        max_steps=1000,
        max_events=6,
        mpc_horizon=2,
        mpc_discount=0.85,
    )
    fixed = _run_city_policy(
        sumo_api=sumo_api,
        city_key="toy",
        city=city,
        policy="fixed_30",
        model=None,
        actions=[0.0, 30.0],
        base_dwell_sec=8.0,
        max_steps=1000,
        max_events=6,
        mpc_horizon=2,
        mpc_discount=0.85,
    )

    assert no_hold["ok"] is True
    assert fixed["ok"] is True
    assert no_hold["mean_hold_seconds"] == 0.0
    assert fixed["mean_hold_seconds"] == 30.0
    assert fixed["mean_dwell_seconds"] > no_hold["mean_dwell_seconds"]
    assert len(no_hold["line_rows"]) == 1
    assert no_hold["line_rows"][0]["events"] == no_hold["events"]
