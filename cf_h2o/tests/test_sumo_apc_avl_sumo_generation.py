import shutil

import pandas as pd
import pytest

from cf_h2o.eval.sumo_apc_avl_sumo_generation import _safe_id, build_city_sumo, run_sumo_smoke


def _write_line(line_dir) -> None:
    line_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "stop_name": ["S1", "S2", "S3"],
            "latitude": [42.0, 42.001, 42.002],
            "longitude": [-71.0, -71.001, -71.002],
        }
    ).to_excel(line_dir / "stop_news.xlsx", index=False)
    pd.DataFrame(
        {
            "start_stop": ["S1", "S2"],
            "end_stop": ["S2", "S3"],
            "distance": [100.0, 150.0],
            "V_max": [8.0, 9.0],
            **{f"{hour:02d}:00:00": [8.0, 9.0] for hour in range(24)},
        }
    ).to_excel(line_dir / "route_news.xlsx", index=False)
    pd.DataFrame({"launch_time": [0, 120], "direction": [1, 1]}).to_excel(
        line_dir / "time_table.xlsx",
        index=False,
    )


def test_safe_id_is_sumo_xml_friendly():
    assert _safe_id("123 / a-b") == "id_123_a_b"


def test_build_city_sumo_and_smoke(tmp_path):
    if not shutil.which("netconvert") or not shutil.which("sumo"):
        pytest.skip("SUMO binaries are not installed")
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

    result = build_city_sumo(
        city_key="toy",
        line_manifest=manifest,
        out_dir=tmp_path / "sumo",
        max_lines=0,
        max_departures_per_line=2,
        duration_sec=300.0,
    )
    smoke = run_sumo_smoke(result)

    assert result["lines_built"] == 1
    assert result["edges"] == 2
    assert result["vehicles"] == 2
    assert smoke["ok"] is True
    assert smoke["tripinfo_count"] >= 1
