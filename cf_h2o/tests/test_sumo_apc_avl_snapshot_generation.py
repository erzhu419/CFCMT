import shutil

import pandas as pd
import pytest

from cf_h2o.eval.sumo_apc_avl_snapshot_generation import _hourly_od, _load_libsumo, _snapshot_sumocfg, generate_city_snapshots
from cf_h2o.eval.sumo_apc_avl_sumo_generation import build_city_sumo


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
    pd.DataFrame(
        {
            "time_period": ["00:00:00", "00:00:00", "00:00:00"],
            "stop_name": ["S1", "S2", "S3"],
            "S1": [0.0, 0.0, 0.0],
            "S2": [2.0, 0.0, 0.0],
            "S3": [1.0, 3.0, 0.0],
        }
    ).to_excel(line_dir / "passenger_OD.xlsx", index=False)


def test_loads_libsumo_not_traci():
    api = _load_libsumo()
    assert api.__name__ == "libsumo"
    assert hasattr(api, "vehicle")


def test_hourly_od_collapses_duplicate_ordered_stop_labels():
    passenger_od = pd.DataFrame(
        {
            "time_period": ["00:00:00", "00:00:00"],
            "stop_name": ["S1", "S2"],
            "S1": [0.0, 0.0],
            "S2": [2.0, 0.0],
            "S3": [1.0, 3.0],
        }
    )

    hourly = _hourly_od(passenger_od, ["S1", "S2", "S1", "S3"])

    matrix = hourly[0]
    assert list(matrix.index) == ["S1", "S2", "S3"]
    assert list(matrix.columns) == ["S1", "S2", "S3"]
    assert float(matrix.loc["S1", "S2"]) == 2.0


def test_generate_city_snapshots_with_libsumo(tmp_path):
    if not shutil.which("netconvert") or not shutil.which("sumo"):
        pytest.skip("SUMO binaries are not installed")
    _load_libsumo()
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
    city_result = build_city_sumo(
        city_key="toy",
        line_manifest=manifest,
        out_dir=tmp_path / "sumo",
        max_lines=0,
        max_departures_per_line=2,
        duration_sec=300.0,
    )

    summary = generate_city_snapshots(
        city_key="toy",
        city_result=city_result,
        line_manifest=manifest,
        snapshot_period=30.0,
        vehicle_capacity=4.0,
    )
    avl = pd.read_csv(summary["avl_path"])
    apc = pd.read_csv(summary["apc_path"])

    assert summary["ok"] is True
    assert len(avl) > 0
    assert len(apc) > 0
    assert avl["occupancy"].min() >= 0.0
    assert avl["occupancy"].max() <= 4.0
    assert apc[["boardings", "alightings", "occupancy_after"]].min().min() >= 0.0
    assert summary["keep_sumo_output"] is False
    assert summary["sumocfg_path"].endswith("simulation.snapshot.sumocfg")


def test_snapshot_sumocfg_disables_summary_and_tripinfo(tmp_path):
    cfg = tmp_path / "simulation.sumocfg"
    cfg.write_text(
        """<?xml version='1.0' encoding='utf-8'?>
<configuration>
  <input>
    <net-file value="net.net.xml" />
    <route-files value="routes.rou.xml" />
  </input>
  <time>
    <begin value="0.0" />
    <end value="60.0" />
  </time>
  <output>
    <summary-output value="summary.xml" />
    <tripinfo-output value="tripinfo.xml" />
    <tripinfo-output.write-unfinished value="true" />
  </output>
</configuration>
""",
        encoding="utf-8",
    )
    path = _snapshot_sumocfg({"sumocfg_path": str(cfg), "out_dir": str(tmp_path)}, keep_sumo_output=False)
    text = path.read_text(encoding="utf-8")

    assert path.name == "simulation.snapshot.sumocfg"
    assert "summary-output" not in text
    assert "tripinfo-output" not in text
    assert "no-step-log" in text
