import json
from argparse import Namespace

import pandas as pd

from cf_h2o.eval.sumo_apc_avl_input_validation import LineTask, run, validate_line_task


def _write_line(line_dir, *, negative_od: bool = False) -> None:
    line_dir.mkdir(parents=True)
    stops = pd.DataFrame(
        {
            "stop_id": [0, 1, 2],
            "stop_name": ["S1", "S2", "S3"],
            "latitude": [42.0, 42.001, 42.002],
            "longitude": [-71.0, -71.001, -71.002],
        }
    )
    route = pd.DataFrame(
        {
            "route_id": [0, 1],
            "start_stop": ["S1", "S2"],
            "end_stop": ["S2", "S3"],
            "distance": [100.0, 150.0],
            "V_max": [8.0, 9.0],
            **{f"{hour:02d}:00:00": [8.0, 9.0] for hour in range(24)},
        }
    )
    timetable = pd.DataFrame({"launch_time": [0, 600, 1200], "direction": [1, 1, 1]})
    od_value = -1.0 if negative_od else 2.0
    passenger_od = pd.DataFrame(
        {
            "time_period": ["00:00:00", "00:00:00", "00:00:00"],
            "stop_name": ["S1", "S2", "S3"],
            "S1": [0.0, 0.0, 0.0],
            "S2": [od_value, 0.0, 0.0],
            "S3": [0.0, 1.0, 0.0],
        }
    )
    stops.to_excel(line_dir / "stop_news.xlsx", index=False)
    route.to_excel(line_dir / "route_news.xlsx", index=False)
    timetable.to_excel(line_dir / "time_table.xlsx", index=False)
    passenger_od.to_excel(line_dir / "passenger_OD.xlsx", index=False)


def test_validate_line_task_extracts_sumo_input_contract(tmp_path):
    line_dir = tmp_path / "env" / "data" / "L1"
    _write_line(line_dir)
    task = LineTask(
        city_key="toy",
        city_name="Toy",
        env_path=str(tmp_path / "env"),
        line_key="L1",
        line_dir=str(line_dir),
        manifest_line={"line_key": "L1", "stops": 3, "segments": 2, "timetable_rows": 3},
    )

    summary = validate_line_task(task)

    assert summary["ok"] is True
    assert summary["stops"] == 3
    assert summary["segments"] == 2
    assert summary["schedule"]["valid_departures"] == 3
    assert summary["demand"]["total_od_demand"] == 3.0
    assert summary["sumo_input_contract"]["segments_are_chain"] is True


def test_validate_line_task_rejects_negative_od(tmp_path):
    line_dir = tmp_path / "env" / "data" / "L1"
    _write_line(line_dir, negative_od=True)
    task = LineTask(
        city_key="toy",
        city_name="Toy",
        env_path=str(tmp_path / "env"),
        line_key="L1",
        line_dir=str(line_dir),
        manifest_line={},
    )

    summary = validate_line_task(task)

    assert summary["ok"] is False
    assert any("negative" in error for error in summary["errors"])


def test_run_writes_line_manifest_and_report(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    env = root / "city_env"
    line_dir = env / "data" / "L1"
    _write_line(line_dir)
    (env / "gtfs_city_manifest.json").write_text(
        json.dumps({"line_count": 1, "failure_count": 0, "lines": [{"line_key": "L1", "stops": 3, "segments": 2}]}),
        encoding="utf-8",
    )
    config = root / "config.json"
    config.write_text(
        json.dumps({"generated_envs": {"toy": {"city": "Toy", "env_path": "city_env", "line_count": 1}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr("cf_h2o.eval.sumo_apc_avl_input_validation._repo_root", lambda: root)

    out = root / "results.json"
    md_out = root / "results.md"
    result = run(
        Namespace(
            config=config,
            out=out,
            md_out=md_out,
            input_dir=root / "inputs",
            cities=["toy"],
            workers=1,
            max_lines_per_city=0,
            progress_every=0,
        )
    )

    assert result["ok"] is True
    assert out.exists()
    assert md_out.exists()
    manifest_path = root / "inputs" / "sumo_apc_avl_line_inputs_full.jsonl"
    assert manifest_path.exists()
    assert len(manifest_path.read_text(encoding="utf-8").splitlines()) == 1
