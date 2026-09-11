from copy import deepcopy
import json
import sys

import pytest

from scripts.data import summarize_cologne_local_fraction_refit as summary
from scripts.data.summarize_cologne_repaired_controller_comparison import SERVICE_KEYS


def inputs():
    baseline = {
        "protocol": summary.BASELINE_PROTOCOL, "repaired_sumocfg": "/fixed/cologne1.sumocfg",
        "source_bindings": {"runtime": "/old/runtime.py"}, "executor_source": "/old/executor.py",
        "cells": [{"seed": seed, "arm": arm, "run_valid": True, "status": "PASS",
                   "zero_incident_status": "PASS", "source_result": f"/old/{seed}/{arm}/result.json",
                   "task_id": f"old-{seed}-{arm}", "elapsed_sec": 2,
                   "metrics": {**dict.fromkeys(SERVICE_KEYS, 12 if arm == summary.ARMS[0] else 8),
                               **dict.fromkeys(("collision_incidents", "collision_events",
                                                "starting_teleports", "ending_teleports"), 0)}}
                  for seed in summary.SEEDS for arm in summary.ARMS],
    }
    cells = [{**deepcopy(row), "protocol": summary.PROTOCOL, "scenario": "cologne1",
              "repaired_sumocfg": baseline["repaired_sumocfg"], "source_root": "/new/source",
              "model_path": "/new/model.pkl", "occupancy_equation_protocol": "fraction-v1",
              "source_result": f"/new/{row['seed']}/result.json", "task_id": f"new-{row['seed']}",
              "metrics": {**row["metrics"], "mean_tripinfo_waiting_time": 10}}
             for row in baseline["cells"] if row["arm"] == summary.ARMS[0]]
    fit = {"protocol": summary.PROTOCOL, "status": "PASS", "training_group_count": 100,
           "training_row_count": 800, "selected_groups": [f"g{i}" for i in range(100)],
           "frozen_candidate": "constant_alpha_0", "model_path": "/new/model.pkl",
           "occupancy_equation_protocol": "fraction-v1", "pool_groups": 150,
           "pool_rows": 1200, "collection_elapsed_seconds_sum": 90,
           "new_collection_elapsed_seconds_sum": 60, "elapsed_sec": 1,
           "selection": {"2027": {"pool_groups": 50, "quota": 34, "selected_groups": list(range(34))}},
           "cells": [{"result": "/collection/result.json", "reused": True, "elapsed_sec": 30,
                      "retained_groups": ["a", "b"], "censored_groups": [], "unavailable_groups": ["c"]}]}
    return cells, fit, baseline


def test_pairing_training_costs_and_original_control_provenance_are_preserved():
    cells, fit, baseline = inputs()
    baseline["cells"].reverse()
    result = summary.aggregate(list(reversed(cells)), fit, baseline)
    assert result["matrix_status"] == "COMPLETE"
    assert result["all_seed_service"]["metrics"]["mean_tripinfo_waiting_time"]["paired_difference_mean"] == 2
    assert result["additional_seed_service"]["seeds"] == [52282, 63792]
    assert [item["metrics"]["mean_tripinfo_waiting_time"]["after_minus_before"]
            for item in result["changes_from_v154i_rigid"]] == [-2, -2, -2]
    assert result["baseline_source_bindings"] == {"runtime": "/old/runtime.py"}
    assert result["source_root"] == "/new/source"
    assert result["cells"][0]["model_path"] == "/new/model.pkl"
    assert all(row["source_result"].startswith("/old/") and "source_root" not in row
               for row in result["cells"][3:])
    assert result["reused_phase_pressure_task_ids"] == [f"old-{seed}-phase_pressure" for seed in summary.SEEDS]
    assert result["new_simulation_elapsed_seconds_sum"] == 6
    assert result["training"]["collection_elapsed_seconds_sum"] == 90
    assert result["training"]["new_collection_elapsed_seconds_sum"] == 60
    assert result["training"]["selection"]["2027"]["selected_group_count"] == 34
    assert result["training"]["collection_cells"][0]["unavailable_groups_count"] == 1


def test_valid_collision_fail_stays_in_both_means():
    cells, fit, baseline = inputs()
    cells[0].update(status="FAIL", zero_incident_status="FAIL")
    cells[0]["metrics"].update(collision_incidents=1, collision_events=2, mean_tripinfo_waiting_time=16)
    result = summary.aggregate(cells, fit, baseline)
    assert result["matrix_status"] == "COMPLETE"
    assert result["safety_by_arm"][summary.ARMS[0]]["zero_incident_status"] == "FAIL"
    assert result["all_seed_service"]["metrics"]["mean_tripinfo_waiting_time"]["paired_difference_mean"] == 4
    assert result["additional_seed_service"]["metrics"]["mean_tripinfo_waiting_time"]["paired_difference_mean"] == 5


@pytest.mark.parametrize("invalid", ["new", "old_rigid", "pp", "fit"])
def test_invalid_input_cannot_produce_complete_service_means(invalid):
    cells, fit, baseline = inputs()
    if invalid == "fit":
        fit["training_group_count"] = 99
    else:
        row = cells[0] if invalid == "new" else baseline["cells"][int(invalid == "pp")]
        row.update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = summary.aggregate(cells, fit, baseline)
    assert result["matrix_status"] == "INVALID"
    assert result["all_seed_service"] is result["additional_seed_service"] is None


@pytest.mark.parametrize("key", ["seed", "source_root", "model_path", "repaired_sumocfg", "protocol"])
def test_mismatched_new_cell_is_rejected(key):
    cells, fit, baseline = inputs()
    cells[0][key] = "different"
    with pytest.raises(ValueError):
        summary.aggregate(cells, fit, baseline)


def test_cli_reads_fixed_result_paths_and_binds_evaluation_task_ids(tmp_path, monkeypatch):
    cells, fit, baseline = inputs()
    paths = []
    for row in cells:
        path = tmp_path / "evaluation" / f"seed_{row['seed']}" / summary.ARMS[0] / "result.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row)); paths.append(path)
    fit_path = tmp_path / "fit_v2" / "result.json"
    fit_path.parent.mkdir(); fit_path.write_text(json.dumps(fit))
    launch, old, output = [tmp_path / name for name in ("launch.json", "baseline.json", "summary.json")]
    launch.write_text(json.dumps({"scheduler_stdout": json.dumps({"submitted": [
        {"signature": f"CFCMT/v154j/local-fraction-v2/evaluate/{seed}", "id": f"eval-{seed}"}
        for seed in summary.SEEDS]})}))
    old.write_text(json.dumps(baseline))
    monkeypatch.setattr(sys, "argv", ["summary", "--results-root", str(tmp_path), "--launch-record", str(launch),
                                      "--baseline-summary", str(old), "--out", str(output)])
    summary.main()
    result = json.loads(output.read_text())
    assert result["refs"]["fit_result"] == str(fit_path)
    assert result["retrieved_result_bytes"] == sum(path.stat().st_size for path in [*paths, fit_path])
    assert [row["task_id"] for row in result["cells"][:3]] == [f"eval-{seed}" for seed in summary.SEEDS]
