from copy import deepcopy
import json
import sys

import pytest

from scripts.data import summarize_cologne_threshold_closed_loop as summary


def inputs():
    baseline = {"protocol": summary.BASELINE_PROTOCOL, "repaired_sumocfg": "/fixed/cologne1.sumocfg",
                "source_root": "/j/source_v2", "baseline_source_bindings": {"runtime": "/i/runtime.py"},
                "baseline_executor_source": "/i/executor.py", "cells": [],
                "training": {"status": "PASS", "model_path": "/j/model.pkl", "training_group_count": 100,
                             "frozen_candidate": "constant_alpha_0", "occupancy_equation_protocol": "fraction"}}
    frozen = {"protocol": summary.THRESHOLD_PROTOCOL, "threshold": summary.THRESHOLD,
              "mode": "minimum_advantage", "comparison": "predicted_advantage > threshold + 1e-12",
              "model_path": "/j/model.pkl", "source_root": "/j/source_v2", "selection_data": "original B100 OOF"}
    new = []
    for index, seed in enumerate(summary.SEEDS, 1):
        for arm in (summary.UNGATED, summary.PP):
            row = {"seed": seed, "arm": arm, "run_valid": True, "status": "PASS",
                   "zero_incident_status": "PASS", "source_result": f"/old/{seed}/{arm}/result.json",
                   "source_protocol": summary.BASELINE_PROTOCOL if arm == summary.UNGATED else "v154i",
                   "reused": arm == summary.PP, "task_id": f"old-{seed}-{arm}", "elapsed_sec": 7,
                   "metrics": {**dict.fromkeys(summary.SERVICE_KEYS, index * 20 if arm == summary.UNGATED else 8),
                               **dict.fromkeys(("collision_incidents", "collision_events",
                                                "starting_teleports", "ending_teleports"), 0)}}
            if arm == summary.UNGATED:
                row.update(source_root="/j/source_v2", model_path="/j/model.pkl")
            baseline["cells"].append(row)
        row = deepcopy(baseline["cells"][-2])
        row.update(protocol=summary.PROTOCOL, scenario="cologne1", arm=summary.GATED, reused=False,
                   repaired_sumocfg=baseline["repaired_sumocfg"], occupancy_equation_protocol="fraction",
                   frozen_threshold=deepcopy(frozen), elapsed_sec=2, task_id=f"new-{seed}",
                   collision_inventory={"incident_counts_by_first_participant_lane_pair": []})
        row["metrics"].update(dict.fromkeys(summary.SERVICE_KEYS, index * 10))
        new.append(row)
    return new, baseline


def test_three_arm_seed_pairing_percentage_means_and_reused_provenance():
    rows, baseline = inputs()
    original_baseline = deepcopy(baseline)
    result = summary.aggregate(list(reversed(rows)), baseline)
    service = result["all_seed_service"]
    metric = "mean_tripinfo_waiting_time"
    assert result["matrix_status"] == "COMPLETE"
    assert [service["means_by_arm"][arm][metric] for arm in summary.ARMS] == [40, 20, 8]
    assert service["new_vs_ungated"][metric]["relative_difference_of_means_percent"] == -50
    assert service["new_vs_phase_pressure"][metric]["relative_difference_of_means_percent"] == 150
    assert [row["new_minus_reference"] for row in service["new_vs_ungated"][metric]["paired_differences"]] == [-10, -20, -30]
    assert result["additional_seed_service"]["seeds"] == [52282, 63792]
    assert result["new_simulation_elapsed_seconds_sum"] == 6
    assert result["reused_cell_count"] == 6
    assert result["new_task_ids"] == [f"new-{seed}" for seed in summary.SEEDS]
    assert result["reused_task_ids"][summary.PP] == [f"old-{seed}-phase_pressure" for seed in summary.SEEDS]
    assert all(row["source_protocol"] == "v154i" and "source_root" not in row
               for row in result["cells"] if row["arm"] == summary.PP)
    assert baseline == original_baseline


def test_valid_collision_fail_remains_in_all_and_additional_seed_means():
    rows, baseline = inputs()
    rows[0].update(status="FAIL", zero_incident_status="FAIL")
    rows[0]["metrics"].update(collision_incidents=1, collision_events=2, mean_tripinfo_waiting_time=100)
    result = summary.aggregate(rows, baseline)
    assert result["matrix_status"] == "COMPLETE"
    assert result["safety_by_arm"][summary.GATED]["collision_incidents_total"] == 1
    assert result["safety_by_arm"][summary.GATED]["zero_incident_status"] == "FAIL"
    assert result["all_seed_service"]["means_by_arm"][summary.GATED]["mean_tripinfo_waiting_time"] == 50
    assert result["additional_seed_service"]["means_by_arm"][summary.GATED]["mean_tripinfo_waiting_time"] == 65


@pytest.mark.parametrize("arm", summary.ARMS)
def test_invalid_run_prevents_complete_means(arm):
    rows, baseline = inputs()
    row = rows[0] if arm == summary.GATED else next(r for r in baseline["cells"] if r["arm"] == arm)
    row.update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = summary.aggregate(rows, baseline)
    assert result["matrix_status"] == "INVALID"
    assert result["all_seed_service"] is result["additional_seed_service"] is None
    assert result["safety_by_arm"][arm]["zero_incident_status"] == "UNAVAILABLE"
    assert len(result["cells"]) == 9


@pytest.mark.parametrize("key", ["seed", "repaired_sumocfg", "source_root", "model_path", "protocol"])
def test_mismatched_new_run_is_rejected(key):
    rows, baseline = inputs()
    rows[0][key] = "different"
    with pytest.raises(ValueError):
        summary.aggregate(rows, baseline)


def test_exact_frozen_threshold_and_original_b100_are_required():
    rows, baseline = inputs()
    for row in rows:
        row["frozen_threshold"]["threshold"] += 0.001
    with pytest.raises(ValueError):
        summary.aggregate(rows, baseline)
    rows, baseline = inputs()
    rows[1]["frozen_threshold"]["selection_data"] = "different calibration"
    with pytest.raises(ValueError):
        summary.aggregate(rows, baseline)
    rows, baseline = inputs()
    baseline["training"]["training_group_count"] = 99
    with pytest.raises(ValueError):
        summary.aggregate(rows, baseline)


def test_cli_reads_frozen_paths_and_new_scheduler_ids(tmp_path, monkeypatch):
    rows, baseline = inputs()
    paths = []
    for row in rows:
        path = tmp_path / "evaluation" / f"seed_{row['seed']}" / summary.GATED / "result.json"
        path.parent.mkdir(parents=True); path.write_text(json.dumps(row)); paths.append(path)
    launch, old, out = [tmp_path / name for name in ("launch.json", "baseline.json", "summary.json")]
    launch.write_text(json.dumps({"scheduler_stdout": json.dumps({"submitted": [
        {"signature": f"CFCMT/v154m/threshold-closed-loop-v1/evaluate/{seed}", "id": f"eval-{seed}"}
        for seed in summary.SEEDS]})}))
    old.write_text(json.dumps(baseline))
    monkeypatch.setattr(sys, "argv", ["summary", "--results-root", str(tmp_path), "--launch-record", str(launch),
                                      "--baseline-summary", str(old), "--out", str(out)])
    summary.main()
    result = json.loads(out.read_text())
    assert result["new_task_ids"] == [f"eval-{seed}" for seed in summary.SEEDS]
    assert result["retrieved_result_bytes"] == sum(path.stat().st_size for path in paths)
    assert result["refs"]["baseline_summary"] == str(old)
