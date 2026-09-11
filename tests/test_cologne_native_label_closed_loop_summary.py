"""Use completed M trajectories as fixtures, never as native-model evaluation claims."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_native_label_closed_loop as summary

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "cf_h2o/results"


def inputs():
    protocol = json.loads((RESULTS / "cluster/tsc_v154q_cologne_native_label_closed_loop_20260909/protocol_v1.json").read_text())
    baseline = json.loads((RESULTS / "paper_artifacts/tsc_v154m_cologne_threshold_closed_loop_v1.json").read_text())
    rows = []
    for seed in summary.SEEDS:
        path = RESULTS / f"cluster/tsc_v154m_cologne_threshold_closed_loop_20260909/evaluation/seed_{seed}/rigid_thresholded/result.json"
        row = json.loads(path.read_text())
        row.update(protocol=summary.PROTOCOL, model_path=protocol["frozen_threshold"]["model_path"],
                   frozen_threshold=deepcopy(protocol["frozen_threshold"]), task_id=f"fixture-{seed}",
                   source_result=str(path))
        row["originator_diagnostics"].update(threshold=0.0, threshold_model_path=row["model_path"])
        rows.append(row)
    return protocol, rows, baseline


def test_actual_nine_cell_metric_arithmetic_and_provenance():
    protocol, rows, baseline = inputs()
    original = deepcopy(baseline)
    result = summary.aggregate(protocol, list(reversed(rows)), baseline)
    service = result["all_seed_service"]
    waiting = "mean_tripinfo_waiting_time"
    assert result["matrix_status"] == "COMPLETE"
    assert len(result["cells"]) == 9
    assert result["reused_cell_count"] == 6
    assert service["means_by_arm"][summary.NEW][waiting] == pytest.approx(15.701571546732836)
    assert service["means_by_arm"][summary.PP][waiting] == pytest.approx(14.037386269644335)
    assert service["new_vs_original_thresholded"][waiting]["difference_mean"] == 0
    assert service["new_vs_phase_pressure"][waiting]["relative_difference_of_means_percent"] == pytest.approx(11.855378523616466)
    assert result["additional_seed_service"]["seeds"] == [52282, 63792]
    assert result["reused_task_ids"][summary.OLD] == ["t91746", "t91747", "t91748"]
    assert result["reused_task_ids"][summary.PP] == ["t91636", "t91638", "t91640"]
    assert result["threshold_application"]["totals"]["accepted_overrides"] == 207
    assert all(row["source_protocol"] == summary.PP_PROTOCOL for row in result["cells"] if row["arm"] == summary.PP)
    assert baseline == original


@pytest.mark.parametrize("change", ["seed", "model", "threshold", "originator", "baseline_source"])
def test_wrong_roster_model_threshold_or_source_is_rejected(change):
    protocol, rows, baseline = inputs()
    if change == "seed":
        rows[0]["seed"] = rows[1]["seed"]
    elif change == "model":
        rows[0]["model_path"] = baseline["model_path"]
    elif change == "threshold":
        rows[0]["frozen_threshold"]["threshold"] = 0.1
    elif change == "originator":
        rows[0]["threshold_originator_path"] = "/different.py"
    else:
        next(row for row in baseline["cells"] if row["arm"] == summary.PP)["source_protocol"] = "v154f"
    with pytest.raises(ValueError):
        summary.aggregate(protocol, rows, baseline)


def test_valid_collision_kept_but_invalid_run_disables_means():
    protocol, rows, baseline = inputs()
    rows[0].update(status="FAIL", zero_incident_status="FAIL")
    rows[0]["metrics"].update(collision_incidents=1, collision_events=2)
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "COMPLETE"
    assert result["all_seed_service"] is not None
    assert result["safety_by_arm"][summary.NEW]["zero_incident_status"] == "FAIL"
    assert result["safety_by_arm"][summary.NEW]["collision_incidents_total"] == 1
    rows[0].update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "INVALID"
    assert len(result["cells"]) == 9
    assert result["all_seed_service"] is result["additional_seed_service"] is None


def test_completed_native_trajectories_keep_large_service_regression():
    protocol, _, baseline = inputs()
    rows = []
    for seed in summary.SEEDS:
        path = RESULTS / f"cluster/tsc_v154q_cologne_native_label_closed_loop_20260909/evaluation/seed_{seed}/rigid_thresholded/result.json"
        row = json.loads(path.read_text())
        row.update(source_result=str(path), task_id=f"native-{seed}")
        rows.append(row)
    result = summary.aggregate(protocol, rows, baseline)
    means = result["all_seed_service"]["means_by_arm"][summary.NEW]
    assert means["mean_tripinfo_waiting_time"] == pytest.approx(274.701104121418)
    assert means["system_vehicle_hours"] == pytest.approx(295.88916666666665)
    assert means["arrived"] == 1012
    assert means["pending_vehicles_at_horizon"] == 802
    assert result["threshold_application"]["totals"]["accepted_overrides"] == 888
    assert result["threshold_application"]["totals"]["threshold_rejected_proposals"] == 0
    assert result["safety_by_arm"][summary.NEW]["zero_incident_status"] == "PASS"
