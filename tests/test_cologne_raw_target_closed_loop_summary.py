"""X metric fixtures test B summary logic, not claims about B simulations."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_raw_target_closed_loop as summary

RESULTS = Path(__file__).resolve().parents[1] / "cf_h2o/results"


def inputs():
    protocol = json.loads((RESULTS / "cluster/tsc_v155b_cologne_raw_target_closed_loop_20260910/protocol_v1.json").read_text())
    baseline = json.loads((RESULTS / "paper_artifacts/tsc_v154x_cologne_spaced_no_stay_v1.json").read_text())
    rows = []
    for seed in summary.SEEDS:
        path = RESULTS / f"cluster/tsc_v154x_cologne_spaced_no_stay_20260910/evaluation/seed_{seed}/rigid_thresholded/result.json"
        row = json.loads(path.read_text())
        row.update(protocol=summary.PROTOCOL, model_path=protocol["frozen_threshold"]["model_path"],
                   frozen_threshold=deepcopy(protocol["frozen_threshold"]), status="COMPLETE",
                   task_id=f"fixture-{seed}", source_result=str(path))
        diag = row["originator_diagnostics"]
        diag.update(threshold=0, threshold_model_path=protocol["frozen_threshold"]["model_path"])
        diag["model_stay_veto_count"] += diag["threshold_rejected_proposal_count"]
        diag["threshold_rejected_proposal_count"] = 0
        rows.append(row)
    return protocol, rows, baseline


def test_primary_scalar_cost_and_nine_cell_arithmetic():
    protocol, rows, baseline = inputs()
    original = deepcopy(baseline)
    result = summary.aggregate(protocol, list(reversed(rows)), baseline)
    service = result["all_seed_service"]
    assert result["matrix_status"] == "COMPLETE" and len(result["cells"]) == 9
    assert result["primary_efficiency"]["metric"] == "mean_queue"
    assert service["means_by_arm"][summary.NEW]["mean_queue"] == pytest.approx(7.282876776805046)
    assert service["means_by_arm"][summary.NEW]["mean_queue_per_lane"] == pytest.approx(7.282876776805046 / 8)
    assert service["raw_target_vs_normalized_target"]["mean_queue"]["difference_mean"] == 0
    assert service["raw_target_vs_phase_pressure"]["mean_queue"]["relative_difference_of_means_percent"] == pytest.approx(.6806085134625217)
    assert result["additional_seed_service"]["seeds"] == [52282, 63792]
    assert result["reused_task_ids"][summary.NORMALIZED] == ["t91944", "t91945", "t91946"]
    assert result["execution_diagnosis"]["totals"]["actual_accepted_overrides"] == 24
    assert result["execution_diagnosis"]["normalized_target_baseline"]["totals"]["actual_accepted_overrides"] == 24
    assert baseline == original


@pytest.mark.parametrize("change", ["old_model", "score_units", "sample_denominator", "pp_task", "spacing"])
def test_frozen_model_units_and_cost_denominator_rejected(change):
    protocol, rows, baseline = inputs()
    if change == "old_model":
        rows[0]["model_path"] = baseline["model_path"]
    elif change == "score_units":
        rows[0]["frozen_threshold"]["score_units"] = "group_normalized"
    elif change == "sample_denominator":
        rows[0]["metrics"]["fixed_horizon_sample_seconds"] -= 1
    elif change == "pp_task":
        next(row for row in baseline["cells"] if row["arm"] == summary.PP)["task_id"] = "different"
    else:
        rows[0]["metrics"]["accepted_intervention_trace"][1]["time_sec"] = 25300
    with pytest.raises(ValueError):
        summary.aggregate(protocol, rows, baseline)


def test_valid_collisions_are_diagnostic_and_invalid_runs_disable_efficiency_means():
    protocol, rows, baseline = inputs()
    rows[0].update(status="COMPLETE", zero_incident_status="FAIL")
    rows[0]["metrics"].update(collision_incidents=1, collision_events=2)
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "COMPLETE" and result["all_seed_service"] is not None
    assert result["safety_by_arm"][summary.NEW]["zero_incident_status"] == "FAIL"
    assert result["safety_by_arm"][summary.NORMALIZED]["zero_incident_status"] == "FAIL"
    rows[0].update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "INVALID" and len(result["cells"]) == 9
    assert result["all_seed_service"] is result["additional_seed_service"] is None
