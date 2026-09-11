"""Synthetic spaced actions on U metrics test arithmetic without asserting X outcomes."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_spaced_no_stay as summary

RESULTS = Path(__file__).resolve().parents[1] / "cf_h2o/results"


def inputs():
    baseline = json.loads((RESULTS / "paper_artifacts/tsc_v154u_cologne_recalibrated_no_stay_v1.json").read_text())
    protocol = {key: baseline[key] for key in ("source_root", "threshold_originator_path", "ablation_originator_path", "frozen_threshold")}
    protocol.update(protocol=summary.PROTOCOL, sumocfg=baseline["repaired_sumocfg"], spacing_contract=deepcopy(summary.SPACING),
                    controls={"continuous_model_tasks": baseline["new_task_ids"],
                              "phase_pressure_tasks": baseline["reused_task_ids"][summary.PP]})
    rows = []
    for seed in summary.SEEDS:
        path = RESULTS / f"cluster/tsc_v154u_cologne_recalibrated_no_stay_20260910/evaluation/seed_{seed}/rigid_thresholded/result.json"
        row = json.loads(path.read_text())
        row.update(protocol=summary.PROTOCOL, spacing_contract=deepcopy(summary.SPACING),
                   task_id=f"fixture-{seed}", source_result=str(path))
        metrics, diag = row["metrics"], row["originator_diagnostics"]
        metrics["residual_deployment"].update(cooldown_intervals=44, cooldown_intervals_override=44)
        # The frozen source need not report this newer descriptive field.
        metrics["residual_deployment"].pop("minimum_same_or_conflicting_intervention_gap_sec", None)
        trace = deepcopy(metrics["accepted_intervention_trace"][:8])
        for index, item in enumerate(trace):
            item["time_sec"] = 25260 + index * 450
        metrics["accepted_intervention_trace"] = trace
        for key in ("accepted_overrides", "executed_overrides", "accepted_switch_overrides", "executed_switch_overrides"):
            metrics["guard_audit"][key] = 8
        metrics["guard_audit"]["rejected_cooldown"] = diag["phase_pressure_override_count"] - 8
        rows.append(row)
    return protocol, rows, baseline


def test_nine_cell_means_and_two_stage_proposal_accounting():
    protocol, rows, baseline = inputs()
    original = deepcopy(baseline)
    result = summary.aggregate(protocol, list(reversed(rows)), baseline)
    assert result["matrix_status"] == "COMPLETE" and len(result["cells"]) == 9
    assert result["reused_cell_count"] == 6 and result["new_simulation_count"] == 3
    service = result["all_seed_service"]
    assert service["means_by_arm"][summary.NEW]["mean_tripinfo_waiting_time"] == pytest.approx(16.57781480376274)
    assert service["spaced_vs_continuous"]["system_vehicle_hours"]["difference_mean"] == 0
    assert result["additional_seed_service"]["seeds"] == [52282, 63792]
    execution = result["execution_diagnosis"]
    assert execution["totals"]["raw_model_proposals"] == 734
    assert execution["totals"]["originator_passed_proposals"] == 209
    assert execution["totals"]["actual_accepted_overrides"] == 24
    assert execution["totals"]["cooldown_rejected_proposals"] == 185
    assert all(item["minimum_intervention_gap_sec"] == 450 for item in execution["spacing_by_seed"])
    assert all(len(item["accepted_intervention_trace"]) == 8 for item in execution["spacing_by_seed"])
    assert result["reused_task_ids"][summary.CONTINUOUS] == ["t91893", "t91894", "t91895"]
    assert baseline == original


@pytest.mark.parametrize("change", ["threshold", "network", "task", "originator", "roster"])
def test_frozen_bindings_rejected(change):
    protocol, rows, baseline = inputs()
    if change == "threshold":
        rows[0]["frozen_threshold"]["threshold"] = 0
    elif change == "network":
        baseline["repaired_sumocfg"] = "different"
    elif change == "task":
        next(row for row in baseline["cells"] if row["arm"] == summary.PP)["task_id"] = "different"
    elif change == "originator":
        rows[0]["ablation_originator_path"] = "different"
    else:
        rows[0]["seed"] = rows[1]["seed"]
    with pytest.raises(ValueError):
        summary.aggregate(protocol, rows, baseline)


@pytest.mark.parametrize("change", ["short_gap", "truncated_trace", "stay", "cooldown_binding", "other_rejection", "lost_proposal"])
def test_spacing_and_execution_failures_rejected(change):
    protocol, rows, baseline = inputs()
    metrics = rows[0]["metrics"]
    if change == "short_gap":
        metrics["accepted_intervention_trace"][1]["time_sec"] -= 10
    elif change == "truncated_trace":
        metrics["accepted_intervention_trace"].pop()
    elif change == "stay":
        metrics["accepted_intervention_trace"][0]["override_kind"] = "stay"
    elif change == "cooldown_binding":
        metrics["residual_deployment"]["cooldown_intervals_override"] = 0
    elif change == "other_rejection":
        metrics["guard_audit"]["rejected_executor"] = 1
    else:
        metrics["guard_audit"]["rejected_cooldown"] -= 1
    with pytest.raises(ValueError):
        summary.aggregate(protocol, rows, baseline)


def test_complete_collision_kept_but_invalid_run_disables_means():
    protocol, rows, baseline = inputs()
    rows[0].update(status="FAIL", zero_incident_status="FAIL")
    rows[0]["metrics"].update(collision_incidents=1, collision_events=2)
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "COMPLETE" and result["all_seed_service"] is not None
    assert result["safety_by_arm"][summary.NEW]["zero_incident_status"] == "FAIL"
    rows[0].update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "INVALID" and len(result["cells"]) == 9
    assert result["all_seed_service"] is result["additional_seed_service"] is None
