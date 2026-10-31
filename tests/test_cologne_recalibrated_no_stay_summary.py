"""S metrics exercise summary arithmetic; fixtures are not V154U simulation evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_recalibrated_no_stay as summary

RESULTS = Path(__file__).resolve().parents[1] / "cf_h2o/results"


def inputs():
    baseline = json.loads((RESULTS / "paper_artifacts/tsc_v154s_cologne_no_stay_closed_loop_v1.json").read_text())
    calibration = json.loads((RESULTS / "paper_artifacts/tsc_v154t_cologne_no_stay_oof_threshold_v1.json").read_text())
    protocol = {key: baseline[key] for key in ("source_root", "threshold_originator_path", "ablation_originator_path")}
    protocol.update(protocol=summary.PROTOCOL, sumocfg=baseline["repaired_sumocfg"],
                    frozen_threshold=calibration["frozen_threshold"],
                    controls={"no_stay_threshold_zero_tasks": baseline["new_task_ids"],
                              "phase_pressure_tasks": baseline["reused_task_ids"][summary.PP]})
    rows = []
    for seed in summary.SEEDS:
        path = RESULTS / f"cluster/tsc_v154s_cologne_no_stay_closed_loop_20260909/evaluation/seed_{seed}/rigid_thresholded/result.json"
        row = json.loads(path.read_text())
        row.update(protocol=summary.PROTOCOL, frozen_threshold=deepcopy(calibration["frozen_threshold"]),
                   task_id=f"fixture-{seed}", source_result=str(path))
        row["originator_diagnostics"].update(threshold=summary.THRESHOLD, threshold_rejected_proposal_count=4)
        row["originator_diagnostics"]["phase_pressure_override_count"] -= 4
        rows.append(row)
    return protocol, rows, baseline


def test_nine_cells_means_and_uncapped_accounting():
    protocol, rows, baseline = inputs()
    original = deepcopy(baseline)
    result = summary.aggregate(protocol, list(reversed(rows)), baseline)
    service, waiting = result["all_seed_service"], "mean_tripinfo_waiting_time"
    assert result["matrix_status"] == "COMPLETE"
    assert len(result["cells"]) == 9 and result["reused_cell_count"] == 6
    assert service["means_by_arm"][summary.NEW][waiting] == pytest.approx(17.10784533957594)
    assert service["means_by_arm"][summary.PP][waiting] == pytest.approx(14.037386269644335)
    assert service["recalibrated_vs_threshold_zero"][waiting]["difference_mean"] == 0
    assert service["recalibrated_vs_phase_pressure"]["arrived"]["difference_mean"] == pytest.approx(-4.3333333333333)
    assert result["additional_seed_service"]["seeds"] == [52282, 63792]
    assert result["reused_task_ids"] == {summary.ZERO: ["t91871", "t91872", "t91873"],
                                         summary.PP: ["t91636", "t91638", "t91640"]}
    totals = result["execution_diagnosis"]["totals"]
    assert totals["ungated_rigid_proposals_on_no_stay_trajectory"] == 692
    assert totals["accepted_overrides"] == 239
    assert totals["threshold_rejected_proposals"] == 12 and totals["stay_veto_count"] == 441
    assert totals["executed_stay_overrides"] == 0
    assert result["execution_diagnosis"]["threshold_zero_baseline"]["totals"]["accepted_overrides"] == 251
    assert result["execution_diagnosis"]["maximum_green_elapsed_sec_by_seed"][0]["by_tls"] == {"cluster_357187_359543": 20.0}
    assert baseline == original


@pytest.mark.parametrize("change", ["duplicate_seed", "model", "network", "threshold", "originator", "task", "pp_protocol"])
def test_wrong_frozen_comparison_rejected(change):
    protocol, rows, baseline = inputs()
    if change == "duplicate_seed":
        rows[0]["seed"] = rows[1]["seed"]
    elif change == "model":
        rows[0]["model_path"] = "/different.pkl"
    elif change == "network":
        baseline["repaired_sumocfg"] = "/different.sumocfg"
    elif change == "threshold":
        rows[0]["frozen_threshold"]["threshold"] = 0
    elif change == "originator":
        rows[0]["ablation_originator_path"] = "/different.py"
    else:
        row = next(row for row in baseline["cells"] if row["arm"] == summary.PP)
        row["task_id" if change == "task" else "source_protocol"] = "different"
    with pytest.raises(ValueError):
        summary.aggregate(protocol, rows, baseline)


@pytest.mark.parametrize("change", ["observed_threshold", "lost_veto", "executed_stay"])
def test_execution_binding_and_accounting_rejected(change):
    protocol, rows, baseline = inputs()
    if change == "observed_threshold":
        rows[0]["originator_diagnostics"]["threshold"] = 0
    elif change == "lost_veto":
        rows[0]["originator_diagnostics"]["model_stay_veto_count"] = 0
    else:
        rows[0]["metrics"]["guard_audit"]["executed_stay_overrides"] = 1
    with pytest.raises(ValueError):
        summary.aggregate(protocol, rows, baseline)


def test_complete_collision_kept_and_invalid_run_disables_means():
    protocol, rows, baseline = inputs()
    rows[0].update(status="FAIL", zero_incident_status="FAIL")
    rows[0]["metrics"].update(collision_incidents=1, collision_events=2)
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "COMPLETE" and result["all_seed_service"] is not None
    assert result["safety_by_arm"][summary.NEW]["zero_incident_status"] == "FAIL"
    assert result["safety_by_arm"][summary.NEW]["collision_incidents_total"] == 1
    rows[0].update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = summary.aggregate(protocol, rows, baseline)
    assert result["matrix_status"] == "INVALID" and len(result["cells"]) == 9
    assert result["all_seed_service"] is result["additional_seed_service"] is None
