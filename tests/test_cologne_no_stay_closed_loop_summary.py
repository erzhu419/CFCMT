"""Arithmetic/provenance fixtures reuse Q metrics, not claims of new S evaluation."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_no_stay_closed_loop as summary

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "cf_h2o/results"


def inputs():
    protocol = json.loads((RESULTS / "cluster/tsc_v154s_cologne_no_stay_closed_loop_20260909/protocol_v1.json").read_text())
    baseline = json.loads((RESULTS / "paper_artifacts/tsc_v154q_cologne_native_label_closed_loop_v1.json").read_text())
    rows = []
    for seed in summary.SEEDS:
        path = RESULTS / f"cluster/tsc_v154q_cologne_native_label_closed_loop_20260909/evaluation/seed_{seed}/rigid_thresholded/result.json"
        row = json.loads(path.read_text())
        row.update(protocol=summary.PROTOCOL, ablation_originator_path=protocol["ablation_originator_path"],
                   task_id=f"fixture-{seed}", source_result=str(path), maximum_green_elapsed_sec_by_tls={"fixture": 50})
        row["originator_diagnostics"].update(protocol=summary.ORIGINATOR_PROTOCOL, model_stay_veto_count=3)
        row["originator_diagnostics"]["phase_pressure_override_count"] -= 3
        row["metrics"]["guard_audit"]["executed_stay_overrides"] = 0
        rows.append(row)
    return protocol, rows, baseline


def test_twelve_cell_arithmetic_and_original_bindings():
    protocol, rows, baseline = inputs()
    original = deepcopy(baseline)
    result = summary.aggregate(protocol, list(reversed(rows)), baseline)
    service = result["all_seed_service"]
    waiting = "mean_tripinfo_waiting_time"
    assert result["matrix_status"] == "COMPLETE"
    assert len(result["cells"]) == 12 and result["reused_cell_count"] == 9
    assert service["means_by_arm"][summary.NEW][waiting] == pytest.approx(274.701104121418)
    assert service["means_by_arm"][summary.PP][waiting] == pytest.approx(14.037386269644335)
    assert service["no_stay_vs_native"][waiting]["difference_mean"] == 0
    assert service["no_stay_vs_phase_pressure"]["arrived"]["difference_mean"] == pytest.approx(1012 - 1997.6666666666667)
    assert result["additional_seed_service"]["seeds"] == [52282, 63792]
    assert result["reused_task_ids"][summary.NATIVE] == ["t91851", "t91852", "t91853"]
    assert result["reused_task_ids"][summary.OLD] == ["t91746", "t91747", "t91748"]
    assert result["reused_task_ids"][summary.PP] == ["t91636", "t91638", "t91640"]
    assert result["execution_diagnosis"]["totals"]["accepted_overrides"] == 879
    assert result["execution_diagnosis"]["totals"]["stay_veto_count"] == 9
    assert result["execution_diagnosis"]["native_baseline"]["executed_stay_overrides_total"] == 808
    assert baseline == original


@pytest.mark.parametrize("change", ["seed", "model", "q_row", "ablation", "baseline_source"])
def test_wrong_roster_model_or_mixed_comparison_is_rejected(change):
    protocol, rows, baseline = inputs()
    if change == "seed":
        rows[0]["seed"] = rows[1]["seed"]
    elif change == "model":
        rows[0]["model_path"] = baseline["original_model_path"]
    elif change == "q_row":
        rows[0]["protocol"] = summary.BASELINE_PROTOCOL
    elif change == "ablation":
        rows[0]["ablation_originator_path"] = "/different.py"
    else:
        next(row for row in baseline["cells"] if row["arm"] == summary.PP)["source_protocol"] = "v154f"
    with pytest.raises(ValueError):
        summary.aggregate(protocol, rows, baseline)


@pytest.mark.parametrize("change", ["lost_veto", "executed_stay"])
def test_incomplete_proposal_accounting_or_executed_stay_is_rejected(change):
    protocol, rows, baseline = inputs()
    if change == "lost_veto":
        rows[0]["originator_diagnostics"]["model_stay_veto_count"] = 0
    else:
        rows[0]["metrics"]["guard_audit"]["executed_stay_overrides"] = 1
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
    assert result["matrix_status"] == "INVALID" and len(result["cells"]) == 12
    assert result["all_seed_service"] is result["additional_seed_service"] is None
