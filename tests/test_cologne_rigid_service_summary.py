from copy import deepcopy
import json

from scripts.data.summarize_cologne_rigid_service_failure import (
    ARMS, DEFAULT_INPUT, TRACE_LIMIT, sampled_runs, summarize_results,
)


def _trace(time_sec, *, speed=0.0, vehicles=10.0, phase="Gr"):
    return {
        "time_sec": time_sec, "tls_id": "tls", "selected_phase_state": phase,
        "current_green_state_before": phase, "green_elapsed_sec_before": time_sec,
        "total_vehicles": vehicles, "mean_speed": speed, "total_queue": 13.5,
        "override_kind": "stay",
    }


def test_stationary_runs_do_not_bridge_missing_decisions_or_count_empty_lanes():
    # Override-only traces omit pressure agreements; a gap cannot prove a hold.
    trace = [_trace(0), _trace(10), _trace(30), _trace(40, vehicles=0), _trace(50)]
    runs = sampled_runs(trace, 10, stationary=True)
    assert [(row["start_time_sec"], row["end_time_sec"]) for row in runs] == [
        (0, 10), (30, 30), (50, 50)
    ]
    assert runs[0]["sampled_span_sec"] == 10
    assert runs[0]["last_queue_proxy"] == 13.5


def test_retained_cologne_evidence_preserves_full_run_and_trace_boundaries():
    # Regression against retained evidence detects capped traces being treated
    # as full runs, source changes being lost, or pending demand being dropped.
    summary = summarize_results(DEFAULT_INPUT)
    rigid = summary["arms"]["rigid_target_only"]
    source = summary["arms"]["selected_source_corrected_rigid"]
    assert rigid["trace_coverage"]["retained_rows"] == TRACE_LIMIT
    assert rigid["trace_coverage"]["unretained_accepted_overrides"] == 96
    assert rigid["phase_execution_audit"]["same_phase_requests"] == 338
    assert rigid["full_run_guard_counts"]["executed_stay_overrides"] == 333
    suffix = rigid["stationary_suffix_of_retained_trace"]
    assert (suffix["start_time_sec"], suffix["end_time_sec"]) == (26700.0, 27830.0)
    assert suffix["sampled_span_sec"] == 1130.0
    assert rigid["endpoints"]["arrived_plus_active_plus_pending"] == 2015
    assert source["trace_coverage"]["source_changes_in_full_run"] == 2
    assert source["trace_coverage"]["source_changes_in_retained_override_rows"] == 1
    assert summary["arms"]["phase_pressure"]["trace_coverage"]["retained_rows"] == 0


def test_mixed_seeds_are_not_reported_as_paired_arms(tmp_path):
    import pytest

    for arm in ARMS:
        payload = json.loads((DEFAULT_INPUT / arm / "result.json").read_text())
        if arm == "phase_pressure":
            payload = deepcopy(payload)
            payload["seed"] += 1
        output = tmp_path / arm / "result.json"
        output.parent.mkdir()
        output.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="one city/scenario/seed"):
        summarize_results(tmp_path)
