from copy import deepcopy

import pytest

from scripts.data.summarize_cologne_priority_yellow_comparison import (
    ARMS, BASELINE_PROTOCOL, EXPECTED, PROTOCOL, SERVICE_KEYS, aggregate,
)


def cells():
    return [{"protocol": PROTOCOL, "seed": seed, "arm": arm, "run_valid": True,
             "status": "PASS", "zero_incident_status": "PASS", "elapsed_sec": 2.0,
             "repaired_sumocfg": "/v154d/cologne1.sumocfg", "source_bindings": {"runtime": "old85"},
             "executor_source": "/v154i/safe_phase_controller.py",
             "metrics": {**{key: 10.0 if arm == ARMS[0] else 8.0 for key in SERVICE_KEYS},
                         "collision_incidents": 0, "collision_events": 0},
             "collision_inventory": {"incident_counts_by_first_participant_lane_pair": []}}
            for seed, arm in EXPECTED]


@pytest.mark.parametrize("change", ["missing", "duplicate", "executor_source", "source_bindings", "repaired_sumocfg", "protocol", "reused"])
def test_refuses_incomplete_or_mismatched_experiments(change):
    rows = cells()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    else:
        rows[0][change] = True if change == "reused" else "different"
    with pytest.raises(ValueError):
        aggregate(rows)


def test_valid_collision_remains_in_both_prespecified_service_summaries():
    rows = cells()
    rows[0]["metrics"].update(collision_incidents=1, collision_events=2, mean_tripinfo_waiting_time=16)
    rows[0].update(status="FAIL", zero_incident_status="FAIL")
    result = aggregate(rows)
    assert result["matrix_status"] == "COMPLETE"
    assert result["safety_by_arm"][ARMS[0]]["zero_collision_status"] == "FAIL"
    assert result["safety_by_arm"][ARMS[0]]["zero_collision_seed_count"] == 2
    assert result["all_seed_service"]["metrics"]["mean_tripinfo_waiting_time"]["paired_difference_mean"] == 4
    assert result["additional_seed_service"]["metrics"]["mean_tripinfo_waiting_time"]["paired_difference_mean"] == 5
    assert result["new_simulation_count"] == 6
    assert result["new_simulation_elapsed_seconds_sum"] == 12


def test_invalid_cell_prevents_complete_service_comparison_without_dropping_seed():
    rows = cells()
    rows[0].update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = aggregate(rows)
    assert result["matrix_status"] == "INVALID"
    assert result["all_seed_service"] is result["additional_seed_service"] is None
    assert len(result["cells"]) == 6
    assert result["safety_by_arm"][ARMS[0]]["zero_collision_status"] == "UNAVAILABLE"


def test_before_after_pairs_by_seed_and_arm_and_keeps_collision_delta():
    rows = cells()
    old = deepcopy(rows)
    old[0]["metrics"].update(mean_tripinfo_waiting_time=14, collision_incidents=1)
    baseline = {"protocol": BASELINE_PROTOCOL, "repaired_sumocfg": rows[0]["repaired_sumocfg"],
                "cells": list(reversed(old))}
    change = aggregate(rows, baseline)["changes_from_v154f"][0]
    assert (change["seed"], change["arm"]) == EXPECTED[0]
    assert change["metrics"]["mean_tripinfo_waiting_time"]["after_minus_before"] == -4
    assert change["metrics"]["collision_incidents"]["after_minus_before"] == -1
    rows[0]["run_valid"] = False
    change = aggregate(rows, baseline)["changes_from_v154f"][0]
    assert change["metrics"]["mean_tripinfo_waiting_time"]["after_minus_before"] is None
