from copy import deepcopy

import pytest

from scripts.data.summarize_cologne_repaired_controller_comparison import (
    ARMS, PROTOCOL, SEEDS, SERVICE_KEYS, aggregate, reused_cell,
)


def cells():
    result = []
    for seed in SEEDS:
        for arm in ARMS:
            metrics = {key: 10.0 if arm == ARMS[0] else 8.0 for key in SERVICE_KEYS}
            metrics.update(collision_incidents=0, collision_events=0)
            result.append({"protocol": PROTOCOL, "arm": arm, "seed": seed,
                "run_valid": True, "status": "PASS", "zero_incident_status": "PASS",
                "metrics": metrics, "collision_inventory": {"incident_counts_by_first_participant_lane_pair": []},
                "focal_passages": {}, "source_bindings": {"runtime": "old85"},
                "repaired_sumocfg": "/same/v154d.sumocfg", "source_result": f"/{seed}/{arm}.json",
                "reused": (seed, arm) == (41242, ARMS[0]), "elapsed_sec": 2.0})
    return result


@pytest.mark.parametrize("kind", ["missing", "duplicate"])
def test_requires_every_prespecified_pair_exactly_once(kind):
    rows = cells()
    if kind == "missing":
        rows.pop()
    else:
        rows[-1] = deepcopy(rows[0])
    with pytest.raises(ValueError, match="six frozen cells"):
        aggregate(rows)


def test_collision_failure_is_retained_in_service_means_and_seed_status():
    rows = cells()
    rows[0]["metrics"].update(collision_incidents=2, collision_events=4, mean_tripinfo_waiting_time=16)
    rows[0].update(status="FAIL", zero_incident_status="FAIL")
    result = aggregate(rows)
    assert result["matrix_status"] == "COMPLETE"
    assert result["safety_by_arm"][ARMS[0]]["zero_collision_status"] == "FAIL"
    assert result["safety_by_arm"][ARMS[0]]["zero_collision_seed_count"] == 2
    assert result["all_seed_service"]["metrics"]["mean_tripinfo_waiting_time"]["paired_difference_mean"] == 4
    assert result["additional_seed_service"]["metrics"]["mean_tripinfo_waiting_time"]["paired_difference_mean"] == 5
    assert result["new_simulation_count"] == 5
    assert result["new_simulation_elapsed_seconds_sum"] == 10


def test_invalid_run_prevents_paired_mean_instead_of_dropping_its_seed():
    rows = cells()
    rows[0].update(run_valid=False, status="INVALID", zero_incident_status="UNAVAILABLE")
    result = aggregate(rows)
    assert result["matrix_status"] == "INVALID"
    assert result["all_seed_service"] is None
    assert result["additional_seed_service"] is None
    assert len(result["cells"]) == 6


def test_reuse_selects_repaired_branch_not_original_baseline():
    fixture = cells()[2]
    prior = {"protocol": "tsc-v154e-cologne-bilateral-full-validation-v1", "status": "PASS",
             "repair": {key: fixture[key] for key in ["metrics", "collision_inventory", "focal_passages"]},
             "baseline": {"metrics": {"collision_incidents": 18}},
             "source_bindings": fixture["source_bindings"], "repaired_sumocfg": fixture["repaired_sumocfg"]}
    row = reused_cell(prior, "/retained/v154e.json")
    assert row["seed"] == 41242 and row["arm"] == "rigid_target_only"
    assert row["metrics"]["collision_incidents"] == 0
    assert row["reused"] and row["source_branch"] == "repair"
