"""Statistics and roster tests for the predeclared fresh-seed comparison."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("fresh_dense_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def fixture(effect):
    protocol = dict(name=report.NAME, city_scenarios=deepcopy(report.CITIES), seeds=list(range(10)),
        policies=list(report.POLICIES), bootstrap=dict(replicates=20000, seed=20261022,
            generator="numpy_PCG64", unit="complete_paired_simulator_seed_block"))
    records = []
    for city, scenarios in protocol["city_scenarios"].items():
        for scenario in scenarios:
            for seed in protocol["seeds"]:
                for policy in report.POLICIES:
                    waiting = 100 + (effect(city, seed) if policy == report.POLICIES[0] else 0)
                    records.append(dict(city=city, scenario=scenario, seed=seed, policy=policy,
                        evaluation_horizon_sec=3600, sumo_version="1.22.0", metrics=dict(
                            mean_tripinfo_waiting_time=waiting, mean_completed_tripinfo_waiting_time=waiting,
                            system_vehicle_hours=50., completion_ratio=.9, tripinfo_count=10,
                            tripinfo_completed_count=9, tripinfo_unfinished_count=1,
                            starting_teleports=0, ending_teleports=0, collision_incidents=0)))
    return records, protocol


def test_city_weights_do_not_triple_weight_jinan():
    records, protocol = fixture(lambda city, seed: -8 if city == "los_angeles" else 2)
    for row in records:
        row["metrics"]["mean_tripinfo_waiting_time"] += 10 * row["seed"]
    result = report.summarize(records, protocol)
    assert result["macro"]["cfcmt_wait_sec"] == 142
    assert result["macro"]["dense_wait_sec"] == 145
    assert result["macro"]["difference_sec"] == -3
    assert result["macro"]["relative_difference_percent"] == pytest.approx(-300 / 145)
    assert result["macro"]["relative_difference_percent"] != pytest.approx(np.mean(-300 / (100 + 10 * np.arange(10))))
    assert result["macro"]["ci95_difference_sec"] == [-3, -3]
    assert result["better_seed_count"] == 10 and result["advantage_confirmed"]
    assert result["by_city"]["jinan"]["better_seed_count"] == 0


def test_opposite_city_effects_cancel_in_every_bootstrap_seed_block():
    # Independent city/cell resampling would destroy this exact cancellation.
    records, protocol = fixture(lambda city, seed: (-1 if city == "los_angeles" else 1) * (seed + 1))
    result = report.summarize(records, protocol)
    assert result["macro"]["difference_sec"] == 0
    assert result["macro"]["ci95_difference_sec"] == [0, 0]
    assert result["macro"]["ci95_relative_difference_percent"] == [0, 0]
    assert not result["advantage_confirmed"]
    assert result["bootstrap"]["independent_seed_blocks"] == 10


def test_missing_rollout_has_no_scientific_estimate():
    records, protocol = fixture(lambda city, seed: -1)
    result = report.summarize(records[:-1], protocol)
    assert not result["complete"] and result["ready"] == 79 and len(result["missing"]) == 1
    assert result["records"] == records[:-1]
    assert not {"macro", "secondary", "advantage_confirmed", "by_seed"}.intersection(result)


def test_duplicate_and_invalid_rollouts_cannot_enter_estimates():
    records, protocol = fixture(lambda city, seed: -1)
    with pytest.raises(ValueError, match="Duplicate"):
        report.summarize(records + [records[0]], protocol)
    for field, value in (("seed", 100), ("evaluation_horizon_sec", 600), ("sumo_version", "1.25.0")):
        invalid = deepcopy(records)
        invalid[0][field] = value
        with pytest.raises(ValueError):
            report.summarize(invalid, protocol)
    for field, value in (("mean_tripinfo_waiting_time", np.nan), ("starting_teleports", 1)):
        invalid = deepcopy(records)
        invalid[0]["metrics"][field] = value
        with pytest.raises(ValueError):
            report.summarize(invalid, protocol)
