from __future__ import annotations

import json
from pathlib import Path

from scripts.cluster.audit_tsc_target_offline_source_guard_fresh_confirmation import (
    _aggregate,
    _comparison,
    _comparison_gates,
    _validate_candidate_contract,
    _validate_offline_contract,
)


def test_v111_frozen_candidate_and_offline_contracts() -> None:
    root = Path(__file__).resolve().parents[2]
    path = (
        root
        / "cf_h2o/config/traffic_signal_tsc_v111_target_offline_source_guard_confirmation_v1.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["_candidate_path"] = str(path)
    guard = {
        "enabled": True,
        "risk_multiplier": 0.0,
        "min_context_trust": 0.1,
        "margin": 0.0,
        "max_relative_rule_gap": 1.0,
    }
    selection = {
        "protocol": "tsc-v110-target-offline-source-and-guard-selection-v2",
        "city": "jinan",
        "target_data_contract": {
            "target_closed_loop_rollouts_used_for_selection": False,
        },
        "selection": {
            "selected_source_city_group": "hangzhou",
            "selected_source_weight": 1.0,
        },
        "guard_selection": {
            "selected_profile_key": "risk_0__gap_1",
            "guard_config": guard,
        },
    }

    _validate_candidate_contract(spec)
    _validate_offline_contract(
        spec,
        selection,
        selection_sha256=spec["offline_source_guard_contract"][
            "jinan_result_sha256"
        ],
    )


def test_v111_three_arm_paired_comparisons() -> None:
    scenarios = (
        "jinan_3x4_real",
        "jinan_3x4_real_2000",
        "jinan_3x4_real_2500",
    )
    seeds = tuple(range(101, 109))
    waiting = {
        "pressure_rule": 100.0,
        "target_only_guard": 95.0,
        "source_guard": 90.0,
    }
    rows = []
    for scenario in scenarios:
        for seed in seeds:
            for method, value in waiting.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "seed": seed,
                        "candidate_key": method,
                        "metrics": {
                            "mean_tripinfo_waiting_time": value + seed / 1000.0,
                            "collision_incidents": 0,
                            "starting_teleports": 0,
                            "ending_teleports": 0,
                        },
                    }
                )

    aggregate = _aggregate(rows, seeds=seeds)
    combined = _comparison(
        rows,
        aggregate,
        seeds=seeds,
        candidate="source_guard",
        reference="pressure_rule",
        bootstrap_seed=111,
    )
    contribution = _comparison(
        rows,
        aggregate,
        seeds=seeds,
        candidate="source_guard",
        reference="target_only_guard",
        bootstrap_seed=112,
    )

    assert all(_comparison_gates(combined).values())
    assert all(_comparison_gates(contribution).values())
