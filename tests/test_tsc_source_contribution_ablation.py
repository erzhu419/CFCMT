from __future__ import annotations

import json
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    DIAGNOSTIC_BASELINE_POLICY_TO_RUNTIME,
    POLICIES as CONFIRMATION_POLICIES,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_ablation import (
    POLICY,
    SPEC,
)
from scripts.cluster import audit_tsc_external_source_contribution_ablation as audit
from scripts.cluster.launch_tsc_external_source_contribution_ablation import (
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_source_contribution_ablation_shard import (
    all_identities,
    shard_identities,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v29_external_full_budget_confirmation.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_target_only_ablation_does_not_change_confirmation_policy_set() -> None:
    assert len(CONFIRMATION_POLICIES) == 7
    assert POLICY not in CONFIRMATION_POLICIES
    assert DIAGNOSTIC_BASELINE_POLICY_TO_RUNTIME[POLICY] == (
        "causal_target_only_contrast_raw",
        ("baseline", "causal_target_only"),
    )
    assert SPEC.source_policy == POLICY
    assert SPEC.coordination_mode == "sparse"
    assert SPEC.analysis_status == (
        "post_freeze_ablation_frozen_before_original_confirmation"
    )


def test_six_shards_cover_the_eight_original_closed_loop_cells_once() -> None:
    protocol = _protocol()
    identities = all_identities(protocol)
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert len(identities) == len(set(identities)) == 8
    assert [len(shard) for shard in shards] == [2, 2, 1, 1, 1, 1]
    assert len(flattened) == len(set(flattened)) == 8
    assert set(flattened) == set(identities)


def _row(
    *, city: str, scenario: str, seed: int, policy: str, waiting: float
) -> dict[str, object]:
    return {
        "city": city,
        "scenario": scenario,
        "seed": seed,
        "policy": policy,
        "metrics": {
            "mean_tripinfo_waiting_time": waiting,
            "collision_incidents": 0,
            "starting_teleports": 0,
            "ending_teleports": 0,
            "tripinfo_count": 100,
            "tripinfo_completed_count": 90,
            "tripinfo_unfinished_count": 10,
            "demand_population_at_horizon": 120,
        },
    }


def test_audit_weights_cities_equally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audit, "BOOTSTRAP_REPLICATES", 100)
    method_rows = []
    target_rows = []
    for city, scenarios in audit.CITY_SCENARIOS.items():
        for scenario in scenarios:
            for seed in audit.SEEDS:
                method_waiting = 100.0 if city == "los_angeles" else 200.0
                target_waiting = 120.0 if city == "los_angeles" else 210.0
                method_rows.append(
                    _row(
                        city=city,
                        scenario=scenario,
                        seed=seed,
                        policy=audit.METHOD_POLICY,
                        waiting=method_waiting,
                    )
                )
                target_rows.append(
                    _row(
                        city=city,
                        scenario=scenario,
                        seed=seed,
                        policy=POLICY,
                        waiting=target_waiting,
                    )
                )

    parent = {
        "protocol": audit.AGGREGATE_PROTOCOL,
        "validity_gate": {"passed": True},
        "matrix": {"rollout_count": 56},
        "records": method_rows,
    }
    launch = {
        "protocol": LAUNCH_PROTOCOL,
        "complete": True,
        "matrix_size": 8,
        "full_rollouts_remain_remote": True,
    }
    result = audit.build_audit(
        launch=launch,
        parent=parent,
        target_rows=target_rows,
    )

    assert result["macro_summary"][audit.METHOD_POLICY] == pytest.approx(150.0)
    assert result["macro_summary"][POLICY] == pytest.approx(165.0)
    assert result["cfcmt_oriented_improvement_seconds"] == pytest.approx(15.0)
    assert result["city_oriented_improvements_seconds"] == {
        "los_angeles": pytest.approx(20.0),
        "jinan": pytest.approx(10.0),
    }
    assert result["improved_cell_count"] == 8
