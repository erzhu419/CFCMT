from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_bounded_stay_redevelopment import (
    BOUNDED_STAY_POLICY,
    bounded_stay_candidate,
    all_rollout_identities,
    phase_transition_alignment,
)
from scripts.cluster import (
    run_tsc_external_bounded_stay_redevelopment_shard as shard_runner,
)
from scripts.cluster.audit_tsc_external_interference_aware_redevelopment import (
    _bootstrap_indices,
)
from scripts.cluster.audit_tsc_external_bounded_stay_redevelopment import (
    _candidate_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V86_PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v86_external_v9_"
    "interference_aware_redevelopment.json"
)
V88_PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v88_external_v9_"
    "bounded_stay_redevelopment.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v88_freezes_one_stay_aware_candidate_on_54_development_seeds() -> None:
    v86 = _read(V86_PROTOCOL_PATH)
    v88 = _read(V88_PROTOCOL_PATH)
    candidate = bounded_stay_candidate(v88, v86)
    identities = all_rollout_identities(v88)

    assert candidate.key == BOUNDED_STAY_POLICY
    assert candidate.execution_trust_region.allow_stay_during_global_cooldown
    assert (
        candidate.execution_trust_region.max_stay_overrides_per_global_cooldown
        == 1
    )
    assert candidate.execution_trust_region.global_cooldown_intervals == 11
    assert candidate.execution_trust_region.max_simultaneous_overrides == 1
    assert candidate.coordination_mode == "direct"
    assert len(identities) == len(set(identities)) == 54
    assert {identity[3] for identity in identities} == {BOUNDED_STAY_POLICY}
    assert set(v88["development"]["seeds"]).isdisjoint(
        v88["development"]["sealed_prospective_seeds"]
    )


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"accepted_intervention_trace": rows}


def test_phase_transition_alignment_allows_stays_inside_switch_horizon() -> None:
    alignment = phase_transition_alignment(
        _metrics(
            [
                {
                    "time_sec": 60.0,
                    "execution_effective": True,
                    "override_kind": "stay",
                },
                {
                    "time_sec": 70.0,
                    "execution_effective": True,
                    "override_kind": "switch",
                },
                {
                    "time_sec": 130.0,
                    "execution_effective": True,
                    "override_kind": "stay",
                    "global_cooldown_active_before": True,
                    "stay_bypass_remaining_before": 1,
                },
                {
                    "time_sec": 190.0,
                    "execution_effective": True,
                    "override_kind": "switch",
                },
            ]
        ),
        expected_horizon_sec=120,
    )

    assert alignment["passed"] is True
    assert alignment["effective_override_count"] == 4
    assert alignment["switch_override_count"] == 2
    assert alignment["stay_override_count"] == 2
    assert alignment["bounded_stay_override_count"] == 1
    assert alignment["stay_budget_violation_count"] == 0
    assert alignment["minimum_observed_switch_gap_sec"] == 120.0


def test_phase_transition_alignment_rejects_switches_inside_horizon() -> None:
    alignment = phase_transition_alignment(
        _metrics(
            [
                {
                    "time_sec": 70.0,
                    "execution_effective": True,
                    "override_kind": "switch",
                },
                {
                    "time_sec": 180.0,
                    "execution_effective": True,
                    "override_kind": "switch",
                },
            ]
        ),
        expected_horizon_sec=120,
    )

    assert alignment["passed"] is False
    assert alignment["switch_gap_violation_count"] == 1
    assert alignment["minimum_observed_switch_gap_sec"] == 110.0


def test_v88_six_shards_partition_every_seed_once() -> None:
    protocol = _read(V88_PROTOCOL_PATH)
    shards = [
        shard_runner.shard_identities(
            protocol, shard_index=index, shard_count=6
        )
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert [len(shard) for shard in shards] == [9, 9, 9, 9, 9, 9]
    assert len(flattened) == len(set(flattened)) == 54
    assert set(flattened) == set(all_rollout_identities(protocol))


def test_v88_spawn_workers_start_from_snapshot_root(monkeypatch) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        shard_runner.os,
        "chdir",
        lambda path: observed.append(Path(path)),
    )

    selected = shard_runner._prepare_spawn_working_directory()

    assert selected == PROJECT_ROOT
    assert observed == [PROJECT_ROOT]


def _audit_rows(method_collision_seeds: set[int]) -> list[dict[str, object]]:
    return [
        {
            "seed": seed,
            "relative_delta": -0.01,
            "absolute_delta": -1.0,
            "effective_overrides": 2,
            "baseline_teleports": 0,
            "method_teleports": 0,
            "baseline_collision_incidents": int(seed == 0),
            "method_collision_incidents": int(seed in method_collision_seeds),
            "baseline_departed": 100,
            "method_departed": 100,
            "phase_execution_accounting_closed": True,
        }
        for seed in range(8)
    ]


def _selection_rule() -> dict[str, object]:
    return _read(V88_PROTOCOL_PATH)["development"]["selection_rule"]


def test_v88_aggregate_collision_gate_reports_seed_discordance() -> None:
    summary = _candidate_summary(
        policy="candidate",
        rows=_audit_rows({1}),
        rule=_selection_rule(),
        bootstrap_indices=_bootstrap_indices(
            count=8, replicates=1000, seed=20260830
        ),
    )

    assert summary["gates"]["aggregate_collision_safety"] is True
    assert summary["gates"]["passed"] is True
    assert summary["collision_discordance"]["baseline_only_seed_count"] == 1
    assert summary["collision_discordance"]["method_only_seed_count"] == 1


def test_v88_aggregate_collision_gate_rejects_total_method_excess() -> None:
    summary = _candidate_summary(
        policy="candidate",
        rows=_audit_rows({1, 2}),
        rule=_selection_rule(),
        bootstrap_indices=_bootstrap_indices(
            count=8, replicates=1000, seed=20260830
        ),
    )

    assert summary["gates"]["aggregate_collision_safety"] is False
    assert summary["gates"]["passed"] is False
