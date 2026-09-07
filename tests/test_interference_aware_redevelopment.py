from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (
    _partial_interference_alignment,
    all_rollout_identities,
    redevelopment_candidates,
    stay_aware_execution_candidate,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    _minimum_intervention_gap_sec,
)
from scripts.cluster.freeze_tsc_external_v9_interference_aware_redevelopment import (
    METHOD_KEYS,
)
from scripts.cluster import (
    run_tsc_external_interference_aware_redevelopment_shard as shard_runner,
)
from scripts.cluster.audit_tsc_external_interference_aware_redevelopment import (
    _bootstrap_indices,
    _candidate_summary,
)
from scripts.cluster.run_tsc_external_interference_aware_redevelopment_shard import (
    shard_identities,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v86_external_v9_"
    "interference_aware_redevelopment.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v86_progressive_candidate_grid_is_exact() -> None:
    protocol = _protocol()
    candidates = redevelopment_candidates(protocol)
    identities = all_rollout_identities(protocol)

    assert tuple(row.key for row in candidates) == METHOD_KEYS
    assert tuple(row.coordination_mode for row in candidates) == (
        "direct",
        "sparse",
        "sparse",
        "sparse",
    )
    assert tuple(row.cooldown_intervals for row in candidates) == (0, 11, 11, 11)
    assert tuple(
        row.execution_trust_region.global_cooldown_intervals
        for row in candidates
    ) == (11, 0, 0, 0)
    assert tuple(
        row.execution_trust_region.max_simultaneous_overrides
        for row in candidates
    ) == (1, 1, 2, None)
    assert not any(
        row.execution_trust_region.allow_stay_during_global_cooldown
        for row in candidates
    )
    assert len(identities) == len(set(identities)) == 270
    assert len({row[2] for row in identities}) == 54


def test_stay_aware_candidate_changes_only_declared_execution_semantics() -> None:
    base = redevelopment_candidates(_protocol())[0]
    candidate = stay_aware_execution_candidate(
        base,
        key="uh_global_h120_stay_aware",
        candidate_role="phase_transition_aware_redevelopment",
    )

    assert candidate.key == "uh_global_h120_stay_aware"
    assert candidate.candidate_role == "phase_transition_aware_redevelopment"
    assert candidate.execution_trust_region.allow_stay_during_global_cooldown
    assert (
        candidate.execution_trust_region.global_cooldown_intervals
        == base.execution_trust_region.global_cooldown_intervals
    )
    assert candidate.artifact_key == base.artifact_key
    assert candidate.prediction_horizon_sec == base.prediction_horizon_sec
    assert candidate.coordination_mode == base.coordination_mode
    assert candidate.cooldown_intervals == base.cooldown_intervals


def test_stay_aware_candidate_can_bound_one_continuation_veto() -> None:
    base = redevelopment_candidates(_protocol())[0]
    candidate = stay_aware_execution_candidate(
        base,
        key="uh_global_h120_stay_budget1",
        candidate_role="bounded_phase_transition_aware_redevelopment",
        max_stay_overrides_per_global_cooldown=1,
    )

    assert candidate.execution_trust_region.allow_stay_during_global_cooldown
    assert (
        candidate.execution_trust_region.max_stay_overrides_per_global_cooldown
        == 1
    )
    assert candidate.execution_trust_region.global_cooldown_intervals == 11


def test_cooldown_reports_the_earliest_next_control_epoch() -> None:
    assert _minimum_intervention_gap_sec(0, 10) == 10
    assert _minimum_intervention_gap_sec(5, 10) == 60
    assert _minimum_intervention_gap_sec(11, 10) == 120


def test_v86_six_shards_partition_every_identity_once() -> None:
    protocol = _protocol()
    shards = [
        shard_identities(protocol, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert [len(shard) for shard in shards] == [45, 45, 45, 45, 45, 45]
    assert len(flattened) == len(set(flattened)) == 270
    assert set(flattened) == set(all_rollout_identities(protocol))


def test_v86_spawn_workers_start_from_snapshot_root(monkeypatch) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        shard_runner.os,
        "chdir",
        lambda path: observed.append(Path(path)),
    )

    selected = shard_runner._prepare_spawn_working_directory()

    assert selected == PROJECT_ROOT
    assert observed == [PROJECT_ROOT]


def test_v86_candidate_gate_rejects_an_extra_collision() -> None:
    rule = _protocol()["development"]["selection_rule"]
    rows = [
        {
            "relative_delta": -0.01,
            "baseline_waiting_time": 100.0,
            "method_waiting_time": 99.0,
            "absolute_delta": -1.0,
            "executed_overrides": 2,
            "independent_overlap_pair_count": 0,
            "intervention_alignment_passed": True,
            "baseline_teleports": 0,
            "method_teleports": 0,
            "baseline_collision_incidents": 0,
            "method_collision_incidents": int(index == 0),
        }
        for index in range(8)
    ]
    indices = _bootstrap_indices(count=8, replicates=1000, seed=20260830)

    unsafe = _candidate_summary(
        policy="candidate",
        rows=rows,
        rule=rule,
        bootstrap_indices=indices,
        bootstrap_seed=20260830,
    )
    for row in rows:
        row["method_collision_incidents"] = 0
    safe = _candidate_summary(
        policy="candidate",
        rows=rows,
        rule=rule,
        bootstrap_indices=indices,
        bootstrap_seed=20260830,
    )

    assert unsafe["gates"]["collision_safety"] is False
    assert unsafe["gates"]["passed"] is False
    assert safe["gates"]["collision_safety"] is True
    assert safe["gates"]["passed"] is True


def _metrics(trace: list[dict[str, object]]) -> dict[str, object]:
    return {
        "accepted_intervention_trace": trace,
        "residual_deployment": {
            "intervention_graph": {
                "conflicts": {
                    "a": ["b"],
                    "b": ["a"],
                    "c": [],
                }
            }
        },
    }


def test_partial_interference_alignment_rejects_neighbor_overlap() -> None:
    alignment = _partial_interference_alignment(
        _metrics(
            [
                {"time_sec": 100.0, "tls_id": "a", "request_accepted": True},
                {"time_sec": 210.0, "tls_id": "b", "request_accepted": True},
            ]
        ),
        expected_horizon_sec=120,
    )

    assert alignment["passed"] is False
    assert alignment["violation_count"] == 1
    assert alignment["minimum_observed_conflicting_gap_sec"] == 110.0


def test_partial_interference_alignment_allows_independent_overlap() -> None:
    alignment = _partial_interference_alignment(
        _metrics(
            [
                {"time_sec": 100.0, "tls_id": "a", "request_accepted": True},
                {"time_sec": 100.0, "tls_id": "c", "request_accepted": True},
                {"time_sec": 220.0, "tls_id": "b", "request_accepted": True},
            ]
        ),
        expected_horizon_sec=120,
    )

    assert alignment["passed"] is True
    assert alignment["violation_count"] == 0
    assert alignment["independent_overlap_pair_count"] == 1
    assert alignment["minimum_observed_conflicting_gap_sec"] == 120.0
