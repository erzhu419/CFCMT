from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_fresh_confirmation import (
    all_rollout_identities,
    global_switch_alignment,
    selected_candidate,
)
from scripts.cluster import run_tsc_external_fresh_confirmation_shard as shard_runner
from scripts.cluster.audit_tsc_external_fresh_confirmation import (
    _confirmation_summary,
)
from scripts.cluster.audit_tsc_external_interference_aware_redevelopment import (
    _bootstrap_indices,
)
from scripts.cluster.freeze_tsc_external_v9_fresh_confirmation import (
    METHOD_POLICY,
    PHASE_POLICY,
    generate_confirmation_seeds,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V86_PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v86_external_v9_"
    "interference_aware_redevelopment.json"
)
V89_PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v89_external_v9_fresh_confirmation.json"
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v89_freezes_64_disjoint_seeds_and_two_policies() -> None:
    v86 = _read(V86_PROTOCOL_PATH)
    v89 = _read(V89_PROTOCOL_PATH)
    candidate = selected_candidate(v89, v86)
    identities = all_rollout_identities(v89)
    confirmation = v89["confirmation"]

    assert candidate.key == METHOD_POLICY
    assert candidate.execution_trust_region.global_cooldown_intervals == 11
    assert candidate.execution_trust_region.max_simultaneous_overrides == 1
    assert not candidate.execution_trust_region.allow_stay_during_global_cooldown
    assert (
        candidate.execution_trust_region.max_stay_overrides_per_global_cooldown
        is None
    )
    assert len(identities) == len(set(identities)) == 128
    assert {identity[3] for identity in identities} == {
        PHASE_POLICY,
        METHOD_POLICY,
    }
    assert len(set(confirmation["seeds"])) == 64
    assert set(confirmation["seeds"]).isdisjoint(
        confirmation["adaptive_seeds_excluded"]
    )
    assert set(confirmation["seeds"]).isdisjoint(
        confirmation["sealed_v85_seeds_excluded"]
    )


def test_v89_seed_generator_is_deterministic_and_excludes_inputs() -> None:
    first = generate_confirmation_seeds(excluded={100_001, 100_002})
    second = generate_confirmation_seeds(excluded={100_001, 100_002})

    assert first == second
    assert len(first) == len(set(first)) == 64
    assert not set(first).intersection({100_001, 100_002})


def _metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    return {"accepted_intervention_trace": rows}


def test_global_switch_alignment_allows_only_outside_cooldown_stays() -> None:
    alignment = global_switch_alignment(
        _metrics(
            [
                {
                    "time_sec": 60.0,
                    "execution_effective": True,
                    "override_kind": "stay",
                    "global_cooldown_active_before": False,
                },
                {
                    "time_sec": 70.0,
                    "execution_effective": True,
                    "override_kind": "switch",
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
    assert alignment["effective_override_count"] == 3
    assert alignment["switch_override_count"] == 2
    assert alignment["stay_override_count"] == 1
    assert alignment["minimum_observed_switch_gap_sec"] == 120.0


def test_global_switch_alignment_rejects_cooldown_stay() -> None:
    alignment = global_switch_alignment(
        _metrics(
            [
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
                },
            ]
        ),
        expected_horizon_sec=120,
    )

    assert alignment["passed"] is False
    assert alignment["stay_during_global_cooldown_count"] == 1


def test_v89_six_shards_partition_every_identity_once() -> None:
    protocol = _read(V89_PROTOCOL_PATH)
    shards = [
        shard_runner.shard_identities(
            protocol, shard_index=index, shard_count=6
        )
        for index in range(6)
    ]
    flattened = [identity for shard in shards for identity in shard]

    assert [len(shard) for shard in shards] == [22, 22, 21, 21, 21, 21]
    assert len(flattened) == len(set(flattened)) == 128
    assert set(flattened) == set(all_rollout_identities(protocol))


def test_v89_spawn_workers_start_from_snapshot_root(monkeypatch) -> None:
    observed: list[Path] = []
    monkeypatch.setattr(
        shard_runner.os,
        "chdir",
        lambda path: observed.append(Path(path)),
    )

    selected = shard_runner._prepare_spawn_working_directory()

    assert selected == PROJECT_ROOT
    assert observed == [PROJECT_ROOT]


def _audit_rows(*, method_collision_seeds: set[int]) -> list[dict[str, object]]:
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
            "phase_execution_accounting_closed": True,
            "baseline_system_vehicle_hours": 100.0,
            "method_system_vehicle_hours": 99.0,
            "baseline_halted_vehicle_hours": 50.0,
            "method_halted_vehicle_hours": 49.0,
            "baseline_mean_queue": 20.0,
            "method_mean_queue": 19.0,
            "baseline_completion_ratio": 0.8,
            "method_completion_ratio": 0.81,
        }
        for seed in range(64)
    ]


def _rule() -> dict[str, object]:
    return _read(V89_PROTOCOL_PATH)["confirmation"]["preregistered_gate"]


def test_v89_confirmation_gate_passes_a_uniform_safe_improvement() -> None:
    summary = _confirmation_summary(
        rows=_audit_rows(method_collision_seeds={1}),
        rule=_rule(),
        bootstrap_indices=_bootstrap_indices(
            count=64, replicates=1000, seed=20260831
        ),
    )

    assert summary["gates"]["aggregate_collision_safety"] is True
    assert summary["gates"]["passed"] is True
    assert summary["exact_sign_test"]["one_sided_p"] < 0.05


def test_v89_confirmation_gate_rejects_collision_excess() -> None:
    summary = _confirmation_summary(
        rows=_audit_rows(method_collision_seeds={1, 2}),
        rule=_rule(),
        bootstrap_indices=_bootstrap_indices(
            count=64, replicates=1000, seed=20260831
        ),
    )

    assert summary["gates"]["aggregate_collision_safety"] is False
    assert summary["gates"]["passed"] is False
