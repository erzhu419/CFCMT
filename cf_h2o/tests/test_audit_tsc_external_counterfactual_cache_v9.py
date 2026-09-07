import json
from pathlib import Path

from scripts.cluster.audit_tsc_external_counterfactual_cache_v9 import (
    audit_cache,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import ASSIGNMENTS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_v9_cache_audit_accepts_complete_safe_matrix(tmp_path: Path) -> None:
    manifest_sha = "a" * 64
    tree_sha = "b" * 64
    source_tree_sha = "c" * 64
    cache_root = tmp_path / "cache"
    results_root = tmp_path / "results"
    cache_root.mkdir()
    specs = []
    for node, scenario, seed in ASSIGNMENTS:
        names = []
        for shard in range(16):
            name = f"{scenario}_seed{seed}_shard{shard}.npz"
            (cache_root / name).write_bytes(f"{scenario}:{seed}:{shard}".encode())
            names.append(f"/remote/cache/{name}")
        summary = {
            "experiment": "traffic_signal_counterfactual_cache_precompute",
            "elapsed_seconds": 1.0,
            "runtime": {
                "hostname": node,
                "source_tree_sha256": source_tree_sha,
                "libsumo_version": "1.22.0",
            },
            "setting": {
                "scenarios": [scenario],
                "seeds": [seed],
                "libsumo_version": "1.22.0",
                "manifest": {
                    "version": 2,
                    "protocol": (
                        "external-la-jinan-v9-monotone-virtual-lane-safe-"
                        "full-network-v1"
                    ),
                },
                "input_provenance": {
                    "manifest_sha256": manifest_sha,
                    "tree_sha256": tree_sha,
                },
                "collection_shards": 16,
                "workers": 16,
                "duration_sec": 600.0,
                "control_interval_sec": 10,
                "warmup_sec": 60.0,
                "counterfactual_horizon_intervals": 6,
                "max_focal_tls": 4,
                "min_tls_coverage": 1.0,
                "behavior_policy": "phase_pressure",
                "cache_version": (
                    "v19-policy-consistent-one-step-mechanisms-two-pass-"
                    "20260809"
                ),
            },
            "diagnostics": {
                scenario: {
                    "cache_files": names,
                    "covered_tls_count": 4,
                    "controllable_tls_count": 4,
                    "tls_coverage_fraction": 1.0,
                    "groups": 10,
                    "rows": 20,
                    "counterfactual_branches": 20,
                    "counterfactual_estimand": (
                        "one-control-interval-local-mechanisms-plus-pressure-"
                        "rollout-value-v1"
                    ),
                    "behavior_safety_audit": {
                        "no_teleport_passed": True,
                        "starting_teleports": 0,
                        "ending_teleports": 0,
                        "unique_collision_incidents": 0,
                        "raw_collision_events": 0,
                    },
                    "counterfactual_replay_audit": {
                        "passed": True,
                        "checks": 2,
                        "max_abs_difference": 0.0,
                    },
                    "counterfactual_safety_audit": {
                        "protocol": (
                            "paired-action-group-symmetric-censor-on-any-"
                            "branch-collision-v1"
                        ),
                        "no_teleport_passed": True,
                        "symmetric_group_censoring_passed": True,
                        "retained_branches": 20,
                        "retained_groups": 10,
                        "censored_groups": 0,
                        "unique_collision_incidents_observed": 0,
                    },
                }
            },
        }
        _write_json(
            results_root / f"{scenario}_seed{seed}" / "cache_summary.json",
            summary,
        )
        specs.append(
            {
                "require_node": node,
                "cmd": (
                    f"run --manifest traffic_signal_tsc_v39_external_la_jinan_"
                    f"v9_manifest.json --expected {manifest_sha} {tree_sha}"
                ),
            }
        )
    launch = {
        "protocol": "v42r38-external-adaptation-cache-v9-submission-v1",
        "submitted": True,
        "scheduler_returncode": 0,
        "snapshot_sha256": "d" * 64,
        "source_tree_sha256": source_tree_sha,
        "conversion_manifest_sha256": manifest_sha,
        "conversion_tree_sha256": tree_sha,
        "collection_shards": 16,
        "workers_per_task": 16,
        "specs": specs,
    }
    launch_path = tmp_path / "launch.json"
    _write_json(launch_path, launch)

    audit = audit_cache(
        cache_root=cache_root,
        results_root=results_root,
        launch_manifest=launch_path,
        expected_manifest_sha256=manifest_sha,
        expected_tree_sha256=tree_sha,
    )

    assert audit["status"] == "PASS"
    assert audit["decision"] == "authorize_v9_external_model_refit"
    assert audit["cache_file_count"] == 128
    assert audit["integrity_gate"]["passed"] is True
