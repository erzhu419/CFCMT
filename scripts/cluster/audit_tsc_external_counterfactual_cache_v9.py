#!/usr/bin/env python3
"""Audit the repaired-network v9 LA/Jinan adaptation cache matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (  # noqa: E402
    aggregate_cache_sha256,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    ASSIGNMENTS,
    _sha256,
)


AUDIT_PROTOCOL = "tsc-v53r49-external-v9-counterfactual-cache-audit-v1"
LAUNCH_PROTOCOL = "v42r38-external-adaptation-cache-v9-submission-v1"
MANIFEST_PROTOCOL = (
    "external-la-jinan-v9-monotone-virtual-lane-safe-full-network-v1"
)
CACHE_VERSION = "v19-policy-consistent-one-step-mechanisms-two-pass-20260809"
ESTIMAND = "one-control-interval-local-mechanisms-plus-pressure-rollout-value-v1"
SAFETY_PROTOCOL = "paired-action-group-symmetric-censor-on-any-branch-collision-v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_cache(
    *,
    cache_root: Path,
    results_root: Path,
    launch_manifest: Path,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    specs = tuple(launch.get("specs", ()))
    expected_pairs = {
        (str(scenario), int(seed)) for _, scenario, seed in ASSIGNMENTS
    }
    expected_nodes = {
        (str(scenario), int(seed)): str(node)
        for node, scenario, seed in ASSIGNMENTS
    }
    command_text = "\n".join(str(spec.get("cmd", "")) for spec in specs)
    launch_gate = {
        "protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "submitted": launch.get("submitted") is True,
        "scheduler_returncode_zero": int(launch.get("scheduler_returncode", -1)) == 0,
        "snapshot_bound": bool(launch.get("snapshot_sha256"))
        and bool(launch.get("source_tree_sha256")),
        "eight_specs": len(specs) == len(expected_pairs) == 8,
        "six_physical_nodes": {
            str(spec.get("require_node")) for spec in specs
        }
        == {"node001", "node002", "node003", "node004", "node005", "node006"},
        "manifest_hash_bound": launch.get("conversion_manifest_sha256")
        == expected_manifest_sha256
        and expected_manifest_sha256 in command_text,
        "tree_hash_bound": launch.get("conversion_tree_sha256")
        == expected_tree_sha256
        and expected_tree_sha256 in command_text,
        "v9_manifest_bound": command_text.count(
            "traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
        )
        == len(expected_pairs),
        "one_worker_per_shard": int(launch.get("collection_shards", -1)) == 16
        and int(launch.get("workers_per_task", -1)) == 16,
    }
    launch_gate["passed"] = all(launch_gate.values())

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    declared_cache_basenames: list[str] = []
    total_behavior_collision_incidents = 0
    total_counterfactual_collision_incidents = 0
    total_counterfactual_censored_groups = 0
    total_counterfactual_groups = 0
    for scenario, seed in sorted(expected_pairs):
        path = Path(results_root) / f"{scenario}_seed{seed}" / "cache_summary.json"
        if not path.is_file():
            errors.append(f"missing cache summary: {scenario}/{seed}")
            continue
        result = _read_json(path)
        setting = dict(result.get("setting", {}))
        runtime = dict(result.get("runtime", {}))
        diagnostics_by_scenario = dict(result.get("diagnostics", {}))
        diagnostics = dict(diagnostics_by_scenario.get(scenario, {}))
        behavior = dict(diagnostics.get("behavior_safety_audit", {}))
        replay = dict(diagnostics.get("counterfactual_replay_audit", {}))
        branch_safety = dict(diagnostics.get("counterfactual_safety_audit", {}))
        provenance = dict(setting.get("input_provenance", {}))
        manifest = dict(setting.get("manifest", {}))
        cache_files = tuple(str(value) for value in diagnostics.get("cache_files", ()))
        cache_basenames = tuple(Path(value).name for value in cache_files)
        declared_cache_basenames.extend(cache_basenames)
        row_gate = {
            "experiment_matches": result.get("experiment")
            == "traffic_signal_counterfactual_cache_precompute",
            "identity_matches": tuple(setting.get("scenarios", ())) == (scenario,)
            and tuple(int(value) for value in setting.get("seeds", ())) == (seed,),
            "physical_host_matches": runtime.get("hostname")
            == expected_nodes[(scenario, seed)],
            "source_tree_matches": runtime.get("source_tree_sha256")
            == launch.get("source_tree_sha256"),
            "sumo_1_22": runtime.get("libsumo_version") == "1.22.0"
            and setting.get("libsumo_version") == "1.22.0",
            "manifest_protocol_matches": manifest.get("protocol")
            == MANIFEST_PROTOCOL
            and int(manifest.get("version", -1)) == 2,
            "conversion_manifest_matches": provenance.get("manifest_sha256")
            == expected_manifest_sha256,
            "conversion_tree_matches": provenance.get("tree_sha256")
            == expected_tree_sha256,
            "frozen_collection_contract": int(setting.get("collection_shards", -1))
            == 16
            and int(setting.get("workers", -1)) == 16
            and float(setting.get("duration_sec", -1.0)) == 600.0
            and int(setting.get("control_interval_sec", -1)) == 10
            and float(setting.get("warmup_sec", -1.0)) == 60.0
            and int(setting.get("counterfactual_horizon_intervals", -1)) == 6
            and int(setting.get("max_focal_tls", -1)) == 4
            and float(setting.get("min_tls_coverage", -1.0)) == 1.0
            and setting.get("behavior_policy") == "phase_pressure"
            and setting.get("cache_version") == CACHE_VERSION,
            "sixteen_unique_shards": len(cache_basenames) == 16
            and len(set(cache_basenames)) == 16,
            "full_tls_coverage": float(diagnostics.get("tls_coverage_fraction", -1.0))
            == 1.0
            and int(diagnostics.get("covered_tls_count", -1))
            == int(diagnostics.get("controllable_tls_count", -2)),
            "behavior_safe": behavior.get("no_teleport_passed") is True
            and int(behavior.get("starting_teleports", -1)) == 0
            and int(behavior.get("ending_teleports", -1)) == 0
            and int(behavior.get("unique_collision_incidents", -1)) == 0
            and int(behavior.get("raw_collision_events", -1)) == 0,
            "deterministic_replay": replay.get("passed") is True
            and int(replay.get("checks", 0)) > 0
            and float(replay.get("max_abs_difference", -1.0)) == 0.0,
            "estimand_matches": diagnostics.get("counterfactual_estimand")
            == ESTIMAND,
            "symmetric_safety_censoring": branch_safety.get("protocol")
            == SAFETY_PROTOCOL
            and branch_safety.get("no_teleport_passed") is True
            and branch_safety.get("symmetric_group_censoring_passed") is True
            and int(branch_safety.get("retained_branches", -1))
            == int(diagnostics.get("counterfactual_branches", -2)),
            "nonempty_training_support": int(diagnostics.get("groups", 0)) > 0
            and int(diagnostics.get("rows", 0)) > 0,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"cache row gate failed: {scenario}/{seed}")
        total_behavior_collision_incidents += int(
            behavior.get("unique_collision_incidents", 0)
        )
        total_counterfactual_collision_incidents += int(
            branch_safety.get("unique_collision_incidents_observed", 0)
        )
        total_counterfactual_censored_groups += int(
            branch_safety.get("censored_groups", 0)
        )
        total_counterfactual_groups += int(branch_safety.get("retained_groups", 0))
        total_counterfactual_groups += int(branch_safety.get("censored_groups", 0))
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "hostname": runtime.get("hostname"),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "cache_file_count": len(cache_basenames),
                "groups": diagnostics.get("groups"),
                "rows": diagnostics.get("rows"),
                "counterfactual_branches": diagnostics.get(
                    "counterfactual_branches"
                ),
                "counterfactual_collision_incidents": branch_safety.get(
                    "unique_collision_incidents_observed"
                ),
                "counterfactual_censored_groups": branch_safety.get(
                    "censored_groups"
                ),
                "gate": row_gate,
            }
        )

    cache_sha256 = None
    cache_file_count = 0
    try:
        cache_sha256, cache_file_count = aggregate_cache_sha256(cache_root)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot hash cache: {exc}")
    observed_cache_basenames = {
        path.name for path in Path(cache_root).glob("*.npz") if path.is_file()
    }
    observed_summaries = set(Path(results_root).rglob("cache_summary.json"))
    integrity_gate = {
        "launch_gate_passed": launch_gate["passed"],
        "all_expected_rows_present": len(rows) == len(expected_pairs),
        "no_extra_rows": len(observed_summaries) == len(expected_pairs),
        "all_row_gates_passed": bool(rows)
        and all(row["gate"]["passed"] for row in rows),
        "exactly_128_declared_shards": len(declared_cache_basenames) == 128
        and len(set(declared_cache_basenames)) == 128,
        "exactly_128_observed_shards": cache_file_count == 128
        and len(observed_cache_basenames) == 128,
        "declared_and_observed_shards_match": set(declared_cache_basenames)
        == observed_cache_basenames,
        "behavior_collision_free": total_behavior_collision_incidents == 0,
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": (
            "authorize_v9_external_model_refit"
            if integrity_gate["passed"]
            else "block_v9_external_model_refit"
        ),
        "claim_boundary": (
            "Training-cache provenance, coverage, determinism, and safety only; "
            "no downstream model or policy efficacy was inspected."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "conversion_manifest_sha256": expected_manifest_sha256,
        "conversion_tree_sha256": expected_tree_sha256,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "cache_sha256": cache_sha256,
        "cache_file_count": cache_file_count,
        "launch_gate": launch_gate,
        "rows": rows,
        "safety_summary": {
            "behavior_collision_incidents": total_behavior_collision_incidents,
            "counterfactual_collision_incidents": (
                total_counterfactual_collision_incidents
            ),
            "counterfactual_censored_groups": total_counterfactual_censored_groups,
            "counterfactual_candidate_groups": total_counterfactual_groups,
        },
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tree-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v9 cache audit: {args.out}")
    payload = audit_cache(
        cache_root=args.cache_root,
        results_root=args.results_root,
        launch_manifest=args.launch_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        expected_tree_sha256=args.expected_tree_sha256,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "rows": len(payload["rows"]),
                "cache_file_count": payload["cache_file_count"],
                "cache_sha256": payload["cache_sha256"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
