#!/usr/bin/env python3
"""Audit the six-seed, four-scenario v61 counterfactual cache."""

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
from scripts.cluster.audit_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    AUDIT_DECISION as FREEZE_AUDIT_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_hierarchical_protocol import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_offline_cache import (  # noqa: E402
    LAUNCH_PROTOCOL,
    NODES,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _sha256,
)


AUDIT_PROTOCOL = "tsc-v61r57-external-v9-hierarchical-offline-cache-audit-v1"
AUDIT_DECISION = "authorize_hierarchical_action_level_confirmation"
CACHE_VERSION = "v19-policy-consistent-one-step-mechanisms-two-pass-20260809"
MANIFEST_PROTOCOL = (
    "external-la-jinan-v9-monotone-virtual-lane-safe-full-network-v1"
)
SAFETY_PROTOCOL = "paired-action-group-symmetric-censor-on-any-branch-collision-v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_hierarchical_cache(
    *,
    cache_root: Path,
    results_root: Path,
    launch_manifest: Path,
    protocol_path: Path,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    protocol = _read_json(protocol_path)
    offline = dict(protocol["fresh_offline_confirmation"])
    seeds = tuple(int(value) for value in offline["seeds"])
    scenarios = tuple(
        str(scenario)
        for city in sorted(protocol["city_scenarios"])
        for scenario in protocol["city_scenarios"][city]
    )
    specs = tuple(launch.get("specs", ()))
    expected_nodes = dict(zip(seeds, NODES))
    direct_results = tuple(launch.get("direct_results", ()))
    launch_gate = {
        "protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "frozen_protocol_matches": protocol.get("protocol") == PROTOCOL
        and launch.get("frozen_protocol_sha256") == _sha256(protocol_path),
        "submitted": launch.get("submitted") is True,
        "six_specs": len(specs) == len(seeds) == 6,
        "six_physical_nodes": {str(row.get("require_node")) for row in specs}
        == set(NODES),
        "direct_results_clean": launch.get("execution_mode")
        == "direct_run_on_six_physical_nodes"
        and len(direct_results) == 6
        and all(int(row.get("returncode", -1)) == 0 for row in direct_results),
        "authorization_bound": launch.get("authorization_decision")
        == FREEZE_AUDIT_DECISION,
        "conversion_hashes_bound": bool(launch.get("conversion_manifest_sha256"))
        and bool(launch.get("conversion_tree_sha256")),
        "one_worker_per_scenario_shard": int(
            launch.get("workers_per_task", -1)
        )
        == len(scenarios) * int(offline["collection_shards_per_scenario_seed"]),
    }
    launch_gate["passed"] = all(launch_gate.values())
    rows = []
    errors = []
    declared_files: list[str] = []
    behavior_collisions = 0
    counterfactual_collisions = 0
    censored_groups = 0
    for seed in seeds:
        path = Path(results_root) / f"seed{seed}" / "cache_summary.json"
        if not path.is_file():
            errors.append(f"missing cache summary: {seed}")
            continue
        result = _read_json(path)
        setting = dict(result.get("setting", {}))
        runtime = dict(result.get("runtime", {}))
        diagnostics = dict(result.get("diagnostics", {}))
        setting_gate = {
            "experiment_matches": result.get("experiment")
            == "traffic_signal_counterfactual_cache_precompute",
            "seed_matches": tuple(setting.get("seeds", ())) == (seed,),
            "scenarios_match": tuple(setting.get("scenarios", ())) == scenarios,
            "physical_host_matches": runtime.get("hostname")
            == expected_nodes[seed],
            "source_tree_matches": runtime.get("source_tree_sha256")
            == launch.get("source_tree_sha256"),
            "sumo_version_matches": runtime.get("libsumo_version") == "1.22.0"
            and setting.get("libsumo_version") == "1.22.0",
            "manifest_matches": setting.get("manifest", {}).get("protocol")
            == MANIFEST_PROTOCOL,
            "collection_contract_matches": float(setting.get("duration_sec", -1))
            == float(offline["duration_sec"])
            and int(setting.get("control_interval_sec", -1))
            == int(offline["control_interval_sec"])
            and float(setting.get("warmup_sec", -1))
            == float(offline["warmup_sec"])
            and int(setting.get("counterfactual_horizon_intervals", -1))
            == int(offline["counterfactual_horizon_intervals"])
            and int(setting.get("collection_shards", -1))
            == int(offline["collection_shards_per_scenario_seed"])
            and int(setting.get("workers", -1))
            == int(launch["workers_per_task"])
            and setting.get("behavior_policy") == offline["behavior_policy"]
            and setting.get("cache_version") == CACHE_VERSION,
            "authorization_matches": setting.get("authorization_provenance", {}).get(
                "sha256"
            )
            == launch.get("authorization_audit_sha256")
            and setting.get("authorization_provenance", {}).get("decision")
            == FREEZE_AUDIT_DECISION,
            "conversion_matches": setting.get("input_provenance", {}).get(
                "manifest_sha256"
            )
            == launch.get("conversion_manifest_sha256")
            and setting.get("input_provenance", {}).get("tree_sha256")
            == launch.get("conversion_tree_sha256"),
            "all_scenarios_present": set(diagnostics) == set(scenarios),
        }
        scenario_rows = []
        for scenario in scenarios:
            diagnostic = dict(diagnostics.get(scenario, {}))
            behavior = dict(diagnostic.get("behavior_safety_audit", {}))
            replay = dict(diagnostic.get("counterfactual_replay_audit", {}))
            safety = dict(diagnostic.get("counterfactual_safety_audit", {}))
            files = tuple(Path(value).name for value in diagnostic.get("cache_files", ()))
            declared_files.extend(files)
            scenario_gate = {
                "sixteen_unique_shards": len(files) == 16
                and len(set(files)) == 16,
                "full_tls_coverage": float(
                    diagnostic.get("tls_coverage_fraction", -1)
                )
                == 1.0,
                "behavior_safe": behavior.get("no_teleport_passed") is True
                and int(behavior.get("starting_teleports", -1)) == 0
                and int(behavior.get("ending_teleports", -1)) == 0
                and int(behavior.get("unique_collision_incidents", -1)) == 0,
                "deterministic_replay": replay.get("passed") is True
                and float(replay.get("max_abs_difference", -1)) == 0.0,
                "symmetric_censoring": safety.get("protocol") == SAFETY_PROTOCOL
                and safety.get("no_teleport_passed") is True
                and safety.get("symmetric_group_censoring_passed") is True,
                "nonempty": int(diagnostic.get("groups", 0)) > 0
                and int(diagnostic.get("rows", 0)) > 0,
            }
            scenario_gate["passed"] = all(scenario_gate.values())
            behavior_collisions += int(behavior.get("unique_collision_incidents", 0))
            counterfactual_collisions += int(
                safety.get("unique_collision_incidents_observed", 0)
            )
            censored_groups += int(safety.get("censored_groups", 0))
            scenario_rows.append(
                {
                    "scenario": scenario,
                    "groups": diagnostic.get("groups"),
                    "rows": diagnostic.get("rows"),
                    "counterfactual_collisions": safety.get(
                        "unique_collision_incidents_observed"
                    ),
                    "censored_groups": safety.get("censored_groups"),
                    "gate": scenario_gate,
                }
            )
        setting_gate["passed"] = all(setting_gate.values()) and all(
            row["gate"]["passed"] for row in scenario_rows
        )
        if not setting_gate["passed"]:
            errors.append(f"cache gate failed: seed {seed}")
        rows.append(
            {
                "seed": seed,
                "hostname": runtime.get("hostname"),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "scenarios": scenario_rows,
                "gate": setting_gate,
            }
        )
    cache_sha = None
    cache_count = 0
    try:
        cache_sha, cache_count = aggregate_cache_sha256(cache_root)
    except (OSError, ValueError) as exc:
        errors.append(f"cannot hash hierarchical cache: {exc}")
    observed_files = {path.name for path in Path(cache_root).glob("*.npz")}
    expected_file_count = len(seeds) * len(scenarios) * 16
    integrity_gate = {
        "launch_gate_passed": bool(launch_gate["passed"]),
        "all_seed_rows_present": len(rows) == len(seeds),
        "all_row_gates_passed": bool(rows)
        and all(row["gate"]["passed"] for row in rows),
        "declared_file_count_exact": len(declared_files) == expected_file_count
        and len(set(declared_files)) == expected_file_count,
        "observed_file_count_exact": cache_count == expected_file_count
        and len(observed_files) == expected_file_count,
        "declared_observed_match": set(declared_files) == observed_files,
        "behavior_collision_free": behavior_collisions == 0,
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": AUDIT_DECISION if integrity_gate["passed"] else "reject",
        "claim_boundary": (
            "Cache provenance, physical execution, coverage, determinism, and safety "
            "only. No successor action outcome is evaluated here."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "frozen_protocol_sha256": _sha256(protocol_path),
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "seeds": list(seeds),
        "scenarios": list(scenarios),
        "cache_sha256": cache_sha,
        "cache_file_count": cache_count,
        "launch_gate": launch_gate,
        "rows": rows,
        "safety_summary": {
            "behavior_collision_incidents": behavior_collisions,
            "counterfactual_collision_incidents": counterfactual_collisions,
            "counterfactual_censored_groups": censored_groups,
        },
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite hierarchical cache audit: {args.out}")
    payload = audit_hierarchical_cache(
        cache_root=args.cache_root,
        results_root=args.results_root,
        launch_manifest=args.launch_manifest,
        protocol_path=args.protocol,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
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
