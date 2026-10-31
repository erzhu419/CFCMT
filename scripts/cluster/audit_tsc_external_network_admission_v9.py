#!/usr/bin/env python3
"""Audit the authoritative v9 LA/Jinan full-horizon network admission."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    ASSIGNMENTS,
    _sha256,
)


AUDIT_PROTOCOL = "tsc-v53r49-external-network-v9-admission-audit-v1"
LAUNCH_PROTOCOL = "v42r38-external-la-jinan-admission-v9-submission-v1"
RESULT_PROTOCOL = "full-horizon-libsumo-demand-conserving-diagnostic-admission-v4"
CONVERSION_PROTOCOL = "cfcmt-cityflow-full-external-sumo-conversion-v9"
VIRTUAL_LANE_PROTOCOL = (
    "cityflow-virtual-cartesian-lane-link-monotone-rank-v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_admission(
    *,
    results_root: Path,
    launch_manifest: Path,
    expected_manifest_sha256: str,
    expected_tree_sha256: str,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    specs = tuple(launch.get("specs", ()))
    expected = {(scenario, int(seed)) for _, scenario, seed in ASSIGNMENTS}
    expected_nodes = {
        (scenario, int(seed)): node for node, scenario, seed in ASSIGNMENTS
    }
    command_text = "\n".join(str(spec.get("cmd", "")) for spec in specs)
    launch_gate = {
        "protocol_matches": launch.get("protocol") == LAUNCH_PROTOCOL,
        "submitted": launch.get("submitted") is True,
        "scheduler_returncode_zero": int(
            launch.get("scheduler_returncode", -1)
        )
        == 0,
        "snapshot_bound": bool(launch.get("snapshot_sha256"))
        and bool(launch.get("source_tree_sha256")),
        "eight_specs": len(specs) == len(expected) == 8,
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
        "strict_zero_collision": command_text.count(
            "--maximum-collision-count 0"
        )
        == len(expected),
        "strict_zero_teleport": command_text.count(
            "--maximum-teleport-fraction 0"
        )
        == len(expected),
    }
    launch_gate["passed"] = all(launch_gate.values())

    rows = []
    errors = []
    for scenario, seed in sorted(expected):
        path = Path(results_root) / f"{scenario}_seed{seed}" / "admission.json"
        if not path.is_file():
            errors.append(f"missing admission result: {scenario}/{seed}")
            continue
        result = _read_json(path)
        conversion = dict(result.get("conversion", {}))
        network = dict(conversion.get("network", {}))
        reduction = dict(network.get("virtual_lane_link_reduction", {}))
        observed = dict(result.get("observed", {}))
        runtime = dict(result.get("runtime", {}))
        row_gate = {
            "result_passed": result.get("passed") is True,
            "protocol_matches": result.get("protocol") == RESULT_PROTOCOL,
            "identity_matches": (
                str(result.get("scenario")),
                int(result.get("seed", -1)),
            )
            == (scenario, seed),
            "physical_host_matches": runtime.get("hostname")
            == expected_nodes[(scenario, seed)],
            "source_tree_matches": runtime.get("source_tree_sha256")
            == launch.get("source_tree_sha256"),
            "sumo_1_22": runtime.get("libsumo_version") == "1.22.0",
            "conversion_protocol_matches": conversion.get("protocol")
            == CONVERSION_PROTOCOL,
            "conversion_manifest_matches": conversion.get("manifest_sha256")
            == expected_manifest_sha256,
            "conversion_tree_matches": conversion.get("tree_sha256")
            == expected_tree_sha256,
            "virtual_lane_protocol_matches": reduction.get("protocol")
            == VIRTUAL_LANE_PROTOCOL,
            "virtual_crossings_zero": int(
                reduction.get("post_reduction_crossing_road_link_count", -1)
            )
            == 0,
            "virtual_lane_coverage_complete": int(
                reduction.get("source_lane_coverage_failures", -1)
            )
            == 0
            and int(reduction.get("target_lane_coverage_failures", -1)) == 0,
            "zero_collision_incidents": int(
                observed.get("unique_collision_incidents", -1)
            )
            == 0,
            "zero_raw_collision_events": int(
                observed.get("raw_collision_events", -1)
            )
            == 0,
            "zero_teleports": int(observed.get("starting_teleports", -1)) == 0
            and int(observed.get("ending_teleports", -1)) == 0,
            "full_horizon": float(observed.get("final_time_sec", -1.0))
            >= 3600.0,
        }
        row_gate["passed"] = all(row_gate.values())
        if not row_gate["passed"]:
            errors.append(f"admission row gate failed: {scenario}/{seed}")
        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "hostname": runtime.get("hostname"),
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "arrived": observed.get("arrived"),
                "collision_incidents": observed.get(
                    "unique_collision_incidents"
                ),
                "teleports": observed.get("ending_teleports"),
                "gate": row_gate,
            }
        )

    observed_files = set(Path(results_root).rglob("admission.json"))
    integrity_gate = {
        "launch_gate_passed": launch_gate["passed"],
        "all_expected_rows_present": len(rows) == len(expected),
        "no_extra_rows": len(observed_files) == len(expected),
        "all_row_gates_passed": bool(rows)
        and all(row["gate"]["passed"] for row in rows),
        "no_errors": not errors,
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": (
            "authorize_v9_counterfactual_cache_recollection"
            if integrity_gate["passed"]
            else "reject_v9_conversion_and_block_downstream_experiments"
        ),
        "claim_boundary": (
            "Network integrity and simulator safety admission only; no efficacy "
            "metric was used for admission."
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "conversion_manifest_sha256": expected_manifest_sha256,
        "conversion_tree_sha256": expected_tree_sha256,
        "snapshot_sha256": launch.get("snapshot_sha256"),
        "source_tree_sha256": launch.get("source_tree_sha256"),
        "launch_gate": launch_gate,
        "rows": rows,
        "integrity_gate": integrity_gate,
        "errors": errors,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-tree-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v9 admission audit: {args.out}")
    payload = audit_admission(
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
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
