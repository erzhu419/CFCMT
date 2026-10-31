#!/usr/bin/env python3
"""Audit the two-city OOF trust-region freeze and emit a joint certificate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    ADAPTATION_SEEDS,
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (  # noqa: E402
    RESULT_PROTOCOL,
    select_oof_trust_region,
)
from scripts.cluster.launch_tsc_external_trust_region_freeze import (  # noqa: E402
    CITY_NODES,
    LAUNCH_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v44r40-external-oof-trust-region-joint-audit-v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def audit_trust_region_freeze(
    *,
    freeze_root: Path,
    launch_manifest: Path,
    parent_joint_audit: Path,
) -> dict[str, Any]:
    launch = _read_json(launch_manifest)
    parent = _read_json(parent_joint_audit)
    if (
        launch.get("protocol") != LAUNCH_PROTOCOL
        or not bool(launch.get("submitted", False))
        or int(launch.get("task_count", -1)) != len(CITY_NODES)
        or parent.get("status") != "PASS"
        or launch.get("joint_freeze_audit_sha256") != _sha256(parent_joint_audit)
    ):
        raise ValueError("trust-region launch or parent freeze audit changed")
    launch_nodes = {
        str(spec["signature"]).rsplit("/", 1)[-1]: str(spec["require_node"])
        for spec in launch["specs"]
    }
    if launch_nodes != CITY_NODES:
        raise ValueError("trust-region physical node assignment changed")

    cities: dict[str, Any] = {}
    for city, expected_node in CITY_NODES.items():
        path = Path(freeze_root) / city / "trust_region_freeze.json"
        payload = _read_json(path)
        records = list(payload.get("oof_action_records", ()))
        recomputed_guard, recomputed_selection = select_oof_trust_region(records)
        recomputed_guard_payload = {
            "enabled": recomputed_guard.enabled,
            "risk_multiplier": recomputed_guard.risk_multiplier,
            "min_context_trust": recomputed_guard.min_context_trust,
            "margin": recomputed_guard.margin,
            "max_relative_rule_gap": recomputed_guard.max_relative_rule_gap,
        }
        identities = {
            (int(row["fold_index"]), str(row["scenario"]), str(row["group_id"]))
            for row in records
        }
        validation_seeds = {int(row["validation_seed"]) for row in records}
        parent_city = parent["cities"][city]
        if (
            payload.get("protocol") != RESULT_PROTOCOL
            or payload.get("city") != city
            or payload.get("hostname") != expected_node
            or payload.get("runtime", {}).get("source_tree_sha256")
            != launch.get("source_tree_sha256")
            or payload.get("joint_freeze_audit_sha256") != _sha256(parent_joint_audit)
            or payload.get("frozen_method_model_sha256")
            != parent_city["method_model"]["sha256"]
            or payload.get("frozen_method_result_sha256")
            != parent_city["method_result"]["sha256"]
            or validation_seeds != set(ADAPTATION_SEEDS)
            or len(identities) != len(records)
            or int(payload.get("oof_action_record_count", -1)) != len(records)
            or payload.get("deployment", {}).get("guard") != recomputed_guard_payload
            or payload.get("selection") != recomputed_selection
            or not bool(payload.get("freeze_gate", {}).get("passed", False))
        ):
            raise ValueError(f"trust-region certificate audit failed for {city}")
        cities[city] = {
            "certificate_path": str(path.resolve()),
            "certificate_sha256": _sha256(path),
            "hostname": payload["hostname"],
            "oof_action_record_count": len(records),
            "selected_candidate": payload["selected_candidate"],
            "decision": payload["selection"]["decision"],
            "deployment": payload["deployment"],
            "selected_oof_diagnostics": payload["selection"]["selected"],
            "recomputed_selector_exact_match": True,
        }
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "decision": "authorize_contaminated_seed_development_rollouts_only",
        "claim_boundary": (
            "guard selection is frozen without held-out or closed-loop seeds; "
            "the next 8081/9091 rollouts are redesign diagnostics, not confirmation"
        ),
        "launch_manifest_sha256": _sha256(launch_manifest),
        "parent_joint_freeze_audit_sha256": _sha256(parent_joint_audit),
        "snapshot_sha256": launch["snapshot_sha256"],
        "source_tree_sha256": launch["source_tree_sha256"],
        "cities": cities,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--parent-joint-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite trust-region audit: {args.out}")
    payload = audit_trust_region_freeze(
        freeze_root=args.freeze_root,
        launch_manifest=args.launch_manifest,
        parent_joint_audit=args.parent_joint_audit,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({"status": payload["status"], "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
