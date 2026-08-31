#!/usr/bin/env python3
"""Run one topology-repaired source-weight rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json
from cf_h2o.eval.traffic_signal_topology_repaired_source_weight_rollout import (
    run_topology_repaired_source_weight_rollout,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--source-weight", type=float, required=True)
    parser.add_argument("--allowed-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--guard-risk-multiplier", type=float)
    parser.add_argument("--guard-min-context-trust", type=float, default=0.0)
    parser.add_argument("--guard-margin", type=float, default=0.0)
    parser.add_argument("--guard-max-relative-rule-gap", type=float, default=1.0)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--offline-authorization-sha256", required=True)
    parser.add_argument("--joint-audit", type=Path, required=True)
    parser.add_argument("--joint-audit-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest-sha256", required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite source-weight rollout")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    guard = (
        ContrastGuardConfig(
            enabled=True,
            risk_multiplier=args.guard_risk_multiplier,
            min_context_trust=args.guard_min_context_trust,
            margin=args.guard_margin,
            max_relative_rule_gap=args.guard_max_relative_rule_gap,
        )
        if args.guard_risk_multiplier is not None
        else None
    )
    result = run_topology_repaired_source_weight_rollout(
        model_path=args.model,
        expected_model_sha256=args.model_sha256,
        source_weight=args.source_weight,
        allowed_seeds=tuple(args.allowed_seeds),
        guard_config=guard,
        rollout_kwargs={
            "protocol_spec_path": args.protocol,
            "offline_authorization_path": args.offline_authorization,
            "expected_offline_authorization_sha256": args.offline_authorization_sha256,
            "joint_freeze_audit_path": args.joint_audit,
            "expected_joint_freeze_sha256": args.joint_audit_sha256,
            "freeze_root": args.freeze_root,
            "external_manifest_path": args.external_manifest,
            "conversion_root": args.conversion_root,
            "expected_external_manifest_sha256": args.external_manifest_sha256,
            "conversion_manifest_path": args.conversion_manifest,
            "expected_conversion_manifest_sha256": args.conversion_manifest_sha256,
            "expected_conversion_tree_sha256": args.conversion_tree_sha256,
            "expected_sumo_version": "1.22.0",
            "scenario": args.scenario,
            "seed": args.seed,
            "tripinfo_out": args.tripinfo_out,
        },
    )
    _atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "source_weight": args.source_weight,
                "waiting": result["metrics"]["mean_tripinfo_waiting_time"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
