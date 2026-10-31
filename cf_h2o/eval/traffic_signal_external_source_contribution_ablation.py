"""Run one frozen target-only comparator for the v43 source-contribution ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    run_external_closed_loop_rollout,
)


RESULT_PROTOCOL = "tsc-v90r86-external-source-contribution-ablation-rollout-v1"
ANALYSIS_PROTOCOL = "tsc-v90r86-post-freeze-source-contribution-ablation-v1"
POLICY = "causal_target_only_mpc"
SPEC = ClosedLoopDiagnosticSpec(
    name=POLICY,
    source_policy=POLICY,
    coordination_mode="sparse",
    result_protocol=RESULT_PROTOCOL,
    analysis_status="post_freeze_ablation_frozen_before_original_confirmation",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--expected-offline-authorization-sha256", required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--expected-external-manifest-sha256", required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--expected-conversion-manifest-sha256", required=True)
    parser.add_argument("--expected-conversion-tree-sha256", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite source-contribution evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_external_closed_loop_rollout(
        protocol_spec_path=args.protocol_spec,
        offline_authorization_path=args.offline_authorization,
        expected_offline_authorization_sha256=(
            args.expected_offline_authorization_sha256
        ),
        joint_freeze_audit_path=args.joint_freeze_audit,
        expected_joint_freeze_sha256=args.expected_joint_freeze_sha256,
        freeze_root=args.freeze_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        expected_external_manifest_sha256=(
            args.expected_external_manifest_sha256
        ),
        conversion_manifest_path=args.conversion_manifest,
        expected_conversion_manifest_sha256=(
            args.expected_conversion_manifest_sha256
        ),
        expected_conversion_tree_sha256=args.expected_conversion_tree_sha256,
        expected_sumo_version=args.expected_sumo_version,
        scenario=args.scenario,
        seed=args.seed,
        policy=POLICY,
        tripinfo_out=args.tripinfo_out,
        diagnostic_spec=SPEC,
    )
    payload["analysis_protocol"] = ANALYSIS_PROTOCOL
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "policy": POLICY,
                "mean_waiting_time": payload["metrics"][
                    "mean_tripinfo_waiting_time"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
