"""Run one post-hoc deployment-mechanism diagnostic on the frozen v43 matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    METHOD_POLICY,
    run_external_closed_loop_rollout,
)


RESULT_PROTOCOL = "tsc-v43r39-external-closed-loop-deployment-diagnostic-v1"
ANALYSIS_PROTOCOL = "tsc-v43r39-post-hoc-deployment-mechanism-diagnostic-v1"


def _spec(
    name: str,
    source_policy: str,
    coordination_mode: str,
) -> ClosedLoopDiagnosticSpec:
    return ClosedLoopDiagnosticSpec(
        name=name,
        source_policy=source_policy,
        coordination_mode=coordination_mode,
        result_protocol=RESULT_PROTOCOL,
    )


DIAGNOSTIC_SPECS = {
    spec.name: spec
    for spec in (
        _spec("selected_source_prior", "selected_source_prior", "sparse"),
        _spec("cfcmt_selected_spatial_only", METHOD_POLICY, "spatial_only"),
        _spec("cfcmt_selected_direct", METHOD_POLICY, "direct"),
        _spec("rigid_anchor_direct", "rigid_anchor_mpc", "direct"),
        _spec("simulator_only_direct", "simulator_only_mpc", "direct"),
        _spec(
            "h2oplus_style_dense_residual_direct",
            "h2oplus_style_dense_residual_mpc",
            "direct",
        ),
    )
}
POLICIES = tuple(DIAGNOSTIC_SPECS)


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
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite diagnostic evidence")
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
        policy=args.policy,
        tripinfo_out=args.tripinfo_out,
        diagnostic_spec=DIAGNOSTIC_SPECS[args.policy],
    )
    payload["analysis_protocol"] = ANALYSIS_PROTOCOL
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "policy": args.policy,
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
