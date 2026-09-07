#!/usr/bin/env python3
"""Audit the v69 450-second counterfactual cache and physical execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster import (  # noqa: E402
    audit_tsc_external_long_horizon_target_veto_cache as base,
)
from scripts.cluster.authorize_tsc_external_450s_veto_cache import (  # noqa: E402
    AUDIT_DECISION as AUTHORIZATION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_estimand_aligned_450s_veto_protocol import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_450s_veto_cache import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import _sha256  # noqa: E402
from scripts.cluster.recover_tsc_external_450s_veto_cache_seed import (  # noqa: E402
    RECOVERY_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v69r65-external-v9-450s-veto-cache-audit-v1"
AUDIT_DECISION = "authorize_demand_conditioned_450s_veto_oof_training_only"


def audit(
    *, cache_root: Path, results_root: Path, launch_manifest: Path, protocol_path: Path
) -> dict:
    base.PROTOCOL = PROTOCOL
    base.LAUNCH_PROTOCOL = LAUNCH_PROTOCOL
    base.AUTHORIZATION_DECISION = AUTHORIZATION_DECISION
    base.AUDIT_PROTOCOL = AUDIT_PROTOCOL
    base.AUDIT_DECISION = AUDIT_DECISION
    base.RECOVERY_PROTOCOL = RECOVERY_PROTOCOL
    payload = base.audit_long_horizon_cache(
        cache_root=cache_root,
        results_root=results_root,
        launch_manifest=launch_manifest,
        protocol_path=protocol_path,
    )
    for row in payload.get("rows", ()):
        for scenario in row.get("scenarios", ()):
            gate = scenario.get("gate", {})
            if "three_hundred_second_rollout_value" in gate:
                gate["four_hundred_fifty_second_rollout_value"] = gate.pop(
                    "three_hundred_second_rollout_value"
                )
    payload["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["claim_boundary"] = (
        "Cache provenance, exact 450-second estimand, full TLS coverage, deterministic "
        "replay, symmetric collision censoring, and physical execution only."
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--launch-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v69 cache audit: {args.out}")
    payload = audit(
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
                "audit_sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
