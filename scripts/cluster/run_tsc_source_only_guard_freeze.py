#!/usr/bin/env python3
"""Freeze the target-label-free TSC deployment guard on source cities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_source_only_guard import (
    run_source_only_guard_freeze,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-repair-audit", type=Path, required=True)
    parser.add_argument("--prior-policy", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--fit-workers", type=int, default=7)
    parser.add_argument("--familywise-alpha", type=float, default=0.05)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = run_source_only_guard_freeze(
        source_cache_root=args.source_cache_root,
        source_manifest_path=args.source_manifest,
        source_repair_audit_path=args.source_repair_audit,
        prior_policy=args.prior_policy,
        workers=args.workers,
        fit_workers=args.fit_workers,
        familywise_alpha=args.familywise_alpha,
        out=args.out,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "guard_config": payload["guard_config"],
                "source_domains": payload["source_domains"],
                "uncertainty_scale": payload["uncertainty_scale"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
