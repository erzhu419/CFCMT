#!/usr/bin/env python3
"""Run the frozen historical-prefix demand diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_historical_demand_model_diagnostic import (  # noqa: E402
    run_diagnostic,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite historical diagnostic: {args.out}")
    payload = run_diagnostic(
        diagnostic_protocol_path=args.diagnostic_protocol,
        cache_root=args.cache_root,
        cache_audit_path=args.cache_audit,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        workers=args.workers,
    )
    _atomic_json(args.out, payload)
    summary = {
        city: {
            variant: {
                "passed": result["passed_selection_rule"],
                "mean_delta": result["selected_candidate"]["mean_delta"],
                "quantile": result["selected_candidate"]["acceptance_quantile"],
                "retained_groups": result["selected_candidate"][
                    "retained_group_count"
                ],
                "spearman": result["oof_spearman"],
            }
            for variant, result in row["variants"].items()
        }
        for city, row in payload["cities"].items()
    }
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "summary": summary,
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
