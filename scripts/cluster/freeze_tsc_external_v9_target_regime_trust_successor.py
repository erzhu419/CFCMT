#!/usr/bin/env python3
"""Fit and freeze the v80 target-regime trust runtime artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_target_regime_trust_freeze import (  # noqa: E402
    build_full_fit_artifact,
    write_full_fit_freeze,
)
from scripts.cluster.freeze_tsc_external_v9_target_regime_trust_diagnostic import (  # noqa: E402
    PROTOCOL,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--freeze-result", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostic.read_text(encoding="utf-8"))
    if (
        protocol.get("protocol") != PROTOCOL
        or diagnostic.get("input_hashes", {}).get("protocol") != _sha256(args.protocol)
        or _sha256(args.diagnostic)
        != _sha256(Path(protocol["outputs"]["diagnostic"]))
        or diagnostic.get("classification")
        != "adaptive_post_v78_six_seed_method_development_not_confirmatory"
        or any(
            Path(protocol["outputs"][key]).exists()
            for key in ("validation_root", "prospective_root")
        )
    ):
        raise ValueError("v80 target regime freeze evidence changed")
    spec = protocol["diagnostic"]
    artifact = build_full_fit_artifact(
        diagnostic,
        city=spec["city"],
        scenarios=spec["scenarios"],
        seeds=spec["seeds"],
        l2=float(spec["ridge_l2"]),
    )
    payload = write_full_fit_freeze(
        artifact=artifact,
        artifact_path=args.artifact,
        freeze_path=args.freeze_result,
        protocol_path=args.protocol,
        diagnostic_path=args.diagnostic,
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_scenarios": payload["active_scenarios"],
                "artifact_sha256": payload["artifact"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
