#!/usr/bin/env python3
"""Run the frozen v79 target-regime trust cross-fitted diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_target_regime_trust_diagnostic import (  # noqa: E402
    run_cross_fitted_diagnostic,
)
from cf_h2o.traffic_signal.target_regime_trust import (  # noqa: E402
    phase_history_feature,
)
from scripts.cluster.freeze_tsc_external_v9_target_regime_trust_diagnostic import (  # noqa: E402
    PROTOCOL,
)


RUN_PROTOCOL = "tsc-v79r75-target-regime-trust-diagnostic-run-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(*, protocol_path: Path) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("v79 target-regime diagnostic protocol changed")
    parent = protocol["parent"]
    audit_path = Path(parent["audit_path"])
    audit = _read_json(audit_path)
    if (
        _sha256(audit_path) != parent["audit_sha256"]
        or audit.get("integrity_gate", {}).get("passed") is not True
        or audit.get("advance_gate", {}).get("passed") is not False
    ):
        raise ValueError("v79 parent audit changed")
    row_index = {
        (row["city"], row["scenario"], int(row["seed"]), row["policy"]): row
        for row in audit["rows"]
    }
    spec = protocol["diagnostic"]
    city = str(spec["city"])
    rows = []
    for paired in audit["paired_rows"]:
        if paired["city"] != city:
            continue
        key = (city, paired["scenario"], int(paired["seed"]), "phase_pressure")
        phase_row = row_index[key]
        phase_path = Path(phase_row["result_path"])
        if _sha256(phase_path) != phase_row["result_sha256"]:
            raise ValueError(f"PhasePressure history changed: {phase_path}")
        phase = _read_json(phase_path)
        metrics = phase["metrics"]
        rows.append(
            {
                "seed": int(paired["seed"]),
                "scenario": str(paired["scenario"]),
                "phase_mean_queue_per_lane": phase_history_feature(metrics),
                "relative_delta": float(paired["relative_delta"]),
                "phase_result_path": str(phase_path.resolve()),
                "phase_result_sha256": phase_row["result_sha256"],
                "candidate_outcome_source": "v78_paired_full_horizon_waiting_time",
            }
        )
    diagnostic = run_cross_fitted_diagnostic(
        rows,
        seeds=spec["seeds"],
        scenarios=spec["scenarios"],
        l2=float(spec["ridge_l2"]),
        bootstrap=spec["bootstrap"],
        advancement_rule=spec["advancement_rule"],
    )
    sealed_paths = [
        Path(protocol["outputs"][key])
        for key in ("artifact_root", "validation_root", "prospective_root")
    ]
    integrity = {
        "parent_audit_bound": _sha256(audit_path) == parent["audit_sha256"],
        "matrix_complete": len(rows) == int(spec["matrix_size"]),
        "phase_history_only": all(
            row["candidate_outcome_source"]
            == "v78_paired_full_horizon_waiting_time"
            for row in rows
        ),
        "successor_outputs_still_absent": all(not path.exists() for path in sealed_paths),
    }
    integrity["passed"] = all(integrity.values())
    advance = bool(integrity["passed"] and diagnostic["advancement_gate"]["passed"])
    return {
        "protocol": RUN_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": (
            "authorize_target_regime_trust_full_fit_freeze_only"
            if advance
            else "reject_target_regime_trust_and_preserve_sealed_seeds"
        ),
        "classification": (
            "adaptive_post_v78_six_seed_method_development_not_confirmatory"
        ),
        "input_hashes": {
            "protocol": _sha256(protocol_path),
            "parent_audit": _sha256(audit_path),
        },
        "source_rows": rows,
        "diagnostic": diagnostic,
        "integrity_gate": integrity,
        "advance_gate": {"passed": advance},
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v79 diagnostic: {args.out}")
    payload = run(protocol_path=args.protocol)
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "advance": payload["advance_gate"]["passed"],
                "always_cfcmt_mean_delta": payload["diagnostic"]["always_cfcmt"][
                    "mean_relative_delta"
                ],
                "gated_mean_delta": payload["diagnostic"][
                    "cross_fitted_regime_gate"
                ]["mean_relative_delta"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
