#!/usr/bin/env python3
"""Audit the historical-prefix OOF diagnostic and its advancement gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_historical_demand_model_diagnostic import (  # noqa: E402
    BASE_VARIANT,
    DIAGNOSTIC_PROTOCOL,
    HISTORICAL_VARIANT,
    RESULT_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v76r72-historical-prefix-model-diagnostic-audit-v1"
AUTHORIZATION = "authorize_historical_prefix_model_freeze_only"


def _selected(variant: Mapping[str, Any]) -> Mapping[str, Any]:
    return variant["selected_candidate"]


def audit(protocol_path: Path, result_path: Path) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    city_decisions = {}
    for city, payload in sorted(result.get("cities", {}).items()):
        baseline = payload["variants"][BASE_VARIANT]
        historical = payload["variants"][HISTORICAL_VARIANT]
        base_selected = _selected(baseline)
        history_selected = _selected(historical)
        history_advances = (
            bool(historical["passed_selection_rule"])
            and float(history_selected["robust_score"])
            < float(base_selected["robust_score"])
            and float(history_selected["mean_delta"])
            < float(base_selected["mean_delta"])
        )
        city_decisions[str(city)] = {
            "decision": (
                "freeze_active_historical_veto"
                if history_advances
                else "force_phase_pressure_fallback"
            ),
            "baseline_selected": base_selected,
            "historical_selected": history_selected,
            "historical_passed_selection_rule": bool(
                historical["passed_selection_rule"]
            ),
            "historical_beats_baseline_robust_score": float(
                history_selected["robust_score"]
            )
            < float(base_selected["robust_score"]),
            "historical_beats_baseline_mean_delta": float(
                history_selected["mean_delta"]
            )
            < float(base_selected["mean_delta"]),
            "proposal_count": int(payload["proposal_count"]),
            "historical_oof_prediction_sha256": str(
                historical["oof_prediction_sha256"]
            ),
        }
    adaptation = {int(value) for value in protocol["diagnostic"]["seeds"]}
    validation = {int(value) for value in protocol["validation"]["closed_loop_seeds"]}
    prospective = {
        int(value)
        for value in protocol["prospective_confirmation"]["closed_loop_seeds"]
    }
    gates = {
        "protocol_identity": protocol.get("protocol") == DIAGNOSTIC_PROTOCOL,
        "result_identity": result.get("protocol") == RESULT_PROTOCOL,
        "result_passed": result.get("status") == "PASS",
        "diagnostic_boundary_preserved": result.get("decision")
        == "diagnostic_only_no_closed_loop_or_sealed_seed_authorization",
        "result_binds_protocol": result.get("diagnostic_protocol_sha256")
        == _sha256(protocol_path),
        "integrity_passed": result.get("integrity_gate", {}).get("passed") is True,
        "expected_city_set": set(city_decisions)
        == set(protocol["diagnostic"]["city_scenarios"]),
        "jinan_historical_module_advances": city_decisions.get("jinan", {}).get(
            "decision"
        )
        == "freeze_active_historical_veto",
        "los_angeles_forced_fallback": city_decisions.get(
            "los_angeles", {}
        ).get("decision")
        == "force_phase_pressure_fallback",
        "runtime_features_exclude_future_terms": all(
            marker not in name
            for name in protocol["diagnostic"]["variants"][HISTORICAL_VARIANT][
                "feature_names"
            ]
            for marker in ("next", "remaining", "future")
        ),
        "abandoned_v75_had_no_results": int(
            protocol["abandoned_v75"]["result_file_count_at_correction_freeze"]
        )
        == 0,
        "adaptation_validation_disjoint": not adaptation & validation,
        "adaptation_prospective_disjoint": not adaptation & prospective,
        "validation_prospective_disjoint": not validation & prospective,
        "sealed_outputs_absent_at_freeze": all(
            int(value) == 0
            for value in protocol["future_file_counts_at_freeze"].values()
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "status": "PASS" if gates["passed"] else "FAIL",
        "decision": AUTHORIZATION if gates["passed"] else "reject",
        "diagnostic_protocol": {
            "path": str(protocol_path),
            "sha256": _sha256(protocol_path),
        },
        "diagnostic_result": {
            "path": str(result_path),
            "sha256": _sha256(result_path),
        },
        "city_decisions": city_decisions,
        "integrity_gate": gates,
        "claim_boundary": (
            "Authorizes fitting a historical-prefix model on disclosed adaptation "
            "seeds only. It does not authorize closed-loop or sealed-seed claims."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite historical audit: {args.out}")
    payload = audit(args.protocol, args.result)
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "city_decisions": {
                    city: row["decision"]
                    for city, row in payload["city_decisions"].items()
                },
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
