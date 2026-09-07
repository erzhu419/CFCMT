#!/usr/bin/env python3
"""Audit v73 and authorize only OOF-qualified city-specific successors."""

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
from cf_h2o.eval.traffic_signal_external_local_demand_model_diagnostic import (  # noqa: E402
    DIAGNOSTIC_PROTOCOL,
    RESULT_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v73r69-local-demand-model-diagnostic-audit-v1"
AUTHORIZATION = "authorize_oof_qualified_city_successor_freeze_only"
WINNER_VARIANT = "topology_local_squared_raw_scaled"


def _selected_feasible(variant: Mapping[str, Any]) -> Mapping[str, Any] | None:
    feasible = [row for row in variant["grid"] if bool(row["feasible"])]
    return min(
        feasible,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_delta"]),
            float(row["harmful_group_fraction"]),
            -int(row["retained_group_count"]),
            float(row["acceptance_quantile"]),
        ),
        default=None,
    )


def audit(protocol_path: Path, result_path: Path) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    city_decisions = {}
    for city, payload in sorted(result.get("cities", {}).items()):
        variant = payload["variants"][WINNER_VARIANT]
        selected = _selected_feasible(variant)
        city_decisions[str(city)] = {
            "decision": (
                "freeze_active_veto" if selected is not None else "force_phase_pressure_fallback"
            ),
            "variant": WINNER_VARIANT,
            "selected_candidate": selected,
            "proposal_count": int(payload["proposal_count"]),
            "feasible_candidate_count": int(variant["feasible_candidate_count"]),
            "oof_prediction_sha256": str(variant["oof_prediction_sha256"]),
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
        "diagnostic_only_boundary_preserved": result.get("decision")
        == "diagnostic_only_no_closed_loop_or_validation_authorization",
        "result_binds_protocol": result.get("diagnostic_protocol_sha256")
        == _sha256(protocol_path),
        "integrity_passed": result.get("integrity_gate", {}).get("passed") is True,
        "source_gate_passed": bool(result.get("source_gate"))
        and all(bool(value) for value in result["source_gate"].values()),
        "topology_baseline_reproduced": result.get("integrity_gate", {}).get(
            "topology_baseline_reproduces_v72"
        )
        is True,
        "expected_city_set": set(city_decisions)
        == set(protocol["diagnostic"]["city_scenarios"]),
        "jinan_winner_active": city_decisions.get("jinan", {}).get("decision")
        == "freeze_active_veto",
        "los_angeles_forced_fallback": city_decisions.get("los_angeles", {}).get(
            "decision"
        )
        == "force_phase_pressure_fallback",
        "winner_not_baseline": WINNER_VARIANT != "topology_squared_normalized",
        "winner_uses_label_free_local_context": bool(
            protocol["diagnostic"]["variants"][WINNER_VARIANT]["local_demand"]
        ),
        "winner_target_transform_predeclared": protocol["diagnostic"]["variants"][
            WINNER_VARIANT
        ]["target"]
        == "raw_scenario_robust_scaled",
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
        "winner_selection_rule": (
            "feasible_only_then_lower_robust_score_mean_harm_more_groups_quantile"
        ),
        "city_decisions": city_decisions,
        "integrity_gate": gates,
        "claim_boundary": (
            "Authorizes fitting frozen full-adaptation models only. It does not "
            "authorize closed-loop, validation-seed, prospective, or superiority claims."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v73 audit: {args.out}")
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
