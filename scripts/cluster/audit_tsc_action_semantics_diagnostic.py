#!/usr/bin/env python3
"""Audit the seed-99518 stay/switch execution-semantics diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_interference_aware_redevelopment import (  # noqa: E402
    RESULT_PROTOCOL as V86_RESULT_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v87-action-semantics-diagnostic-audit-v1"
SEED = 99518


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _trace_row_at(result: Mapping[str, Any], time_sec: float) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in result["metrics"]["accepted_intervention_trace"]
        if abs(float(row["time_sec"]) - float(time_sec)) <= 1e-9
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one residual trace row at t={time_sec}, got {len(rows)}")
    return rows[0]


def _compact_result(path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
    metrics = dict(result["metrics"])
    guard = dict(metrics.get("guard_audit", {}))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "protocol": str(result["protocol"]),
        "policy": str(result["policy"]),
        "seed": int(result["seed"]),
        "mean_waiting_time": float(metrics["mean_tripinfo_waiting_time"]),
        "collision_incidents": int(metrics["collision_incidents"]),
        "collision_events": int(metrics["collision_events"]),
        "executed_switch_overrides": int(
            guard.get("executed_switch_overrides", guard.get("executed_overrides", 0))
        ),
        "executed_stay_overrides": int(guard.get("executed_stay_overrides", 0)),
        "effective_overrides": int(
            guard.get("effective_overrides", guard.get("executed_overrides", 0))
        ),
    }


def audit_diagnostic(
    *,
    global_result_path: Path,
    local_result_path: Path,
    stay_refresh_result_path: Path,
    transition_only_result_path: Path,
) -> dict[str, Any]:
    global_result = _read_json(global_result_path)
    local_result = _read_json(local_result_path)
    stay_refresh = _read_json(stay_refresh_result_path)
    transition_only = _read_json(transition_only_result_path)

    if not (
        global_result.get("protocol") == V86_RESULT_PROTOCOL
        and local_result.get("protocol") == V86_RESULT_PROTOCOL
        and stay_refresh.get("protocol") == "tsc-v87-action-aware-seed-diagnostic-v1"
        and transition_only.get("protocol")
        == "tsc-v87-action-aware-seed-diagnostic-v2"
        and all(
            int(result.get("seed", -1)) == SEED
            for result in (global_result, local_result, stay_refresh, transition_only)
        )
        and global_result.get("policy") == "uh_global_h120_m1"
        and local_result.get("policy") == "uh_local_h120_m1"
        and stay_refresh.get("diagnostic_only_evidence") is True
        and stay_refresh.get("selection_eligible_evidence") is False
        and transition_only.get("diagnostic_only_evidence") is True
        and transition_only.get("selection_eligible_evidence") is False
    ):
        raise ValueError("action-semantics diagnostic identity or evidence class changed")

    global_t70 = _trace_row_at(global_result, 70.0)
    local_t70 = _trace_row_at(local_result, 70.0)
    local_t130 = _trace_row_at(local_result, 130.0)
    transition_t70 = _trace_row_at(transition_only, 70.0)
    transition_t130 = _trace_row_at(transition_only, 130.0)
    global_has_t130 = any(
        abs(float(row["time_sec"]) - 130.0) <= 1e-9
        for row in global_result["metrics"]["accepted_intervention_trace"]
    )
    mechanism_gate = {
        "global_and_local_share_t70_switch": bool(
            global_t70["request_accepted"]
            and local_t70["request_accepted"]
            and global_t70["tls_id"] == local_t70["tls_id"] == "intersection_1_1"
        ),
        "legacy_global_suppresses_t130_stay": not global_has_t130,
        "legacy_local_exposes_t130_stay": bool(
            local_t130["request_accepted"] is False
            and local_t130["tls_id"] == "intersection_4_1"
            and local_t130["selected_phase_state"]
            == local_t130["current_green_state_before"]
            and local_t130["selected_phase_state"] != local_t130["prior_phase_state"]
        ),
        "transition_only_preserves_t70_switch": bool(
            transition_t70.get("execution_effective") is True
            and transition_t70.get("override_kind") == "switch"
        ),
        "transition_only_executes_t130_stay": bool(
            transition_t130.get("execution_effective") is True
            and transition_t130.get("override_kind") == "stay"
            and transition_t130["tls_id"] == "intersection_4_1"
        ),
        "legacy_collision_is_removed_in_both_action_aware_variants": bool(
            int(global_result["metrics"]["collision_incidents"]) == 1
            and int(local_result["metrics"]["collision_incidents"]) == 0
            and int(stay_refresh["metrics"]["collision_incidents"]) == 0
            and int(transition_only["metrics"]["collision_incidents"]) == 0
        ),
    }
    mechanism_gate["passed"] = all(mechanism_gate.values())
    if not mechanism_gate["passed"]:
        raise ValueError("action-semantics mechanism gate failed")

    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "decision": "authorize_phase_transition_aware_development_only",
        "seed": SEED,
        "mechanism_gate": mechanism_gate,
        "results": {
            "legacy_global": _compact_result(global_result_path, global_result),
            "legacy_local": _compact_result(local_result_path, local_result),
            "stay_refresh": _compact_result(stay_refresh_result_path, stay_refresh),
            "phase_transition_only": _compact_result(
                transition_only_result_path, transition_only
            ),
        },
        "claim_boundary": (
            "This single-seed diagnostic identifies an execution-semantics failure and "
            "authorizes adaptive development only. It is neither efficacy selection nor "
            "confirmation evidence."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-result", type=Path, required=True)
    parser.add_argument("--local-result", type=Path, required=True)
    parser.add_argument("--stay-refresh-result", type=Path, required=True)
    parser.add_argument("--transition-only-result", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic audit: {args.out}")
    payload = audit_diagnostic(
        global_result_path=args.global_result,
        local_result_path=args.local_result,
        stay_refresh_result_path=args.stay_refresh_result,
        transition_only_result_path=args.transition_only_result,
    )
    _atomic_json(args.out, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
