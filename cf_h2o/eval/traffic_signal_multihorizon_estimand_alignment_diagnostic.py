"""Diagnose v80 rollout/estimand mismatch before freezing v81 development."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_multihorizon_state_latent_closed_loop_development import (
    RESULT_PROTOCOL as V80_ROLLOUT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.audit_tsc_external_multihorizon_state_latent_closed_loop import (
    AUDIT_PROTOCOL as V80_AUDIT_PROTOCOL,
    REJECTION_DECISION as V80_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_state_latent_closed_loop_development import (
    PROTOCOL as V80_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-multihorizon-estimand-alignment-diagnostic-v1"
AUTHORIZATION_DECISION = "authorize_estimand_aligned_global_cooldown_development"
SOURCE_POLICY = "mh_compact_cd0"
PRIORITY_THRESHOLDS = (0.0, 0.02, 0.05, 0.10, 0.15)
GLOBAL_COOLDOWN_CONTROLS = (0, 2, 5)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def run_diagnostic(
    *, protocol_path: Path, audit_path: Path, results_root: Path
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    audit = _read_json(audit_path)
    if (
        protocol.get("protocol") != V80_PROTOCOL
        or audit.get("protocol") != V80_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision") != V80_REJECTION_DECISION
        or audit.get("integrity_gate", {}).get("passed") is not True
        or audit.get("protocol_sha256") != _sha256(protocol_path)
        or int(audit.get("matrix_size", -1)) != 154
    ):
        raise ValueError("v80 estimand diagnostic parent evidence changed")
    summaries = {
        str(row["policy"]): dict(row) for row in audit["candidate_summaries"]
    }
    source = summaries[SOURCE_POLICY]
    if source.get("feasible") is not False or float(source["mean_relative_delta"]) <= 0:
        raise ValueError("v80 source policy no longer represents failed development")

    traces = []
    seed_rows = []
    protocol_sha = _sha256(protocol_path)
    for pair in source["seed_rows"]:
        seed = int(pair["seed"])
        result_path = (
            Path(results_root)
            / "jinan"
            / "jinan_3x4_real"
            / f"seed_{seed}"
            / SOURCE_POLICY
            / "result.json"
        )
        result = _read_json(result_path)
        metrics = dict(result["metrics"])
        trace = [
            dict(row)
            for row in metrics["accepted_intervention_trace"]
            if bool(row["request_accepted"])
        ]
        if not (
            result.get("protocol") == V80_ROLLOUT_PROTOCOL
            and result.get("protocol_sha256") == protocol_sha
            and int(result.get("seed", -1)) == seed
            and result.get("policy") == SOURCE_POLICY
            and int(metrics["guard_audit"]["executed_overrides"]) == len(trace)
        ):
            raise ValueError(f"v80 intervention trace changed for seed {seed}")
        times = np.asarray(sorted(float(row["time_sec"]) for row in trace))
        gaps = np.diff(times)
        seed_rows.append(
            {
                "seed": seed,
                "relative_delta": float(pair["relative_delta"]),
                "executed_overrides": len(trace),
                "adjacent_gap_count": int(gaps.size),
                "adjacent_gap_below_60s_count": int(np.sum(gaps < 60.0)),
                "median_adjacent_gap_sec": (
                    float(np.median(gaps)) if gaps.size else None
                ),
                "result_sha256": _sha256(result_path),
            }
        )
        traces.extend(
            {
                "seed": seed,
                "relative_delta": float(pair["relative_delta"]),
                "time_sec": float(row["time_sec"]),
                "priority": float(row["priority"]),
                "total_queue": float(row["total_queue"]),
                "mean_speed": float(row["mean_speed"]),
            }
            for row in trace
        )
    priorities = np.asarray([row["priority"] for row in traces], dtype=float)
    gaps_below = sum(row["adjacent_gap_below_60s_count"] for row in seed_rows)
    gap_count = sum(row["adjacent_gap_count"] for row in seed_rows)
    expected_cooldown = (
        int(protocol["environment"]["prediction_horizon_sec"])
        // int(protocol["environment"]["control_interval_sec"])
    ) - 1
    priority_quantiles = {
        str(quantile): float(np.quantile(priorities, quantile))
        for quantile in (0.10, 0.25, 0.50, 0.75, 0.85, 0.90)
    }
    threshold_activity = {
        str(threshold): {
            "retained_trace_count": int(np.sum(priorities >= threshold)),
            "retained_trace_fraction": float(np.mean(priorities >= threshold)),
            "active_seed_count": sum(
                any(
                    row["seed"] == seed and row["priority"] >= threshold
                    for row in traces
                )
                for seed in protocol["development"]["seeds"]
            ),
        }
        for threshold in PRIORITY_THRESHOLDS
    }
    integrity = {
        "v80_rejection_chain_exact": True,
        "development_seed_count_exact": len(seed_rows) == 22,
        "trace_count_matches_executed_overrides": len(traces)
        == int(source["total_executed_overrides"]),
        "estimand_horizon_exact": int(
            protocol["environment"]["prediction_horizon_sec"]
        )
        == 60,
        "control_interval_exact": int(
            protocol["environment"]["control_interval_sec"]
        )
        == 10,
        "derived_global_cooldown_exact": expected_cooldown == 5,
        "repeated_intervention_mismatch_observed": gaps_below > 0,
        "priority_grid_has_full_seed_activity_before_global_cooldown": all(
            row["active_seed_count"] == 22 for row in threshold_activity.values()
        ),
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": (
            AUTHORIZATION_DECISION
            if integrity["passed"]
            else "retain_v80_rejection_and_stop_estimand_alignment"
        ),
        "v80_protocol_sha256": _sha256(protocol_path),
        "v80_audit_sha256": _sha256(audit_path),
        "source_policy": SOURCE_POLICY,
        "source_mean_relative_delta": float(source["mean_relative_delta"]),
        "executed_trace_count": len(traces),
        "adjacent_gap_count": int(gap_count),
        "adjacent_gap_below_60s_count": int(gaps_below),
        "adjacent_gap_below_60s_fraction": float(gaps_below / max(gap_count, 1)),
        "priority_quantiles": priority_quantiles,
        "priority_thresholds": list(PRIORITY_THRESHOLDS),
        "threshold_activity": threshold_activity,
        "global_cooldown_controls": list(GLOBAL_COOLDOWN_CONTROLS),
        "estimand_aligned_global_cooldown_intervals": expected_cooldown,
        "seed_rows": seed_rows,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": bool(integrity["passed"]),
            "next_stage": "v81_estimand_aligned_closed_loop_development",
        },
        "claim_boundary": (
            "This diagnostic uses only v80 adaptive-development traces. It "
            "identifies rollout/estimand mismatch and freezes a successor grid; "
            "it is not efficacy evidence and does not unseal v84 or v85."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite estimand alignment diagnostic")
    payload = run_diagnostic(
        protocol_path=args.protocol,
        audit_path=args.audit,
        results_root=args.results_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "trace_count": payload["executed_trace_count"],
                "gap_below_60_fraction": payload[
                    "adjacent_gap_below_60s_fraction"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
