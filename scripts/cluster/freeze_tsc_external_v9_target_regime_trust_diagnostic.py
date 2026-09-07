#!/usr/bin/env python3
"""Freeze the disclosed post-v78 target-regime trust diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.traffic_signal.target_regime_trust import FEATURE_NAME  # noqa: E402


PROTOCOL = "tsc-v79r75-external-v9-target-regime-trust-diagnostic-v1"
V62_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v62_external_v9_"
    "topology_veto_closed_loop_development.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _file_count(path: Path) -> int:
    path = Path(path)
    if path.is_file():
        return 1
    return sum(1 for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def build_protocol(
    *,
    v78_audit_path: Path,
    v78_results_root: Path,
    diagnostic_output: Path,
    artifact_root: Path,
    validation_root: Path,
    prospective_root: Path,
) -> dict[str, Any]:
    parent = _read_json(V62_PATH)
    audit = _read_json(v78_audit_path)
    old_sealed = [Path(value) for value in parent["future_outputs"].values()]
    future_paths = old_sealed + [
        Path(diagnostic_output),
        Path(artifact_root),
        Path(validation_root),
        Path(prospective_root),
    ]
    future_counts = {str(path): _file_count(path) for path in future_paths}
    if (
        parent.get("protocol")
        != "tsc-v78r74-external-v9-topology-veto-closed-loop-development-v2"
        or audit.get("protocol")
        != "tsc-v78r74-topology-veto-closed-loop-development-audit-v2"
        or audit.get("status") != "PASS"
        or audit.get("decision")
        != "reject_topology_veto_development_and_preserve_sealed_seeds"
        or audit.get("integrity_gate", {}).get("passed") is not True
        or audit.get("advance_gate", {}).get("passed") is not False
        or _file_count(v78_results_root) != int(audit["result_file_count"])
        or any(future_counts.values())
    ):
        raise ValueError(
            f"v79 target-regime diagnostic parent or sealed evidence changed: {future_counts}"
        )
    sources = (
        "cf_h2o/traffic_signal/target_regime_trust.py",
        "cf_h2o/eval/traffic_signal_target_regime_trust_diagnostic.py",
        "scripts/cluster/freeze_tsc_external_v9_target_regime_trust_diagnostic.py",
        "scripts/cluster/run_tsc_external_target_regime_trust_diagnostic.py",
    )
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v78_failure_disclosed_development_pre_formal_cross_fit",
        "parent": {
            "protocol_path": str(V62_PATH),
            "protocol_sha256": _sha256(V62_PATH),
            "audit_path": str(Path(v78_audit_path).resolve()),
            "audit_sha256": _sha256(v78_audit_path),
            "results_root": str(Path(v78_results_root).resolve()),
            "results_tree_sha256": audit["result_tree_sha256"],
        },
        "feature_selection_disclosure": (
            "The sole queue-regime feature was selected after inspecting the failed "
            "v78 six-seed development outcomes. Therefore v79 remains adaptive method "
            "development and cannot provide confirmatory efficacy evidence."
        ),
        "diagnostic": {
            "city": "jinan",
            "seeds": parent["development"]["seeds"],
            "scenarios": parent["development"]["city_scenarios"]["jinan"],
            "matrix_size": 18,
            "target": "paired_full_horizon_waiting_time_relative_delta_vs_phase_pressure",
            "features": [FEATURE_NAME],
            "forbidden_model_features": [
                "scenario_id",
                "seed",
                "candidate_intervention_trace",
                "heldout_day_full_horizon_metrics",
                "future_vehicle_arrivals",
            ],
            "history_source": (
                "full-horizon passive PhasePressure logs from training seeds in the "
                "same target demand configuration"
            ),
            "heldout_feature": (
                "mean training-seed PhasePressure queue per lane for the heldout "
                "configuration; no metric from the heldout seed"
            ),
            "fold": "leave_one_simulator_seed_out",
            "ridge_l2": 1.0,
            "trust_rule": "predicted_relative_delta_strictly_less_than_zero",
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
            },
            "advancement_rule": {
                "maximum_mean_relative_delta": 0.0,
                "maximum_bootstrap_95pct_upper_relative_delta": 0.01,
                "maximum_seed_mean_relative_regression": 0.01,
                "minimum_improved_seed_fraction": 0.5,
                "minimum_mean_correction_vs_always_cfcmt": 0.002,
                "minimum_selection_rate": 0.1,
                "maximum_selection_rate": 0.9,
                "negative_mechanistic_slope_required_in_every_fold": True,
            },
        },
        "runtime_executable_sources": {path: _sha256(Path(path)) for path in sources},
        "future_file_counts_at_freeze": future_counts,
        "outputs": {
            "diagnostic": str(Path(diagnostic_output).resolve()),
            "artifact_root": str(Path(artifact_root).resolve()),
            "validation_root": str(Path(validation_root).resolve()),
            "prospective_root": str(Path(prospective_root).resolve()),
        },
        "validation": parent["validation"],
        "prospective_confirmation": parent["prospective_confirmation"],
        "claim_boundary": (
            "Only six disclosed adaptation seeds are used. The old and successor "
            "validation/prospective outputs remain absent until this diagnostic passes "
            "and a full-fit runtime artifact is frozen."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v78-audit", type=Path, required=True)
    parser.add_argument("--v78-results-root", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v79 protocol: {args.out}")
    payload = build_protocol(
        v78_audit_path=args.v78_audit,
        v78_results_root=args.v78_results_root,
        diagnostic_output=args.diagnostic_output,
        artifact_root=args.artifact_root,
        validation_root=args.validation_root,
        prospective_root=args.prospective_root,
    )
    _atomic_json(args.out, payload)
    print(json.dumps({"protocol": PROTOCOL, "sha256": _sha256(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
