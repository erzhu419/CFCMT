#!/usr/bin/env python3
"""Freeze the untouched-seed v81 target-regime validation protocol."""

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
from cf_h2o.eval.traffic_signal_target_regime_trust_freeze import (  # noqa: E402
    ARTIFACT_PROTOCOL,
    FREEZE_PROTOCOL,
    load_frozen_regime_trust,
)
from cf_h2o.eval.traffic_signal_external_target_regime_trust_validation import (  # noqa: E402
    CANDIDATE_KEY,
    PHASE_POLICY,
    PROTOCOL,
)


V62_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v62_external_v9_"
    "topology_veto_closed_loop_development.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_protocol(
    *,
    diagnostic_protocol_path: Path,
    diagnostic_result_path: Path,
    regime_freeze_path: Path,
    regime_artifact_path: Path,
    validation_results_root: Path,
    launch_output: Path,
    validation_audit_output: Path,
    prospective_root: Path,
) -> dict[str, Any]:
    parent = _read_json(V62_PATH)
    diagnostic_protocol = _read_json(diagnostic_protocol_path)
    diagnostic = _read_json(diagnostic_result_path)
    freeze = _read_json(regime_freeze_path)
    artifact, model = load_frozen_regime_trust(
        artifact_path=regime_artifact_path,
        expected_sha256=freeze["artifact"]["sha256"],
    )
    old_sealed_paths = [Path(value) for value in parent["future_outputs"].values()]
    future_paths = old_sealed_paths + [
        Path(validation_results_root),
        Path(launch_output),
        Path(validation_audit_output),
        Path(prospective_root),
    ]
    if (
        diagnostic.get("decision")
        != "authorize_target_regime_trust_full_fit_freeze_only"
        or diagnostic.get("advance_gate", {}).get("passed") is not True
        or diagnostic.get("input_hashes", {}).get("protocol")
        != _sha256(diagnostic_protocol_path)
        or freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("status") != "PASS"
        or freeze.get("decision")
        != "authorize_untouched_validation_protocol_freeze_only"
        or freeze.get("diagnostic", {}).get("sha256")
        != _sha256(diagnostic_result_path)
        or artifact.get("protocol") != ARTIFACT_PROTOCOL
        or any(path.exists() for path in future_paths)
    ):
        raise ValueError("v81 validation parent or sealed evidence changed")
    development_seeds = [int(value) for value in diagnostic_protocol["diagnostic"]["seeds"]]
    validation_seeds = [int(value) for value in parent["validation"]["closed_loop_seeds"]]
    prospective_seeds = [
        int(value) for value in parent["prospective_confirmation"]["closed_loop_seeds"]
    ]
    if (
        set(development_seeds) & set(validation_seeds)
        or set(development_seeds) & set(prospective_seeds)
        or set(validation_seeds) & set(prospective_seeds)
    ):
        raise ValueError("target regime seed partitions overlap")
    city_scenarios = parent["validation"]["city_scenarios"]
    scenario_gate = {}
    for city, scenarios in city_scenarios.items():
        for scenario in scenarios:
            if city == artifact["city"]:
                row = artifact["scenario_history"][scenario]
                feature = float(row["historical_phase_mean_queue_per_lane"])
                decision = model.trusts_local_controller(feature)
                if decision != bool(row["trust_local_controller"]):
                    raise ValueError("target regime artifact decision changed")
            else:
                decision = False
            scenario_gate[str(scenario)] = bool(decision)
    if sum(scenario_gate.values()) != 1:
        raise ValueError("v81 validation requires one frozen active scenario")
    sources = (
        "cf_h2o/eval/traffic_signal_external_target_regime_trust_validation.py",
        "cf_h2o/eval/traffic_signal_target_regime_trust_freeze.py",
        "cf_h2o/traffic_signal/target_regime_trust.py",
        "scripts/cluster/run_tsc_external_target_regime_trust_validation_shard.py",
        "scripts/cluster/launch_tsc_external_target_regime_trust_validation.py",
        "scripts/cluster/audit_tsc_external_target_regime_trust_validation.py",
    )
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_full_fit_freeze_pre_untouched_validation_outcome",
        "diagnostic_protocol": {
            "path": str(Path(diagnostic_protocol_path).resolve()),
            "sha256": _sha256(diagnostic_protocol_path),
        },
        "diagnostic_result": {
            "path": str(Path(diagnostic_result_path).resolve()),
            "sha256": _sha256(diagnostic_result_path),
            "adaptive_development_not_confirmatory": True,
        },
        "regime_trust_freeze": {
            "path": str(Path(regime_freeze_path).resolve()),
            "sha256": _sha256(regime_freeze_path),
        },
        "regime_trust_artifact": {
            "path": str(Path(regime_artifact_path).resolve()),
            "sha256": _sha256(regime_artifact_path),
        },
        "development_evidence": {
            "seeds": development_seeds,
            "feature_selection_disclosed_post_v78": True,
        },
        "execution_evidence": {
            key: parent[key]
            for key in (
                "proposal_parent_protocol",
                "hierarchical_freeze_audit",
                "topology_veto_freeze_result",
                "topology_veto_joint_audit",
                "topology_veto_artifact_root",
            )
        },
        "proposal": parent["proposal"],
        "validation": {
            "classification": "untouched_seed_confirmatory_validation",
            "seeds": validation_seeds,
            "city_scenarios": city_scenarios,
            "duration_sec": 3600,
            "primary_metric": "mean_tripinfo_waiting_time",
            "phase_policy": PHASE_POLICY,
            "candidate": {
                "key": CANDIDATE_KEY,
                "runtime_policy": parent["development"]["candidate"]["runtime_policy"],
                "coordination_mode": "direct",
                "cooldown_intervals": 44,
                "max_simultaneous_overrides": 1,
            },
            "policies": [PHASE_POLICY, CANDIDATE_KEY],
            "scenario_gate": scenario_gate,
            "matrix_size": 2
            * len(validation_seeds)
            * sum(len(values) for values in city_scenarios.values()),
            "analysis": {
                "bootstrap_replicates": 10000,
                "bootstrap_seed": 20260831,
                "active_scenario": {
                    "minimum_mean_relative_improvement": 0.0025,
                    "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.005,
                    "maximum_seed_relative_regression": 0.03,
                    "minimum_improved_seed_fraction": 0.5,
                    "minimum_executed_interventions_per_seed": 1,
                },
                "jinan_all_scenarios": {
                    "maximum_mean_relative_delta": 0.0,
                    "maximum_bootstrap_95pct_upper_mean_relative_delta": 0.003,
                    "maximum_seed_mean_relative_regression": 0.01,
                    "minimum_improved_seed_fraction": 0.5,
                },
                "fallback": {
                    "mean_waiting_time_equal_atol": 1e-12,
                    "mean_travel_time_equal_atol": 1e-12,
                    "zero_interventions": True,
                    "safety_counts_equal": True,
                },
                "collisions_noninferior_to_phase": True,
                "teleports_noninferior_to_phase": True,
            },
        },
        "environment": parent["environment"],
        "runtime_executable_sources": {path: _sha256(Path(path)) for path in sources},
        "outputs": {
            "validation_results_root": str(Path(validation_results_root).resolve()),
            "launch": str(Path(launch_output).resolve()),
            "audit": str(Path(validation_audit_output).resolve()),
            "prospective_root": str(Path(prospective_root).resolve()),
        },
        "prospective_confirmation": parent["prospective_confirmation"],
        "claim_boundary": (
            "The 16 validation seeds are untouched confirmatory evidence for the "
            "already frozen regime gate. The three prospective seeds remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--regime-freeze", type=Path, required=True)
    parser.add_argument("--regime-artifact", type=Path, required=True)
    parser.add_argument("--validation-results-root", type=Path, required=True)
    parser.add_argument("--launch-output", type=Path, required=True)
    parser.add_argument("--validation-audit-output", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v81 validation protocol: {args.out}")
    payload = build_protocol(
        diagnostic_protocol_path=args.diagnostic_protocol,
        diagnostic_result_path=args.diagnostic_result,
        regime_freeze_path=args.regime_freeze,
        regime_artifact_path=args.regime_artifact,
        validation_results_root=args.validation_results_root,
        launch_output=args.launch_output,
        validation_audit_output=args.validation_audit_output,
        prospective_root=args.prospective_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "matrix_size": payload["validation"]["matrix_size"],
                "scenario_gate": payload["validation"]["scenario_gate"],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
