#!/usr/bin/env python3
"""Freeze the waiting-aligned proposal-veto OOF diagnostic."""

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
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PROTOCOL as CACHE_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-waiting-aligned-veto-diagnostic-v1"
PARENT_PROTOCOL = (
    "tsc-v64r60-external-v9-hierarchical-closed-loop-development-v2"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def build_protocol(
    *,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    manifest_path: Path,
    diagnostic_result_root: Path,
    artifact_root: Path,
    closed_loop_root: Path,
    confirmatory_root: Path,
    prospective_root: Path,
) -> dict[str, Any]:
    cache = _read_json(cache_protocol_path)
    audit = _read_json(cache_audit_path)
    parent = _read_json(proposal_parent_protocol_path)
    parent_audit = _read_json(hierarchical_freeze_audit_path)
    if (
        cache.get("protocol") != CACHE_PROTOCOL
        or audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision")
        != "authorize_waiting_aligned_seed_blocked_oof_training"
        or audit.get("gate", {}).get("passed") is not True
        or parent.get("protocol") != PARENT_PROTOCOL
        or parent_audit.get("status") != "PASS"
    ):
        raise ValueError("waiting-aligned veto diagnostic parent evidence changed")
    future_roots = [
        Path(diagnostic_result_root),
        Path(artifact_root),
        Path(closed_loop_root),
        Path(confirmatory_root),
        Path(prospective_root),
    ]
    if any(path.exists() for path in future_roots):
        raise ValueError("waiting-aligned diagnostic or sealed output root already exists")
    collection = dict(cache["collection"])
    if (
        collection.get("scenario") != "jinan_3x4_real"
        or collection.get("counterfactual_cost_mode") != "halted_queue"
        or int(collection.get("seed_count", 0)) != 22
    ):
        raise ValueError("waiting-aligned veto diagnostic collection changed")
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "development_only_seed_blocked_waiting_aligned_veto_selection",
        "frozen_inputs": {
            "cache_protocol_path": str(Path(cache_protocol_path).resolve()),
            "cache_protocol_sha256": _sha256(cache_protocol_path),
            "cache_audit_path": str(Path(cache_audit_path).resolve()),
            "cache_audit_sha256": _sha256(cache_audit_path),
            "proposal_parent_protocol_path": str(
                Path(proposal_parent_protocol_path).resolve()
            ),
            "proposal_parent_protocol_sha256": _sha256(
                proposal_parent_protocol_path
            ),
            "hierarchical_freeze_audit_path": str(
                Path(hierarchical_freeze_audit_path).resolve()
            ),
            "hierarchical_freeze_audit_sha256": _sha256(
                hierarchical_freeze_audit_path
            ),
            "manifest_path": str(Path(manifest_path).resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "training": {
            "city": "jinan",
            "scenarios": [str(collection["scenario"])],
            "seeds": [int(value) for value in collection["seeds"]],
            "collection_shards": int(collection["collection_shards_per_seed"]),
            "cross_fitting": "leave_one_simulator_seed_out_22_folds",
            "proposal_role": "immutable_v60_hierarchical_cfcmt",
            "veto_role": "accept_or_fallback_to_phase_pressure_only",
            "feature_variant": "state_topology_raw_scaled",
            "target": (
                "450_second_global_halted_queue_integral_candidate_minus_"
                "phase_pressure"
            ),
            "target_transform": "raw_scenario_robust_scaled_for_fit",
            "evaluation_target": "group_normalized_waiting_aligned_delta",
        },
        "diagnostic": {
            "acceptance_quantiles": [
                0.025,
                0.05,
                0.075,
                0.1,
                0.125,
                0.15,
                0.2,
                0.25,
                0.3,
                0.4,
                0.5,
            ],
            "model": {
                "learning_rate": 0.05,
                "max_iter": 120,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 12,
                "l2_regularization": 24.0,
                "random_state": 20260803,
                "uncertainty_quantile": 0.9,
            },
            "selection_rule": {
                "minimum_retained_groups_per_city": 50,
                "minimum_retained_seed_fraction": 0.8,
                "maximum_equal_seed_scenario_mean_delta": -0.001,
                "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
                "maximum_worst_seed_mean_delta": 0.01,
                "maximum_harmful_group_fraction": 0.45,
                "fallback": "always_veto_to_phase_pressure_if_no_gate_is_feasible",
            },
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260830,
                "unit": "simulator_seed",
            },
        },
        "future_roots_absent_at_freeze": [str(path.resolve()) for path in future_roots],
        "claim_boundary": (
            "OOF selection uses only the 22 disclosed redevelopment seeds. No "
            "confirmatory or prospective seed may be read at this stage."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--diagnostic-result-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--closed-loop-root", type=Path, required=True)
    parser.add_argument("--confirmatory-root", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite veto diagnostic protocol: {args.out}")
    payload = build_protocol(
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        manifest_path=args.manifest,
        diagnostic_result_root=args.diagnostic_result_root,
        artifact_root=args.artifact_root,
        closed_loop_root=args.closed_loop_root,
        confirmatory_root=args.confirmatory_root,
        prospective_root=args.prospective_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "folds": len(payload["training"]["seeds"]),
                "quantiles": len(payload["diagnostic"]["acceptance_quantiles"]),
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
