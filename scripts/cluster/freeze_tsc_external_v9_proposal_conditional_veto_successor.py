#!/usr/bin/env python3
"""Freeze the post-v66 proposal-conditional target-veto successor."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (  # noqa: E402
    FAILURE_DECISION as V66_FAILURE_DECISION,
    RESULT_PROTOCOL as V66_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_proposal_conditional_veto_freeze import (  # noqa: E402
    _model_config,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (  # noqa: E402
    TARGET_SUPPORT_FEATURES,
)
from scripts.cluster.audit_tsc_external_long_horizon_target_veto_cache import (  # noqa: E402
    AUDIT_DECISION as CACHE_AUDIT_DECISION,
    AUDIT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (  # noqa: E402
    PROTOCOL as COLLECTION_PROTOCOL,
    _file_count,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_training import (  # noqa: E402
    PROTOCOL as V66_TRAINING_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


PROTOCOL = "tsc-v67r63-external-v9-proposal-conditional-veto-successor-v1"
QUANTILES = (0.05, 0.075, 0.10, 0.125, 0.15)


def _reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_protocol(
    *,
    collection_protocol_path: Path,
    v66_training_protocol_path: Path,
    failed_v66_result_path: Path,
    failed_v66_artifact_root: Path,
    cache_audit_path: Path,
    future_development_root: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    collection = _read_json(collection_protocol_path)
    training = _read_json(v66_training_protocol_path)
    failed = _read_json(failed_v66_result_path)
    cache_audit = _read_json(cache_audit_path)
    if (
        collection.get("protocol") != COLLECTION_PROTOCOL
        or training.get("protocol") != V66_TRAINING_PROTOCOL
        or training["parent_protocol"]["sha256"]
        != _sha256(collection_protocol_path)
        or failed.get("protocol") != V66_RESULT_PROTOCOL
        or failed.get("status") != "FAIL"
        or failed.get("decision") != V66_FAILURE_DECISION
        or failed.get("active_cities") != []
        or failed.get("executable_training_freeze", {}).get(
            "training_protocol_sha256"
        )
        != _sha256(v66_training_protocol_path)
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision") != CACHE_AUDIT_DECISION
        or failed.get("cache_audit_sha256") != _sha256(cache_audit_path)
        or _file_count(future_development_root) != 0
    ):
        raise ValueError("proposal-conditional successor authorization changed")
    city_artifacts = {}
    for city, declared in failed["city_artifacts"].items():
        certificate = Path(failed_v66_artifact_root) / city / "freeze.json"
        model = Path(failed_v66_artifact_root) / city / "model.pkl"
        if (
            _sha256(certificate) != declared["certificate"]["sha256"]
            or _sha256(model) != declared["model"]["sha256"]
        ):
            raise ValueError(f"failed v66 artifact identity changed for {city}")
        city_artifacts[city] = {
            "certificate": {
                "path": _reference(certificate),
                "sha256": _sha256(certificate),
            },
            "model": {"path": _reference(model), "sha256": _sha256(model)},
            "v66_active": False,
        }
    executable_sources = {
        path: _sha256(PROJECT_ROOT / path)
        for path in (
            "cf_h2o/eval/traffic_signal_external_proposal_conditional_veto_freeze.py",
            "cf_h2o/traffic_signal/proposal_conditional_target_veto.py",
        )
    }
    selection_rule = dict(collection["target_veto_training"]["selection_rule"])
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "stage": "post_v66_failure_outcome_informed_method_development_freeze",
        "collection_protocol": {
            "path": _reference(collection_protocol_path),
            "sha256": _sha256(collection_protocol_path),
        },
        "v66_training_protocol": {
            "path": _reference(v66_training_protocol_path),
            "sha256": _sha256(v66_training_protocol_path),
        },
        "failed_v66_result": {
            "path": _reference(failed_v66_result_path),
            "sha256": _sha256(failed_v66_result_path),
            "status": "FAIL",
            "decision": V66_FAILURE_DECISION,
            "active_cities": [],
        },
        "failed_v66_artifacts": city_artifacts,
        "cache_audit": {
            "path": _reference(cache_audit_path),
            "sha256": _sha256(cache_audit_path),
            "decision": CACHE_AUDIT_DECISION,
        },
        "executable_sources": executable_sources,
        "training": {
            "classification": "target_simulator_labeled_few_shot_adaptation",
            "zero_shot_claim_permitted": False,
            "seeds": collection["long_horizon_collection"]["seeds"],
            "city_scenarios": collection["city_scenarios"],
            "source_records": "v66_frozen_short_proposal_records_only",
            "all_action_groups_used_only_as_deployed_effect_denominators": True,
            "target": "300_second_group_normalized_candidate_minus_phase_cost",
            "features": list(TARGET_SUPPORT_FEATURES),
            "feature_count": len(TARGET_SUPPORT_FEATURES),
            "excludes_city_id_context_and_target_outcome_features": True,
            "model": asdict(_model_config()),
            "cross_fitting": "leave_one_simulator_seed_out",
            "fold_threshold_rule": (
                "quantile_of_training_fold_predictions_on_short_proposals_only"
            ),
            "acceptance_quantiles": list(QUANTILES),
            "fallback_quantile": 0.125,
            "selection_rule": selection_rule,
            "bootstrap": {
                "replicates": 10000,
                "seed": 20260819,
                "unit": "simulator_seed_equal_scenario_mean",
            },
            "final_fit": (
                "refit_on_all_six_development_seeds_and_calibrate_selected_"
                "quantile_on_final_training_predictions"
            ),
            "scope": (
                "veto_only_after_frozen_v60_CFCMT_proposes; never originate, "
                "redirect, or modify phase-pressure reference"
            ),
        },
        "future_development_root": str(Path(future_development_root).resolve()),
        "future_result_file_count_at_freeze": 0,
        "validation": collection["validation"],
        "prospective_confirmation": collection["prospective_confirmation"],
        "scientific_disclosure": {
            "v65_target_labels_inspected_during_successor_design": True,
            "v66_oof_failure_is_preserved_not_overwritten": True,
            "successor_oof_is_exploratory_method_selection_not_confirmation": True,
            "first_independent_efficacy_test": (
                "frozen_3600_second_closed_loop_development_seeds"
            ),
            "validation_and_prospective_seeds_untouched": True,
        },
        "claim_boundary": (
            "A post-failure successor freeze. Its same-data OOF metrics may select "
            "an implementation but cannot support an efficacy claim."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-protocol", type=Path, required=True)
    parser.add_argument("--v66-training-protocol", type=Path, required=True)
    parser.add_argument("--failed-v66-result", type=Path, required=True)
    parser.add_argument("--failed-v66-artifact-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--future-development-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite successor protocol: {args.out}")
    payload = build_protocol(
        collection_protocol_path=args.collection_protocol,
        v66_training_protocol_path=args.v66_training_protocol,
        failed_v66_result_path=args.failed_v66_result,
        failed_v66_artifact_root=args.failed_v66_artifact_root,
        cache_audit_path=args.cache_audit,
        future_development_root=args.future_development_root,
        frozen_at_utc=args.frozen_at_utc or datetime.now(timezone.utc).isoformat(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "quantiles": payload["training"]["acceptance_quantiles"],
                "feature_count": payload["training"]["feature_count"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
