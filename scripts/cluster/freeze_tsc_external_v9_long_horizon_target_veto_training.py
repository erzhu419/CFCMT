#!/usr/bin/env python3
"""Freeze the executable v65 OOF training analysis before cache inspection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (  # noqa: E402
    AUTHORIZATION_DECISION,
    FAILURE_DECISION,
    MODEL_PROTOCOL,
    MODEL_RANDOM_SEED,
    RESULT_PROTOCOL,
    _target_model_config,
)
from scripts.cluster.audit_tsc_external_long_horizon_target_veto_cache import (  # noqa: E402
    AUDIT_DECISION as REQUIRED_CACHE_AUDIT_DECISION,
    AUDIT_PROTOCOL as REQUIRED_CACHE_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (  # noqa: E402
    PROTOCOL as PARENT_PROTOCOL,
    _file_count,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


PROTOCOL = "tsc-v66r62-external-v9-long-horizon-target-veto-training-v1"
TRAINING_SOURCE = Path(
    "cf_h2o/eval/traffic_signal_external_long_horizon_target_veto_freeze.py"
)
VETO_SOURCE = Path("cf_h2o/traffic_signal/long_horizon_target_veto.py")
RUNTIME_SOURCE = Path("cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py")


def _reference(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_training_protocol(
    *,
    parent_path: Path,
    authorization_path: Path,
    future_cache_audit_path: Path,
    future_training_root: Path,
    future_development_root: Path,
    frozen_at_utc: str,
) -> dict[str, Any]:
    parent = _read_json(parent_path)
    authorization = _read_json(authorization_path)
    if (
        parent.get("protocol") != PARENT_PROTOCOL
        or authorization.get("status") != "PASS"
        or authorization.get("decision")
        != "authorize_full_horizon_target_veto_cache_collection"
        or authorization.get("frozen_protocol_sha256") != _sha256(parent_path)
        or future_cache_audit_path.exists()
        or _file_count(future_training_root) != 0
        or _file_count(future_development_root) != 0
    ):
        raise ValueError("v66 training freeze preconditions changed")
    source_paths = (TRAINING_SOURCE, VETO_SOURCE, RUNTIME_SOURCE)
    if any(not (PROJECT_ROOT / path).is_file() for path in source_paths):
        raise FileNotFoundError("v66 executable training source is incomplete")
    model_config = vars(_target_model_config())
    training = dict(parent["target_veto_training"])
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": str(frozen_at_utc),
        "stage": "pre_cache_label_and_pre_training_outcome_executable_analysis_freeze",
        "parent_protocol": {
            "path": _reference(parent_path),
            "sha256": _sha256(parent_path),
        },
        "collection_authorization": {
            "path": _reference(authorization_path),
            "sha256": _sha256(authorization_path),
            "decision": authorization["decision"],
        },
        "outcome_blinding_statement": {
            "cache_contents_or_labels_read": False,
            "cache_effect_metrics_read": False,
            "allowed_operational_observations": [
                "aggregate shard file count",
                "process presence",
                "node load average",
            ],
            "future_cache_audit_absent": True,
            "future_training_file_count": 0,
            "future_development_file_count": 0,
        },
        "required_cache_audit": {
            "future_path": _reference(future_cache_audit_path),
            "protocol": REQUIRED_CACHE_AUDIT_PROTOCOL,
            "decision": REQUIRED_CACHE_AUDIT_DECISION,
            "must_pass_before_training": True,
        },
        "executable_sources": {
            str(path): _sha256(PROJECT_ROOT / path) for path in source_paths
        },
        "model": {
            "family": training["model_family"],
            "protocol": MODEL_PROTOCOL,
            "random_seed": MODEL_RANDOM_SEED,
            "config": model_config,
            "causal_features_only": True,
            "reference_policy": "phase_pressure",
            "cross_fitting": training["cross_fitting"],
            "all_six_seed_final_refit_after_oof_selection": True,
        },
        "proposal_veto_contract": {
            "proposal": parent["proposal"],
            "veto_can_only_retain_or_reject_exact_proposal_action": True,
            "veto_cannot_rank_or_originate_an_alternative_action": True,
            "veto_fallback": "phase_pressure",
            "runtime_order": [
                "frozen_v60_short_horizon_proposal",
                "frozen_v60_pressure_and_conformal_gate",
                "target_300s_oof_selected_veto",
                "observable_execution_coordination",
            ],
        },
        "selection": {
            "gate_candidates": training["gate_candidates"],
            "selection_rule": training["selection_rule"],
            "bootstrap": training["bootstrap"],
            "city_specific_gate_permitted": training[
                "city_specific_gate_permitted"
            ],
            "selection_source": (
                "leave-one-simulator-seed-out 300-second target labels only"
            ),
            "development_outcomes_used_for_gate_selection": False,
        },
        "outputs": {
            "result_protocol": RESULT_PROTOCOL,
            "success_decision": AUTHORIZATION_DECISION,
            "failure_decision": FAILURE_DECISION,
            "future_training_root": str(Path(future_training_root).resolve()),
            "future_development_root": str(Path(future_development_root).resolve()),
        },
        "validation": parent["validation"],
        "prospective_confirmation": parent["prospective_confirmation"],
        "claim_boundary": parent["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--future-cache-audit", type=Path, required=True)
    parser.add_argument("--future-training-root", type=Path, required=True)
    parser.add_argument("--future-development-root", type=Path, required=True)
    parser.add_argument("--frozen-at-utc")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v66 training protocol: {args.out}")
    payload = build_training_protocol(
        parent_path=args.parent,
        authorization_path=args.authorization,
        future_cache_audit_path=args.future_cache_audit,
        future_training_root=args.future_training_root,
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
                "source_count": len(payload["executable_sources"]),
                "gate_candidate_count": len(payload["selection"]["gate_candidates"]),
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
