#!/usr/bin/env python3
"""Authorize v65 cache collection after checking the frozen evidence chain."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_hierarchical_closed_loop_development import (  # noqa: E402
    AUDIT_PROTOCOL as V64_AUDIT_PROTOCOL,
    REJECTION_DECISION as V64_REJECTION_DECISION,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (  # noqa: E402
    PROTOCOL,
    _file_count,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


AUDIT_PROTOCOL = "tsc-v65r61-external-v9-long-horizon-target-veto-authorization-v1"
AUDIT_DECISION = "authorize_full_horizon_target_veto_cache_collection"


def authorize_cache_collection(
    *,
    protocol_path: Path,
    v64_audit_path: Path,
    future_cache_root: Path,
    future_training_root: Path,
    future_closed_loop_root: Path,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    v64_audit = _read_json(v64_audit_path)
    future_counts = {
        "cache": _file_count(future_cache_root),
        "training": _file_count(future_training_root),
        "closed_loop": _file_count(future_closed_loop_root),
    }
    collection = dict(protocol.get("long_horizon_collection", {}))
    training = dict(protocol.get("target_veto_training", {}))
    gates = {
        "protocol_matches": protocol.get("protocol") == PROTOCOL,
        "v64_audit_hash_matches": _sha256(v64_audit_path)
        == protocol.get("v64_falsification_audit", {}).get("sha256"),
        "v64_rejection_is_integrity_clean": (
            v64_audit.get("protocol") == V64_AUDIT_PROTOCOL
            and v64_audit.get("status") == "PASS"
            and v64_audit.get("decision") == V64_REJECTION_DECISION
            and bool(v64_audit.get("integrity_gate", {}).get("passed", False))
        ),
        "collection_is_full_horizon": (
            int(collection.get("duration_sec", -1)) == 3600
            and int(collection.get("counterfactual_horizon_intervals", -1)) == 30
            and int(collection.get("rollout_value_horizon_sec", -1)) == 300
            and bool(collection.get("full_network_and_full_benchmark_horizon", False))
        ),
        "all_six_training_seeds_declared": len(collection.get("seeds", ())) == 6
        and len(set(collection.get("seeds", ()))) == 6,
        "all_four_scenarios_declared": int(collection.get("scenario_count", -1))
        == 4,
        "exact_cache_matrix_declared": int(
            collection.get("expected_cache_file_count", -1)
        )
        == 384,
        "few_shot_claim_is_explicit": training.get("classification")
        == "target_simulator_labeled_few_shot_adaptation"
        and training.get("zero_shot_claim_permitted") is False,
        "proposal_is_immutable_and_target_label_free": protocol.get("proposal", {}).get(
            "target_labels_used_by_proposal"
        )
        is False,
        "validation_and_prospective_remain_sealed": bool(
            protocol.get("validation", {}).get(
                "must_not_run_until_development_selector_and_analysis_hash_are_frozen",
                False,
            )
        )
        and bool(
            protocol.get("prospective_confirmation", {}).get(
                "must_not_run_until_validation_passes_and_prospective_analysis_is_frozen",
                False,
            )
        ),
        "future_outputs_absent": not any(future_counts.values()),
    }
    gates["passed"] = all(gates.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if gates["passed"] else "FAIL",
        "decision": AUDIT_DECISION if gates["passed"] else "reject",
        "frozen_protocol_sha256": _sha256(protocol_path),
        "v64_falsification_audit_sha256": _sha256(v64_audit_path),
        "future_file_counts_at_authorization": future_counts,
        "gate": gates,
        "claim_boundary": (
            "This audit authorizes only target-labeled 300-second counterfactual "
            "cache collection on the six declared training seeds."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v64-audit", type=Path, required=True)
    parser.add_argument("--future-cache-root", type=Path, required=True)
    parser.add_argument("--future-training-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v65 authorization: {args.out}")
    payload = authorize_cache_collection(
        protocol_path=args.protocol,
        v64_audit_path=args.v64_audit,
        future_cache_root=args.future_cache_root,
        future_training_root=args.future_training_root,
        future_closed_loop_root=args.future_closed_loop_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
