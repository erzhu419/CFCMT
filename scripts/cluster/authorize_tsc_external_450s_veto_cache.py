#!/usr/bin/env python3
"""Authorize only the frozen v69 450-second target-veto cache collection."""

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
from scripts.cluster.freeze_tsc_external_v9_estimand_aligned_450s_veto_protocol import (  # noqa: E402
    PROTOCOL,
    V68_AUDIT_PROTOCOL,
    V68_REJECTION,
    _file_count,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


AUDIT_PROTOCOL = "tsc-v69r65-external-v9-450s-veto-cache-authorization-v1"
AUDIT_DECISION = "authorize_estimand_aligned_450s_veto_cache_collection"


def authorize(
    *,
    protocol_path: Path,
    v68_audit_path: Path,
    future_cache_root: Path,
    future_training_root: Path,
    future_closed_loop_root: Path,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    v68 = _read_json(v68_audit_path)
    collection = protocol.get("long_horizon_collection", {})
    training = protocol.get("target_veto_training", {})
    future_counts = {
        "cache": _file_count(future_cache_root),
        "training": _file_count(future_training_root),
        "closed_loop": _file_count(future_closed_loop_root),
    }
    gates = {
        "protocol_matches": protocol.get("protocol") == PROTOCOL,
        "v68_audit_bound": (
            _sha256(v68_audit_path)
            == protocol.get("v68_falsification_audit", {}).get("sha256")
            and v68.get("protocol") == V68_AUDIT_PROTOCOL
            and v68.get("status") == "PASS"
            and v68.get("decision") == V68_REJECTION
            and bool(v68.get("integrity_gate", {}).get("passed", False))
        ),
        "estimand_aligned": (
            int(collection.get("duration_sec", -1)) == 3600
            and int(collection.get("control_interval_sec", -1)) == 10
            and int(collection.get("counterfactual_horizon_intervals", -1)) == 45
            and int(collection.get("rollout_value_horizon_sec", -1)) == 450
            and protocol.get("closed_loop_development", {})
            .get("candidate", {})
            .get("cooldown_intervals")
            == 44
        ),
        "full_matrix": (
            len(collection.get("seeds", ())) == 6
            and len(set(collection.get("seeds", ()))) == 6
            and int(collection.get("scenario_count", -1)) == 4
            and int(collection.get("collection_shards_per_scenario_seed", -1)) == 32
            and int(collection.get("workers_per_seed", -1)) == 128
            and int(collection.get("expected_cache_file_count", -1)) == 768
        ),
        "demand_context_is_label_free": (
            protocol.get("demand_schedule_context", {}).get("label_free") is True
            and protocol.get("demand_schedule_context", {}).get(
                "scenario_id_is_not_a_feature"
            )
            is True
            and len(protocol.get("demand_schedule_context", {}).get("profiles", {}))
            == 4
        ),
        "few_shot_claim_explicit": (
            training.get("classification")
            == "target_simulator_labeled_few_shot_adaptation"
            and training.get("zero_shot_claim_permitted") is False
        ),
        "sealed_seed_sets_match": (
            sorted(protocol["validation"]["closed_loop_seeds"])
            == sorted(v68["untouched_validation_seeds"])
            and sorted(protocol["prospective_confirmation"]["closed_loop_seeds"])
            == sorted(v68["untouched_prospective_seeds"])
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
        "v68_falsification_audit_sha256": _sha256(v68_audit_path),
        "future_file_counts_at_authorization": future_counts,
        "gate": gates,
        "claim_boundary": (
            "Authorizes only 450-second target-simulator counterfactual cache "
            "collection on the six declared adaptation seeds."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--v68-audit", type=Path, required=True)
    parser.add_argument("--future-cache-root", type=Path, required=True)
    parser.add_argument("--future-training-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v69 authorization: {args.out}")
    payload = authorize(
        protocol_path=args.protocol,
        v68_audit_path=args.v68_audit,
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
