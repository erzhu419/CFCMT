#!/usr/bin/env python3
"""Freeze fresh seed partitions after the failed v81 confirmatory validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)


PROTOCOL = "tsc-v82r78-external-v9-post-validation-redevelopment-partition-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fresh_seeds(*, excluded: set[int], count: int, rng_seed: int) -> list[int]:
    rng = np.random.default_rng(int(rng_seed))
    selected: list[int] = []
    while len(selected) < int(count):
        value = int(rng.integers(10000, 100000))
        if value not in excluded and value not in selected:
            selected.append(value)
    return selected


def build_partition(
    *,
    validation_protocol_path: Path,
    validation_audit_path: Path,
    old_prospective_root: Path,
    redevelopment_root: Path,
    confirmatory_root: Path,
    prospective_root: Path,
) -> dict[str, Any]:
    protocol = _read_json(validation_protocol_path)
    audit = _read_json(validation_audit_path)
    development = [int(value) for value in protocol["development_evidence"]["seeds"]]
    failed_validation = [int(value) for value in protocol["validation"]["seeds"]]
    old_prospective = [
        int(value)
        for value in protocol["prospective_confirmation"]["closed_loop_seeds"]
    ]
    future_paths = [
        Path(old_prospective_root),
        Path(redevelopment_root),
        Path(confirmatory_root),
        Path(prospective_root),
    ]
    if (
        audit.get("status") != "PASS"
        or audit.get("decision")
        != "reject_target_regime_successor_and_preserve_prospective_seeds"
        or audit.get("integrity_gate", {}).get("passed") is not True
        or audit.get("advance_gate", {}).get("passed") is not False
        or _sha256(validation_protocol_path)
        != audit.get("input_hashes", {}).get("protocol")
        or any(path.exists() for path in future_paths)
    ):
        raise ValueError("post-validation partition parent or sealed roots changed")
    excluded = set(development) | set(failed_validation) | set(old_prospective)
    fresh = _fresh_seeds(excluded=excluded, count=37, rng_seed=20260830)
    confirmatory = fresh[:32]
    prospective = old_prospective + fresh[32:]
    redevelopment = development + failed_validation
    partitions = [set(redevelopment), set(confirmatory), set(prospective)]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("post-validation seed partitions overlap")
    return {
        "protocol": PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_failed_v81_pre_action_trace_redevelopment",
        "failed_validation": {
            "protocol_path": str(Path(validation_protocol_path).resolve()),
            "protocol_sha256": _sha256(validation_protocol_path),
            "audit_path": str(Path(validation_audit_path).resolve()),
            "audit_sha256": _sha256(validation_audit_path),
            "decision": audit["decision"],
            "active_mean_relative_delta": audit["active_scenario_summary"][
                "mean_relative_delta"
            ],
            "jinan_mean_relative_delta": audit["jinan_all_scenarios_summary"][
                "mean_relative_delta"
            ],
        },
        "partition_generation": {
            "algorithm": "numpy_pcg64_unique_integers_10000_99999",
            "rng_seed": 20260830,
            "fresh_draw_count": 37,
            "generated_before_action_trace_redevelopment": True,
        },
        "redevelopment": {
            "classification": "development_only_never_confirmatory_again",
            "seeds": redevelopment,
            "original_adaptation_seeds": development,
            "failed_v81_validation_seeds": failed_validation,
            "seed_count": len(redevelopment),
            "root": str(Path(redevelopment_root).resolve()),
        },
        "confirmatory": {
            "classification": "untouched_confirmatory_after_successor_refreeze",
            "seeds": confirmatory,
            "seed_count": len(confirmatory),
            "root": str(Path(confirmatory_root).resolve()),
        },
        "prospective": {
            "classification": "untouched_prospective_after_confirmatory_pass",
            "seeds": prospective,
            "preserved_original_seeds": old_prospective,
            "seed_count": len(prospective),
            "root": str(Path(prospective_root).resolve()),
        },
        "future_roots_absent_at_freeze": [str(path.resolve()) for path in future_paths],
        "claim_boundary": (
            "The failed v81 seeds may now be used only for disclosed redevelopment. "
            "No efficacy claim may use them as confirmatory evidence again."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-protocol", type=Path, required=True)
    parser.add_argument("--validation-audit", type=Path, required=True)
    parser.add_argument("--old-prospective-root", type=Path, required=True)
    parser.add_argument("--redevelopment-root", type=Path, required=True)
    parser.add_argument("--confirmatory-root", type=Path, required=True)
    parser.add_argument("--prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite post-validation partition: {args.out}")
    payload = build_partition(
        validation_protocol_path=args.validation_protocol,
        validation_audit_path=args.validation_audit,
        old_prospective_root=args.old_prospective_root,
        redevelopment_root=args.redevelopment_root,
        confirmatory_root=args.confirmatory_root,
        prospective_root=args.prospective_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "redevelopment_count": payload["redevelopment"]["seed_count"],
                "confirmatory_seeds": payload["confirmatory"]["seeds"],
                "prospective_seeds": payload["prospective"]["seeds"],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
