#!/usr/bin/env python3
"""Audit safe complete counterfactual action-group capacity by target and seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _dataset_tls_ids_v3,
    _group_seed_v3,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


SOURCE_SEEDS = (2027, 3037, 4047)
COLLECTION_SHARDS = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_capacity(
    *,
    cache_root: Path,
    manifest_path: Path,
    expected_cache_sha256: str,
    workers: int,
    required_groups: int,
) -> dict[str, Any]:
    cache_sha256, cache_file_count = aggregate_cache_sha256(cache_root)
    if cache_sha256 != expected_cache_sha256:
        raise ValueError("counterfactual cache hash mismatch")
    manifest = load_traffic_signal_manifest(manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    target_rows = []
    for target in manifest.sumocfgs:
        dataset = bank[target]
        groups = np.asarray(dataset.metadata.get("action_group_ids", ()), dtype=str)
        if groups.shape != (dataset.size,):
            raise ValueError(f"{target}: row-aligned action groups are missing")
        unique_groups = sorted(set(groups.tolist()))
        by_seed = {
            str(seed): sum(_group_seed_v3(group) == seed for group in unique_groups)
            for seed in SOURCE_SEEDS
        }
        target_rows.append(
            {
                "target": target,
                "city_group": manifest.city_groups[target],
                "safe_complete_group_count": len(unique_groups),
                "group_count_by_seed": by_seed,
                "controllable_tls_count": len(_dataset_tls_ids_v3(dataset)),
                "required_group_count": int(required_groups),
                "capacity_passed": len(unique_groups) >= int(required_groups),
            }
        )
    return {
        "audit": "tsc-safe-complete-target-action-group-capacity-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(cache_root.resolve()),
        "cache_aggregate_sha256": cache_sha256,
        "cache_file_count": cache_file_count,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "required_group_count": int(required_groups),
        "all_targets_passed": all(row["capacity_passed"] for row in target_rows),
        "target_rows": target_rows,
        "loader_cache_audit": cache_audit,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--required-groups", type=int, default=60)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit_capacity(
        cache_root=args.cache_root,
        manifest_path=args.manifest,
        expected_cache_sha256=args.expected_cache_sha256,
        workers=args.workers,
        required_groups=args.required_groups,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "all_targets_passed": payload["all_targets_passed"],
                "minimum_group_count": min(
                    row["safe_complete_group_count"]
                    for row in payload["target_rows"]
                ),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
