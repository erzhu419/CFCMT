#!/usr/bin/env python3
"""Freeze the v78 leakage-isolated multihorizon nested selection."""

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
    _sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_counterfactual_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_latent_screen import (  # noqa: E402
    AUTHORIZATION_DECISION as SCREEN_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as SCREEN_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.freeze_tsc_external_v9_multihorizon_latent_screen import (  # noqa: E402
    PROTOCOL as SCREEN_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_waiting_cache import (  # noqa: E402
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (  # noqa: E402
    MANIFEST_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (  # noqa: E402
    PARTITION_PROTOCOL,
)


PROTOCOL = "tsc-v83r79-external-v9-multihorizon-nested-selection-v1"
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_protocol(
    *,
    screen_protocol_path: Path,
    screen_result_path: Path,
    screen_oof_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    partition_path: Path,
    manifest_path: Path,
    remote_shard_root: Path,
    local_shard_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite multihorizon nested protocol: {output_path}"
        )
    screen_protocol = _read_json(screen_protocol_path)
    screen_result = _read_json(screen_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest = _read_json(manifest_path)
    frozen = dict(screen_protocol.get("frozen_inputs", {}))
    screen_oof_sha256 = _file_sha256(screen_oof_path)
    selected = screen_result.get("advance_gate", {}).get("selected_candidate")
    if (
        screen_protocol.get("protocol") != SCREEN_PROTOCOL
        or screen_result.get("protocol") != SCREEN_RESULT_PROTOCOL
        or screen_result.get("status") != "PASS"
        or screen_result.get("decision") != SCREEN_AUTHORIZATION_DECISION
        or screen_result.get("integrity_gate", {}).get("passed") is not True
        or screen_result.get("advance_gate", {}).get("passed") is not True
        or not isinstance(selected, dict)
        or screen_result.get("diagnostic_protocol_sha256")
        != _sha256(screen_protocol_path)
        or screen_result.get("oof_artifact", {}).get("sha256")
        != screen_oof_sha256
        or cache_protocol.get("protocol") != CACHE_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or partition.get("protocol") != PARTITION_PROTOCOL
        or manifest.get("protocol") != MANIFEST_PROTOCOL
        or frozen.get("cache_protocol_sha256") != _sha256(cache_protocol_path)
        or frozen.get("cache_audit_sha256") != _sha256(cache_audit_path)
        or frozen.get("partition_sha256") != _sha256(partition_path)
        or frozen.get("manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("multihorizon nested parent evidence changed")
    training = dict(screen_protocol["training"])
    seeds = tuple(int(value) for value in training["seeds"])
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if (
        len(seeds) != 22
        or len(set(seeds)) != len(seeds)
        or set(seeds) != {
            int(value) for value in partition["redevelopment"]["seeds"]
        }
        or any(path.exists() for path in sealed_roots)
        or int(training["expanded_candidate_count"]) != 25
    ):
        raise ValueError("multihorizon nested development partition changed")
    node_assignments = {
        node: list(seeds[index:: len(NODES)])
        for index, node in enumerate(NODES)
    }
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v77_screen_pre_nested_outcomes",
        "frozen_inputs": {
            "screen_protocol_path": str(screen_protocol_path.resolve()),
            "screen_protocol_sha256": _sha256(screen_protocol_path),
            "screen_result_path": str(screen_result_path.resolve()),
            "screen_result_sha256": _sha256(screen_result_path),
            "screen_oof_path": str(screen_oof_path.resolve()),
            "screen_oof_sha256": screen_oof_sha256,
            "cache_protocol_path": str(cache_protocol_path.resolve()),
            "cache_protocol_sha256": _sha256(cache_protocol_path),
            "cache_audit_path": str(cache_audit_path.resolve()),
            "cache_audit_sha256": _sha256(cache_audit_path),
            "partition_path": str(partition_path.resolve()),
            "partition_sha256": _sha256(partition_path),
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
        },
        "nested_selection": {
            "city": str(training["city"]),
            "scenarios": list(training["scenarios"]),
            "outer_seeds": list(seeds),
            "outer_fold_count": len(seeds),
            "inner_fold_count_per_outer": len(seeds) - 1,
            "training_seed_count_per_inner_fit": len(seeds) - 2,
            "outer_inner_pair_count": len(seeds) * (len(seeds) - 1),
            "collection_shards": int(training["collection_shards"]),
            "reference_policy": str(training["reference_policy"]),
            "expected_action_count_per_group": int(
                training["expected_action_count_per_group"]
            ),
            "rollout_prefix_horizons_sec": list(
                training["rollout_prefix_horizons_sec"]
            ),
            "model_candidates": list(training["model_candidates"]),
            "expanded_candidate_count": int(training["expanded_candidate_count"]),
            "selection": screen_protocol["selection"],
            "selection_threshold_scaling": "none_use_exact_v77_thresholds",
            "outer_prediction_source": "v77_leave_one_seed_out_oof_artifact",
            "final_full_development_candidate": dict(selected),
            "node_assignments": node_assignments,
            "cache_workers": 32,
            "inner_fold_workers": 21,
            "sealed_confirmatory_seeds": list(
                training["sealed_confirmatory_seeds"]
            ),
            "sealed_prospective_seeds": list(training["sealed_prospective_seeds"]),
        },
        "output_roots": {
            "remote_shard_root": str(remote_shard_root),
            "local_shard_root": str(local_shard_root.resolve()),
        },
        "advance_contract": {
            "nested_result_is_development_evidence": True,
            "closed_loop_development_requires_nested_gate": True,
            "confirmatory_and_prospective_roots_remain_sealed": True,
        },
        "claim_boundary": (
            "Each outer seed is excluded from every inner training fit and from "
            "joint model, horizon, and gate selection. The selected procedure is "
            "then evaluated only on that outer seed's v77 OOF predictions. Passing "
            "authorizes closed-loop redevelopment only; v84 and v85 remain sealed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-protocol", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-oof", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--remote-shard-root", type=Path, required=True)
    parser.add_argument("--local-shard-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        screen_protocol_path=args.screen_protocol,
        screen_result_path=args.screen_result,
        screen_oof_path=args.screen_oof,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        partition_path=args.partition,
        manifest_path=args.manifest,
        remote_shard_root=args.remote_shard_root,
        local_shard_root=args.local_shard_root,
        output_path=args.out,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "outer_folds": payload["nested_selection"]["outer_fold_count"],
                "inner_fits": payload["nested_selection"][
                    "outer_inner_pair_count"
                ],
                "sha256": _sha256(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
