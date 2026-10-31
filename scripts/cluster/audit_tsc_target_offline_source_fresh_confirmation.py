#!/usr/bin/env python3
"""Audit fresh-seed closed-loop confirmation of the offline source selector."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from scripts.cluster.audit_tsc_source_city_disjoint_confirmation import (  # noqa: E402
    _paired_summary,
)
from scripts.cluster.launch_tsc_source_city_component_screen import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_source_city_component_shard import (  # noqa: E402
    CANDIDATE_PROTOCOL,
    SHARD_PROTOCOL,
)


PROTOCOL = "tsc-v98-target-offline-source-fresh-confirmation-audit-v2"
SCENARIOS = (
    "jinan_3x4_real",
    "jinan_3x4_real_2000",
    "jinan_3x4_real_2500",
)
METHODS = ("target_only", "offline_selector")
BOOTSTRAP_SEED = 9_800_032


def _load_rows(
    shards_root: Path,
    *,
    expected_candidate_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = sorted(Path(shards_root).glob("shard_*.json"))
    if len(paths) != 6:
        raise ValueError("fresh confirmation must contain six shard summaries")
    rows = []
    hashes = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not (
            payload.get("protocol") == SHARD_PROTOCOL
            and payload.get("candidate_protocol") == CANDIDATE_PROTOCOL
            and payload.get("candidate_spec_sha256")
            == str(expected_candidate_sha256)
            and payload.get("passed") is True
            and int(payload.get("task_count", -1)) == len(payload.get("rows", ()))
        ):
            raise ValueError(f"invalid fresh confirmation shard: {path}")
        rows.extend(dict(row) for row in payload["rows"])
        hashes[path.name] = _sha256(path)
    return rows, hashes


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
) -> dict[int, dict[str, dict[str, float | int]]]:
    by_identity = {
        (str(row["scenario"]), int(row["seed"]), str(row["candidate_key"])): row
        for row in rows
    }
    expected = {
        (scenario, int(seed), method)
        for scenario in SCENARIOS
        for seed in seeds
        for method in METHODS
    }
    if len(by_identity) != len(rows) or set(by_identity) != expected:
        raise ValueError("fresh confirmation scenario/seed/method matrix is incomplete")
    aggregate = {}
    for seed in seeds:
        aggregate[int(seed)] = {}
        for method in METHODS:
            metrics = [
                by_identity[(scenario, int(seed), method)]["metrics"]
                for scenario in SCENARIOS
            ]
            aggregate[int(seed)][method] = {
                "waiting": float(
                    np.mean(
                        [float(row["mean_tripinfo_waiting_time"]) for row in metrics]
                    )
                ),
                "collision_incidents": int(
                    sum(int(row.get("collision_incidents", 0)) for row in metrics)
                ),
                "teleports": int(
                    sum(
                        int(row.get("starting_teleports", 0))
                        + int(row.get("ending_teleports", 0))
                        for row in metrics
                    )
                ),
            }
    return aggregate


def _comparison_summary(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[int, Mapping[str, Mapping[str, float | int]]],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    relative = [
        (
            float(aggregate[int(seed)]["offline_selector"]["waiting"])
            - float(aggregate[int(seed)]["target_only"]["waiting"])
        )
        / max(float(aggregate[int(seed)]["target_only"]["waiting"]), 1e-9)
        for seed in seeds
    ]
    paired = _paired_summary(relative, bootstrap_seed=BOOTSTRAP_SEED)
    by_identity = {
        (str(row["scenario"]), int(row["seed"]), str(row["candidate_key"])): row
        for row in rows
    }
    by_scenario = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        values = []
        for seed in seeds:
            target = float(
                by_identity[(scenario, int(seed), "target_only")]["metrics"][
                    "mean_tripinfo_waiting_time"
                ]
            )
            source = float(
                by_identity[(scenario, int(seed), "offline_selector")]["metrics"][
                    "mean_tripinfo_waiting_time"
                ]
            )
            values.append((source - target) / max(target, 1e-9))
        by_scenario[scenario] = _paired_summary(
            values,
            bootstrap_seed=BOOTSTRAP_SEED + scenario_index + 1,
        )
    safety = {}
    for metric in ("collision_incidents", "teleports"):
        target_total = int(
            sum(int(aggregate[int(seed)]["target_only"][metric]) for seed in seeds)
        )
        source_total = int(
            sum(
                int(aggregate[int(seed)]["offline_selector"][metric])
                for seed in seeds
            )
        )
        safety[metric] = {
            "target_only": target_total,
            "offline_selector": source_total,
            "delta": source_total - target_total,
        }
    return {"paired_seed": paired, "by_scenario": by_scenario, "safety": safety}


def _target_anchor_contract(
    spec: Mapping[str, Any], launch: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    strict = spec.get("strict_target_only_contract")
    if strict is None:
        return (
            {"family": "causal_target_only", "protocol": "legacy-v93-anchor"},
            "This evaluates the B100 offline-selected Hangzhou component against "
            "the frozen legacy v93 target anchor under the same Jinan pressure guard.",
        )
    strict = dict(strict)
    artifact = dict(launch.get("artifact_contract", {})).get(
        "jinan_strict_target_only_model"
    )
    if not isinstance(artifact, Mapping) or (
        strict.get("family") != "causal_target_only_v2"
        or strict.get("model_protocol")
        != "cfcmt-strict-target-only-action-model-v1"
        or int(strict.get("source_rows_consumed", -1)) != 0
        or artifact.get("sha256") != strict.get("jinan_model_sha256")
    ):
        raise ValueError("strict target-only launch/model contract changed")
    return (
        {
            "family": str(strict["family"]),
            "protocol": str(strict["model_protocol"]),
            "model_path": str(artifact["path"]),
            "model_sha256": str(artifact["sha256"]),
            "source_rows_consumed": 0,
        },
        "This evaluates the B100 offline-selected Hangzhou component against "
        "causal_target_only_v2, fitted with zero source rows, under the same "
        "Jinan pressure guard.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--candidate-spec-sha256", required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite fresh confirmation audit")
    if _sha256(args.candidate_spec) != str(args.candidate_spec_sha256):
        raise ValueError("fresh confirmation candidate identity changed")
    spec = json.loads(args.candidate_spec.read_text(encoding="utf-8"))
    launch = json.loads(args.launch.read_text(encoding="utf-8"))
    if not (
        spec.get("protocol") == CANDIDATE_PROTOCOL
        and launch.get("protocol") == LAUNCH_PROTOCOL
        and launch.get("complete") is True
        and launch.get("candidate_spec_sha256") == str(args.candidate_spec_sha256)
        and launch.get("scenarios") == list(SCENARIOS)
    ):
        raise ValueError("fresh confirmation freeze/launch contract changed")
    seeds = tuple(int(value) for value in spec["confirmation_seeds"])
    if tuple(int(value) for value in launch["seeds"]) != seeds:
        raise ValueError("fresh confirmation launch seed list changed")
    target_anchor, claim_boundary = _target_anchor_contract(spec, launch)
    rows, shard_hashes = _load_rows(
        args.shards_root,
        expected_candidate_sha256=args.candidate_spec_sha256,
    )
    aggregate = _aggregate(rows, seeds=seeds)
    statistics = _comparison_summary(rows, aggregate, seeds=seeds)
    paired = statistics["paired_seed"]
    safety = statistics["safety"]
    gates = {
        "mean_waiting_improves": float(paired["mean_relative_delta"]) < 0.0,
        "bootstrap_ci_upper_below_zero": float(paired["bootstrap_mean_ci95"][1])
        < 0.0,
        "wilcoxon_two_sided_p_below_0p05": float(
            paired["wilcoxon_p_two_sided"]
        )
        < 0.05,
        "no_collision_excess": int(safety["collision_incidents"]["delta"]) <= 0,
        "no_teleport_excess": int(safety["teleports"]["delta"]) <= 0,
    }
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "fresh-seed-closed-loop-confirmation",
        "candidate_spec_sha256": str(args.candidate_spec_sha256),
        "launch_sha256": _sha256(args.launch),
        "shard_sha256": shard_hashes,
        "scenario_count": len(SCENARIOS),
        "seed_count": len(seeds),
        "method_count": len(METHODS),
        "matrix_size": len(rows),
        "statistics": statistics,
        "gates": gates,
        "confirmation_passed": all(gates.values()),
        "target_only_anchor": target_anchor,
        "claim_boundary": claim_boundary,
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["confirmation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
