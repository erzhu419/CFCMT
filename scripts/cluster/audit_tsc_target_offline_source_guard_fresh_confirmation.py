#!/usr/bin/env python3
"""Audit the three-arm v111 fresh-seed Jinan closed-loop confirmation."""

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
from scripts.cluster.audit_tsc_target_offline_source_fresh_confirmation import (  # noqa: E402
    _target_anchor_contract,
)
from scripts.cluster.launch_tsc_source_city_component_screen import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_source_city_component_shard import (  # noqa: E402
    CANDIDATE_PROTOCOL,
    SHARD_PROTOCOL,
    load_candidate_specs,
)


PROTOCOL = "tsc-v111-target-offline-source-guard-fresh-confirmation-audit-v1"
SCENARIOS = (
    "jinan_3x4_real",
    "jinan_3x4_real_2000",
    "jinan_3x4_real_2500",
)
METHODS = ("pressure_rule", "target_only_guard", "source_guard")
BOOTSTRAP_SEED = 11_100_032


def _load_rows(
    shards_root: Path,
    *,
    expected_candidate_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    paths = sorted(Path(shards_root).glob("shard_*.json"))
    if len(paths) != 6:
        raise ValueError("v111 confirmation must contain six shard summaries")
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
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
            raise ValueError(f"invalid v111 confirmation shard: {path}")
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
        raise ValueError("v111 scenario/seed/method matrix is incomplete")
    aggregate: dict[int, dict[str, dict[str, float | int]]] = {}
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


def _comparison(
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[int, Mapping[str, Mapping[str, float | int]]],
    *,
    seeds: Sequence[int],
    candidate: str,
    reference: str,
    bootstrap_seed: int,
) -> dict[str, Any]:
    relative = [
        (
            float(aggregate[int(seed)][candidate]["waiting"])
            - float(aggregate[int(seed)][reference]["waiting"])
        )
        / max(float(aggregate[int(seed)][reference]["waiting"]), 1e-9)
        for seed in seeds
    ]
    by_identity = {
        (str(row["scenario"]), int(row["seed"]), str(row["candidate_key"])): row
        for row in rows
    }
    by_scenario = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        values = []
        for seed in seeds:
            candidate_waiting = float(
                by_identity[(scenario, int(seed), candidate)]["metrics"][
                    "mean_tripinfo_waiting_time"
                ]
            )
            reference_waiting = float(
                by_identity[(scenario, int(seed), reference)]["metrics"][
                    "mean_tripinfo_waiting_time"
                ]
            )
            values.append(
                (candidate_waiting - reference_waiting)
                / max(reference_waiting, 1e-9)
            )
        by_scenario[scenario] = _paired_summary(
            values,
            bootstrap_seed=bootstrap_seed + scenario_index + 1,
        )
    safety = {}
    for metric in ("collision_incidents", "teleports"):
        candidate_total = int(
            sum(int(aggregate[int(seed)][candidate][metric]) for seed in seeds)
        )
        reference_total = int(
            sum(int(aggregate[int(seed)][reference][metric]) for seed in seeds)
        )
        safety[metric] = {
            "candidate": candidate_total,
            "reference": reference_total,
            "delta": candidate_total - reference_total,
        }
    return {
        "candidate": candidate,
        "reference": reference,
        "paired_seed": _paired_summary(relative, bootstrap_seed=bootstrap_seed),
        "by_scenario": by_scenario,
        "safety": safety,
    }


def _comparison_gates(comparison: Mapping[str, Any]) -> dict[str, bool]:
    paired = comparison["paired_seed"]
    safety = comparison["safety"]
    return {
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


def _validate_offline_contract(
    spec: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    selection_sha256: str,
) -> None:
    contract = dict(spec.get("offline_source_guard_contract", {}))
    source = dict(selection.get("selection", {}))
    guard = dict(selection.get("guard_selection", {}))
    expected_guard = {
        "enabled": True,
        "risk_multiplier": 0.0,
        "min_context_trust": 0.1,
        "margin": 0.0,
        "max_relative_rule_gap": 1.0,
    }
    if not (
        selection.get("protocol")
        == "tsc-v110-target-offline-source-and-guard-selection-v2"
        and selection.get("city") == "jinan"
        and selection.get("target_data_contract", {}).get(
            "target_closed_loop_rollouts_used_for_selection"
        )
        is False
        and source.get("selected_source_city_group") == "hangzhou"
        and float(source.get("selected_source_weight", -1.0)) == 1.0
        and guard.get("selected_profile_key") == "risk_0__gap_1"
        and guard.get("guard_config") == expected_guard
        and contract.get("jinan_result_sha256") == selection_sha256
        and contract.get("selected_source_city_group") == "hangzhou"
        and float(contract.get("selected_source_weight", -1.0)) == 1.0
        and contract.get("selected_guard_profile_key") == "risk_0__gap_1"
        and contract.get("selected_guard_config") == expected_guard
        and contract.get("target_closed_loop_rollouts_used_for_selection") is False
    ):
        raise ValueError("v111 target-offline source/guard contract changed")


def _validate_candidate_contract(spec: Mapping[str, Any]) -> None:
    candidates = {
        str(row["key"]): dict(row)
        for row in load_candidate_specs(Path(str(spec["_candidate_path"])))
    }
    expected_guard = {
        "enabled": True,
        "risk_multiplier": 0.0,
        "min_context_trust": 0.1,
        "margin": 0.0,
        "max_relative_rule_gap": 1.0,
    }
    if (
        tuple(candidates) != METHODS
        or candidates["pressure_rule"].get("baseline_policy") != "phase_pressure"
        or candidates["target_only_guard"]["weights_by_city"]["jinan"] != {}
        or candidates["source_guard"]["weights_by_city"]["jinan"]
        != {"hangzhou": 1.0}
        or candidates["target_only_guard"]["guard_by_city"]["jinan"]
        != expected_guard
        or candidates["source_guard"]["guard_by_city"]["jinan"]
        != expected_guard
    ):
        raise ValueError("v111 three-arm candidate contract changed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--candidate-spec-sha256", required=True)
    parser.add_argument("--offline-selection-result", type=Path, required=True)
    parser.add_argument("--offline-selection-result-sha256", required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite v111 confirmation audit")
    if _sha256(args.candidate_spec) != str(args.candidate_spec_sha256):
        raise ValueError("v111 candidate identity changed")
    if _sha256(args.offline_selection_result) != str(
        args.offline_selection_result_sha256
    ):
        raise ValueError("v111 offline selection identity changed")

    spec = json.loads(args.candidate_spec.read_text(encoding="utf-8"))
    spec["_candidate_path"] = str(args.candidate_spec)
    selection = json.loads(args.offline_selection_result.read_text(encoding="utf-8"))
    launch = json.loads(args.launch.read_text(encoding="utf-8"))
    if not (
        spec.get("protocol") == CANDIDATE_PROTOCOL
        and spec.get("scientific_status")
        == "frozen-before-v111-fresh-seed-closed-loop-confirmation"
        and launch.get("protocol") == LAUNCH_PROTOCOL
        and launch.get("complete") is True
        and launch.get("candidate_spec_sha256") == str(args.candidate_spec_sha256)
        and launch.get("scenarios") == list(SCENARIOS)
    ):
        raise ValueError("v111 freeze/launch contract changed")
    _validate_candidate_contract(spec)
    _validate_offline_contract(
        spec,
        selection,
        selection_sha256=str(args.offline_selection_result_sha256),
    )
    target_anchor, _ = _target_anchor_contract(spec, launch)

    seeds = tuple(int(value) for value in spec["confirmation_seeds"])
    if tuple(int(value) for value in launch["seeds"]) != seeds:
        raise ValueError("v111 launch seed list changed")
    rows, shard_hashes = _load_rows(
        args.shards_root,
        expected_candidate_sha256=args.candidate_spec_sha256,
    )
    aggregate = _aggregate(rows, seeds=seeds)
    combined_vs_rule = _comparison(
        rows,
        aggregate,
        seeds=seeds,
        candidate="source_guard",
        reference="pressure_rule",
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    source_contribution = _comparison(
        rows,
        aggregate,
        seeds=seeds,
        candidate="source_guard",
        reference="target_only_guard",
        bootstrap_seed=BOOTSTRAP_SEED + 10,
    )
    target_guard_vs_rule = _comparison(
        rows,
        aggregate,
        seeds=seeds,
        candidate="target_only_guard",
        reference="pressure_rule",
        bootstrap_seed=BOOTSTRAP_SEED + 20,
    )
    primary_gates = _comparison_gates(combined_vs_rule)
    source_contribution_gates = _comparison_gates(source_contribution)
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "fresh-seed-closed-loop-confirmation",
        "candidate_spec_sha256": str(args.candidate_spec_sha256),
        "offline_selection_result_sha256": str(
            args.offline_selection_result_sha256
        ),
        "launch_sha256": _sha256(args.launch),
        "shard_sha256": shard_hashes,
        "scenario_count": len(SCENARIOS),
        "seed_count": len(seeds),
        "method_count": len(METHODS),
        "matrix_size": len(rows),
        "comparisons": {
            "source_guard_vs_pressure_rule": combined_vs_rule,
            "source_guard_vs_target_only_guard": source_contribution,
            "target_only_guard_vs_pressure_rule": target_guard_vs_rule,
        },
        "primary_gates": primary_gates,
        "source_contribution_gates": source_contribution_gates,
        "confirmation_passed": all(primary_gates.values())
        and all(source_contribution_gates.values()),
        "target_only_anchor": target_anchor,
        "claim_boundary": (
            "B100 target-offline selection fixed the Hangzhou source weight and "
            "deployment guard without target closed-loop outcomes; this audit "
            "uses disjoint fresh seeds to compare the combined policy with exact "
            "phase pressure and to isolate the added source contribution."
        ),
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["confirmation_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
