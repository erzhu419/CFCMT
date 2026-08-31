#!/usr/bin/env python3
"""Audit the post-freeze target-only ablation against frozen v43 CFCMT records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json
from cf_h2o.eval.traffic_signal_external_closed_loop_aggregate import (
    AGGREGATE_PROTOCOL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    METHOD_POLICY,
    PRIMARY_METRIC,
    _bootstrap_primary_improvement,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_ablation import POLICY
from scripts.cluster.launch_tsc_external_source_contribution_ablation import (
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_source_contribution_ablation_shard import (
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v90r86-source-contribution-ablation-audit-v1"
CITY_SCENARIOS = {
    "los_angeles": ("la_1x4",),
    "jinan": (
        "jinan_3x4_real",
        "jinan_3x4_real_2000",
        "jinan_3x4_real_2500",
    ),
}
SEEDS = (8081, 9091)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def _load_target_rows(shards_root: Path) -> list[dict[str, Any]]:
    paths = sorted(shards_root.glob("shard_*.json"))
    if len(paths) != 6:
        raise ValueError("expected six source-contribution shard summaries")
    rows: list[dict[str, Any]] = []
    observed_indices = set()
    for path in paths:
        shard = _read_json(path)
        observed_indices.add(int(shard.get("shard_index", -1)))
        if not (
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and int(shard.get("shard_count", -1)) == 6
            and int(shard.get("task_count", -1)) == len(shard.get("rows", ()))
        ):
            raise ValueError(f"invalid source-contribution shard: {path}")
        rows.extend(dict(row) for row in shard["rows"])
    if observed_indices != set(range(6)):
        raise ValueError("source-contribution shard indices are incomplete")

    expected = {
        (city, scenario, seed, POLICY)
        for city, scenarios in CITY_SCENARIOS.items()
        for scenario in scenarios
        for seed in SEEDS
    }
    observed = {
        (str(row["city"]), str(row["scenario"]), int(row["seed"]), str(row["policy"]))
        for row in rows
    }
    if len(rows) != 8 or observed != expected:
        raise ValueError("source-contribution rollout identities are incomplete")
    for row in rows:
        metrics = dict(row["metrics"])
        _finite(metrics[PRIMARY_METRIC], label="target-only waiting")
        if int(metrics["starting_teleports"]) != 0 or int(
            metrics["ending_teleports"]
        ) != 0:
            raise ValueError("target-only ablation contains teleports")
        if not (
            0
            <= int(metrics["tripinfo_completed_count"])
            <= int(metrics["tripinfo_count"])
            <= int(metrics["demand_population_at_horizon"])
        ):
            raise ValueError("target-only trip population accounting failed")
    return rows


def _parent_method_rows(parent: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not (
        parent.get("protocol") == AGGREGATE_PROTOCOL
        and parent.get("validity_gate", {}).get("passed") is True
        and int(parent.get("matrix", {}).get("rollout_count", -1)) == 56
    ):
        raise ValueError("parent v43 aggregate is not valid")
    rows = [
        dict(row)
        for row in parent["records"]
        if row.get("policy") == METHOD_POLICY
    ]
    expected = {
        (city, scenario, seed, METHOD_POLICY)
        for city, scenarios in CITY_SCENARIOS.items()
        for scenario in scenarios
        for seed in SEEDS
    }
    observed = {
        (str(row["city"]), str(row["scenario"]), int(row["seed"]), str(row["policy"]))
        for row in rows
    }
    if len(rows) != 8 or observed != expected:
        raise ValueError("parent CFCMT comparison cells are incomplete")
    return rows


def _cell_records(
    method_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records = []
    for row in (*method_rows, *target_rows):
        metrics = dict(row["metrics"])
        records.append(
            {
                "city": str(row["city"]),
                "scenario": str(row["scenario"]),
                "seed": int(row["seed"]),
                "policy": str(row["policy"]),
                "metrics": {
                    PRIMARY_METRIC: _finite(
                        metrics[PRIMARY_METRIC], label="paired waiting"
                    ),
                    "collision_incidents": int(metrics["collision_incidents"]),
                },
            }
        )
    return records


def _mean(
    records: Sequence[Mapping[str, Any]], *, city: str, scenario: str, policy: str
) -> float:
    values = [
        float(row["metrics"][PRIMARY_METRIC])
        for row in records
        if row["city"] == city
        and row["scenario"] == scenario
        and row["policy"] == policy
    ]
    if len(values) != len(SEEDS):
        raise ValueError("paired scenario summary lost a seed")
    return float(np.mean(values))


def build_audit(
    *, launch: Mapping[str, Any], parent: Mapping[str, Any], target_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if not (
        launch.get("protocol") == LAUNCH_PROTOCOL
        and launch.get("complete") is True
        and int(launch.get("matrix_size", -1)) == 8
        and launch.get("full_rollouts_remain_remote") is True
    ):
        raise ValueError("source-contribution launch is incomplete")
    method_rows = _parent_method_rows(parent)
    records = _cell_records(method_rows, target_rows)
    policies = (METHOD_POLICY, POLICY)
    city_summary: dict[str, Any] = {}
    for city, scenarios in CITY_SCENARIOS.items():
        city_summary[city] = {}
        for policy in policies:
            scenario_means = [
                _mean(records, city=city, scenario=scenario, policy=policy)
                for scenario in scenarios
            ]
            city_summary[city][policy] = float(np.mean(scenario_means))
    macro = {
        policy: float(
            np.mean([city_summary[city][policy] for city in CITY_SCENARIOS])
        )
        for policy in policies
    }
    improvement = float(macro[POLICY] - macro[METHOD_POLICY])
    paired = {
        (row["city"], row["scenario"], int(row["seed"])): float(
            row["metrics"][PRIMARY_METRIC]
        )
        for row in records
        if row["policy"] == METHOD_POLICY
    }
    target = {
        (row["city"], row["scenario"], int(row["seed"])): float(
            row["metrics"][PRIMARY_METRIC]
        )
        for row in records
        if row["policy"] == POLICY
    }
    cell_improvements = {
        f"{city}/{scenario}/seed_{seed}": target[(city, scenario, seed)]
        - paired[(city, scenario, seed)]
        for city, scenarios in CITY_SCENARIOS.items()
        for scenario in scenarios
        for seed in SEEDS
    }
    bootstrap = _bootstrap_primary_improvement(
        records,
        CITY_SCENARIOS,
        baseline=POLICY,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED + 90,
    )
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "analysis_classification": "post_freeze_frozen_comparator_ablation",
        "original_confirmation_unchanged": True,
        "target_only_model_frozen_before_original_confirmation": True,
        "comparator_selected_for_closed_loop_after_main_outcomes": True,
        "matrix": {
            "cities": list(CITY_SCENARIOS),
            "scenarios": [
                scenario for scenarios in CITY_SCENARIOS.values() for scenario in scenarios
            ],
            "seeds": list(SEEDS),
            "target_only_rollouts": len(target_rows),
            "paired_cells": len(cell_improvements),
        },
        "primary_metric": PRIMARY_METRIC,
        "city_summary": city_summary,
        "macro_summary": macro,
        "cfcmt_oriented_improvement_seconds": improvement,
        "cfcmt_relative_improvement": improvement / max(abs(macro[POLICY]), 1e-12),
        "city_oriented_improvements_seconds": {
            city: city_summary[city][POLICY] - city_summary[city][METHOD_POLICY]
            for city in CITY_SCENARIOS
        },
        "cell_oriented_improvements_seconds": cell_improvements,
        "improved_cell_count": int(sum(value > 0 for value in cell_improvements.values())),
        "descriptive_bootstrap": bootstrap,
        "safety": {
            METHOD_POLICY: {
                "teleports": 0,
                "collision_incidents": int(
                    sum(row["metrics"]["collision_incidents"] for row in method_rows)
                ),
            },
            POLICY: {
                "teleports": 0,
                "collision_incidents": int(
                    sum(row["metrics"]["collision_incidents"] for row in target_rows)
                ),
            },
        },
        "claim_boundary": (
            "This post-freeze ablation reuses a baseline model frozen before the original "
            "v43 confirmation and the same eight scenario-seed cells. The decision to add "
            "the closed-loop comparator was made after the main outcomes, so intervals are "
            "descriptive and the result is not a new preregistered confirmation."
        ),
    }


def _write_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    city = audit["city_summary"]
    macro = audit["macro_summary"]
    boot = audit["descriptive_bootstrap"]
    lines = [
        "# V90 source-contribution ablation",
        "",
        f"- status: `{audit['status']}`",
        f"- CFCMT macro waiting: {macro[METHOD_POLICY]:.6f} s",
        f"- target-only macro waiting: {macro[POLICY]:.6f} s",
        f"- CFCMT-oriented improvement: {audit['cfcmt_oriented_improvement_seconds']:.6f} s "
        f"({100.0 * audit['cfcmt_relative_improvement']:.3f}%)",
        f"- descriptive 95% interval: [{boot['improvement_95pct_ci'][0]:.6f}, "
        f"{boot['improvement_95pct_ci'][1]:.6f}] s",
        f"- improved cells: {audit['improved_cell_count']}/8",
        f"- Los Angeles improvement: {audit['city_oriented_improvements_seconds']['los_angeles']:.6f} s",
        f"- Jinan improvement: {audit['city_oriented_improvements_seconds']['jinan']:.6f} s",
        "",
        "## Boundary",
        "",
        str(audit["claim_boundary"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--parent-aggregate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = build_audit(
        launch=_read_json(args.launch),
        parent=_read_json(args.parent_aggregate),
        target_rows=_load_target_rows(args.shards_root),
    )
    _atomic_json(args.out, audit)
    _write_markdown(args.markdown_out, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
