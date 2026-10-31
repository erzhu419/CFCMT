#!/usr/bin/env python3
"""Audit the frozen V91 CFCMT versus target-only fresh-seed confirmation."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (  # noqa: E402
    METHOD_POLICY,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_ablation import (  # noqa: E402
    POLICY as TARGET_ONLY_POLICY,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_confirmation import (  # noqa: E402
    all_rollout_identities,
)
from scripts.cluster.freeze_tsc_external_source_contribution_confirmation import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_source_contribution_confirmation import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_external_source_contribution_confirmation_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v91r87-source-contribution-fresh-confirmation-audit-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _finite(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite {label}: {value!r}")
    return number


def _exact_one_sided_sign_p(*, improved: int, total: int) -> float:
    if not 0 <= improved <= total:
        raise ValueError("invalid sign-test counts")
    return float(
        sum(math.comb(total, count) for count in range(improved, total + 1))
        / (2**total)
    )


def _load_rows(
    *, shards_root: Path, protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    paths = sorted(shards_root.glob("shard_*.json"))
    if len(paths) != 6:
        raise ValueError("V91 requires six shard summaries")
    rows: list[dict[str, Any]] = []
    indices = set()
    protocol_sha = _sha256(
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v91_external_source_contribution_confirmation.json"
    )
    for path in paths:
        shard = _read_json(path)
        indices.add(int(shard.get("shard_index", -1)))
        if not (
            shard.get("protocol") == SHARD_PROTOCOL
            and shard.get("passed") is True
            and int(shard.get("shard_count", -1)) == 6
            and shard.get("confirmation_protocol_sha256") == protocol_sha
            and int(shard.get("task_count", -1)) == len(shard.get("rows", ()))
        ):
            raise ValueError(f"invalid V91 shard: {path}")
        rows.extend(dict(row) for row in shard["rows"])
    expected = set(all_rollout_identities(dict(protocol)))
    observed = {
        (str(row["city"]), str(row["scenario"]), int(row["seed"]), str(row["policy"]))
        for row in rows
    }
    if indices != set(range(6)) or len(rows) != len(expected) or observed != expected:
        raise ValueError("V91 shard identity coverage failed")
    for row in rows:
        metrics = dict(row["metrics"])
        _finite(metrics["mean_tripinfo_waiting_time"], label="V91 waiting")
        if int(metrics["starting_teleports"]) != 0 or int(
            metrics["ending_teleports"]
        ) != 0:
            raise ValueError("V91 contains teleports")
        if not (
            0
            <= int(metrics["tripinfo_completed_count"])
            <= int(metrics["tripinfo_count"])
            <= int(metrics["demand_population_at_horizon"])
        ):
            raise ValueError("V91 population accounting failed")
        if row["policy"] == METHOD_POLICY and int(
            row["method_prediction_calls"]
        ) <= 0:
            raise ValueError("V91 CFCMT model was not active")
    return rows


def build_audit(
    *,
    protocol: Mapping[str, Any],
    launch: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not (
        protocol.get("protocol") == PROTOCOL
        and protocol.get("scientific_status", {}).get("outcomes_available_at_freeze")
        is False
        and launch.get("protocol") == LAUNCH_PROTOCOL
        and launch.get("complete") is True
        and int(launch.get("matrix_size", -1)) == 512
        and launch.get("full_rollouts_remain_remote") is True
        and launch.get("confirmation_protocol_sha256")
        == _sha256(
            PROJECT_ROOT
            / "cf_h2o/config/traffic_signal_tsc_v91_external_source_contribution_confirmation.json"
        )
    ):
        raise ValueError("V91 freeze or launch integrity failed")

    confirmation = protocol["confirmation"]
    city_scenarios = {
        str(city): tuple(str(scenario) for scenario in scenarios)
        for city, scenarios in confirmation["city_scenarios"].items()
    }
    seeds = tuple(int(seed) for seed in confirmation["seeds"])
    lookup = {
        (str(row["city"]), str(row["scenario"]), int(row["seed"]), str(row["policy"])): float(
            row["metrics"]["mean_tripinfo_waiting_time"]
        )
        for row in rows
    }
    seed_rows = []
    for seed in seeds:
        policy_city_values: dict[str, dict[str, float]] = {}
        for policy in (METHOD_POLICY, TARGET_ONLY_POLICY):
            policy_city_values[policy] = {
                city: float(
                    np.mean(
                        [
                            lookup[(city, scenario, seed, policy)]
                            for scenario in scenarios
                        ]
                    )
                )
                for city, scenarios in city_scenarios.items()
            }
        method_macro = float(
            np.mean(list(policy_city_values[METHOD_POLICY].values()))
        )
        target_macro = float(
            np.mean(list(policy_city_values[TARGET_ONLY_POLICY].values()))
        )
        seed_rows.append(
            {
                "seed": seed,
                "method_macro_waiting": method_macro,
                "target_only_macro_waiting": target_macro,
                "relative_delta": (method_macro - target_macro)
                / max(abs(target_macro), 1e-12),
                "city_relative_deltas": {
                    city: (
                        policy_city_values[METHOD_POLICY][city]
                        - policy_city_values[TARGET_ONLY_POLICY][city]
                    )
                    / max(abs(policy_city_values[TARGET_ONLY_POLICY][city]), 1e-12)
                    for city in city_scenarios
                },
            }
        )

    relative = np.asarray([row["relative_delta"] for row in seed_rows], dtype=float)
    method_waiting = np.asarray(
        [row["method_macro_waiting"] for row in seed_rows], dtype=float
    )
    target_waiting = np.asarray(
        [row["target_only_macro_waiting"] for row in seed_rows], dtype=float
    )
    bootstrap_spec = confirmation["bootstrap"]
    rng = np.random.default_rng(int(bootstrap_spec["seed"]))
    indices = rng.integers(
        0,
        len(seeds),
        size=(int(bootstrap_spec["replicates"]), len(seeds)),
    )
    draws = np.mean(relative[indices], axis=1)
    ci = np.quantile(draws, [0.025, 0.975])
    improved = int(np.sum(relative < 0.0))
    sign_p = _exact_one_sided_sign_p(improved=improved, total=len(seeds))
    city_means = {
        city: float(
            np.mean([row["city_relative_deltas"][city] for row in seed_rows])
        )
        for city in city_scenarios
    }
    collisions = {
        policy: int(
            sum(
                int(row["metrics"]["collision_incidents"])
                for row in rows
                if row["policy"] == policy
            )
        )
        for policy in (METHOD_POLICY, TARGET_ONLY_POLICY)
    }
    rule = confirmation["preregistered_gate"]
    gates = {
        "mean_relative_delta": float(np.mean(relative))
        <= float(rule["maximum_mean_relative_delta"]),
        "bootstrap_upper": float(ci[1])
        <= float(rule["maximum_paired_bootstrap_95pct_upper_relative_delta"]),
        "each_city_mean": max(city_means.values())
        <= float(rule["maximum_each_city_mean_relative_delta"]),
        "worst_seed": float(np.max(relative))
        <= float(rule["maximum_worst_seed_relative_delta"]),
        "improved_seed_fraction": improved / len(seeds)
        >= float(rule["minimum_improved_seed_fraction"]),
        "one_sided_sign_test": sign_p
        <= float(rule["maximum_one_sided_exact_sign_test_p"]),
        "aggregate_teleports": True,
        "aggregate_collision_safety": collisions[METHOD_POLICY]
        <= collisions[TARGET_ONLY_POLICY],
        "complete_identity_matrix": len(rows) == int(confirmation["matrix_size"]),
        "no_failed_seed_exclusion": len(seed_rows) == len(seeds),
        "method_active": all(
            int(row["method_prediction_calls"]) > 0
            for row in rows
            if row["policy"] == METHOD_POLICY
        ),
    }
    gate_passed = all(gates.values())
    summary = {
        "seed_count": len(seeds),
        "method_mean_macro_waiting": float(np.mean(method_waiting)),
        "target_only_mean_macro_waiting": float(np.mean(target_waiting)),
        "mean_relative_delta": float(np.mean(relative)),
        "bootstrap": {
            "replicates": int(bootstrap_spec["replicates"]),
            "seed": int(bootstrap_spec["seed"]),
            "ci95": [float(ci[0]), float(ci[1])],
            "probability_below_zero": float(np.mean(draws < 0.0)),
        },
        "improved_seed_count": improved,
        "improved_seed_fraction": improved / len(seeds),
        "one_sided_exact_sign_test_p": sign_p,
        "worst_seed_relative_delta": float(np.max(relative)),
        "best_seed_relative_delta": float(np.min(relative)),
        "city_mean_relative_deltas": city_means,
        "collision_incidents": collisions,
    }
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "integrity_gate": {"passed": True},
        "confirmation_summary": summary,
        "confirmation_gate": {"passed": gate_passed, "gates": gates},
        "decision": (
            "accept_fresh_source_contribution_confirmation"
            if gate_passed
            else "reject_fresh_source_contribution_confirmation"
        ),
        "seed_rows": seed_rows,
        "scientific_status": {
            "fresh_seeds": True,
            "models_refit": False,
            "same_two_external_city_geometries": True,
            "population_level_city_transfer": False,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def _write_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    summary = audit["confirmation_summary"]
    ci = summary["bootstrap"]["ci95"]
    lines = [
        "# V91 fresh source-contribution confirmation",
        "",
        f"- evidence integrity: `{audit['status']}`",
        f"- joint scientific gate: `{audit['confirmation_gate']['passed']}`",
        f"- decision: `{audit['decision']}`",
        f"- fresh seeds: {summary['seed_count']}",
        f"- CFCMT macro waiting: {summary['method_mean_macro_waiting']:.6f} s",
        f"- target-only macro waiting: {summary['target_only_mean_macro_waiting']:.6f} s",
        f"- mean relative delta: {100.0 * summary['mean_relative_delta']:.4f}%",
        f"- paired-seed 95% interval: [{100.0 * ci[0]:.4f}%, {100.0 * ci[1]:.4f}%]",
        f"- improved seeds: {summary['improved_seed_count']}/{summary['seed_count']}",
        f"- one-sided sign-test p: {summary['one_sided_exact_sign_test_p']:.6f}",
        f"- worst seed: {100.0 * summary['worst_seed_relative_delta']:.4f}%",
        f"- city mean deltas: {json.dumps(summary['city_mean_relative_deltas'], sort_keys=True)}",
        f"- collision incidents: {json.dumps(summary['collision_incidents'], sort_keys=True)}",
        "",
        "## Gate details",
        "",
    ]
    lines.extend(
        f"- {name}: `{passed}`"
        for name, passed in audit["confirmation_gate"]["gates"].items()
    )
    lines.extend(["", "## Boundary", "", str(audit["claim_boundary"])])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = _read_json(args.protocol)
    audit = build_audit(
        protocol=protocol,
        launch=_read_json(args.launch),
        rows=_load_rows(shards_root=args.shards_root, protocol=protocol),
    )
    _atomic_json(args.out, audit)
    _write_markdown(args.markdown_out, audit)
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
