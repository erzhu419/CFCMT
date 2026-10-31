#!/usr/bin/env python3
"""Audit the V93 guard profile screen after source and weight freezing."""

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


from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    select_source_city_candidate_from_closed_loop,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from scripts.cluster.audit_tsc_source_city_component_screen import (
    SCENARIOS_BY_CITY,
    _load_rows,
)
from scripts.cluster.launch_tsc_source_city_component_screen import (
    GUARD_PROFILE_PROTOCOL,
    LAUNCH_PROTOCOL,
)
from scripts.cluster.run_tsc_source_city_component_shard import (
    SCENARIOS,
    load_candidate_specs,
)


AUDIT_PROTOCOL = "tsc-v93-source-city-guard-profile-screen-audit-v1"


def _normalize_guard_config(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "enabled": True,
        "risk_multiplier": float(value["risk_multiplier"]),
        "min_context_trust": float(value["min_context_trust"]),
        "margin": float(value["margin"]),
        "max_relative_rule_gap": float(value["max_relative_rule_gap"]),
    }


def select_guard_profile_from_closed_loop(
    rows_by_profile: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    city: str,
    scenarios: Sequence[str],
    profile_keys: Sequence[str],
    baseline_key: str = "raw",
    candidate_key: str = "selected_source",
    minimum_mean_improvement_without_safety_gain: float = 0.005,
    maximum_worst_seed_waiting_regression: float = 0.01,
) -> dict[str, Any]:
    """Choose a guard from paired rollouts without trading away safety."""

    profiles = tuple(str(value) for value in profile_keys)
    expected_scenarios = tuple(str(value) for value in scenarios)
    if (
        not profiles
        or len(profiles) != len(set(profiles))
        or baseline_key not in profiles
        or not expected_scenarios
    ):
        raise ValueError("invalid guard profile grid")
    normalized: dict[tuple[str, str, int], dict[str, Any]] = {}
    seed_sets = []
    for profile in profiles:
        rows = []
        for source_row in rows_by_profile[profile]:
            if str(source_row["city"]) != str(city):
                continue
            if str(source_row["candidate_key"]) != str(candidate_key):
                continue
            scenario = str(source_row["scenario"])
            if scenario not in expected_scenarios:
                raise ValueError("guard row is outside the city scenario grid")
            metrics = source_row.get("metrics", source_row)
            waiting = float(metrics["mean_tripinfo_waiting_time"])
            if not np.isfinite(waiting) or waiting < 0.0:
                raise ValueError("guard waiting must be finite and nonnegative")
            row = {
                "scenario": scenario,
                "seed": int(source_row["seed"]),
                "waiting": waiting,
                "collision_incidents": int(metrics.get("collision_incidents", 0)),
                "teleports": int(metrics.get("starting_teleports", 0))
                + int(metrics.get("ending_teleports", 0))
                + int(metrics.get("teleports", 0)),
            }
            identity = (profile, scenario, row["seed"])
            if identity in normalized:
                raise ValueError("guard profile rollout identity is duplicated")
            normalized[identity] = row
            rows.append(row)
        seed_sets.append({row["seed"] for row in rows})
    if not seed_sets or any(values != seed_sets[0] for values in seed_sets[1:]):
        raise ValueError("guard profiles use different seed sets")
    seeds = tuple(sorted(seed_sets[0]))
    expected = {
        (profile, scenario, seed)
        for profile in profiles
        for scenario in expected_scenarios
        for seed in seeds
    }
    if not seeds or set(normalized) != expected:
        raise ValueError("guard closed-loop matrix is incomplete")

    aggregate: dict[tuple[str, int], dict[str, Any]] = {}
    for profile in profiles:
        for seed in seeds:
            scenario_rows = [
                normalized[(profile, scenario, seed)]
                for scenario in expected_scenarios
            ]
            aggregate[(profile, seed)] = {
                "waiting": float(np.mean([row["waiting"] for row in scenario_rows])),
                "collision_incidents": int(
                    sum(row["collision_incidents"] for row in scenario_rows)
                ),
                "teleports": int(sum(row["teleports"] for row in scenario_rows)),
            }

    baseline_collision = sum(
        aggregate[(baseline_key, seed)]["collision_incidents"] for seed in seeds
    )
    baseline_teleports = sum(
        aggregate[(baseline_key, seed)]["teleports"] for seed in seeds
    )
    grid: dict[str, Any] = {}
    eligible = []
    for profile in profiles:
        relative = np.asarray(
            [
                (
                    aggregate[(profile, seed)]["waiting"]
                    - aggregate[(baseline_key, seed)]["waiting"]
                )
                / max(aggregate[(baseline_key, seed)]["waiting"], 1e-9)
                for seed in seeds
            ],
            dtype=float,
        )
        collision_delta = int(
            sum(
                aggregate[(profile, seed)]["collision_incidents"] for seed in seeds
            )
            - baseline_collision
        )
        teleport_delta = int(
            sum(aggregate[(profile, seed)]["teleports"] for seed in seeds)
            - baseline_teleports
        )
        mean_delta = float(np.mean(relative))
        worst_delta = float(np.max(relative))
        robust_score = float(
            mean_delta + 0.5 * np.std(relative) + max(worst_delta, 0.0)
        )
        safety_gain = bool(collision_delta < 0 or teleport_delta < 0)
        is_eligible = bool(
            profile != baseline_key
            and collision_delta <= 0
            and teleport_delta <= 0
            and worst_delta <= float(maximum_worst_seed_waiting_regression)
            and (
                safety_gain
                or mean_delta
                <= -float(minimum_mean_improvement_without_safety_gain)
            )
        )
        summary = {
            "profile_key": profile,
            "seed_relative_deltas": relative.tolist(),
            "mean_relative_delta": mean_delta,
            "worst_seed_relative_delta": worst_delta,
            "robust_score": robust_score,
            "collision_delta": collision_delta,
            "teleport_delta": teleport_delta,
            "safety_gain": safety_gain,
            "eligible": is_eligible,
        }
        grid[profile] = summary
        if is_eligible:
            eligible.append(summary)
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                int(row["collision_delta"] + row["teleport_delta"]),
                float(row["robust_score"]),
                str(row["profile_key"]),
            ),
        )
        reason = (
            "safety_improving_guard"
            if selected["safety_gain"]
            else "performance_improving_guard"
        )
    else:
        selected = grid[baseline_key]
        reason = "raw_guard_fallback"
    return {
        "protocol": "source-city-guard-closed-loop-selector-v1",
        "city": str(city),
        "scenarios": list(expected_scenarios),
        "seeds": list(seeds),
        "baseline_key": str(baseline_key),
        "candidate_key": str(candidate_key),
        "selected_profile_key": str(selected["profile_key"]),
        "selection_reason": reason,
        "minimum_mean_improvement_without_safety_gain": float(
            minimum_mean_improvement_without_safety_gain
        ),
        "maximum_worst_seed_waiting_regression": float(
            maximum_worst_seed_waiting_regression
        ),
        "grid": grid,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--guard-profile-spec", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite guard audit: {args.out}")

    candidate_sha = _sha256(args.candidate_spec)
    candidates = load_candidate_specs(args.candidate_spec)
    candidate_keys = tuple(str(row["key"]) for row in candidates)
    if candidate_keys != ("target_only", "selected_source"):
        raise ValueError("guard screen candidate pair changed")
    profile_sha = _sha256(args.guard_profile_spec)
    profile_payload = json.loads(args.guard_profile_spec.read_text(encoding="utf-8"))
    if profile_payload.get("protocol") != GUARD_PROFILE_PROTOCOL:
        raise ValueError("guard profile specification protocol changed")
    profiles = tuple(dict(row) for row in profile_payload.get("profiles", ()))
    profile_keys = tuple(str(row["key"]) for row in profiles)
    if not profiles or len(profile_keys) != len(set(profile_keys)) or "raw" not in profile_keys:
        raise ValueError("guard profile keys are invalid")
    profile_by_key = {str(row["key"]): row for row in profiles}

    rows_by_profile: dict[str, list[dict[str, Any]]] = {}
    launch_sha256 = {}
    for profile in profile_keys:
        root = args.results_root / profile
        launch_path = root / "launch.json"
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
        expected_guard = _normalize_guard_config(
            profile_by_key[profile].get("guard_config")
        )
        if not (
            launch.get("protocol") == LAUNCH_PROTOCOL
            and launch.get("complete") is True
            and launch.get("candidate_spec_sha256") == candidate_sha
            and launch.get("guard_config") == expected_guard
            and launch.get("guard_profile", {}).get("key") == profile
            and launch.get("guard_profile", {}).get("spec_sha256") == profile_sha
            and tuple(int(value) for value in launch.get("seeds", ()))
            == tuple(int(value) for value in args.seeds)
        ):
            raise ValueError(f"guard launch contract failed: {launch_path}")
        shard_count = len(tuple(launch.get("shard_indices", ())))
        rows = _load_rows(
            shards_root=root / "_shards",
            expected_shard_count=shard_count,
            candidate_spec_sha256=candidate_sha,
        )
        if len(rows) != len(SCENARIOS) * len(args.seeds) * len(candidate_keys):
            raise ValueError(f"guard profile matrix size changed: {profile}")
        if any(row.get("guard_config") != expected_guard for row in rows):
            raise ValueError(f"guard row configuration changed: {profile}")
        rows_by_profile[profile] = rows
        launch_sha256[profile] = _sha256(launch_path)

    source_effect = {
        profile: {
            city: select_source_city_candidate_from_closed_loop(
                rows,
                city=city,
                scenarios=scenarios,
                candidate_keys=candidate_keys,
            )
            for city, scenarios in SCENARIOS_BY_CITY.items()
        }
        for profile, rows in rows_by_profile.items()
    }
    guard_selection = {
        city: select_guard_profile_from_closed_loop(
            rows_by_profile,
            city=city,
            scenarios=scenarios,
            profile_keys=profile_keys,
        )
        for city, scenarios in SCENARIOS_BY_CITY.items()
    }
    selected_guard_by_city = {
        city: {
            "profile_key": selection["selected_profile_key"],
            "guard_config": profile_by_key[selection["selected_profile_key"]].get(
                "guard_config"
            ),
        }
        for city, selection in guard_selection.items()
    }
    payload = {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "development-only-not-confirmation",
        "candidate_spec_sha256": candidate_sha,
        "guard_profile_spec_sha256": profile_sha,
        "seed_count": len(args.seeds),
        "scenario_count": len(SCENARIOS),
        "profile_count": len(profile_keys),
        "matrix_size": sum(len(rows) for rows in rows_by_profile.values()),
        "launch_sha256_by_profile": launch_sha256,
        "source_effect_by_profile": source_effect,
        "guard_selection_by_city": guard_selection,
        "selected_guard_by_city": selected_guard_by_city,
        "claim_boundary": (
            "Source identity, source mass, and guard are development choices. "
            "Only a disjoint confirmation matrix can support the final effect claim."
        ),
    }
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "matrix_size": payload["matrix_size"],
                "selected_guard_by_city": selected_guard_by_city,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
