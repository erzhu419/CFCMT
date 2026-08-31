#!/usr/bin/env python3
"""Freeze an independent-seed CFCMT versus target-only confirmation protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (  # noqa: E402
    METHOD_POLICY,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (  # noqa: E402
    PROTOCOL as V43_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_heldout_evaluation import (  # noqa: E402
    RESULT_PROTOCOL as V43_OFFLINE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_ablation import (  # noqa: E402
    POLICY as TARGET_ONLY_POLICY,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.audit_tsc_external_source_contribution_ablation import (  # noqa: E402
    AUDIT_PROTOCOL as V90_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_fresh_confirmation import (  # noqa: E402
    PROTOCOL as V89_PROTOCOL,
)


PROTOCOL = "tsc-v91r87-external-source-contribution-fresh-confirmation-v1"
SEED_GENERATOR_SEED = 20260901
SEED_COUNT = 64
BOOTSTRAP_REPLICATES = 20_000
NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _seed_values(value: Any, *, key: str = "") -> set[int]:
    seeds: set[int] = set()
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            seeds.update(_seed_values(child, key=str(child_key)))
    else:
        normalized_key = key.lower()
        key_tokens = normalized_key.replace("-", "_").split("_")
        is_non_seed_field = normalized_key.startswith("non_seed")
        is_seed_collection = (
            not is_non_seed_field
            and any(token in {"seed", "seeds"} for token in key_tokens)
        )
        is_seed_scalar = (
            not is_non_seed_field
            and (
                normalized_key == "seed"
                or (
                    normalized_key.endswith("_seed")
                    and "_per_" not in normalized_key
                )
            )
        )
        if isinstance(value, (list, tuple)) and is_seed_collection:
            for item in value:
                if isinstance(item, int) and not isinstance(item, bool):
                    seeds.add(int(item))
        elif (
            isinstance(value, int)
            and not isinstance(value, bool)
            and is_seed_scalar
        ):
            seeds.add(int(value))
    return seeds


def generate_seeds(*, excluded: Iterable[int]) -> tuple[int, ...]:
    excluded_set = {int(seed) for seed in excluded}
    rng = random.Random(SEED_GENERATOR_SEED)
    selected: list[int] = []
    while len(selected) < SEED_COUNT:
        candidate = rng.randrange(1_000_000, 10_000_000)
        if candidate not in excluded_set and candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def freeze_protocol(
    *,
    v43_protocol_path: Path,
    v43_offline_result_path: Path,
    v43_joint_audit_path: Path,
    v89_protocol_path: Path,
    v90_audit_path: Path,
    remote_results_root: Path,
    local_results_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite V91 protocol: {output_path}")
    v43 = _read_json(v43_protocol_path)
    offline = _read_json(v43_offline_result_path)
    joint = _read_json(v43_joint_audit_path)
    v89 = _read_json(v89_protocol_path)
    v90 = _read_json(v90_audit_path)
    if not (
        v43.get("protocol") == V43_PROTOCOL
        and offline.get("protocol") == V43_OFFLINE_PROTOCOL
        and offline.get("decision") == "authorize_closed_loop_external_confirmation"
        and offline.get("offline_confirmation_gate", {}).get("passed") is True
        and joint.get("status") == "PASS"
        and v89.get("protocol") == V89_PROTOCOL
        and v90.get("protocol") == V90_AUDIT_PROTOCOL
        and v90.get("status") == "PASS"
        and v90.get("target_only_model_frozen_before_original_confirmation") is True
        and v90.get("original_confirmation_unchanged") is True
    ):
        raise ValueError("V91 authorization chain changed")

    city_scenarios = {
        str(city): [str(scenario) for scenario in scenarios]
        for city, scenarios in v43["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    if len(city_scenarios) != 2 or sum(map(len, city_scenarios.values())) != 4:
        raise ValueError("V91 city-scenario matrix changed")
    excluded = _seed_values(v43) | _seed_values(v89)
    excluded.update({8081, 9091})
    seeds = generate_seeds(excluded=excluded)
    if not (
        len(seeds) == len(set(seeds)) == SEED_COUNT
        and not set(seeds).intersection(excluded)
    ):
        raise ValueError("V91 fresh seed partition failed")

    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "pre_outcome_fresh_source_contribution_confirmation",
        "parents": {
            "v43_protocol": _record(v43_protocol_path),
            "v43_offline_authorization": _record(v43_offline_result_path),
            "v43_joint_freeze_audit": _record(v43_joint_audit_path),
            "v89_execution_stress_protocol": _record(v89_protocol_path),
            "v90_post_freeze_ablation": _record(v90_audit_path),
        },
        "confirmation": {
            "classification": "fresh_independent_simulator_seed_confirmation",
            "city_scenarios": city_scenarios,
            "policies": [METHOD_POLICY, TARGET_ONLY_POLICY],
            "method_policy": METHOD_POLICY,
            "target_only_policy": TARGET_ONLY_POLICY,
            "seeds": list(seeds),
            "seed_count": len(seeds),
            "scenario_count": sum(map(len, city_scenarios.values())),
            "matrix_size": len(seeds) * sum(map(len, city_scenarios.values())) * 2,
            "pairing_unit": "same_city_scenario_fresh_simulator_seed",
            "primary_metric": "mean_tripinfo_waiting_time",
            "primary_aggregation": (
                "within-seed equal-scenario-within-city then equal-city relative delta"
            ),
            "negative_relative_delta_favours_cfcmt": True,
            "seed_generation": {
                "algorithm": "python_random_mt19937_unique_randrange_1000000_10000000",
                "generator_seed": SEED_GENERATOR_SEED,
                "generated_before_any_v91_rollout": True,
                "excluded_seed_count": len(excluded),
            },
            "excluded_seeds": sorted(excluded),
            "preregistered_gate": {
                "maximum_mean_relative_delta": 0.0,
                "maximum_paired_bootstrap_95pct_upper_relative_delta": 0.0,
                "maximum_each_city_mean_relative_delta": 0.0,
                "maximum_worst_seed_relative_delta": 0.05,
                "minimum_improved_seed_fraction": 0.5,
                "maximum_one_sided_exact_sign_test_p": 0.05,
                "method_teleports_must_not_exceed_target_only": True,
                "method_collision_incidents_must_not_exceed_target_only": True,
                "all_frozen_identities_must_complete": True,
                "no_failed_seed_exclusion": True,
                "joint_rule": "all_gates_must_pass",
            },
            "bootstrap": {
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": SEED_GENERATOR_SEED,
                "unit": "paired_fresh_simulator_seed",
            },
        },
        "output_roots": {
            "remote_results_root": str(remote_results_root),
            "local_results_root": str(local_results_root.resolve()),
        },
        "scientific_status": {
            "outcomes_available_at_freeze": False,
            "cfcmt_model_refit": False,
            "target_only_model_refit": False,
            "target_labels_added": False,
            "same_two_external_city_geometries": True,
            "fresh_seed_generalization_only": True,
            "population_level_cross_city_claim_authorized": False,
        },
        "claim_boundary": (
            "V91 independently tests whether the frozen v43 source-plus-target CFCMT "
            "controller improves over the already frozen target-only family on new "
            "simulator seeds in the same Los Angeles and Jinan benchmark geometries. "
            "It can establish seed-level source contribution in these targets, not "
            "population-level transfer to unseen cities or field effectiveness."
        ),
    }
    atomic_write_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v43-protocol", type=Path, required=True)
    parser.add_argument("--v43-offline-result", type=Path, required=True)
    parser.add_argument("--v43-joint-audit", type=Path, required=True)
    parser.add_argument("--v89-protocol", type=Path, required=True)
    parser.add_argument("--v90-audit", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--local-results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = freeze_protocol(
        v43_protocol_path=args.v43_protocol,
        v43_offline_result_path=args.v43_offline_result,
        v43_joint_audit_path=args.v43_joint_audit,
        v89_protocol_path=args.v89_protocol,
        v90_audit_path=args.v90_audit,
        remote_results_root=args.remote_results_root,
        local_results_root=args.local_results_root,
        output_path=args.out,
    )
    confirmation = payload["confirmation"]
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "seed_count": confirmation["seed_count"],
                "matrix_size": confirmation["matrix_size"],
                "outcomes_available_at_freeze": payload["scientific_status"][
                    "outcomes_available_at_freeze"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
