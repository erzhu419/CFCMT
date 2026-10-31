"""Evaluate frozen method and baselines on the untouched external seed 7079."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CORRECTION_FAMILY,
    CONSTANT_ALPHAS,
    DISAGREEMENT_CAPS,
    CONFIDENCE_MULTIPLIERS,
    _combine_anchored_ensemble_scores,
    _local_group_alpha,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    CANDIDATE_KEYS,
    constant_candidate_key,
    local_candidate_key,
)
from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
    MODEL_ARTIFACT_PROTOCOL as BASELINE_MODEL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    EVALUATION_SEEDS,
    FROZEN_SELECTOR_KEY,
    MODEL_ARTIFACT_PROTOCOL as METHOD_MODEL_PROTOCOL,
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_seed_v3
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


EVALUATION_SEED = 7079
COLLECTION_SHARDS = 16
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260809


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _load_model_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_protocol: str,
    city: str,
    expected_families: Sequence[str],
) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError(f"frozen model SHA-256 mismatch: {path}")
    payload = pickle.loads(Path(path).read_bytes())
    models = tuple(payload.get("model_ensemble", ()))
    if (
        payload.get("protocol") != expected_protocol
        or payload.get("city") != city
        or len(models) != 5
        or any(
            set(model.family_models) != set(expected_families) for model in models
        )
    ):
        raise ValueError(f"frozen model artifact contract failed: {path}")
    return payload


def _ensemble_family_scores(contrast, models, family: str):
    member_scores = []
    member_uncertainties = []
    member_trust = []
    for model_bundle in models:
        score, uncertainty, trust, _ = _group_adjusted_scores(
            contrast,
            model_bundle.family_models[family].predict(contrast),
            objective_mode=model_bundle.objective_modes[family],
        )
        member_scores.append(score)
        member_uncertainties.append(uncertainty)
        member_trust.append(trust)
    return _combine_anchored_ensemble_scores(
        member_scores, member_uncertainties, member_trust
    )


def evaluate_frozen_ensembles(dataset, *, method_payload, baseline_payload):
    method_models = tuple(method_payload["model_ensemble"])
    baseline_models = tuple(baseline_payload["model_ensemble"])
    prior_keys = {
        str(getattr(model.prior_spec, "key", ""))
        for model in (*method_models, *baseline_models)
    }
    if len(prior_keys) != 1:
        raise ValueError("method and baseline ensembles use different source priors")
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=method_models[0].prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual = np.asarray(contrast.targets["interval_cost"], dtype=float)
    groups = action_group_ids(contrast)
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    boundaries = np.concatenate(
        [
            np.asarray([0], dtype=int),
            np.flatnonzero(sorted_groups[1:] != sorted_groups[:-1]) + 1,
            np.asarray([contrast.size], dtype=int),
        ]
    )
    grouped_rows = [
        order[start:end]
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    group_ids = [str(groups[rows[0]]) for rows in grouped_rows]

    method_scores = {
        family: _ensemble_family_scores(contrast, method_models, family)
        for family in BASE_FAMILIES
    }
    anchor_score, anchor_uncertainty, _ = method_scores[ANCHOR_FAMILY]
    correction_score, correction_uncertainty, _ = method_scores[CORRECTION_FAMILY]
    score_delta = correction_score - anchor_score
    policy_scores: dict[str, np.ndarray] = {}
    policy_alphas: dict[str, list[float]] = {}
    for alpha in CONSTANT_ALPHAS:
        key = constant_candidate_key(alpha)
        policy_scores[f"method/{key}"] = anchor_score + float(alpha) * score_delta
        policy_alphas[f"method/{key}"] = [float(alpha)] * len(grouped_rows)
    for cap in DISAGREEMENT_CAPS:
        for confidence in CONFIDENCE_MULTIPLIERS:
            key = local_candidate_key(cap, confidence)
            row_alpha = np.zeros(contrast.size, dtype=float)
            group_alphas = []
            for rows in grouped_rows:
                alpha, _ = _local_group_alpha(
                    anchor_score=anchor_score,
                    correction_score=correction_score,
                    anchor_uncertainty=anchor_uncertainty,
                    correction_uncertainty=correction_uncertainty,
                    group_rows=rows,
                    disagreement_cap=cap,
                    confidence_multiplier=confidence,
                )
                row_alpha[rows] = alpha
                group_alphas.append(float(alpha))
            policy_scores[f"method/{key}"] = anchor_score + row_alpha * score_delta
            policy_alphas[f"method/{key}"] = group_alphas
    if {
        key.removeprefix("method/") for key in policy_scores if key.startswith("method/")
    } != set(CANDIDATE_KEYS):
        raise AssertionError("held-out method candidate grid changed")

    for family in BENCHMARK_FAMILIES:
        score, _, _ = _ensemble_family_scores(contrast, baseline_models, family)
        policy_scores[f"baseline/{family}"] = score
        policy_alphas[f"baseline/{family}"] = [float("nan")] * len(grouped_rows)

    group_rows = []
    for group_index, (group_id, rows) in enumerate(
        zip(group_ids, grouped_rows, strict=True)
    ):
        values = actual[rows]
        best = float(np.min(values))
        scale = action_group_range(values)
        policies = {}
        for policy, score in policy_scores.items():
            selected = int(rows[int(np.argmin(score[rows]))])
            policies[policy] = {
                "normalized_action_regret": (
                    float(actual[selected]) - best
                )
                / scale,
                "optimal_action": bool(
                    np.isclose(float(actual[selected]), best, rtol=0.0, atol=1e-12)
                ),
                "alpha": policy_alphas[policy][group_index],
            }
        group_rows.append({"group_id": group_id, "policies": policies})

    summaries = {}
    for policy in policy_scores:
        regrets = np.asarray(
            [row["policies"][policy]["normalized_action_regret"] for row in group_rows],
            dtype=float,
        )
        alphas = np.asarray(
            [row["policies"][policy]["alpha"] for row in group_rows], dtype=float
        )
        summaries[policy] = {
            "group_count": len(group_rows),
            "mean_normalized_action_regret": float(np.mean(regrets)),
            "p90_normalized_action_regret": float(np.quantile(regrets, 0.90)),
            "worst_normalized_action_regret": float(np.max(regrets)),
            "optimal_action_rate": float(
                np.mean(
                    [row["policies"][policy]["optimal_action"] for row in group_rows]
                )
            ),
            "mean_alpha": (
                float(np.mean(alphas)) if np.all(np.isfinite(alphas)) else None
            ),
        }
    return {
        "protocol": "frozen-score-ensemble-heldout-action-regret-v1",
        "prior_policy": next(iter(prior_keys)),
        "group_count": len(group_rows),
        "summaries": summaries,
        "group_rows": group_rows,
    }


def _equal_scenario_city_summary(
    scenario_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    scenarios = tuple(scenario_results)
    policies = tuple(next(iter(scenario_results.values()))["summaries"])
    if any(set(result["summaries"]) != set(policies) for result in scenario_results.values()):
        raise ValueError("external scenario policy sets differ within a city")
    return {
        policy: {
            "mean_normalized_action_regret": float(
                np.mean(
                    [
                        scenario_results[scenario]["summaries"][policy][
                            "mean_normalized_action_regret"
                        ]
                        for scenario in scenarios
                    ]
                )
            ),
            "optimal_action_rate": float(
                np.mean(
                    [
                        scenario_results[scenario]["summaries"][policy][
                            "optimal_action_rate"
                        ]
                        for scenario in scenarios
                    ]
                )
            ),
        }
        for policy in policies
    }


def _paired_bootstrap(
    scenario_results: Mapping[str, Mapping[str, Any]],
    *,
    policy: str,
    reference: str,
    seed: int,
) -> dict[str, Any]:
    arrays = []
    for result in scenario_results.values():
        arrays.append(
            np.asarray(
                [
                    row["policies"][reference]["normalized_action_regret"]
                    - row["policies"][policy]["normalized_action_regret"]
                    for row in result["group_rows"]
                ],
                dtype=float,
            )
        )
    rng = np.random.default_rng(int(seed))
    replicates = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        scenario_means = [
            float(np.mean(values[rng.integers(0, len(values), size=len(values))]))
            for values in arrays
        ]
        replicates[index] = float(np.mean(scenario_means))
    observed = float(np.mean([np.mean(values) for values in arrays]))
    return {
        "protocol": "paired-within-scenario-group-bootstrap-v1",
        "policy": policy,
        "reference": reference,
        "positive_means_policy_improves": True,
        "observed_improvement": observed,
        "ci95": [
            float(np.quantile(replicates, 0.025)),
            float(np.quantile(replicates, 0.975)),
        ],
        "two_sided_bootstrap_p": float(
            min(
                1.0,
                2.0
                * min(
                    float(np.mean(replicates <= 0.0)),
                    float(np.mean(replicates >= 0.0)),
                ),
            )
        ),
        "replicates": BOOTSTRAP_REPLICATES,
        "scenario_count": len(arrays),
        "group_counts": [len(values) for values in arrays],
    }


def run_external_heldout_evaluation(
    *,
    evaluation_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_spec_path: Path,
    joint_freeze_audit_path: Path,
    evaluation_launch_manifest_path: Path,
    evaluation_task_root: Path,
    method_root: Path,
    baseline_root: Path,
    expected_evaluation_cache_sha256: str,
    expected_joint_freeze_sha256: str,
    workers: int,
) -> dict[str, Any]:
    protocol = _load_json(protocol_spec_path)
    joint = _load_json(joint_freeze_audit_path)
    launch = _load_json(evaluation_launch_manifest_path)
    if (
        _sha256(joint_freeze_audit_path) != str(expected_joint_freeze_sha256)
        or joint.get("status") != "PASS"
        or joint.get("decision") != "authorize_external_seed7079_collection"
        or launch.get("authorization_audit_sha256")
        != str(expected_joint_freeze_sha256)
        or not bool(launch.get("submitted", False))
    ):
        raise ValueError("held-out evaluation is not bound to the joint freeze")
    cache_sha, cache_files = aggregate_cache_sha256(evaluation_cache_root)
    expected_files = (
        sum(
            len(names)
            for names in protocol["target_protocol"][
                "external_city_scenarios"
            ].values()
        )
        * len(EVALUATION_SEEDS)
        * COLLECTION_SHARDS
    )
    if (
        cache_sha != str(expected_evaluation_cache_sha256)
        or cache_files != expected_files
    ):
        raise ValueError("held-out evaluation cache identity changed")
    task_summaries = sorted(Path(evaluation_task_root).glob("*/cache_summary.json"))
    if len(task_summaries) != 4:
        raise ValueError("held-out evaluation task summaries are incomplete")
    for path in task_summaries:
        summary = _load_json(path)
        authorization = summary.get("setting", {}).get("authorization_provenance")
        if (
            not authorization
            or authorization.get("sha256") != str(expected_joint_freeze_sha256)
            or authorization.get("decision")
            != "authorize_external_seed7079_collection"
        ):
            raise ValueError(f"held-out cache task lacks freeze authorization: {path}")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    manifest = load_traffic_signal_manifest(external_manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        evaluation_cache_root,
        manifest,
        seeds=(EVALUATION_SEED,),
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    city_scenarios = {
        str(city): tuple(str(value) for value in names)
        for city, names in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    city_results = {}
    for city, scenarios in city_scenarios.items():
        method_freeze = _load_json(Path(method_root) / city / "freeze.json")
        baseline_freeze = _load_json(Path(baseline_root) / city / "freeze.json")
        joint_city = joint["cities"][city]
        if (
            _sha256(Path(method_root) / city / "freeze.json")
            != joint_city["method_result"]["sha256"]
            or _sha256(Path(baseline_root) / city / "freeze.json")
            != joint_city["baseline_result"]["sha256"]
        ):
            raise ValueError(f"held-out freeze result SHA mismatch for {city}")
        method_payload = _load_model_artifact(
            Path(method_root) / city / "model_ensemble.pkl",
            expected_sha256=joint_city["method_model"]["sha256"],
            expected_protocol=METHOD_MODEL_PROTOCOL,
            city=city,
            expected_families=BASE_FAMILIES,
        )
        baseline_payload = _load_model_artifact(
            Path(baseline_root) / city / "baseline_model_ensemble.pkl",
            expected_sha256=joint_city["baseline_model"]["sha256"],
            expected_protocol=BASELINE_MODEL_PROTOCOL,
            city=city,
            expected_families=BENCHMARK_FAMILIES,
        )
        if (
            method_payload["selected_group_ids"]
            != baseline_payload["selected_group_ids"]
            or method_payload["fold_assignment"]
            != baseline_payload["fold_assignment"]
            or method_freeze["selected_candidate"]
            != method_payload["selected_candidate"]
            or method_freeze["frozen_selector_key"] != FROZEN_SELECTOR_KEY
        ):
            raise ValueError(f"held-out method/baseline freeze mismatch for {city}")
        selected = str(method_freeze["selected_candidate"])
        scenario_results = {}
        for scenario in scenarios:
            dataset = bank[scenario]
            observed_seeds = {
                _group_seed_v3(str(group))
                for group in dataset.metadata["action_group_ids"]
            }
            if observed_seeds != {EVALUATION_SEED}:
                raise ValueError(f"held-out seed leakage for {scenario}")
            scenario_results[scenario] = evaluate_frozen_ensembles(
                dataset,
                method_payload=method_payload,
                baseline_payload=baseline_payload,
            )
        city_summary = _equal_scenario_city_summary(scenario_results)
        selected_policy = f"method/{selected}"
        anchor_policy = "method/constant_alpha_0"
        inference = {
            "selected_vs_anchor": _paired_bootstrap(
                scenario_results,
                policy=selected_policy,
                reference=anchor_policy,
                seed=BOOTSTRAP_SEED + len(city),
            ),
            "selected_vs_baselines": {
                family: _paired_bootstrap(
                    scenario_results,
                    policy=selected_policy,
                    reference=f"baseline/{family}",
                    seed=BOOTSTRAP_SEED
                    + len(city)
                    + 100 * (index + 1),
                )
                for index, family in enumerate(BENCHMARK_FAMILIES)
            },
        }
        city_results[city] = {
            "scenarios": scenario_results,
            "equal_scenario_summary": city_summary,
            "selected_candidate": selected,
            "selected_policy": selected_policy,
            "anchor_policy": anchor_policy,
            "inference": inference,
        }

    anchor_values = np.asarray(
        [
            row["equal_scenario_summary"][row["anchor_policy"]][
                "mean_normalized_action_regret"
            ]
            for row in city_results.values()
        ],
        dtype=float,
    )
    selected_values = np.asarray(
        [
            row["equal_scenario_summary"][row["selected_policy"]][
                "mean_normalized_action_regret"
            ]
            for row in city_results.values()
        ],
        dtype=float,
    )
    improvements = anchor_values - selected_values
    gate_spec = dict(protocol["offline_confirmation_gate"])
    anchor_macro = float(np.mean(anchor_values))
    selected_macro = float(np.mean(selected_values))
    relative = (
        (anchor_macro - selected_macro) / anchor_macro
        if anchor_macro > 1e-12
        else 0.0
    )
    non_anchor_count = sum(
        row["selected_candidate"] != "constant_alpha_0"
        for row in city_results.values()
    )
    gate = {
        "macro_relative_improvement": relative,
        "improved_external_cities": int(np.sum(improvements > 1e-12)),
        "maximum_absolute_city_regression": float(
            np.max(np.maximum(-improvements, 0.0))
        ),
        "non_anchor_targets": non_anchor_count,
    }
    gate["macro_relative_improvement_passed"] = (
        relative + 1e-12
        >= float(gate_spec["minimum_macro_relative_improvement"])
    )
    gate["improved_external_cities_passed"] = (
        gate["improved_external_cities"]
        >= int(gate_spec["minimum_improved_external_cities"])
    )
    gate["maximum_city_regression_passed"] = (
        gate["maximum_absolute_city_regression"]
        <= float(gate_spec["maximum_absolute_city_regression"]) + 1e-12
    )
    gate["non_anchor_targets_passed"] = (
        non_anchor_count >= int(gate_spec["minimum_non_anchor_targets"])
    )
    gate["passed"] = all(
        gate[key]
        for key in (
            "macro_relative_improvement_passed",
            "improved_external_cities_passed",
            "maximum_city_regression_passed",
            "non_anchor_targets_passed",
        )
    )
    return {
        "protocol": "tsc-v42r38-external-seed7079-heldout-evaluation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "joint_freeze_audit_sha256": _sha256(joint_freeze_audit_path),
        "evaluation_launch_manifest_sha256": _sha256(
            evaluation_launch_manifest_path
        ),
        "evaluation_cache_aggregate_sha256": cache_sha,
        "evaluation_cache_file_count": cache_files,
        "evaluation_cache_audit": cache_audit,
        "evaluation_seed": EVALUATION_SEED,
        "city_results": city_results,
        "offline_confirmation_gate": gate,
        "decision": (
            "authorize_closed_loop_external_confirmation"
            if gate["passed"]
            else "retain_offline_failure_and_prohibit_closed_loop_claim"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--evaluation-launch-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-task-root", type=Path, required=True)
    parser.add_argument("--method-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--expected-evaluation-cache-sha256", required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_external_heldout_evaluation(
        evaluation_cache_root=args.evaluation_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_spec_path=args.protocol_spec,
        joint_freeze_audit_path=args.joint_freeze_audit,
        evaluation_launch_manifest_path=args.evaluation_launch_manifest,
        evaluation_task_root=args.evaluation_task_root,
        method_root=args.method_root,
        baseline_root=args.baseline_root,
        expected_evaluation_cache_sha256=args.expected_evaluation_cache_sha256,
        expected_joint_freeze_sha256=args.expected_joint_freeze_sha256,
        workers=args.workers,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "gate_passed": payload["offline_confirmation_gate"]["passed"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
