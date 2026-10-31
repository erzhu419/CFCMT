"""Evaluate v43 full-refit models on the untouched external seed 8171."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import BASE_FAMILIES
from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    EXTERNAL_COLLECTION_SHARDS,
    FROZEN_SELECTOR_KEY,
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    BASELINE_FREEZE_PROTOCOL,
    BASELINE_MODEL_PROTOCOL,
    EVALUATION_SEED,
    METHOD_FREEZE_PROTOCOL,
    METHOD_MODEL_PROTOCOL,
    PROTOCOL,
    REFIT_PROTOCOL,
    V9_BASELINE_FREEZE_PROTOCOL,
    V9_BASELINE_MODEL_PROTOCOL,
    V9_EVALUATION_SEED,
    V9_METHOD_FREEZE_PROTOCOL,
    V9_METHOD_MODEL_PROTOCOL,
    V9_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_heldout_evaluation import (
    BOOTSTRAP_SEED,
    _equal_scenario_city_summary,
    _load_json,
    _paired_bootstrap,
    evaluate_frozen_ensembles,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_seed_v3
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


JOINT_AUDIT_PROTOCOL = (
    "tsc-v43r39-external-full-budget-joint-freeze-audit-v1"
)
AUTHORIZATION_DECISION = "authorize_external_seed8171_collection"
RESULT_PROTOCOL = "tsc-v43r39-external-seed8171-heldout-evaluation-v1"
V9_JOINT_AUDIT_PROTOCOL = (
    "tsc-v54r50-external-v9-full-budget-joint-freeze-audit-v1"
)
V9_AUTHORIZATION_DECISION = "authorize_external_v9_seed9277_collection"
V9_RESULT_PROTOCOL = "tsc-v57r53-external-v9-seed9277-heldout-evaluation-v1"
V9_CLOSED_LOOP_DECISION = "authorize_external_v9_closed_loop_development_replay"


def _load_full_refit_model_artifact(
    path: Path,
    *,
    expected_sha256: str,
    expected_protocol: str,
    city: str,
    expected_families: Sequence[str],
    expected_group_ids: Sequence[str],
) -> dict[str, Any]:
    if _sha256(path) != str(expected_sha256):
        raise ValueError(f"full-refit model SHA-256 mismatch: {path}")
    payload = pickle.loads(Path(path).read_bytes())
    models = tuple(payload.get("model_ensemble", ()))
    expected_groups = tuple(str(value) for value in expected_group_ids)
    if (
        payload.get("protocol") != expected_protocol
        or payload.get("city") != city
        or len(models) != 1
        or set(models[0].family_models) != set(expected_families)
        or tuple(payload.get("selected_group_ids", ())) != expected_groups
        or tuple(payload.get("deployment_refit_group_ids", ()))
        != expected_groups
        or payload.get("deployment_refit_protocol") != REFIT_PROTOCOL
    ):
        raise ValueError(f"full-refit model artifact contract failed: {path}")
    return payload


def _confirmation_gate(
    city_results: Mapping[str, Mapping[str, Any]],
    gate_spec: Mapping[str, Any],
) -> dict[str, Any]:
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
        "anchor_macro_mean_normalized_action_regret": anchor_macro,
        "selected_macro_mean_normalized_action_regret": selected_macro,
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
    gate["improved_external_cities_passed"] = gate[
        "improved_external_cities"
    ] >= int(gate_spec["minimum_improved_external_cities"])
    gate["maximum_city_regression_passed"] = gate[
        "maximum_absolute_city_regression"
    ] <= float(gate_spec["maximum_absolute_city_regression"]) + 1e-12
    gate["non_anchor_targets_passed"] = non_anchor_count >= int(
        gate_spec["minimum_non_anchor_targets"]
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
    return gate


def run_external_full_heldout_evaluation(
    *,
    evaluation_cache_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_spec_path: Path,
    joint_freeze_audit_path: Path,
    evaluation_launch_manifest_path: Path,
    evaluation_task_root: Path,
    freeze_root: Path,
    expected_evaluation_cache_sha256: str,
    expected_joint_freeze_sha256: str,
    workers: int,
) -> dict[str, Any]:
    protocol = _load_json(protocol_spec_path)
    joint = _load_json(joint_freeze_audit_path)
    launch = _load_json(evaluation_launch_manifest_path)
    protocol_name = str(protocol.get("protocol", ""))
    if protocol_name == PROTOCOL:
        joint_audit_protocol = JOINT_AUDIT_PROTOCOL
        authorization_decision = AUTHORIZATION_DECISION
        evaluation_seed = EVALUATION_SEED
        method_freeze_protocol = METHOD_FREEZE_PROTOCOL
        baseline_freeze_protocol = BASELINE_FREEZE_PROTOCOL
        method_model_protocol = METHOD_MODEL_PROTOCOL
        baseline_model_protocol = BASELINE_MODEL_PROTOCOL
        result_protocol = RESULT_PROTOCOL
        closed_loop_decision = "authorize_closed_loop_external_confirmation"
        generation = "v8"
    elif protocol_name == V9_PROTOCOL:
        joint_audit_protocol = V9_JOINT_AUDIT_PROTOCOL
        authorization_decision = V9_AUTHORIZATION_DECISION
        evaluation_seed = V9_EVALUATION_SEED
        method_freeze_protocol = V9_METHOD_FREEZE_PROTOCOL
        baseline_freeze_protocol = V9_BASELINE_FREEZE_PROTOCOL
        method_model_protocol = V9_METHOD_MODEL_PROTOCOL
        baseline_model_protocol = V9_BASELINE_MODEL_PROTOCOL
        result_protocol = V9_RESULT_PROTOCOL
        closed_loop_decision = V9_CLOSED_LOOP_DECISION
        generation = "v9"
    else:
        raise ValueError("full-budget held-out protocol changed")
    if (
        _sha256(joint_freeze_audit_path) != str(expected_joint_freeze_sha256)
        or joint.get("protocol") != joint_audit_protocol
        or joint.get("protocol_spec_sha256") != _sha256(protocol_spec_path)
        or joint.get("status") != "PASS"
        or not bool(joint.get("gate", {}).get("passed", False))
        or joint.get("decision") != authorization_decision
        or int(joint.get("evaluation_cache_file_count_at_freeze", -1)) != 0
        or launch.get("authorization_audit_sha256")
        != str(expected_joint_freeze_sha256)
        or launch.get("authorization_decision") != authorization_decision
        or int(launch.get("evaluation_seed", -1)) != evaluation_seed
        or not bool(launch.get("submitted", False))
    ):
        raise ValueError("held-out evaluation is not bound to the joint freeze")

    city_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    expected_files = (
        sum(len(names) for names in city_scenarios.values())
        * EXTERNAL_COLLECTION_SHARDS
    )
    cache_sha, cache_files = aggregate_cache_sha256(evaluation_cache_root)
    if (
        cache_sha != str(expected_evaluation_cache_sha256)
        or cache_files != expected_files
    ):
        raise ValueError("seed-8171 evaluation cache identity changed")

    task_summaries = sorted(Path(evaluation_task_root).glob("*/cache_summary.json"))
    if len(task_summaries) != sum(len(value) for value in city_scenarios.values()):
        raise ValueError("seed-8171 evaluation task summaries are incomplete")
    for path in task_summaries:
        summary = _load_json(path)
        authorization = summary.get("setting", {}).get(
            "authorization_provenance"
        )
        if (
            not authorization
            or authorization.get("sha256")
            != str(expected_joint_freeze_sha256)
            or authorization.get("decision") != authorization_decision
        ):
            raise ValueError(f"evaluation task lacks freeze authorization: {path}")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(
        Path(conversion_root).resolve()
    )
    manifest = load_traffic_signal_manifest(external_manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        evaluation_cache_root,
        manifest,
        seeds=(evaluation_seed,),
        collection_shards=EXTERNAL_COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )

    city_results: dict[str, Any] = {}
    for city, scenarios in city_scenarios.items():
        city_root = Path(freeze_root) / city
        method_result_path = city_root / "method_freeze.json"
        baseline_result_path = city_root / "baseline_freeze.json"
        method_model_path = city_root / "method_model.pkl"
        baseline_model_path = city_root / "baseline_model.pkl"
        method_freeze = _load_json(method_result_path)
        baseline_freeze = _load_json(baseline_result_path)
        joint_city = joint["cities"][city]
        if (
            _sha256(method_result_path)
            != joint_city["method_result"]["sha256"]
            or _sha256(baseline_result_path)
            != joint_city["baseline_result"]["sha256"]
            or method_freeze.get("protocol") != method_freeze_protocol
            or baseline_freeze.get("protocol") != baseline_freeze_protocol
        ):
            raise ValueError(f"seed-8171 freeze result mismatch for {city}")

        method_groups = tuple(method_freeze["selected_group_ids"])
        baseline_groups = tuple(baseline_freeze["selected_group_ids"])
        method_payload = _load_full_refit_model_artifact(
            method_model_path,
            expected_sha256=joint_city["method_model"]["sha256"],
            expected_protocol=method_model_protocol,
            city=city,
            expected_families=BASE_FAMILIES,
            expected_group_ids=method_groups,
        )
        baseline_payload = _load_full_refit_model_artifact(
            baseline_model_path,
            expected_sha256=joint_city["baseline_model"]["sha256"],
            expected_protocol=baseline_model_protocol,
            city=city,
            expected_families=BENCHMARK_FAMILIES,
            expected_group_ids=baseline_groups,
        )
        if (
            method_groups != baseline_groups
            or method_freeze["fold_assignment"]
            != baseline_freeze["fold_assignment"]
            or method_payload["fold_assignment"]
            != baseline_payload["fold_assignment"]
            or method_payload["fold_assignment"]
            != method_freeze["fold_assignment"]["assignments"]
            or method_freeze["selected_candidate"]
            != method_payload["selected_candidate"]
            or method_freeze["frozen_selector_key"] != FROZEN_SELECTOR_KEY
            or method_payload["selector_key"] != FROZEN_SELECTOR_KEY
        ):
            raise ValueError(f"method/baseline refit contract mismatch for {city}")

        selected = str(method_freeze["selected_candidate"])
        scenario_results: dict[str, Any] = {}
        for scenario in scenarios:
            dataset = bank[scenario]
            observed_seeds = {
                _group_seed_v3(str(group))
                for group in dataset.metadata["action_group_ids"]
            }
            if observed_seeds != {evaluation_seed}:
                raise ValueError(f"held-out-seed leakage for {scenario}")
            scenario_results[scenario] = evaluate_frozen_ensembles(
                dataset,
                method_payload=method_payload,
                baseline_payload=baseline_payload,
            )

        city_summary = _equal_scenario_city_summary(scenario_results)
        selected_policy = f"method/{selected}"
        anchor_policy = "method/constant_alpha_0"
        city_results[city] = {
            "scenarios": scenario_results,
            "equal_scenario_summary": city_summary,
            "selected_candidate": selected,
            "selected_policy": selected_policy,
            "anchor_policy": anchor_policy,
            "inference": {
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
            },
        }

    gate = _confirmation_gate(
        city_results, protocol["offline_confirmation_gate"]
    )
    return {
        "protocol": result_protocol,
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
        "evaluation_seed": evaluation_seed,
        "generation": generation,
        "deployment_model_ensemble_size": 1,
        "city_results": city_results,
        "offline_confirmation_gate": gate,
        "decision": (
            closed_loop_decision
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
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--expected-evaluation-cache-sha256", required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite held-out result: {args.out}")
    payload = run_external_full_heldout_evaluation(
        evaluation_cache_root=args.evaluation_cache_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_spec_path=args.protocol_spec,
        joint_freeze_audit_path=args.joint_freeze_audit,
        evaluation_launch_manifest_path=args.evaluation_launch_manifest,
        evaluation_task_root=args.evaluation_task_root,
        freeze_root=args.freeze_root,
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
