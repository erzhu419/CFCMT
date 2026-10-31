"""Run the frozen v41 seed-heldout B100 cross-fitted ensemble development stage."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    BASE_FAMILIES,
    COLLECTION_SHARDS,
    EXPECTED_CACHE_FILE_COUNT,
    SOURCE_SEEDS,
    aggregate_cache_sha256,
    evaluate_anchored_candidates,
    evaluate_anchored_ensemble_candidates,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import CANDIDATE_KEYS
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    CANDIDATE_SCOPES,
    MAXIMUM_OOF_FOLD_REGRESSIONS,
    MINIMUM_OOF_RELATIVE_IMPROVEMENTS,
    STANDARD_ERROR_MULTIPLIERS,
)
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import (
    assign_seed_stratified_folds,
    target_group_records,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _coverage_first_group_order_v3,
    _group_seed_v3,
    _group_subset_v3,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


TARGET_GROUP_BUDGET = 100
ADAPTATION_SEEDS = (2027, 3037)
EVALUATION_SEEDS = (4047,)
FOLD_COUNT = 5
GROUPS_PER_FOLD = TARGET_GROUP_BUDGET // FOLD_COUNT
TRAINING_GROUPS_PER_FOLD = TARGET_GROUP_BUDGET - GROUPS_PER_FOLD
SELECTION_SEED = 20260803


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    development = payload.get("development", {})
    expected_scalars = {
        "target_group_budget": TARGET_GROUP_BUDGET,
        "cross_fit_folds": FOLD_COUNT,
        "groups_per_fold": GROUPS_PER_FOLD,
        "model_training_groups_per_fold": TRAINING_GROUPS_PER_FOLD,
        "selection_seed": SELECTION_SEED,
    }
    for key, expected in expected_scalars.items():
        if int(development.get(key, -1)) != expected:
            raise ValueError(f"v41 {key} changed")
    if tuple(int(value) for value in development.get("adaptation_seeds", ())) != ADAPTATION_SEEDS:
        raise ValueError("v41 adaptation seed partition changed")
    if tuple(int(value) for value in development.get("evaluation_seeds", ())) != EVALUATION_SEEDS:
        raise ValueError("v41 evaluation seed partition changed")
    if set(ADAPTATION_SEEDS) & set(EVALUATION_SEEDS):
        raise ValueError("v41 adaptation and evaluation seeds overlap")
    if tuple(str(value) for value in development.get("base_families", ())) != BASE_FAMILIES:
        raise ValueError("v41 base family identities changed")
    grid = development.get("selector_grid", {})
    expected_grid = {
        "candidate_scopes": list(CANDIDATE_SCOPES),
        "minimum_oof_relative_improvement": list(MINIMUM_OOF_RELATIVE_IMPROVEMENTS),
        "maximum_oof_fold_regression": list(MAXIMUM_OOF_FOLD_REGRESSIONS),
        "standard_error_multipliers": list(STANDARD_ERROR_MULTIPLIERS),
    }
    for key, expected in expected_grid.items():
        observed = list(grid.get(key, ()))
        if key != "candidate_scopes":
            observed = [float(value) for value in observed]
        if observed != expected:
            raise ValueError(f"v41 selector grid changed: {key}")
    targets = [str(value) for value in development.get("evaluation_targets", ())]
    exclusions = development.get("source_only_capacity_exclusions", {})
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("v41 evaluation targets must be unique and nonempty")
    if not isinstance(exclusions, dict) or set(targets) & set(exclusions):
        raise ValueError("v41 capacity partition overlaps")
    return payload


def _groups_for_seeds(dataset, seeds: Sequence[int]) -> tuple[str, ...]:
    seed_set = {int(value) for value in seeds}
    groups = sorted(
        set(str(value) for value in dataset.metadata.get("action_group_ids", ()))
    )
    return tuple(group for group in groups if _group_seed_v3(group) in seed_set)


def _select_adaptation_groups(dataset) -> tuple[str, ...]:
    candidates = _groups_for_seeds(dataset, ADAPTATION_SEEDS)
    if len(candidates) < TARGET_GROUP_BUDGET:
        raise ValueError("v41 target has fewer than B100 adaptation-seed groups")
    ordered = _coverage_first_group_order_v3(
        dataset,
        candidates,
        selection_seed=SELECTION_SEED,
        role="v41_seed_heldout_adaptation",
    )
    selected = tuple(sorted(ordered[:TARGET_GROUP_BUDGET]))
    if len(selected) != TARGET_GROUP_BUDGET or len(set(selected)) != TARGET_GROUP_BUDGET:
        raise AssertionError("v41 B100 selection failed")
    return selected


def _fit_explicit_target_groups(
    *,
    bank,
    target: str,
    target_group_ids: Sequence[str],
    manifest,
    source_rule_costs,
    source_rule_specs,
):
    return fit_target_screening_models(
        bank,
        target,
        fit_config=MechanismFitConfig(),
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
        model_families=BASE_FAMILIES,
        target_group_budget=len(target_group_ids),
        target_adaptation_selection_seed=SELECTION_SEED,
        target_calibration_seeds=(),
        target_calibration_fraction=0.0,
        source_selector_min_context_domains=5,
        scenario_city_groups=manifest.city_groups,
        target_adaptation_group_ids=target_group_ids,
    )


def run_target_seed_heldout(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest_path: Path,
    protocol_spec_path: Path,
    target: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol_spec = _load_protocol(protocol_spec_path)
    baseline_sha256 = _sha256(baseline_result)
    if baseline_sha256 != expected_baseline_sha256:
        raise ValueError("v41 expanded source-rule baseline hash mismatch")
    cache_sha256, cache_file_count = aggregate_cache_sha256(cache_root)
    if cache_sha256 != expected_cache_sha256:
        raise ValueError("v41 combined counterfactual cache hash mismatch")
    if cache_file_count != EXPECTED_CACHE_FILE_COUNT:
        raise ValueError("v41 combined counterfactual cache file count changed")
    manifest = load_traffic_signal_manifest(manifest_path)
    development = protocol_spec["development"]
    evaluation_targets = [str(value) for value in development["evaluation_targets"]]
    source_only_exclusions = {
        str(name): int(count)
        for name, count in development["source_only_capacity_exclusions"].items()
    }
    if set(evaluation_targets) | set(source_only_exclusions) != set(manifest.sumocfgs):
        raise ValueError("v41 capacity partition does not cover the manifest")
    if target not in evaluation_targets:
        raise ValueError(f"v41 target is not B100 seed-heldout admitted: {target}")
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    selected_group_ids = _select_adaptation_groups(bank[target])
    evaluation_group_ids = _groups_for_seeds(bank[target], EVALUATION_SEEDS)
    if not evaluation_group_ids:
        raise ValueError("v41 target has no held-out evaluation-seed groups")
    if set(selected_group_ids) & set(evaluation_group_ids):
        raise AssertionError("v41 adaptation/evaluation group leakage")
    selected_dataset = _group_subset_v3(
        bank[target],
        selected_groups=set(selected_group_ids),
        metadata_updates={"target_data_role": "v41_b100_adaptation_feature_pool"},
    )
    fit_bank = dict(bank)
    fit_bank[target] = selected_dataset
    evaluation_dataset = _group_subset_v3(
        bank[target],
        selected_groups=set(evaluation_group_ids),
        metadata_updates={"target_data_role": "v41_seed4047_evaluation"},
    )
    records = target_group_records(
        bank[target],
        selected_group_ids=selected_group_ids,
        expected_group_count=TARGET_GROUP_BUDGET,
    )
    fold_assignment = assign_seed_stratified_folds(
        records,
        expected_group_count=TARGET_GROUP_BUDGET,
        fold_count=FOLD_COUNT,
    )
    baseline = json.loads(baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    fold_results = []
    model_ensemble = []
    for fold_index in range(FOLD_COUNT):
        fold_started = time.monotonic()
        validation_ids = tuple(
            sorted(
                group
                for group, fold in fold_assignment["assignments"].items()
                if int(fold) == fold_index
            )
        )
        validation_set = set(validation_ids)
        training_ids = tuple(
            group for group in selected_group_ids if group not in validation_set
        )
        if (
            len(training_ids) != TRAINING_GROUPS_PER_FOLD
            or len(validation_ids) != GROUPS_PER_FOLD
        ):
            raise ValueError("v41 fold train/validation counts changed")
        models = _fit_explicit_target_groups(
            bank=fit_bank,
            target=target,
            target_group_ids=training_ids,
            manifest=manifest,
            source_rule_costs=source_rule_costs,
            source_rule_specs=source_rule_specs,
        )
        validation = _group_subset_v3(
            selected_dataset,
            selected_groups=validation_set,
            metadata_updates={
                "target_data_role": "v41_cross_fit_out_of_fold_validation",
                "cross_fit_fold_index": fold_index,
            },
        )
        oof_evaluation = evaluate_anchored_candidates(
            validation,
            models=models,
            excluded_group_ids=(),
        )
        heldout_evaluation = evaluate_anchored_candidates(
            evaluation_dataset,
            models=models,
            excluded_group_ids=(),
        )
        if (
            int(oof_evaluation["evaluation_group_count"]) != GROUPS_PER_FOLD
            or set(oof_evaluation["candidates"]) != set(CANDIDATE_KEYS)
            or set(heldout_evaluation["candidates"]) != set(CANDIDATE_KEYS)
        ):
            raise ValueError("v41 fold candidate evaluation contract changed")
        model_ensemble.append(models)
        fold_results.append(
            {
                "fold_index": fold_index,
                "training_group_ids": list(training_ids),
                "validation_group_ids": list(validation_ids),
                "model_diagnostics": models.diagnostics,
                "oof_candidate_metrics": oof_evaluation["candidates"],
                "heldout_seed_candidate_metrics": heldout_evaluation["candidates"],
                "elapsed_sec": float(time.monotonic() - fold_started),
            }
        )
        gc.collect()
    ensemble_evaluation = evaluate_anchored_ensemble_candidates(
        evaluation_dataset,
        model_ensemble=model_ensemble,
        excluded_group_ids=(),
    )
    if (
        int(ensemble_evaluation["model_ensemble_size"]) != FOLD_COUNT
        or int(ensemble_evaluation["evaluation_group_count"])
        != len(evaluation_group_ids)
        or set(ensemble_evaluation["candidates"]) != set(CANDIDATE_KEYS)
    ):
        raise ValueError("v41 ensemble evaluation contract changed")
    oof_candidates = {}
    for candidate in CANDIDATE_KEYS:
        fold_regrets = [
            float(fold["oof_candidate_metrics"][candidate]["mean_normalized_action_regret"])
            for fold in fold_results
        ]
        fold_alphas = [
            float(fold["oof_candidate_metrics"][candidate]["mean_alpha"])
            for fold in fold_results
        ]
        oof_candidates[candidate] = {
            "fold_regrets": fold_regrets,
            "mean_regret": float(np.mean(fold_regrets)),
            "mean_alpha": float(np.mean(fold_alphas)),
        }
    target_city = manifest.city_groups[target]
    heldout = sorted(
        name for name, city in manifest.city_groups.items() if city == target_city
    )
    del model_ensemble
    gc.collect()
    return {
        "protocol": "tsc-v41r37-seed-heldout-b100-cross-fitted-ensemble-target-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "protocol_spec_path": str(protocol_spec_path.resolve()),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest": manifest.to_dict(),
        "cache_root": str(cache_root.resolve()),
        "cache_aggregate_sha256": cache_sha256,
        "cache_file_count": cache_file_count,
        "cache_audit": cache_audit,
        "baseline_result": str(baseline_result.resolve()),
        "baseline_sha256": baseline_sha256,
        "target": target,
        "target_city_group": target_city,
        "heldout_city_scenarios": heldout,
        "source_scenarios": sorted(set(manifest.sumocfgs) - set(heldout)),
        "development_evaluation_targets": evaluation_targets,
        "source_only_capacity_exclusions": source_only_exclusions,
        "target_group_budget": TARGET_GROUP_BUDGET,
        "adaptation_seeds": list(ADAPTATION_SEEDS),
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "selected_group_ids": list(selected_group_ids),
        "evaluation_group_ids": list(evaluation_group_ids),
        "model_families": list(BASE_FAMILIES),
        "fold_assignment": fold_assignment,
        "fold_results": fold_results,
        "oof_candidate_count": len(oof_candidates),
        "oof_candidates": oof_candidates,
        "ensemble_evaluation": ensemble_evaluation,
        "elapsed_sec": float(time.monotonic() - started),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_target_seed_heldout(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest_path=args.manifest,
        protocol_spec_path=args.protocol_spec,
        target=args.target,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "target": payload["target"],
                "fold_count": len(payload["fold_results"]),
                "candidate_count": payload["oof_candidate_count"],
                "evaluation_group_count": payload["ensemble_evaluation"][
                    "evaluation_group_count"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
