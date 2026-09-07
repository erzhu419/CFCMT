"""Generate target-only five-fold OOF evidence for the frozen v40 selector."""

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
    EXPECTED_CACHE_FILE_COUNT,
    SOURCE_SEEDS,
    TARGET_GROUP_BUDGET,
    aggregate_cache_sha256,
    evaluate_anchored_candidates,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import CANDIDATE_KEYS
from cf_h2o.eval.traffic_signal_cross_fitted_anchored_selection import (
    CANDIDATE_SCOPES,
    FOLD_COUNT,
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
    _group_subset_v3,
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


COLLECTION_SHARDS = 16
GROUPS_PER_FOLD = TARGET_GROUP_BUDGET // FOLD_COUNT


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    development = payload.get("development", {})
    if int(development.get("target_group_budget", -1)) != TARGET_GROUP_BUDGET:
        raise ValueError("v40 target group budget changed")
    if int(development.get("cross_fit_folds", -1)) != FOLD_COUNT:
        raise ValueError("v40 fold count changed")
    if int(development.get("groups_per_fold", -1)) != GROUPS_PER_FOLD:
        raise ValueError("v40 groups per fold changed")
    grid = development.get("selector_grid", {})
    expected = {
        "candidate_scopes": list(CANDIDATE_SCOPES),
        "minimum_oof_relative_improvement": list(
            MINIMUM_OOF_RELATIVE_IMPROVEMENTS
        ),
        "maximum_oof_fold_regression": list(MAXIMUM_OOF_FOLD_REGRESSIONS),
        "standard_error_multipliers": list(STANDARD_ERROR_MULTIPLIERS),
    }
    for key, values in expected.items():
        observed = list(grid.get(key, ()))
        if key != "candidate_scopes":
            observed = [float(value) for value in observed]
        if observed != values:
            raise ValueError(f"v40 selector grid changed: {key}")
    targets = [str(value) for value in development.get("evaluation_targets", ())]
    exclusions = development.get("source_only_capacity_exclusions", {})
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("v40 evaluation targets must be unique and nonempty")
    if not isinstance(exclusions, dict) or set(targets) & set(exclusions):
        raise ValueError("v40 capacity partition overlaps")
    return payload


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
        target_adaptation_selection_seed=20260803,
        target_calibration_seeds=(4047,),
        target_calibration_fraction=0.4,
        source_selector_min_context_domains=5,
        scenario_city_groups=manifest.city_groups,
        target_adaptation_group_ids=target_group_ids,
    )


def run_target_oof(
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
        raise ValueError("v40 expanded source-rule baseline hash mismatch")
    cache_sha256, cache_file_count = aggregate_cache_sha256(cache_root)
    if cache_sha256 != expected_cache_sha256:
        raise ValueError("v40 combined counterfactual cache hash mismatch")
    if cache_file_count != EXPECTED_CACHE_FILE_COUNT:
        raise ValueError("v40 combined counterfactual cache file count changed")
    manifest = load_traffic_signal_manifest(manifest_path)
    development = protocol_spec["development"]
    evaluation_targets = [str(value) for value in development["evaluation_targets"]]
    source_only_exclusions = {
        str(name): int(count)
        for name, count in development["source_only_capacity_exclusions"].items()
    }
    if set(evaluation_targets) | set(source_only_exclusions) != set(manifest.sumocfgs):
        raise ValueError("v40 capacity partition does not cover the manifest")
    if target not in evaluation_targets:
        raise ValueError(f"v40 target is not capacity admitted: {target}")
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=max(int(workers), 1),
    )
    target_groups = np.asarray(
        bank[target].metadata.get("action_group_ids", ()), dtype=str
    )
    if target_groups.shape != (bank[target].size,):
        raise ValueError("v40 target action groups are not row aligned")
    safe_complete_group_count = len(set(target_groups.tolist()))
    if safe_complete_group_count < TARGET_GROUP_BUDGET:
        raise ValueError("v40 target has fewer than B60 safe complete groups")
    split = _target_adaptation_split_v3(
        bank[target],
        group_budget=TARGET_GROUP_BUDGET,
        selection_seed=20260803,
        calibration_seeds=(4047,),
        calibration_fraction=0.4,
    )
    selected_group_ids = tuple(
        sorted(
            str(value)
            for value in split["selected"].metadata[
                "target_adaptation_selected_group_ids"
            ]
        )
    )
    if len(selected_group_ids) != TARGET_GROUP_BUDGET:
        raise ValueError("v40 target selection did not produce B60")
    records = target_group_records(
        bank[target], selected_group_ids=selected_group_ids
    )
    fold_assignment = assign_seed_stratified_folds(records)
    baseline = json.loads(baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    fold_results = []
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
        if len(training_ids) != 48 or len(validation_ids) != GROUPS_PER_FOLD:
            raise ValueError("v40 fold train/validation counts changed")
        models = _fit_explicit_target_groups(
            bank=bank,
            target=target,
            target_group_ids=training_ids,
            manifest=manifest,
            source_rule_costs=source_rule_costs,
            source_rule_specs=source_rule_specs,
        )
        validation = _group_subset_v3(
            bank[target],
            selected_groups=validation_set,
            metadata_updates={
                "target_data_role": "v40_cross_fit_out_of_fold_validation",
                "cross_fit_fold_index": fold_index,
            },
        )
        evaluation = evaluate_anchored_candidates(
            validation,
            models=models,
            excluded_group_ids=(),
        )
        if (
            int(evaluation["evaluation_group_count"]) != GROUPS_PER_FOLD
            or int(evaluation["candidate_count"]) != len(CANDIDATE_KEYS)
            or set(evaluation["candidates"]) != set(CANDIDATE_KEYS)
        ):
            raise ValueError("v40 fold candidate evaluation contract changed")
        fold_results.append(
            {
                "fold_index": fold_index,
                "training_group_ids": list(training_ids),
                "validation_group_ids": list(validation_ids),
                "model_diagnostics": models.diagnostics,
                "candidate_metrics": evaluation["candidates"],
                "elapsed_sec": float(time.monotonic() - fold_started),
            }
        )
        del models
        gc.collect()
    oof_candidates = {}
    for candidate in CANDIDATE_KEYS:
        fold_regrets = [
            float(fold["candidate_metrics"][candidate]["mean_normalized_action_regret"])
            for fold in fold_results
        ]
        fold_alphas = [
            float(fold["candidate_metrics"][candidate]["mean_alpha"])
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
    return {
        "protocol": "tsc-v40r36-cross-fitted-anchored-target-oof-v1",
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
        "target_safe_complete_group_count": safe_complete_group_count,
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": list(selected_group_ids),
        "model_families": list(BASE_FAMILIES),
        "fold_assignment": fold_assignment,
        "fold_results": fold_results,
        "oof_candidate_count": len(oof_candidates),
        "oof_candidates": oof_candidates,
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
    payload = run_target_oof(
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
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
