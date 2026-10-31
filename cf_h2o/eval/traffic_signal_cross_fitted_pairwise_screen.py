"""Run the v36 cross-fitted pairwise deployment screen for one target."""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import (
    CROSS_FIT_FOLD_COUNT,
    assign_seed_stratified_folds,
    select_cross_fitted_pairwise,
    target_group_records,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    TARGET_GROUP_BUDGET,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_screen import (
    DEPLOYMENT_FAMILIES,
    _group_regret_rows,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    evaluate_target_action_groups,
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


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
        model_families=DEPLOYMENT_FAMILIES,
        target_group_budget=len(target_group_ids),
        target_adaptation_selection_seed=20260803,
        target_calibration_seeds=(4047,),
        target_calibration_fraction=0.4,
        source_selector_min_context_domains=5,
        scenario_city_groups=manifest.city_groups,
        target_adaptation_group_ids=target_group_ids,
    )


def run_target_cross_fitted_screen(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest_path: Path,
    target: str,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = load_traffic_signal_manifest(manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=(2027, 3037, 4047),
        collection_shards=16,
        workers=workers,
    )
    if target not in bank:
        raise ValueError(f"unknown target: {target}")
    baseline = json.loads(baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
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
    records = target_group_records(
        bank[target],
        selected_group_ids=selected_group_ids,
    )
    record_by_id = {str(row["group_id"]): row for row in records}
    fold_assignment = assign_seed_stratified_folds(records)

    oof_rows = []
    fold_results = []
    for fold_index in range(CROSS_FIT_FOLD_COUNT):
        fold_started = time.monotonic()
        validation_ids = tuple(
            sorted(
                group
                for group, fold in fold_assignment["assignments"].items()
                if int(fold) == fold_index
            )
        )
        training_ids = tuple(
            group for group in selected_group_ids if group not in set(validation_ids)
        )
        if len(training_ids) != 48 or len(validation_ids) != 12:
            raise ValueError("cross-fit train/validation group counts changed")
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
            selected_groups=set(validation_ids),
            metadata_updates={
                "target_data_role": "cross_fit_out_of_fold_validation",
                "cross_fit_fold_index": fold_index,
            },
        )
        rows = _group_regret_rows(validation, models=models)
        if tuple(sorted(str(row["group_id"]) for row in rows)) != validation_ids:
            raise ValueError("cross-fit validation rows changed group identities")
        for row in rows:
            group_id = str(row["group_id"])
            row["fold_index"] = fold_index
            row["simulator_seed"] = int(record_by_id[group_id]["simulator_seed"])
            row["tls_id"] = str(record_by_id[group_id]["tls_id"])
            oof_rows.append(row)
        fold_results.append(
            {
                "fold_index": fold_index,
                "training_group_ids": list(training_ids),
                "validation_group_ids": list(validation_ids),
                "model_diagnostics": models.diagnostics,
                "mean_paired_gain": float(
                    np.mean([float(row["paired_gain"]) for row in rows])
                ),
                "elapsed_sec": float(time.monotonic() - fold_started),
            }
        )
        del models
        gc.collect()
    if len(oof_rows) != TARGET_GROUP_BUDGET:
        raise ValueError("cross-fit screen did not produce 60 OOF rows")
    decision = select_cross_fitted_pairwise(oof_rows)

    final_fit_started = time.monotonic()
    final_models = _fit_explicit_target_groups(
        bank=bank,
        target=target,
        target_group_ids=selected_group_ids,
        manifest=manifest,
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
    )
    evaluation = evaluate_target_action_groups(
        bank[target],
        models=final_models,
        excluded_group_ids=selected_group_ids,
    )
    selected_family = str(decision["selected_family"])
    return {
        "protocol": "tsc-v36r32-cross-fitted-pairwise-deployment-screen-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "manifest": manifest.to_dict(),
        "manifest_path": str(manifest_path.resolve()),
        "cache_audit": cache_audit,
        "baseline_result": str(baseline_result.resolve()),
        "target": target,
        "city_group": manifest.city_groups[target],
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": list(selected_group_ids),
        "model_families": list(DEPLOYMENT_FAMILIES),
        "fold_assignment": fold_assignment,
        "fold_results": fold_results,
        "oof_rows": oof_rows,
        "deployment_decision": decision,
        "final_fit_group_count": len(selected_group_ids),
        "final_model_diagnostics": final_models.diagnostics,
        "final_fit_elapsed_sec": float(time.monotonic() - final_fit_started),
        "offline_evaluation": evaluation,
        "selected_family": selected_family,
        "selected_evaluation": evaluation["families"][selected_family],
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
    parser.add_argument("--target", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_target_cross_fitted_screen(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest_path=args.manifest,
        target=args.target,
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
                "selected_family": payload["selected_family"],
                "oof_gate_passed": payload["deployment_decision"]["gate"][
                    "passed"
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
