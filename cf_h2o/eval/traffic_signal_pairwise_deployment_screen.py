"""Evaluate the calibration-only pairwise deployment gate on one target."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    TARGET_GROUP_BUDGET,
    select_target_pairwise,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    _group_adjusted_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    evaluate_target_action_groups,
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import (
    MechanismDataset,
    MechanismFitConfig,
)


DEPLOYMENT_FAMILIES = (
    GROUP_NORMALIZED_RIGID_FAMILY,
    PAIRWISE_PREFERENCE_FAMILY,
)


def _group_regret_rows(
    dataset: MechanismDataset,
    *,
    models: Any,
) -> list[dict[str, Any]]:
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=models.prior_spec,
        contrast_features=CONTRAST_FEATURES_V3,
    )
    groups = action_group_ids(contrast)
    row_times = np.asarray(contrast.metadata.get("row_times", ()), dtype=float)
    if row_times.shape != (contrast.size,):
        raise ValueError("calibration contrast is missing row-aligned snapshot times")
    actual = np.asarray(contrast.targets["interval_cost"], dtype=float)
    scores = {}
    for family in DEPLOYMENT_FAMILIES:
        score, _, _, _ = _group_adjusted_scores(
            contrast,
            models.family_models[family].predict(contrast),
            objective_mode=models.objective_modes[family],
        )
        scores[family] = score

    rows = []
    for group in sorted(set(str(value) for value in groups)):
        group_rows = np.flatnonzero(groups == group)
        if group_rows.size < 2:
            raise ValueError(f"calibration group {group!r} has fewer than two actions")
        times = row_times[group_rows]
        if not np.allclose(times, times[0], rtol=0.0, atol=1e-12):
            raise ValueError(f"calibration group {group!r} spans multiple times")
        values = actual[group_rows]
        best = float(np.min(values))
        scale = action_group_range(values)
        family_regrets = {}
        selected_rows = {}
        for family in DEPLOYMENT_FAMILIES:
            selected = int(group_rows[int(np.argmin(scores[family][group_rows]))])
            family_regrets[family] = (float(actual[selected]) - best) / scale
            selected_rows[family] = selected
        rows.append(
            {
                "group_id": group,
                "snapshot_time_sec": float(times[0]),
                "action_count": int(group_rows.size),
                "family_regrets": family_regrets,
                "selected_row_indices": selected_rows,
                "paired_gain": (
                    family_regrets[GROUP_NORMALIZED_RIGID_FAMILY]
                    - family_regrets[PAIRWISE_PREFERENCE_FAMILY]
                ),
            }
        )
    return rows


def run_target_deployment_screen(
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
    models = fit_target_screening_models(
        bank,
        target,
        fit_config=MechanismFitConfig(),
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
        model_families=DEPLOYMENT_FAMILIES,
        target_group_budget=TARGET_GROUP_BUDGET,
        target_adaptation_selection_seed=20260803,
        target_calibration_seeds=(4047,),
        target_calibration_fraction=0.4,
        source_selector_min_context_domains=5,
        scenario_city_groups=manifest.city_groups,
    )
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
    calibration_rows = _group_regret_rows(split["calibration"], models=models)
    decision = select_target_pairwise(calibration_rows)
    evaluation = evaluate_target_action_groups(
        bank[target],
        models=models,
        excluded_group_ids=selected_group_ids,
    )
    selected_family = str(decision["selected_family"])
    selected_metric = evaluation["families"][selected_family]
    return {
        "protocol": "tsc-v35r31-calibration-only-pairwise-deployment-screen-v1",
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
        "adaptation_group_count": int(
            split["adaptation"].metadata["target_role_groups"]
        ),
        "calibration_group_count": int(
            split["calibration"].metadata["target_role_groups"]
        ),
        "selected_group_ids": list(selected_group_ids),
        "model_families": list(DEPLOYMENT_FAMILIES),
        "model_diagnostics": models.diagnostics,
        "calibration_rows": calibration_rows,
        "deployment_decision": decision,
        "offline_evaluation": evaluation,
        "selected_family": selected_family,
        "selected_evaluation": selected_metric,
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
    payload = run_target_deployment_screen(
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
                "calibration_gate_passed": payload["deployment_decision"]["gate"][
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
