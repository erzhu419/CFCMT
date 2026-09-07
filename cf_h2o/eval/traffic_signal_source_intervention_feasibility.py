"""Diagnose oracle headroom and held-out V124 source-guard value."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_conservative_source_intervention_gate import (
    _deployed_seed_value,
    _guard_feature_arrays,
    _load_prediction_artifact,
    _paired_dominance,
    _prediction,
    _select_guard,
    _select_source_candidate,
    _source_candidates,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_stacked_b100_source_gate import _selector_dataset
from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import (
    _paired_bootstrap,
)
from cf_h2o.eval.traffic_signal_target_calibrated_source_gate import (
    _normalized_pressure_targets,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _load_pure_waiting_selector_cache_audit,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v125-source-intervention-feasibility-v1"
PRESSURE_TOLERANCE = 1e-12


def _oracle_rows(
    actual: np.ndarray,
    policy_rows: np.ndarray,
    *,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    rows = np.asarray(policy_rows, dtype=int)
    values = np.asarray(actual[rows], dtype=float)
    if eligible is not None:
        allowed = np.asarray(eligible[rows], dtype=bool)
        if not np.all(np.any(allowed, axis=1)):
            raise ValueError("oracle eligibility removed every action in a group")
        values = np.where(allowed, values, np.inf)
    return rows[np.arange(rows.shape[0]), np.argmin(values, axis=1)]


def _seed_policy_summary(
    actual: np.ndarray,
    selected_rows: np.ndarray,
    reference_rows: np.ndarray,
    group_seeds: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(actual[selected_rows], dtype=float)
    by_seed = {
        str(int(seed)): float(np.mean(values[group_seeds == seed]))
        for seed in sorted(int(value) for value in np.unique(group_seeds))
    }
    interventions = np.asarray(selected_rows != reference_rows, dtype=bool)
    active = interventions
    return {
        "paired_seed_bootstrap": _paired_bootstrap(list(by_seed.values())),
        "seed_values": by_seed,
        "intervention_count": int(np.count_nonzero(interventions)),
        "intervention_fraction": float(np.mean(interventions)),
        "beneficial_intervention_fraction": (
            float(np.mean(values[active] < 0.0)) if np.any(active) else 0.0
        ),
        "harmful_intervention_fraction": (
            float(np.mean(values[active] > 0.0)) if np.any(active) else 0.0
        ),
    }


def _fold_bootstrap(
    folds: Sequence[Mapping[str, Any]], key: str
) -> dict[str, Any]:
    values = [float(row[key]) for row in folds]
    return {
        **_paired_bootstrap(values),
        "intervention_count": int(
            sum(int(row[f"{key}_interventions"]) for row in folds)
        ),
    }


def run_feasibility_diagnostic(
    *,
    prediction_artifact_path: Path,
    expected_prediction_artifact_sha256: str,
    selector_cache_root: Path,
    selector_manifest_path: Path,
    selector_cache_audit_path: Path,
    expected_selector_cache_audit_sha256: str,
    selector_scenario: str,
    selector_seeds: Sequence[int],
    selector_collection_shards: int,
    conversion_root: Path,
    cache_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    artifact = _load_prediction_artifact(
        prediction_artifact_path,
        expected_sha256=expected_prediction_artifact_sha256,
    )
    if (
        tuple(int(value) for value in artifact.get("selector_seeds", ()))
        != tuple(int(value) for value in selector_seeds)
        or artifact.get("selector_cache_audit_sha256")
        != expected_selector_cache_audit_sha256
    ):
        raise ValueError("V125 selector identity differs from V123")
    selector_audit = _load_pure_waiting_selector_cache_audit(
        selector_cache_audit_path,
        expected_sha256=expected_selector_cache_audit_sha256,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
    )
    selector, selector_bank_audit = _selector_dataset(
        cache_root=selector_cache_root,
        manifest_path=selector_manifest_path,
        scenario=selector_scenario,
        seeds=selector_seeds,
        collection_shards=selector_collection_shards,
        cache_workers=cache_workers,
    )
    _assert_pure_waiting_selector_dataset(
        selector, target_name="prefix_mean_cost_450s"
    )
    contrast = build_action_contrast_dataset(
        selector,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    actual, unique_groups, group_rows, references, group_seeds = (
        _normalized_pressure_targets(selector, contrast)
    )
    policy_rows = np.vstack(group_rows).astype(int, copy=False)
    reference_rows = policy_rows[
        np.arange(policy_rows.shape[0]),
        np.argmax(references[policy_rows], axis=1),
    ]
    feature_index = {
        str(name): index for index, name in enumerate(contrast.feature_names)
    }
    pressure_delta = np.asarray(
        contrast.features[
            :, feature_index["delta_service_pressure"]
        ],
        dtype=float,
    )
    oracle = _oracle_rows(actual, policy_rows)
    pressure_oracle = _oracle_rows(
        actual,
        policy_rows,
        eligible=pressure_delta >= -PRESSURE_TOLERANCE,
    )
    all_seeds = tuple(sorted(int(value) for value in np.unique(group_seeds)))
    source_order = tuple(str(value) for value in artifact["source_group_order"])
    budgets = (0, *(int(value) for value in artifact["target_budgets"]))
    budget_results = {}
    for budget in budgets:
        key = str(budget)
        target = (
            None
            if budget == 0
            else _prediction(artifact["target_predictions"][key])
        )
        sources = {
            source: _prediction(artifact["source_predictions"][key][source])
            for source in source_order
        }
        if any(row.score.size != contrast.size for row in sources.values()) or (
            target is not None and target.score.size != contrast.size
        ):
            raise ValueError("V125 prediction rows differ from the selector")
        candidates = _source_candidates(target, sources)
        folds = []
        for heldout_seed in all_seeds:
            training_seeds = set(all_seeds) - {heldout_seed}
            candidate_name, selected_prediction, _ = _select_source_candidate(
                candidates,
                actual=actual,
                policy_rows=policy_rows,
                references=references,
                group_seeds=group_seeds,
                training_seeds=training_seeds,
            )
            source_guard = _select_guard(
                selected_prediction,
                actual=actual,
                policy_rows=policy_rows,
                references=references,
                pressure_delta=pressure_delta,
                group_seeds=group_seeds,
                training_seeds=training_seeds,
                target_prediction=target,
            )
            if target is None:
                target_guard = {
                    "enabled": False,
                    "accepted": np.zeros(group_seeds.size, dtype=bool),
                    "training_seed_values": {
                        seed: 0.0 for seed in training_seeds
                    },
                    "training_mean": 0.0,
                    "training_upper_95_one_sided": 0.0,
                }
                dominance = {
                    "mean_source_minus_target": source_guard["training_mean"],
                    "upper_95_one_sided": source_guard[
                        "training_upper_95_one_sided"
                    ],
                    "source_authorized": bool(source_guard["enabled"]),
                }
            else:
                target_guard = _select_guard(
                    target,
                    actual=actual,
                    policy_rows=policy_rows,
                    references=references,
                    pressure_delta=pressure_delta,
                    group_seeds=group_seeds,
                    training_seeds=training_seeds,
                    target_prediction=None,
                )
                dominance = _paired_dominance(
                    source_guard,
                    target_guard,
                    training_seeds=training_seeds,
                )
            heldout_mask = group_seeds == heldout_seed
            source_value, source_count = _deployed_seed_value(
                source_guard,
                selected_prediction,
                actual=actual,
                policy_rows=policy_rows,
                references=references,
                seed_mask=heldout_mask,
            )
            target_value, target_count = (
                (0.0, 0)
                if target is None
                else _deployed_seed_value(
                    target_guard,
                    target,
                    actual=actual,
                    policy_rows=policy_rows,
                    references=references,
                    seed_mask=heldout_mask,
                )
            )
            raw = _guard_feature_arrays(
                selected_prediction,
                policy_rows=policy_rows,
                references=references,
                pressure_delta=pressure_delta,
                target_prediction=target,
            )
            raw_values = np.asarray(actual[raw["learned_rows"]], dtype=float)
            raw_value = float(np.mean(raw_values[heldout_mask]))
            raw_count = int(
                np.count_nonzero(raw["proposed"] & heldout_mask)
            )
            significance_only = bool(
                source_guard["enabled"]
                and dominance["upper_95_one_sided"] < 0.0
            )
            relaxed_value = source_value if significance_only else target_value
            relaxed_count = source_count if significance_only else target_count
            folds.append(
                {
                    "heldout_seed": int(heldout_seed),
                    "selected_source_candidate": candidate_name,
                    "source_guard_enabled": bool(source_guard["enabled"]),
                    "target_guard_enabled": bool(target_guard["enabled"]),
                    "dominance_mean": float(
                        dominance["mean_source_minus_target"]
                    ),
                    "dominance_upper_95_one_sided": float(
                        dominance["upper_95_one_sided"]
                    ),
                    "v124_source_authorized": bool(
                        dominance["source_authorized"]
                    ),
                    "significance_only_source_authorized": significance_only,
                    "raw_source": raw_value,
                    "raw_source_interventions": raw_count,
                    "source_guard": source_value,
                    "source_guard_interventions": source_count,
                    "target_guard": target_value,
                    "target_guard_interventions": target_count,
                    "source_guard_minus_target": source_value - target_value,
                    "source_guard_minus_target_interventions": source_count,
                    "significance_only_policy": relaxed_value,
                    "significance_only_policy_interventions": relaxed_count,
                }
            )
        budget_results[key] = {
            "target_group_budget": budget,
            "folds": folds,
            "raw_source_vs_phase_pressure": _fold_bootstrap(
                folds, "raw_source"
            ),
            "source_guard_candidate_vs_phase_pressure": _fold_bootstrap(
                folds, "source_guard"
            ),
            "target_guard_vs_phase_pressure": _fold_bootstrap(
                folds, "target_guard"
            ),
            "source_guard_candidate_minus_target": _fold_bootstrap(
                folds, "source_guard_minus_target"
            ),
            "significance_only_policy_vs_phase_pressure": _fold_bootstrap(
                folds, "significance_only_policy"
            ),
            "source_guard_enabled_fold_count": int(
                sum(row["source_guard_enabled"] for row in folds)
            ),
            "significance_only_authorized_fold_count": int(
                sum(
                    row["significance_only_source_authorized"]
                    for row in folds
                )
            ),
        }
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "post-v124-development-diagnostic",
        "city": artifact["city"],
        "estimand": artifact["estimand"],
        "selector_scenario": selector_scenario,
        "selector_seed_count": len(all_seeds),
        "selector_group_count": len(unique_groups),
        "selector_row_count": int(contrast.size),
        "oracle_headroom": {
            "unrestricted": _seed_policy_summary(
                actual, oracle, reference_rows, group_seeds
            ),
            "nondecreasing_instantaneous_pressure": _seed_policy_summary(
                actual, pressure_oracle, reference_rows, group_seeds
            ),
            "pressure_tolerance": PRESSURE_TOLERANCE,
        },
        "budget_results": budget_results,
        "inputs": {
            "prediction_artifact_sha256": expected_prediction_artifact_sha256,
            "selector_cache_audit_sha256": expected_selector_cache_audit_sha256,
        },
        "input_audits": {
            "selector_cache": selector_audit,
            "selector_bank": selector_bank_audit,
        },
        "claim_boundary": (
            "V125 diagnoses available selector headroom and held-out guard "
            "behavior. It cannot authorize a fresh-city or closed-loop claim."
        ),
        "runtime_seconds": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-artifact", type=Path, required=True)
    parser.add_argument("--prediction-artifact-sha256", required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-manifest", type=Path, required=True)
    parser.add_argument("--selector-cache-audit", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-sha256", required=True)
    parser.add_argument("--selector-scenario", required=True)
    parser.add_argument("--selector-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--selector-collection-shards", type=int, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V125 result: {args.out}")
    result = run_feasibility_diagnostic(
        prediction_artifact_path=args.prediction_artifact,
        expected_prediction_artifact_sha256=args.prediction_artifact_sha256,
        selector_cache_root=args.selector_cache_root,
        selector_manifest_path=args.selector_manifest,
        selector_cache_audit_path=args.selector_cache_audit,
        expected_selector_cache_audit_sha256=args.selector_cache_audit_sha256,
        selector_scenario=args.selector_scenario,
        selector_seeds=args.selector_seeds,
        selector_collection_shards=args.selector_collection_shards,
        conversion_root=args.conversion_root,
        cache_workers=args.cache_workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "protocol": result["protocol"],
                "oracle_headroom": result["oracle_headroom"],
                "runtime_seconds": result["runtime_seconds"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
