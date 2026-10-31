"""Local-arrival and robust-target diagnostic after v72 module falsification."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_demand_conditioned_veto_freeze import (
    _model_config,
    _profile,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    frozen_proposal_records,
)
from cf_h2o.eval.traffic_signal_external_proposal_conditional_veto_freeze import (
    _balanced_weights,
    summarize_quantile_candidate,
)
from cf_h2o.eval.traffic_signal_external_spatiotemporal_veto_diagnostic import (
    _canonical_sha256,
    feature_matrix as v72_feature_matrix,
    feature_names as v72_feature_names,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.demand_schedule_context import DemandScheduleProfile
from cf_h2o.traffic_signal.demand_timeline_context import (
    TLSLocalTopologyContext,
    TemporalDemandContext,
    build_temporal_demand_context,
    build_tls_local_topology_context,
)
from cf_h2o.traffic_signal.local_demand_timeline_context import (
    LOCAL_DEMAND_FEATURE_NAMES,
    LocalDemandTimelineContext,
    build_local_demand_timeline_context,
)
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
)


DIAGNOSTIC_PROTOCOL = "tsc-v73r69-external-v9-local-demand-model-diagnostic-v1"
RESULT_PROTOCOL = "tsc-v73r69-external-v9-local-demand-model-diagnostic-result-v1"
DECISION = "diagnostic_only_no_closed_loop_or_validation_authorization"
BASE_FEATURE_VARIANT = "static_plus_topology"
VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "topology_squared_normalized": {
        "local_demand": False,
        "model": "legacy_squared_regressor",
        "target": "normalized",
    },
    "topology_squared_clipped": {
        "local_demand": False,
        "model": "squared_regressor",
        "target": "normalized_clipped_0p2",
    },
    "topology_absolute_normalized": {
        "local_demand": False,
        "model": "absolute_regressor",
        "target": "normalized",
    },
    "topology_benefit_classifier": {
        "local_demand": False,
        "model": "benefit_classifier",
        "target": "normalized_sign",
    },
    "topology_benefit_magnitude_classifier": {
        "local_demand": False,
        "model": "benefit_magnitude_classifier",
        "target": "normalized_sign",
    },
    "topology_local_squared_normalized": {
        "local_demand": True,
        "model": "squared_regressor",
        "target": "normalized",
    },
    "topology_local_squared_clipped": {
        "local_demand": True,
        "model": "squared_regressor",
        "target": "normalized_clipped_0p2",
    },
    "topology_local_absolute_normalized": {
        "local_demand": True,
        "model": "absolute_regressor",
        "target": "normalized",
    },
    "topology_local_squared_raw_scaled": {
        "local_demand": True,
        "model": "squared_regressor",
        "target": "raw_scenario_robust_scaled",
    },
    "topology_local_benefit_classifier": {
        "local_demand": True,
        "model": "benefit_classifier",
        "target": "normalized_sign",
    },
    "topology_local_benefit_magnitude_classifier": {
        "local_demand": True,
        "model": "benefit_magnitude_classifier",
        "target": "normalized_sign",
    },
}


def _canonical(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _finite_spearman(scores: np.ndarray, actual: np.ndarray) -> float:
    if len(scores) < 3 or np.unique(scores).size < 2 or np.unique(actual).size < 2:
        return 0.0
    statistic = float(spearmanr(scores, actual).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def _tls_id(record: Mapping[str, Any]) -> str:
    return str(record["group_id"]).rsplit(":", 1)[-1]


def variant_feature_names(name: str) -> tuple[str, ...]:
    if name not in VARIANT_SPECS:
        raise KeyError(name)
    base = tuple(v72_feature_names(BASE_FEATURE_VARIANT))
    return (
        (*base, *LOCAL_DEMAND_FEATURE_NAMES)
        if bool(VARIANT_SPECS[name]["local_demand"])
        else base
    )


def variant_feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    profiles: Mapping[str, DemandScheduleProfile],
    temporal: Mapping[str, TemporalDemandContext],
    topology: Mapping[str, TLSLocalTopologyContext],
    local_demand: Mapping[str, LocalDemandTimelineContext],
) -> np.ndarray:
    base = v72_feature_matrix(
        records,
        variant=BASE_FEATURE_VARIANT,
        profiles=profiles,
        temporal=temporal,
        topology=topology,
    )
    if not bool(VARIANT_SPECS[variant]["local_demand"]):
        return base
    local_rows = (
        np.asarray(
            [
                local_demand[str(row["scenario"])]
                .vector_at(_tls_id(row), float(row["time_sec"]))
                .tolist()
                for row in records
            ],
            dtype=float,
        )
        if records
        else np.empty((0, len(LOCAL_DEMAND_FEATURE_NAMES)), dtype=float)
    )
    matrix = np.concatenate([base, local_rows], axis=1)
    if matrix.shape != (len(records), len(variant_feature_names(variant))):
        raise ValueError("local-demand diagnostic matrix changed")
    return matrix


def _transformed_target(
    records: Sequence[Mapping[str, Any]], *, target: str
) -> tuple[np.ndarray, dict[str, float]]:
    normalized = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in records],
        dtype=float,
    )
    if target in {"normalized", "normalized_sign"}:
        return normalized, {}
    if target == "normalized_clipped_0p2":
        return np.clip(normalized, -0.2, 0.2), {}
    if target == "raw_scenario_robust_scaled":
        scales = {}
        transformed = np.zeros(len(records), dtype=float)
        for scenario in sorted({str(row["scenario"]) for row in records}):
            indices = np.asarray(
                [
                    index
                    for index, row in enumerate(records)
                    if str(row["scenario"]) == scenario
                ],
                dtype=int,
            )
            raw = np.asarray(
                [float(records[index]["actual_raw_delta"]) for index in indices],
                dtype=float,
            )
            scale = max(float(np.median(np.abs(raw))), 1e-6)
            scales[scenario] = scale
            transformed[indices] = np.clip(raw / scale, -3.0, 3.0)
        return transformed, scales
    raise ValueError(f"unknown local-demand target transform: {target}")


def _fit_scores(
    *,
    variant: str,
    train: Sequence[Mapping[str, Any]],
    heldout: Sequence[Mapping[str, Any]],
    x_train: np.ndarray,
    x_heldout: np.ndarray,
    model_spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spec = VARIANT_SPECS[variant]
    transformed, scales = _transformed_target(train, target=str(spec["target"]))
    weights = _balanced_weights(train)
    model_kind = str(spec["model"])
    if model_kind == "legacy_squared_regressor":
        model = ProposalConditionalRegressor(
            feature_names=variant_feature_names(variant),
            config=_model_config(model_spec),
        )
        fit = model.fit(x_train, transformed, sample_weight=weights)
        return (
            model.predict_matrix(x_train),
            model.predict_matrix(x_heldout) if heldout else np.zeros(0),
            {"model": model_kind, "target_scales": scales, "fit": fit},
        )
    common = {
        "learning_rate": float(model_spec["learning_rate"]),
        "max_iter": int(model_spec["max_iter"]),
        "max_leaf_nodes": int(model_spec["max_leaf_nodes"]),
        "min_samples_leaf": int(model_spec["min_samples_leaf"]),
        "l2_regularization": float(model_spec["l2_regularization"]),
        "early_stopping": False,
        "random_state": int(model_spec["random_state"]),
    }
    if model_kind in {"squared_regressor", "absolute_regressor"}:
        loss = (
            "squared_error"
            if model_kind == "squared_regressor"
            else "absolute_error"
        )
        estimator = HistGradientBoostingRegressor(loss=loss, **common)
        estimator.fit(x_train, transformed, sample_weight=weights)
        train_scores = np.asarray(estimator.predict(x_train), dtype=float)
        heldout_scores = (
            np.asarray(estimator.predict(x_heldout), dtype=float)
            if heldout
            else np.zeros(0, dtype=float)
        )
        return train_scores, heldout_scores, {
            "model": model_kind,
            "target": spec["target"],
            "target_scales": scales,
            "training_rows": len(train),
            "training_score_std": float(np.std(train_scores)),
        }
    harmful = (transformed >= 0.0).astype(int)
    classifier_weights = np.asarray(weights, dtype=float)
    if model_kind == "benefit_magnitude_classifier":
        magnitude = np.clip(np.abs(transformed), 0.02, 0.5)
        classifier_weights *= magnitude / max(float(np.mean(magnitude)), 1e-12)
    if np.unique(harmful).size < 2:
        constant = float(harmful[0])
        train_scores = np.full(len(train), constant, dtype=float)
        heldout_scores = np.full(len(heldout), constant, dtype=float)
    else:
        estimator = HistGradientBoostingClassifier(**common)
        estimator.fit(x_train, harmful, sample_weight=classifier_weights)
        train_scores = np.asarray(estimator.predict_proba(x_train)[:, 1], dtype=float)
        heldout_scores = (
            np.asarray(estimator.predict_proba(x_heldout)[:, 1], dtype=float)
            if heldout
            else np.zeros(0, dtype=float)
        )
    return train_scores, heldout_scores, {
        "model": model_kind,
        "target": spec["target"],
        "target_scales": scales,
        "training_rows": len(train),
        "harmful_fraction": float(np.mean(harmful)),
        "training_score_std": float(np.std(train_scores)),
    }


def _fit_variant(
    *,
    variant: str,
    proposals: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    profiles: Mapping[str, DemandScheduleProfile],
    temporal: Mapping[str, TemporalDemandContext],
    topology: Mapping[str, TLSLocalTopologyContext],
    local_demand: Mapping[str, LocalDemandTimelineContext],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    quantiles = tuple(float(value) for value in diagnostic["acceptance_quantiles"])
    predictions = []
    folds = []
    for heldout_seed in seeds:
        train = [
            row
            for row in proposals
            if int(row["simulator_seed"]) != int(heldout_seed)
        ]
        heldout = [
            row
            for row in proposals
            if int(row["simulator_seed"]) == int(heldout_seed)
        ]
        x_train = variant_feature_matrix(
            train,
            variant=variant,
            profiles=profiles,
            temporal=temporal,
            topology=topology,
            local_demand=local_demand,
        )
        x_heldout = variant_feature_matrix(
            heldout,
            variant=variant,
            profiles=profiles,
            temporal=temporal,
            topology=topology,
            local_demand=local_demand,
        )
        train_scores, heldout_scores, fit = _fit_scores(
            variant=variant,
            train=train,
            heldout=heldout,
            x_train=x_train,
            x_heldout=x_heldout,
            model_spec=diagnostic["model"],
        )
        thresholds = {
            str(quantile): float(np.quantile(train_scores, quantile))
            for quantile in quantiles
        }
        fold_predictions = [
            {
                "group_id": str(row["group_id"]),
                "simulator_seed": int(row["simulator_seed"]),
                "scenario": str(row["scenario"]),
                "actual_group_normalized_delta": float(
                    row["actual_group_normalized_delta"]
                ),
                "predicted_delta": float(score),
                "fold_thresholds": thresholds,
            }
            for row, score in zip(heldout, heldout_scores)
        ]
        predictions.extend(fold_predictions)
        folds.append(
            {
                "heldout_seed": int(heldout_seed),
                "training_seeds": sorted(
                    {int(row["simulator_seed"]) for row in train}
                ),
                "training_rows": len(train),
                "heldout_rows": len(heldout),
                "thresholds": thresholds,
                "fit": fit,
                "prediction_sha256": _canonical(fold_predictions),
            }
        )
    if len(predictions) != len(proposals) or len(
        {str(row["group_id"]) for row in predictions}
    ) != len(proposals):
        raise RuntimeError("local-demand OOF partition changed")
    actual = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in predictions]
    )
    scores = np.asarray([float(row["predicted_delta"]) for row in predictions])
    grid = [
        summarize_quantile_candidate(
            predictions,
            all_records=all_records,
            seeds=seeds,
            scenarios=scenarios,
            acceptance_quantile=quantile,
            selection_rule=diagnostic["selection_rule"],
            bootstrap=diagnostic["bootstrap"],
        )
        for quantile in quantiles
    ]
    feasible = [row for row in grid if bool(row["feasible"])]
    best = min(
        grid,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_delta"]),
            float(row["harmful_group_fraction"]),
        ),
    )
    return {
        "variant": variant,
        "spec": VARIANT_SPECS[variant],
        "feature_names": list(variant_feature_names(variant)),
        "feature_count": len(variant_feature_names(variant)),
        "folds": folds,
        "oof_prediction_sha256": _canonical(predictions),
        "oof_row_count": len(predictions),
        "oof_spearman": _finite_spearman(scores, actual),
        "grid": grid,
        "feasible_candidate_count": len(feasible),
        "passed_original_v70_rule": bool(feasible),
        "best_candidate": best,
    }


def run_diagnostic(
    *,
    diagnostic_protocol_path: Path,
    cache_root: Path,
    cache_audit_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    v72_result_path: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = json.loads(
        Path(diagnostic_protocol_path).read_text(encoding="utf-8")
    )
    cache_protocol_path = Path(protocol["cache_protocol"]["path"])
    cache_protocol = json.loads(cache_protocol_path.read_text(encoding="utf-8"))
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    proposal_parent = json.loads(
        Path(proposal_parent_protocol_path).read_text(encoding="utf-8")
    )
    freeze_audit = json.loads(Path(freeze_audit_path).read_text(encoding="utf-8"))
    v72 = json.loads(Path(v72_result_path).read_text(encoding="utf-8"))
    if (
        protocol.get("protocol") != DIAGNOSTIC_PROTOCOL
        or _sha256(cache_protocol_path) != protocol["cache_protocol"]["sha256"]
        or _sha256(cache_audit_path) != protocol["cache_audit"]["sha256"]
        or cache_audit.get("status") != "PASS"
        or _sha256(v72_result_path) != protocol["v72_diagnostic_result"]["sha256"]
        or v72.get("status") != "PASS"
        or _sha256(proposal_parent_protocol_path)
        != protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != protocol["hierarchical_freeze_audit"]["sha256"]
        or freeze_audit.get("status") != "PASS"
    ):
        raise ValueError("v73 local-demand diagnostic evidence changed")
    source_gate = {
        path: Path(path).is_file() and _sha256(Path(path)) == expected
        for path, expected in protocol["runtime_executable_sources"].items()
    }
    if not source_gate or not all(source_gate.values()):
        raise ValueError(f"v73 local-demand sources changed: {source_gate}")
    cache_sha, cache_count = aggregate_cache_sha256(cache_root)
    if (
        cache_sha != cache_audit["cache_sha256"]
        or cache_count != cache_audit["cache_file_count"]
    ):
        raise ValueError("v73 local-demand cache changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    manifest = load_traffic_signal_manifest(external_manifest_path)
    collection = cache_protocol["long_horizon_collection"]
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=tuple(int(value) for value in collection["seeds"]),
        collection_shards=int(collection["collection_shards_per_scenario_seed"]),
        workers=max(int(workers), 1),
    )
    profiles = {
        scenario: _profile(payload)
        for scenario, payload in cache_protocol["demand_schedule_context"][
            "profiles"
        ].items()
    }
    temporal = {}
    topology = {}
    local_demand = {}
    for scenario in profiles:
        sumocfg = Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        temporal[scenario] = build_temporal_demand_context(sumocfg)
        topology[scenario] = build_tls_local_topology_context(sumocfg)
        local_demand[scenario] = build_local_demand_timeline_context(sumocfg)
    seeds = tuple(int(value) for value in collection["seeds"])
    city_results = {}
    reproduction = {}
    for city, scenarios_raw in sorted(cache_protocol["city_scenarios"].items()):
        scenarios = tuple(str(value) for value in scenarios_raw)
        _, frozen_model = _load_frozen_city_model(
            city=str(city),
            freeze_root=freeze_root,
            protocol=proposal_parent,
            freeze_audit=freeze_audit,
        )
        dataset = _merge_city_datasets(bank, scenarios, city=str(city))
        contrast = build_action_contrast_dataset(
            dataset,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES_V3,
        )
        all_records = frozen_proposal_records(
            contrast, frozen_model=frozen_model, scenarios=scenarios
        )
        proposals = [row for row in all_records if bool(row["short_proposed"])]
        variants = {
            name: _fit_variant(
                variant=name,
                proposals=proposals,
                all_records=all_records,
                scenarios=scenarios,
                seeds=seeds,
                profiles={scenario: profiles[scenario] for scenario in scenarios},
                temporal={scenario: temporal[scenario] for scenario in scenarios},
                topology={scenario: topology[scenario] for scenario in scenarios},
                local_demand={
                    scenario: local_demand[scenario] for scenario in scenarios
                },
                diagnostic=protocol["diagnostic"],
            )
            for name in VARIANT_SPECS
        }
        v72_grid = v72["cities"][str(city)]["variants"][
            "static_plus_topology"
        ]["grid"]
        legacy_grid = variants["topology_squared_normalized"]["grid"]
        reproduction[str(city)] = {
            "topology_grid_matches_v72": _canonical_sha256(legacy_grid)
            == _canonical_sha256(v72_grid),
            "v73_grid_sha256": _canonical_sha256(legacy_grid),
            "v72_grid_sha256": _canonical_sha256(v72_grid),
        }
        city_results[str(city)] = {
            "all_action_group_count": len(all_records),
            "proposal_count": len(proposals),
            "variants": variants,
        }
    integrity = {
        "source_gate_passed": all(source_gate.values()),
        "cache_identity_matches": True,
        "all_cities_present": set(city_results)
        == set(cache_protocol["city_scenarios"]),
        "all_variants_present": all(
            set(row["variants"]) == set(VARIANT_SPECS)
            for row in city_results.values()
        ),
        "topology_baseline_reproduces_v72": all(
            row["topology_grid_matches_v72"] for row in reproduction.values()
        ),
        "sealed_results_absent": all(
            int(value) == 0
            for value in protocol["future_file_counts_at_freeze"].values()
        ),
    }
    integrity["passed"] = all(integrity.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": DECISION,
        "claim_boundary": (
            "Post-v72 adaptation-seed diagnostic only. Variant selection is "
            "development evidence; no closed-loop or sealed-seed claim is authorized."
        ),
        "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
        "cache_sha256": cache_sha,
        "cache_file_count": cache_count,
        "bank_audit": bank_audit,
        "source_gate": source_gate,
        "local_demand_context": {
            scenario: context.to_dict()
            for scenario, context in local_demand.items()
        },
        "reproduction": reproduction,
        "cities": city_results,
        "integrity_gate": integrity,
    }
