"""OOF diagnostic for deployment-causal historical local-demand context."""

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
from sklearn.ensemble import HistGradientBoostingRegressor

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_demand_conditioned_veto_freeze import (
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
    TLS_TOPOLOGY_FEATURE_NAMES,
    TLSLocalTopologyContext,
    TemporalDemandContext,
    build_temporal_demand_context,
    build_tls_local_topology_context,
)
from cf_h2o.traffic_signal.historical_local_demand_context import (
    HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES,
    HistoricalLocalDemandContext,
    build_historical_local_demand_context,
)
from cf_h2o.traffic_signal.target_action_support import TARGET_SUPPORT_FEATURES


DIAGNOSTIC_PROTOCOL = (
    "tsc-v76r72-external-v9-historical-prefix-model-diagnostic-v1"
)
RESULT_PROTOCOL = (
    "tsc-v76r72-external-v9-historical-prefix-model-diagnostic-result-v1"
)
DECISION = "diagnostic_only_no_closed_loop_or_sealed_seed_authorization"
BASE_VARIANT = "state_topology_raw_scaled"
HISTORICAL_VARIANT = "state_topology_historical_raw_scaled"
VARIANTS = (BASE_VARIANT, HISTORICAL_VARIANT)


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
    group = str(record["group_id"])
    if ":" not in group:
        raise ValueError(f"historical diagnostic group lacks TLS: {group}")
    return group.rsplit(":", 1)[-1]


def feature_names(variant: str) -> tuple[str, ...]:
    if variant not in VARIANTS:
        raise KeyError(variant)
    names = (*TARGET_SUPPORT_FEATURES, *TLS_TOPOLOGY_FEATURE_NAMES)
    if variant == HISTORICAL_VARIANT:
        names = (*names, *HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES)
    if len(names) != len(set(names)):
        raise ValueError("historical diagnostic feature names overlap")
    return names


def feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    profiles: Mapping[str, DemandScheduleProfile],
    temporal: Mapping[str, TemporalDemandContext],
    topology: Mapping[str, TLSLocalTopologyContext],
    historical: Mapping[str, HistoricalLocalDemandContext],
) -> np.ndarray:
    state = v72_feature_matrix(
        records,
        variant="state_only",
        profiles=profiles,
        temporal=temporal,
        topology=topology,
    )
    topology_rows = (
        np.asarray(
            [
                topology[str(row["scenario"])].vector_for(_tls_id(row)).tolist()
                for row in records
            ],
            dtype=float,
        )
        if records
        else np.empty((0, len(TLS_TOPOLOGY_FEATURE_NAMES)), dtype=float)
    )
    matrices = [state, topology_rows]
    if variant == HISTORICAL_VARIANT:
        history_rows = (
            np.asarray(
                [
                    historical[str(row["scenario"])]
                    .vector_at(_tls_id(row), float(row["time_sec"]))
                    .tolist()
                    for row in records
                ],
                dtype=float,
            )
            if records
            else np.empty(
                (0, len(HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES)), dtype=float
            )
        )
        matrices.append(history_rows)
    matrix = np.concatenate(matrices, axis=1)
    if (
        matrix.shape != (len(records), len(feature_names(variant)))
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("historical diagnostic matrix changed")
    return matrix


def _raw_scenario_scaled_target(
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, float]]:
    transformed = np.zeros(len(records), dtype=float)
    scales = {}
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
    historical: Mapping[str, HistoricalLocalDemandContext],
    diagnostic: Mapping[str, Any],
) -> dict[str, Any]:
    quantiles = tuple(float(value) for value in diagnostic["acceptance_quantiles"])
    model_spec = diagnostic["model"]
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
        x_train = feature_matrix(
            train,
            variant=variant,
            profiles=profiles,
            temporal=temporal,
            topology=topology,
            historical=historical,
        )
        x_heldout = feature_matrix(
            heldout,
            variant=variant,
            profiles=profiles,
            temporal=temporal,
            topology=topology,
            historical=historical,
        )
        target, scales = _raw_scenario_scaled_target(train)
        estimator = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(model_spec["learning_rate"]),
            max_iter=int(model_spec["max_iter"]),
            max_leaf_nodes=int(model_spec["max_leaf_nodes"]),
            min_samples_leaf=int(model_spec["min_samples_leaf"]),
            l2_regularization=float(model_spec["l2_regularization"]),
            early_stopping=False,
            random_state=int(model_spec["random_state"]),
        )
        estimator.fit(x_train, target, sample_weight=_balanced_weights(train))
        train_scores = np.asarray(estimator.predict(x_train), dtype=float)
        heldout_scores = (
            np.asarray(estimator.predict(x_heldout), dtype=float)
            if heldout
            else np.zeros(0, dtype=float)
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
                "target_scales": scales,
                "training_score_std": float(np.std(train_scores)),
                "prediction_sha256": _canonical(fold_predictions),
            }
        )
    if len(predictions) != len(proposals) or len(
        {str(row["group_id"]) for row in predictions}
    ) != len(proposals):
        raise RuntimeError("historical OOF partition changed")
    actual = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in predictions],
        dtype=float,
    )
    scores = np.asarray(
        [float(row["predicted_delta"]) for row in predictions], dtype=float
    )
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
        feasible or grid,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_delta"]),
            float(row["harmful_group_fraction"]),
            -int(row["retained_group_count"]),
            float(row["acceptance_quantile"]),
        ),
    )
    return {
        "variant": variant,
        "feature_names": list(feature_names(variant)),
        "feature_count": len(feature_names(variant)),
        "information_budget": (
            "current_candidate_state_static_topology_and_arrivals_at_or_before_t"
            if variant == HISTORICAL_VARIANT
            else "current_candidate_state_and_static_topology"
        ),
        "folds": folds,
        "oof_prediction_sha256": _canonical(predictions),
        "oof_row_count": len(predictions),
        "oof_spearman": _finite_spearman(scores, actual),
        "grid": grid,
        "feasible_candidate_count": len(feasible),
        "passed_selection_rule": bool(feasible),
        "selected_candidate": best,
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
    if (
        protocol.get("protocol") != DIAGNOSTIC_PROTOCOL
        or _sha256(cache_protocol_path) != protocol["cache_protocol"]["sha256"]
        or _sha256(cache_audit_path) != protocol["cache_audit"]["sha256"]
        or cache_audit.get("status") != "PASS"
        or _sha256(proposal_parent_protocol_path)
        != protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != protocol["hierarchical_freeze_audit"]["sha256"]
        or freeze_audit.get("status") != "PASS"
    ):
        raise ValueError("historical diagnostic parent evidence changed")
    source_gate = {
        path: Path(path).is_file() and _sha256(Path(path)) == expected
        for path, expected in protocol["runtime_executable_sources"].items()
    }
    if not source_gate or not all(source_gate.values()):
        raise ValueError(f"historical diagnostic sources changed: {source_gate}")
    cache_sha, cache_count = aggregate_cache_sha256(cache_root)
    if (
        cache_sha != cache_audit["cache_sha256"]
        or cache_count != cache_audit["cache_file_count"]
    ):
        raise ValueError("historical diagnostic cache changed")
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
    historical = {}
    for scenario in profiles:
        sumocfg = Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        temporal[scenario] = build_temporal_demand_context(sumocfg)
        topology[scenario] = build_tls_local_topology_context(sumocfg)
        historical[scenario] = build_historical_local_demand_context(sumocfg)
    seeds = tuple(int(value) for value in collection["seeds"])
    city_results = {}
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
            variant: _fit_variant(
                variant=variant,
                proposals=proposals,
                all_records=all_records,
                scenarios=scenarios,
                seeds=seeds,
                profiles={scenario: profiles[scenario] for scenario in scenarios},
                temporal={scenario: temporal[scenario] for scenario in scenarios},
                topology={scenario: topology[scenario] for scenario in scenarios},
                historical={
                    scenario: historical[scenario] for scenario in scenarios
                },
                diagnostic=protocol["diagnostic"],
            )
            for variant in VARIANTS
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
        "only_predeclared_variants_present": all(
            set(row["variants"]) == set(VARIANTS)
            for row in city_results.values()
        ),
        "historical_contract_has_no_future_terms": all(
            marker not in name
            for name in HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES
            for marker in ("next", "remaining", "future")
        ),
        "sealed_outputs_absent_at_freeze": all(
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
            "Adaptation-seed diagnostic only. Exact future departures and routes "
            "are excluded from runtime features; sealed seeds remain untouched."
        ),
        "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
        "cache_sha256": cache_sha,
        "cache_file_count": cache_count,
        "bank_audit": bank_audit,
        "source_gate": source_gate,
        "historical_context": {
            scenario: context.to_dict()
            for scenario, context in historical.items()
        },
        "cities": city_results,
        "integrity_gate": integrity,
    }
