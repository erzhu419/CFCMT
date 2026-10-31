"""Seed-LOO module diagnostic after the disclosed v70 negative result."""

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
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    TARGET_SUPPORT_FEATURES,
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
from cf_h2o.traffic_signal.demand_schedule_context import (
    DEMAND_SCHEDULE_FEATURE_NAMES,
    DemandScheduleProfile,
)
from cf_h2o.traffic_signal.demand_timeline_context import (
    TEMPORAL_DEMAND_FEATURE_NAMES,
    TLS_TOPOLOGY_FEATURE_NAMES,
    TLSLocalTopologyContext,
    TemporalDemandContext,
    build_temporal_demand_context,
    build_tls_local_topology_context,
)
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
)


RESULT_PROTOCOL = "tsc-v72r68-external-v9-spatiotemporal-veto-diagnostic-result-v2"
DIAGNOSTIC_PROTOCOL = "tsc-v72r68-external-v9-spatiotemporal-veto-diagnostic-v2"
DECISION = "diagnostic_only_no_closed_loop_or_validation_authorization"
VARIANTS = (
    "state_only",
    "static_global",
    "static_plus_temporal",
    "static_plus_topology",
    "static_plus_spatiotemporal",
)


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def feature_names(variant: str) -> tuple[str, ...]:
    names = tuple(TARGET_SUPPORT_FEATURES)
    if variant != "state_only":
        names = (*names, *DEMAND_SCHEDULE_FEATURE_NAMES)
    if variant in {"static_plus_temporal", "static_plus_spatiotemporal"}:
        names = (*names, *TEMPORAL_DEMAND_FEATURE_NAMES)
    if variant in {"static_plus_topology", "static_plus_spatiotemporal"}:
        names = (*names, *TLS_TOPOLOGY_FEATURE_NAMES)
    if variant not in VARIANTS or len(names) != len(set(names)):
        raise ValueError(f"invalid spatiotemporal diagnostic variant: {variant}")
    return names


def _tls_id(record: Mapping[str, Any]) -> str:
    group = str(record["group_id"])
    if ":" not in group:
        raise ValueError(f"diagnostic group id lacks TLS: {group}")
    return group.rsplit(":", 1)[-1]


def feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    variant: str,
    profiles: Mapping[str, DemandScheduleProfile],
    temporal: Mapping[str, TemporalDemandContext],
    topology: Mapping[str, TLSLocalTopologyContext],
) -> np.ndarray:
    rows = []
    for record in records:
        scenario = str(record["scenario"])
        candidate = dict(record.get("candidate_features", {}))
        missing = [name for name in TARGET_SUPPORT_FEATURES if name not in candidate]
        if missing:
            raise KeyError(f"diagnostic candidate features missing: {missing}")
        row = [float(candidate[name]) for name in TARGET_SUPPORT_FEATURES]
        if variant != "state_only":
            row.extend(profiles[scenario].vector().tolist())
        if variant in {"static_plus_temporal", "static_plus_spatiotemporal"}:
            row.extend(
                temporal[scenario].vector_at(float(record["time_sec"])).tolist()
            )
        if variant in {"static_plus_topology", "static_plus_spatiotemporal"}:
            row.extend(topology[scenario].vector_for(_tls_id(record)).tolist())
        rows.append(row)
    matrix = (
        np.asarray(rows, dtype=float)
        if rows
        else np.empty((0, len(feature_names(variant))), dtype=float)
    )
    if matrix.shape != (len(records), len(feature_names(variant))):
        raise ValueError("spatiotemporal diagnostic feature matrix changed")
    if not np.isfinite(matrix).all():
        raise ValueError("spatiotemporal diagnostic matrix is nonfinite")
    return matrix


def _fit_variant(
    *,
    variant: str,
    records: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    profiles: Mapping[str, DemandScheduleProfile],
    temporal: Mapping[str, TemporalDemandContext],
    topology: Mapping[str, TLSLocalTopologyContext],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    quantiles = tuple(float(value) for value in training["acceptance_quantiles"])
    predictions = []
    folds = []
    for heldout_seed in seeds:
        train = [
            row for row in records if int(row["simulator_seed"]) != int(heldout_seed)
        ]
        heldout = [
            row for row in records if int(row["simulator_seed"]) == int(heldout_seed)
        ]
        model = ProposalConditionalRegressor(
            feature_names=feature_names(variant),
            config=_model_config(training["model"]),
        )
        train_target = np.asarray(
            [float(row["actual_group_normalized_delta"]) for row in train],
            dtype=float,
        )
        fit = model.fit(
            feature_matrix(
                train,
                variant=variant,
                profiles=profiles,
                temporal=temporal,
                topology=topology,
            ),
            train_target,
            sample_weight=_balanced_weights(train),
        )
        train_scores = model.predict_matrix(
            feature_matrix(
                train,
                variant=variant,
                profiles=profiles,
                temporal=temporal,
                topology=topology,
            )
        )
        thresholds = {
            str(quantile): float(np.quantile(train_scores, quantile))
            for quantile in quantiles
        }
        heldout_scores = (
            model.predict_matrix(
                feature_matrix(
                    heldout,
                    variant=variant,
                    profiles=profiles,
                    temporal=temporal,
                    topology=topology,
                )
            )
            if heldout
            else np.zeros(0, dtype=float)
        )
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
                "fit": fit,
                "thresholds": thresholds,
                "prediction_sha256": _canonical_sha256(fold_predictions),
            }
        )
    if len(predictions) != len(records) or len(
        {str(row["group_id"]) for row in predictions}
    ) != len(records):
        raise RuntimeError("spatiotemporal OOF partition is not exact")
    actual = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in predictions],
        dtype=float,
    )
    scores = np.asarray(
        [float(row["predicted_delta"]) for row in predictions], dtype=float
    )
    rank = float(spearmanr(scores, actual).statistic) if len(scores) >= 3 else 0.0
    grid = [
        summarize_quantile_candidate(
            predictions,
            all_records=all_records,
            seeds=seeds,
            scenarios=scenarios,
            acceptance_quantile=quantile,
            selection_rule=training["selection_rule"],
            bootstrap=training["bootstrap"],
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
        "feature_names": list(feature_names(variant)),
        "feature_count": len(feature_names(variant)),
        "folds": folds,
        "oof_prediction_sha256": _canonical_sha256(predictions),
        "oof_row_count": len(predictions),
        "oof_rmse": float(np.sqrt(np.mean((scores - actual) ** 2))),
        "oof_spearman": rank,
        "grid": grid,
        "feasible_candidate_count": len(feasible),
        "best_candidate": best,
        "passed_original_v70_rule": bool(feasible),
    }


def _oracle_variant(
    *,
    records: Sequence[Mapping[str, Any]],
    all_records: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    quantiles = tuple(float(value) for value in training["acceptance_quantiles"])
    predictions = []
    for heldout_seed in seeds:
        train_values = np.asarray(
            [
                float(row["actual_group_normalized_delta"])
                for row in records
                if int(row["simulator_seed"]) != int(heldout_seed)
            ],
            dtype=float,
        )
        thresholds = {
            str(quantile): float(np.quantile(train_values, quantile))
            for quantile in quantiles
        }
        predictions.extend(
            {
                "group_id": str(row["group_id"]),
                "simulator_seed": int(row["simulator_seed"]),
                "scenario": str(row["scenario"]),
                "actual_group_normalized_delta": float(
                    row["actual_group_normalized_delta"]
                ),
                "predicted_delta": float(row["actual_group_normalized_delta"]),
                "fold_thresholds": thresholds,
            }
            for row in records
            if int(row["simulator_seed"]) == int(heldout_seed)
        )
    grid = [
        summarize_quantile_candidate(
            predictions,
            all_records=all_records,
            seeds=seeds,
            scenarios=scenarios,
            acceptance_quantile=quantile,
            selection_rule=training["selection_rule"],
            bootstrap=training["bootstrap"],
        )
        for quantile in quantiles
    ]
    return {
        "classification": "counterfactual_label_oracle_not_deployable",
        "grid": grid,
        "best_candidate": min(grid, key=lambda row: float(row["robust_score"])),
        "feasible_candidate_count": sum(bool(row["feasible"]) for row in grid),
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
    v70_result_path: Path,
    workers: int,
) -> dict[str, Any]:
    diagnostic = json.loads(
        Path(diagnostic_protocol_path).read_text(encoding="utf-8")
    )
    cache_protocol_path = Path(diagnostic["cache_protocol"]["path"])
    cache_protocol = json.loads(cache_protocol_path.read_text(encoding="utf-8"))
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    proposal_parent = json.loads(
        Path(proposal_parent_protocol_path).read_text(encoding="utf-8")
    )
    freeze_audit = json.loads(Path(freeze_audit_path).read_text(encoding="utf-8"))
    v70 = json.loads(Path(v70_result_path).read_text(encoding="utf-8"))
    if (
        diagnostic.get("protocol") != DIAGNOSTIC_PROTOCOL
        or _sha256(cache_protocol_path) != diagnostic["cache_protocol"]["sha256"]
        or _sha256(cache_audit_path) != diagnostic["cache_audit"]["sha256"]
        or cache_audit.get("status") != "PASS"
        or _sha256(v70_result_path) != diagnostic["v70_negative_result"]["sha256"]
        or v70.get("status") != "FAIL"
        or v70.get("active_cities") != []
        or _sha256(proposal_parent_protocol_path)
        != diagnostic["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != diagnostic["hierarchical_freeze_audit"]["sha256"]
        or freeze_audit.get("status") != "PASS"
    ):
        raise ValueError("v72 diagnostic evidence chain changed")
    source_gate = {
        path: Path(path).is_file() and _sha256(Path(path)) == expected
        for path, expected in diagnostic["runtime_executable_sources"].items()
    }
    if not source_gate or not all(source_gate.values()):
        raise ValueError(f"v72 diagnostic sources changed: {source_gate}")
    cache_sha, cache_count = aggregate_cache_sha256(cache_root)
    if (
        cache_sha != cache_audit["cache_sha256"]
        or cache_count != cache_audit["cache_file_count"]
    ):
        raise ValueError("v72 diagnostic cache changed")
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
    temporal = {
        scenario: build_temporal_demand_context(
            Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        )
        for scenario in profiles
    }
    topology = {
        scenario: build_tls_local_topology_context(
            Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        )
        for scenario in profiles
    }
    city_results = {}
    reproduction = {}
    seeds = tuple(int(value) for value in collection["seeds"])
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
                records=proposals,
                all_records=all_records,
                scenarios=scenarios,
                seeds=seeds,
                profiles={scenario: profiles[scenario] for scenario in scenarios},
                temporal={scenario: temporal[scenario] for scenario in scenarios},
                topology={scenario: topology[scenario] for scenario in scenarios},
                training=diagnostic["diagnostic"],
            )
            for variant in VARIANTS
        }
        v70_grid = v70["city_results"][str(city)]["gate_selection"]["grid"]
        reproduction[str(city)] = {
            "static_global_grid_matches_v70": _canonical_sha256(
                variants["static_global"]["grid"]
            )
            == _canonical_sha256(v70_grid),
            "diagnostic_grid_sha256": _canonical_sha256(
                variants["static_global"]["grid"]
            ),
            "v70_grid_sha256": _canonical_sha256(v70_grid),
        }
        city_results[str(city)] = {
            "all_action_group_count": len(all_records),
            "proposal_count": len(proposals),
            "variants": variants,
            "counterfactual_label_oracle": _oracle_variant(
                records=proposals,
                all_records=all_records,
                scenarios=scenarios,
                seeds=seeds,
                training=diagnostic["diagnostic"],
            ),
        }
    integrity = {
        "source_gate_passed": all(source_gate.values()),
        "cache_identity_matches": True,
        "all_cities_present": set(city_results)
        == set(cache_protocol["city_scenarios"]),
        "all_variants_present": all(
            set(row["variants"]) == set(VARIANTS) for row in city_results.values()
        ),
        "static_global_reproduces_v70": all(
            row["static_global_grid_matches_v70"] for row in reproduction.values()
        ),
        "sealed_results_absent": all(
            int(value) == 0
            for value in diagnostic["future_file_counts_at_freeze"].values()
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
            "Post-v70 disclosed adaptation-seed module diagnostic only. The oracle "
            "uses counterfactual labels and is not deployable. No closed-loop, "
            "validation, or prospective outcome is authorized."
        ),
        "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
        "cache_sha256": cache_sha,
        "cache_file_count": cache_count,
        "bank_audit": bank_audit,
        "source_gate": source_gate,
        "context": {
            "temporal": {
                scenario: value.to_dict() for scenario, value in temporal.items()
            },
            "topology": {
                scenario: value.to_dict() for scenario, value in topology.items()
            },
        },
        "reproduction": reproduction,
        "cities": city_results,
        "integrity_gate": integrity,
    }
