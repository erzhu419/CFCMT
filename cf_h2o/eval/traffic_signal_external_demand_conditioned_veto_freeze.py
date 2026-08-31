"""Cross-fit and freeze the 450-second demand-conditioned execution veto."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _merge_city_datasets,
    _sha256,
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
from cf_h2o.traffic_signal.demand_conditioned_target_veto import (
    ARTIFACT_PROTOCOL,
    DemandConditionedTargetVeto,
    DemandConditionedVetoConfig,
)
from cf_h2o.traffic_signal.demand_schedule_context import (
    DEMAND_SCHEDULE_FEATURE_NAMES,
    DEMAND_SCHEDULE_PROTOCOL,
    DemandScheduleProfile,
    build_demand_schedule_profile,
)
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    MODEL_PROTOCOL,
    ProposalConditionalRegressor,
    ProposalConditionalRegressorConfig,
)


RESULT_PROTOCOL = "tsc-v70r66-external-v9-demand-conditioned-450s-veto-freeze-v1"
TRAINING_PROTOCOL = "tsc-v70r66-external-v9-demand-conditioned-450s-veto-training-v1"
AUTHORIZATION_DECISION = (
    "authorize_demand_conditioned_450s_closed_loop_development_only"
)
FAILURE_DECISION = "retain_phase_pressure_and_prohibit_v70_closed_loop_development"


def _profile(payload: Mapping[str, Any]) -> DemandScheduleProfile:
    return DemandScheduleProfile(
        protocol=str(payload["protocol"]),
        horizon_sec=int(payload["horizon_sec"]),
        tls_count=int(payload["tls_count"]),
        scheduled_vehicle_count=float(payload["scheduled_vehicle_count"]),
        input_sha256=str(payload["input_sha256"]),
        feature_names=tuple(str(value) for value in payload["feature_names"]),
        features=tuple(float(value) for value in payload["features"]),
    )


def _model_config(spec: Mapping[str, Any]) -> ProposalConditionalRegressorConfig:
    return ProposalConditionalRegressorConfig(
        learning_rate=float(spec["learning_rate"]),
        max_iter=int(spec["max_iter"]),
        max_leaf_nodes=int(spec["max_leaf_nodes"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        l2_regularization=float(spec["l2_regularization"]),
        random_state=int(spec["random_state"]),
        uncertainty_quantile=float(spec["uncertainty_quantile"]),
    )


def combined_feature_names() -> tuple[str, ...]:
    return (*TARGET_SUPPORT_FEATURES, *DEMAND_SCHEDULE_FEATURE_NAMES)


def combined_feature_matrix(
    records: Sequence[Mapping[str, Any]],
    *,
    profiles: Mapping[str, DemandScheduleProfile],
) -> np.ndarray:
    rows = []
    for record in records:
        scenario = str(record["scenario"])
        if scenario not in profiles:
            raise KeyError(f"proposal record has no demand profile: {scenario}")
        features = dict(record.get("candidate_features", {}))
        missing = [name for name in TARGET_SUPPORT_FEATURES if name not in features]
        if missing:
            raise KeyError(f"proposal record lacks causal features: {missing}")
        rows.append(
            [float(features[name]) for name in TARGET_SUPPORT_FEATURES]
            + profiles[scenario].vector().tolist()
        )
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (len(records), len(combined_feature_names())):
        raise ValueError("demand-conditioned feature matrix changed")
    if not np.isfinite(matrix).all():
        raise ValueError("demand-conditioned feature matrix is nonfinite")
    return matrix


def _fit_oof_fold(
    *,
    heldout_seed: int,
    proposal_records: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, DemandScheduleProfile],
    quantiles: Sequence[float],
    model_spec: Mapping[str, Any],
) -> dict[str, Any]:
    training = [
        row
        for row in proposal_records
        if int(row["simulator_seed"]) != int(heldout_seed)
    ]
    heldout = [
        row
        for row in proposal_records
        if int(row["simulator_seed"]) == int(heldout_seed)
    ]
    if len(training) < 2:
        raise ValueError("demand-conditioned fold has insufficient training rows")
    model = ProposalConditionalRegressor(
        feature_names=combined_feature_names(), config=_model_config(model_spec)
    )
    target = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in training],
        dtype=float,
    )
    fit = model.fit(
        combined_feature_matrix(training, profiles=profiles),
        target,
        sample_weight=_balanced_weights(training),
    )
    training_scores = model.predict_matrix(
        combined_feature_matrix(training, profiles=profiles)
    )
    thresholds = {
        str(float(quantile)): float(np.quantile(training_scores, float(quantile)))
        for quantile in quantiles
    }
    heldout_scores = (
        model.predict_matrix(combined_feature_matrix(heldout, profiles=profiles))
        if heldout
        else np.zeros(0, dtype=float)
    )
    return {
        "heldout_seed": int(heldout_seed),
        "training_rows": len(training),
        "heldout_rows": len(heldout),
        "training_seeds": sorted(
            {int(row["simulator_seed"]) for row in training}
        ),
        "fit": fit,
        "thresholds": thresholds,
        "predictions": [
            {
                "group_id": str(row["group_id"]),
                "simulator_seed": int(row["simulator_seed"]),
                "scenario": str(row["scenario"]),
                "actual_group_normalized_delta": float(
                    row["actual_group_normalized_delta"]
                ),
                "predicted_delta": float(score),
                "fold_thresholds": thresholds,
                "demand_profile_input_sha256": profiles[
                    str(row["scenario"])
                ].input_sha256,
            }
            for row, score in zip(heldout, heldout_scores)
        ],
    }


def fit_city_demand_conditioned_veto(
    *,
    city: str,
    scenarios: Sequence[str],
    all_records: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    profiles: Mapping[str, DemandScheduleProfile],
    training_spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = tuple(dict(row) for row in all_records)
    proposal_records = tuple(
        row for row in records if bool(row.get("short_proposed", False))
    )
    groups = [str(row.get("group_id")) for row in records]
    if (
        not records
        or len(groups) != len(set(groups))
        or {int(row["simulator_seed"]) for row in records} != set(seeds)
        or {str(row["scenario"]) for row in records} != set(scenarios)
        or set(profiles) != set(scenarios)
    ):
        raise ValueError(f"demand-conditioned source records changed for {city}")
    quantiles = tuple(
        float(value) for value in training_spec["acceptance_quantiles"]
    )
    folds = [
        _fit_oof_fold(
            heldout_seed=int(seed),
            proposal_records=proposal_records,
            profiles=profiles,
            quantiles=quantiles,
            model_spec=training_spec["model"],
        )
        for seed in seeds
    ]
    oof_predictions = [row for fold in folds for row in fold["predictions"]]
    if (
        len(oof_predictions) != len(proposal_records)
        or {str(row["group_id"]) for row in oof_predictions}
        != {str(row["group_id"]) for row in proposal_records}
    ):
        raise RuntimeError("demand-conditioned OOF partition is not exact")
    grid = [
        summarize_quantile_candidate(
            oof_predictions,
            all_records=records,
            seeds=seeds,
            scenarios=scenarios,
            acceptance_quantile=quantile,
            selection_rule=training_spec["selection_rule"],
            bootstrap=training_spec["bootstrap"],
        )
        for quantile in quantiles
    ]
    feasible = [row for row in grid if bool(row["feasible"])]
    selected = min(
        feasible,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_delta"]),
            float(row["harmful_group_fraction"]),
            -int(row["retained_group_count"]),
            float(row["acceptance_quantile"]),
        ),
        default=None,
    )
    final_model = ProposalConditionalRegressor(
        feature_names=combined_feature_names(),
        config=_model_config(training_spec["model"]),
    )
    final_target = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in proposal_records],
        dtype=float,
    )
    final_fit = final_model.fit(
        combined_feature_matrix(proposal_records, profiles=profiles),
        final_target,
        sample_weight=_balanced_weights(proposal_records),
    )
    selected_quantile = (
        float(selected["acceptance_quantile"])
        if selected is not None
        else float(training_spec["fallback_quantile"])
    )
    fitted_scores = final_model.predict_matrix(
        combined_feature_matrix(proposal_records, profiles=profiles)
    )
    score_threshold = float(np.quantile(fitted_scores, selected_quantile))
    config = DemandConditionedVetoConfig(
        enabled=selected is not None,
        acceptance_quantile=selected_quantile,
        score_threshold=score_threshold,
        rollout_value_horizon_sec=450,
    )
    scenario_vetoes = {
        scenario: DemandConditionedTargetVeto(
            model=final_model,
            config=config,
            candidate_feature_names=TARGET_SUPPORT_FEATURES,
            demand_profile=profiles[scenario],
        )
        for scenario in scenarios
    }
    summary = {
        "city": city,
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "all_action_group_count": len(records),
        "proposal_training_row_count": len(proposal_records),
        "proposal_fraction": len(proposal_records) / max(len(records), 1),
        "candidate_feature_names": list(TARGET_SUPPORT_FEATURES),
        "demand_feature_names": list(DEMAND_SCHEDULE_FEATURE_NAMES),
        "combined_feature_names": list(combined_feature_names()),
        "model_config": asdict(_model_config(training_spec["model"])),
        "demand_profiles": {
            scenario: profiles[scenario].to_dict() for scenario in scenarios
        },
        "oof_folds": folds,
        "oof_predictions": oof_predictions,
        "gate_selection": {
            "decision": (
                "deploy_demand_conditioned_veto"
                if selected is not None
                else "always_veto_to_phase_pressure"
            ),
            "selected": selected,
            "feasible_candidate_count": len(feasible),
            "grid": grid,
        },
        "final_fit": final_fit,
        "final_config": asdict(config),
        "scenario_veto_diagnostics": {
            scenario: scenario_vetoes[scenario].diagnostics()
            for scenario in scenarios
        },
    }
    artifact = {
        "protocol": ARTIFACT_PROTOCOL,
        "model_protocol": MODEL_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "classification": "target_simulator_labeled_few_shot_adaptation",
        "zero_shot": False,
        "scenario_id_is_model_feature": False,
        "proposal_role": "immutable_v60_hierarchical_cfcmt",
        "veto_role": "demand_conditioned_veto_only_never_action_originator",
        "scenario_vetoes": scenario_vetoes,
        "model": final_model,
        "config": config,
        "selected_gate": summary["gate_selection"],
        "final_fit": final_fit,
    }
    return summary, artifact


def run_demand_conditioned_veto_freeze(
    *,
    cache_root: Path,
    cache_audit_path: Path,
    training_protocol_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    training_protocol = json.loads(
        Path(training_protocol_path).read_text(encoding="utf-8")
    )
    cache_protocol = json.loads(
        Path(training_protocol["cache_protocol"]["path"]).read_text(encoding="utf-8")
    )
    proposal_parent = json.loads(
        Path(proposal_parent_protocol_path).read_text(encoding="utf-8")
    )
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    freeze_audit = json.loads(Path(freeze_audit_path).read_text(encoding="utf-8"))
    expected = training_protocol["evidence"]
    if (
        training_protocol.get("protocol") != TRAINING_PROTOCOL
        or _sha256(Path(training_protocol["cache_protocol"]["path"]))
        != training_protocol["cache_protocol"]["sha256"]
        or _sha256(cache_audit_path) != expected["cache_audit"]["sha256"]
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision")
        != "authorize_demand_conditioned_450s_veto_oof_training_only"
        or not bool(cache_audit.get("integrity_gate", {}).get("passed", False))
        or cache_audit.get("frozen_protocol_sha256")
        != _sha256(Path(training_protocol["cache_protocol"]["path"]))
        or _sha256(proposal_parent_protocol_path)
        != training_protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != training_protocol["hierarchical_freeze_audit"]["sha256"]
        or freeze_audit.get("status") != "PASS"
        or not bool(freeze_audit.get("integrity_gate", {}).get("passed", False))
    ):
        raise ValueError("v70 training evidence chain changed")
    source_gate = {
        path: _sha256(Path(path)) == expected_sha
        for path, expected_sha in training_protocol[
            "runtime_executable_sources"
        ].items()
    }
    if not source_gate or not all(source_gate.values()):
        raise ValueError(f"v70 executable source freeze changed: {source_gate}")
    cache_sha, cache_count = aggregate_cache_sha256(cache_root)
    if (
        cache_sha != cache_audit.get("cache_sha256")
        or cache_count != int(cache_audit.get("cache_file_count", -1))
        or cache_count
        != int(cache_protocol["long_horizon_collection"]["expected_cache_file_count"])
    ):
        raise ValueError("v70 cache identity changed")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite v70 freeze: {output_root}")
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
    if any(
        int(dataset.metadata.get("rollout_value_horizon_sec", -1)) != 450
        for dataset in bank.values()
    ):
        raise ValueError("v70 cache does not have a 450-second value horizon")
    profiles = {
        scenario: _profile(payload)
        for scenario, payload in cache_protocol["demand_schedule_context"][
            "profiles"
        ].items()
    }
    if any(profile.protocol != DEMAND_SCHEDULE_PROTOCOL for profile in profiles.values()):
        raise ValueError("v70 demand profile protocol changed")
    for scenario, profile in profiles.items():
        observed = build_demand_schedule_profile(
            Path(conversion_root) / scenario / f"{scenario}.sumocfg",
            horizon_sec=profile.horizon_sec,
        )
        if observed != profile:
            raise ValueError(f"v70 demand profile input changed: {scenario}")
    output_root.mkdir(parents=True, exist_ok=True)
    city_results = {}
    city_artifacts = {}
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
        records = frozen_proposal_records(
            contrast, frozen_model=frozen_model, scenarios=scenarios
        )
        summary, artifact = fit_city_demand_conditioned_veto(
            city=str(city),
            scenarios=scenarios,
            all_records=records,
            seeds=seeds,
            profiles={scenario: profiles[scenario] for scenario in scenarios},
            training_spec=training_protocol["training"],
        )
        city_root = output_root / str(city)
        city_root.mkdir(parents=True, exist_ok=False)
        model_path = city_root / "model.pkl"
        model_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
        freeze_payload = {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        freeze_path = city_root / "freeze.json"
        _atomic_json(freeze_path, freeze_payload)
        city_results[str(city)] = freeze_payload
        city_artifacts[str(city)] = {
            "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "certificate": {
                "path": str(freeze_path.resolve()),
                "sha256": _sha256(freeze_path),
            },
            "active": bool(summary["final_config"]["enabled"]),
        }
    active_cities = [city for city, row in city_artifacts.items() if row["active"]]
    integrity_gate = {
        "all_cities_present": set(city_results) == set(cache_protocol["city_scenarios"]),
        "all_artifacts_written": all(
            Path(row["model"]["path"]).is_file()
            and Path(row["certificate"]["path"]).is_file()
            for row in city_artifacts.values()
        ),
        "at_least_one_oof_active_city": bool(active_cities),
        "validation_and_prospective_disjoint": not (
            set(seeds) & set(cache_protocol["validation"]["closed_loop_seeds"])
        )
        and not (
            set(seeds)
            & set(cache_protocol["prospective_confirmation"]["closed_loop_seeds"])
        ),
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": (
            AUTHORIZATION_DECISION if integrity_gate["passed"] else FAILURE_DECISION
        ),
        "claim_boundary": (
            "Target-simulator-labeled 450-second OOF veto training only. "
            "Closed-loop efficacy remains untested."
        ),
        "training_protocol_sha256": _sha256(training_protocol_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "cache_sha256": cache_sha,
        "cache_file_count": cache_count,
        "bank_audit": bank_audit,
        "active_cities": active_cities,
        "city_artifacts": city_artifacts,
        "city_results": city_results,
        "integrity_gate": integrity_gate,
    }
