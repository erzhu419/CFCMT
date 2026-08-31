"""Freeze the post-v66 proposal-conditional long-horizon target veto."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    FAILURE_DECISION as V66_FAILURE_DECISION,
    RESULT_PROTOCOL as V66_RESULT_PROTOCOL,
    _bootstrap_seed_means,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    MODEL_PROTOCOL,
    ProposalConditionalRegressor,
    ProposalConditionalRegressorConfig,
    ProposalConditionalTargetVeto,
    ProposalConditionalVetoConfig,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    TARGET_SUPPORT_FEATURES,
)
from scripts.cluster.audit_tsc_external_long_horizon_target_veto_cache import (
    AUDIT_DECISION as CACHE_AUDIT_DECISION,
    AUDIT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v67r63-external-v9-proposal-conditional-veto-freeze-v1"
ARTIFACT_PROTOCOL = "cfcmt-proposal-conditional-target-veto-model-v1"
AUTHORIZATION_DECISION = (
    "authorize_proposal_conditional_target_veto_closed_loop_development"
)
FAILURE_DECISION = "retain_phase_pressure_and_prohibit_successor_development"
MODEL_RANDOM_SEED = 20260803


def _model_config() -> ProposalConditionalRegressorConfig:
    return ProposalConditionalRegressorConfig(
        max_iter=100,
        max_leaf_nodes=7,
        min_samples_leaf=12,
        l2_regularization=16.0,
        random_state=MODEL_RANDOM_SEED,
    )


def _feature_matrix(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    rows = []
    for record in records:
        features = dict(record.get("candidate_features", {}))
        missing = [name for name in TARGET_SUPPORT_FEATURES if name not in features]
        if missing:
            raise KeyError(f"proposal record lacks causal features: {missing}")
        rows.append([float(features[name]) for name in TARGET_SUPPORT_FEATURES])
    matrix = np.asarray(rows, dtype=float)
    if matrix.shape != (len(records), len(TARGET_SUPPORT_FEATURES)):
        raise ValueError("proposal-conditional feature matrix changed")
    return matrix


def _balanced_weights(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    strata = [
        (int(row["simulator_seed"]), str(row["scenario"])) for row in records
    ]
    weights = np.zeros(len(records), dtype=float)
    unique = sorted(set(strata))
    for stratum in unique:
        indices = [index for index, value in enumerate(strata) if value == stratum]
        weights[indices] = 1.0 / max(len(unique) * len(indices), 1)
    weights *= weights.size / max(float(weights.sum()), 1e-12)
    return weights


def _fit_oof_fold(
    *,
    heldout_seed: int,
    proposal_records: Sequence[Mapping[str, Any]],
    quantiles: Sequence[float],
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
        raise ValueError("proposal-conditional fold has insufficient training rows")
    model = ProposalConditionalRegressor(
        feature_names=TARGET_SUPPORT_FEATURES, config=_model_config()
    )
    target = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in training],
        dtype=float,
    )
    fit = model.fit(
        _feature_matrix(training),
        target,
        sample_weight=_balanced_weights(training),
    )
    training_scores = model.predict_matrix(_feature_matrix(training))
    thresholds = {
        str(float(quantile)): float(np.quantile(training_scores, float(quantile)))
        for quantile in quantiles
    }
    heldout_scores = (
        model.predict_matrix(_feature_matrix(heldout))
        if heldout
        else np.zeros(0, dtype=float)
    )
    predictions = [
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
    return {
        "heldout_seed": int(heldout_seed),
        "training_rows": len(training),
        "heldout_rows": len(heldout),
        "training_seeds": sorted(
            {int(row["simulator_seed"]) for row in training}
        ),
        "fit": fit,
        "thresholds": thresholds,
        "predictions": predictions,
    }


def summarize_quantile_candidate(
    oof_predictions: Sequence[Mapping[str, Any]],
    *,
    all_records: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    scenarios: Sequence[str],
    acceptance_quantile: float,
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    expected_strata = {
        (int(seed), str(scenario)) for seed in seeds for scenario in scenarios
    }
    denominator = {
        stratum: sum(
            int(row["simulator_seed"]) == stratum[0]
            and str(row["scenario"]) == stratum[1]
            for row in all_records
        )
        for stratum in expected_strata
    }
    if not denominator or any(value <= 0 for value in denominator.values()):
        raise ValueError("proposal-conditional denominator matrix is incomplete")
    key = str(float(acceptance_quantile))
    evaluated = [
        {
            **dict(row),
            "accepted": float(row["predicted_delta"])
            <= float(row["fold_thresholds"][key]),
        }
        for row in oof_predictions
    ]
    stratum_means = {
        stratum: float(
            sum(
                float(row["actual_group_normalized_delta"])
                for row in evaluated
                if bool(row["accepted"])
                and (
                    int(row["simulator_seed"]),
                    str(row["scenario"]),
                )
                == stratum
            )
            / denominator[stratum]
        )
        for stratum in sorted(expected_strata)
    }
    seed_means = {
        int(seed): float(
            np.mean(
                [stratum_means[(int(seed), str(scenario))] for scenario in scenarios]
            )
        )
        for seed in seeds
    }
    retained = [row for row in evaluated if bool(row["accepted"])]
    retained_seeds = {int(row["simulator_seed"]) for row in retained}
    harmful_fraction = (
        float(
            np.mean(
                [
                    float(row["actual_group_normalized_delta"]) > 0.0
                    for row in retained
                ]
            )
        )
        if retained
        else 1.0
    )
    interval = _bootstrap_seed_means(
        seed_means,
        replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    mean_delta = float(np.mean(list(seed_means.values())))
    seed_std = float(np.std(list(seed_means.values())))
    worst_seed = float(max(seed_means.values()))
    gates = {
        "minimum_retained_groups": len(retained)
        >= int(selection_rule["minimum_retained_groups_per_city"]),
        "minimum_retained_seed_fraction": len(retained_seeds) / max(len(seeds), 1)
        >= float(selection_rule["minimum_retained_seed_fraction"]),
        "mean_delta": mean_delta
        <= float(selection_rule["maximum_equal_seed_scenario_mean_delta"]),
        "bootstrap_upper": float(interval["ci95"][1])
        <= float(selection_rule["maximum_bootstrap_95pct_upper_mean_delta"]),
        "worst_seed": worst_seed
        <= float(selection_rule["maximum_worst_seed_mean_delta"]),
        "harmful_fraction": harmful_fraction
        <= float(selection_rule["maximum_harmful_group_fraction"]),
    }
    gates["passed"] = all(gates.values())
    return {
        "key": f"q{str(float(acceptance_quantile)).replace('.', 'p')}",
        "acceptance_quantile": float(acceptance_quantile),
        "retained_group_count": len(retained),
        "retained_seed_count": len(retained_seeds),
        "retained_seed_fraction": len(retained_seeds) / max(len(seeds), 1),
        "harmful_group_fraction": harmful_fraction,
        "mean_delta": mean_delta,
        "seed_delta_std": seed_std,
        "worst_seed_delta": worst_seed,
        "robust_score": mean_delta + 0.5 * seed_std + max(worst_seed, 0.0),
        "bootstrap": interval,
        "stratum_mean_delta": {
            f"seed{seed}::{scenario}": value
            for (seed, scenario), value in stratum_means.items()
        },
        "seed_mean_delta": {str(seed): value for seed, value in seed_means.items()},
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def fit_city_proposal_conditional_veto(
    *,
    city: str,
    certificate: Mapping[str, Any],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    successor_spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    all_records = tuple(certificate.get("oof_records", ()))
    proposal_records = tuple(
        row for row in all_records if bool(row.get("short_proposed", False))
    )
    groups = [str(row.get("group_id")) for row in all_records]
    if (
        not all_records
        or len(groups) != len(set(groups))
        or {int(row["simulator_seed"]) for row in all_records} != set(seeds)
        or {str(row["scenario"]) for row in all_records} != set(scenarios)
    ):
        raise ValueError(f"proposal-conditional source records changed for {city}")
    quantiles = tuple(
        float(value) for value in successor_spec["acceptance_quantiles"]
    )
    folds = [
        _fit_oof_fold(
            heldout_seed=int(seed),
            proposal_records=proposal_records,
            quantiles=quantiles,
        )
        for seed in seeds
    ]
    oof_predictions = [row for fold in folds for row in fold["predictions"]]
    if (
        len(oof_predictions) != len(proposal_records)
        or {str(row["group_id"]) for row in oof_predictions}
        != {str(row["group_id"]) for row in proposal_records}
    ):
        raise RuntimeError("proposal-conditional OOF partition is not exact")
    grid = [
        summarize_quantile_candidate(
            oof_predictions,
            all_records=all_records,
            seeds=seeds,
            scenarios=scenarios,
            acceptance_quantile=quantile,
            selection_rule=successor_spec["selection_rule"],
            bootstrap=successor_spec["bootstrap"],
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
        feature_names=TARGET_SUPPORT_FEATURES, config=_model_config()
    )
    final_target = np.asarray(
        [float(row["actual_group_normalized_delta"]) for row in proposal_records],
        dtype=float,
    )
    final_fit = final_model.fit(
        _feature_matrix(proposal_records),
        final_target,
        sample_weight=_balanced_weights(proposal_records),
    )
    selected_quantile = (
        float(selected["acceptance_quantile"])
        if selected is not None
        else float(successor_spec["fallback_quantile"])
    )
    fitted_scores = final_model.predict_matrix(_feature_matrix(proposal_records))
    score_threshold = float(np.quantile(fitted_scores, selected_quantile))
    config = ProposalConditionalVetoConfig(
        enabled=selected is not None,
        acceptance_quantile=selected_quantile,
        score_threshold=score_threshold,
        rollout_value_horizon_sec=300,
    )
    veto = ProposalConditionalTargetVeto(model=final_model, config=config)
    summary = {
        "city": city,
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "all_action_group_count": len(all_records),
        "proposal_training_row_count": len(proposal_records),
        "proposal_fraction": len(proposal_records) / max(len(all_records), 1),
        "feature_names": list(TARGET_SUPPORT_FEATURES),
        "model_config": asdict(_model_config()),
        "oof_folds": folds,
        "oof_predictions": oof_predictions,
        "gate_selection": {
            "decision": (
                "deploy_proposal_conditional_veto"
                if selected is not None
                else "always_veto_to_phase_pressure"
            ),
            "selected": selected,
            "feasible_candidate_count": len(feasible),
            "grid": grid,
        },
        "final_fit": final_fit,
        "final_config": asdict(config),
        "veto_diagnostics": veto.diagnostics(),
    }
    artifact = {
        "protocol": ARTIFACT_PROTOCOL,
        "model_protocol": MODEL_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "classification": "target_simulator_labeled_few_shot_adaptation",
        "zero_shot": False,
        "proposal_role": "immutable_v60_hierarchical_cfcmt",
        "veto_role": "proposal_conditional_veto_only_never_action_originator",
        "proposal_conditional_veto": veto,
        "model": final_model,
        "selected_gate": summary["gate_selection"],
        "final_fit": final_fit,
    }
    return summary, artifact


def run_proposal_conditional_veto_freeze(
    *,
    failed_v66_result_path: Path,
    failed_v66_artifact_root: Path,
    cache_audit_path: Path,
    successor_protocol_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    failed = json.loads(Path(failed_v66_result_path).read_text(encoding="utf-8"))
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    successor = json.loads(Path(successor_protocol_path).read_text(encoding="utf-8"))
    if (
        failed.get("protocol") != V66_RESULT_PROTOCOL
        or failed.get("status") != "FAIL"
        or failed.get("decision") != V66_FAILURE_DECISION
        or failed.get("active_cities") != []
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision") != CACHE_AUDIT_DECISION
        or failed.get("cache_audit_sha256") != _sha256(cache_audit_path)
        or successor.get("failed_v66_result", {}).get("sha256")
        != _sha256(failed_v66_result_path)
        or successor.get("cache_audit", {}).get("sha256")
        != _sha256(cache_audit_path)
    ):
        raise ValueError("proposal-conditional successor evidence chain changed")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite successor freeze: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(value) for value in successor["training"]["seeds"])
    city_results = {}
    city_artifacts = {}
    for city, scenarios in sorted(successor["training"]["city_scenarios"].items()):
        declared = failed["city_artifacts"][city]
        certificate_path = Path(failed_v66_artifact_root) / city / "freeze.json"
        if _sha256(certificate_path) != declared["certificate"]["sha256"]:
            raise ValueError(f"failed v66 certificate changed for {city}")
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        summary, artifact = fit_city_proposal_conditional_veto(
            city=city,
            certificate=certificate,
            scenarios=tuple(str(value) for value in scenarios),
            seeds=seeds,
            successor_spec=successor["training"],
        )
        city_root = output_root / city
        city_root.mkdir(parents=True, exist_ok=False)
        model_path = city_root / "model.pkl"
        model_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
        freeze_payload = {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "failed_v66_certificate_sha256": _sha256(certificate_path),
            **summary,
        }
        freeze_path = city_root / "freeze.json"
        _atomic_json(freeze_path, freeze_payload)
        city_results[city] = freeze_payload
        city_artifacts[city] = {
            "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "certificate": {
                "path": str(freeze_path.resolve()),
                "sha256": _sha256(freeze_path),
            },
            "active": bool(summary["final_config"]["enabled"]),
        }
    active_cities = [city for city, row in city_artifacts.items() if row["active"]]
    integrity_gate = {
        "all_cities_present": set(city_results)
        == set(successor["training"]["city_scenarios"]),
        "all_city_artifacts_written": all(
            Path(row["model"]["path"]).is_file()
            and Path(row["certificate"]["path"]).is_file()
            for row in city_artifacts.values()
        ),
        "at_least_one_city_has_feasible_conditional_veto": bool(active_cities),
        "closed_loop_development_results_absent": int(
            successor["future_result_file_count_at_freeze"]
        )
        == 0,
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
            "Post-v66 outcome-informed method development on the same six target "
            "simulator seeds. Efficacy requires independent closed-loop validation."
        ),
        "failed_v66_result_sha256": _sha256(failed_v66_result_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "successor_protocol_sha256": _sha256(successor_protocol_path),
        "active_cities": active_cities,
        "city_artifacts": city_artifacts,
        "city_results": city_results,
        "integrity_gate": integrity_gate,
    }
