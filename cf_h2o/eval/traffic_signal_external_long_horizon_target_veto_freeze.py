"""Cross-fit and freeze the v65 target-labeled long-horizon execution veto."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    TARGET_SUPPORT_FEATURES,
    _candidate_scores,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    _relative_contrast_rule_gap,
    _subset_contrast,
    hierarchical_guard_decision_v3,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.action_ranker import (
    ActionAdvantageConfig,
    PairwiseActionAdvantageRegressor,
    group_normalized_action_target,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.long_horizon_target_veto import (
    LongHorizonTargetVeto,
    LongHorizonTargetVetoConfig,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.cluster.audit_tsc_external_long_horizon_target_veto_cache import (
    AUDIT_DECISION as CACHE_AUDIT_DECISION,
    AUDIT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_hierarchical_closed_loop_protocol import (
    PROTOCOL as PARENT_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_long_horizon_target_veto_protocol import (
    MODEL_FAMILY,
    PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v66r62-external-v9-long-horizon-target-veto-freeze-v1"
MODEL_PROTOCOL = "cfcmt-long-horizon-target-veto-model-v1"
AUTHORIZATION_DECISION = "authorize_target_veto_closed_loop_development"
FAILURE_DECISION = "retain_phase_pressure_and_prohibit_target_veto_development"
MODEL_RANDOM_SEED = 20260803


def _target_model_config() -> ActionAdvantageConfig:
    return ActionAdvantageConfig(
        max_iter=80,
        max_leaf_nodes=7,
        min_samples_leaf=4,
        l2_regularization=16.0,
        random_state=MODEL_RANDOM_SEED,
    )


def _scenario_for_group(group: str, scenarios: Sequence[str]) -> str:
    matches = [name for name in scenarios if str(group).startswith(f"{name}:")]
    if len(matches) != 1:
        raise ValueError(f"action group has no unique scenario: {group!r}")
    return str(matches[0])


def frozen_proposal_records(
    contrast: MechanismDataset,
    *,
    frozen_model: Mapping[str, Any],
    scenarios: Sequence[str],
) -> list[dict[str, Any]]:
    """Reproduce the immutable v60 proposal and guard on a new label horizon."""

    models = frozen_model["models"]
    if str(models.prior_spec.key) != "phase_pressure":
        raise ValueError("v65 veto requires phase-pressure reference actions")
    score, uncertainty, base_trust = _candidate_scores(
        contrast,
        models=models,
        selected_candidate=str(frozen_model["selected_candidate"]),
    )
    pressure_support = np.asarray(
        frozen_model["pressure_target_support"].support(contrast), dtype=float
    )
    conformal_support = np.asarray(
        frozen_model["conformal_target_support"].support(contrast), dtype=float
    )
    pressure_trust = np.minimum(base_trust, pressure_support)
    conformal_trust = np.minimum(base_trust, conformal_support)
    actual, _ = group_normalized_action_target(contrast, "interval_cost")
    raw_actual = np.asarray(contrast.targets["interval_cost"], dtype=float)
    groups = action_group_ids(contrast).astype(str)
    is_reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    rule_gap = _relative_contrast_rule_gap(contrast)
    feature_index = {name: index for index, name in enumerate(contrast.feature_names)}
    if "total_veh" not in feature_index:
        raise KeyError("v65 proposal contrast lacks total_veh")
    candidate_states = np.asarray(
        contrast.metadata.get("candidate_states", [""] * contrast.size), dtype=str
    )
    row_times = np.asarray(
        contrast.metadata.get("row_times", [float("nan")] * contrast.size),
        dtype=float,
    )
    rows_out = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        references = rows[is_reference[rows]]
        if references.size != 1:
            raise ValueError(f"v65 action group lacks one reference: {group!r}")
        reference = int(references[0])
        learned = int(rows[int(np.argmin(score[rows]))])
        decision = hierarchical_guard_decision_v3(
            learned_differs=learned != reference,
            predicted_delta=float(score[learned]),
            uncertainty=float(uncertainty[learned]),
            relative_rule_gap=float(rule_gap[learned]),
            pressure_context_trust=float(pressure_trust[learned]),
            conformal_context_trust=float(conformal_trust[learned]),
            total_vehicles=float(contrast.features[learned, feature_index["total_veh"]]),
            regularizer=frozen_model["regularizer"],
            guard=frozen_model["guard"],
        )
        rows_out.append(
            {
                "group_id": str(group),
                "scenario": _scenario_for_group(str(group), scenarios),
                "simulator_seed": parse_action_group_seed(str(group)),
                "proposal_row_index": learned,
                "reference_row_index": reference,
                "proposal_candidate_state": str(candidate_states[learned]),
                "reference_candidate_state": str(candidate_states[reference]),
                "time_sec": float(row_times[learned]),
                "short_learned_differs": learned != reference,
                "short_proposed": bool(decision["eligible"]),
                "short_rejection": decision["rejection"],
                "short_priority": float(decision["priority"]),
                "short_predicted_delta": float(score[learned]),
                "short_uncertainty": float(uncertainty[learned]),
                "pressure_context_trust": float(pressure_trust[learned]),
                "conformal_context_trust": float(conformal_trust[learned]),
                "pressure_objective": float(decision["pressure_objective"]),
                "conformal_upper": float(decision["conformal_upper"]),
                "relative_rule_gap": float(rule_gap[learned]),
                "actual_group_normalized_delta": float(
                    actual[learned] - actual[reference]
                ),
                "actual_raw_delta": float(
                    raw_actual[learned] - raw_actual[reference]
                ),
                "candidate_features": {
                    name: float(contrast.features[learned, feature_index[name]])
                    for name in TARGET_SUPPORT_FEATURES
                    if name in feature_index
                },
            }
        )
    return rows_out


def _fit_oof_fold(
    *,
    heldout_seed: int,
    contrast: MechanismDataset,
    proposal_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups = action_group_ids(contrast).astype(str)
    group_seed = {
        str(group): parse_action_group_seed(str(group)) for group in np.unique(groups)
    }
    training_mask = np.asarray(
        [group_seed[str(group)] != int(heldout_seed) for group in groups],
        dtype=bool,
    )
    training = _subset_contrast(contrast, training_mask)
    training_groups = set(str(value) for value in training.metadata["action_group_ids"])
    if any(parse_action_group_seed(group) == int(heldout_seed) for group in training_groups):
        raise RuntimeError("target-veto OOF training contains the held-out seed")
    model = PairwiseActionAdvantageRegressor(
        causal=True, config=_target_model_config()
    )
    fit_diagnostics = model.fit(training)
    prediction = model.predict(contrast)["control_cost"]
    mean = np.asarray(prediction["mean"], dtype=float)
    uncertainty = np.asarray(prediction["uncertainty"], dtype=float)
    trust = np.asarray(prediction["context_trust"], dtype=float)
    records = []
    for proposal in proposal_records:
        if int(proposal["simulator_seed"]) != int(heldout_seed):
            continue
        learned = int(proposal["proposal_row_index"])
        reference = int(proposal["reference_row_index"])
        records.append(
            {
                **dict(proposal),
                "target_oof_heldout_seed": int(heldout_seed),
                "target_predicted_delta": float(mean[learned] - mean[reference]),
                "target_uncertainty": float(
                    uncertainty[learned] + uncertainty[reference]
                ),
                "target_context_trust": float(
                    min(trust[learned], trust[reference])
                ),
            }
        )
    return {
        "heldout_seed": int(heldout_seed),
        "training_group_count": len(training_groups),
        "validation_group_count": len(records),
        "training_group_sha256": _canonical_group_sha256(training_groups),
        "fit_diagnostics": fit_diagnostics,
        "records": records,
    }


def _canonical_group_sha256(groups: Sequence[str] | set[str]) -> str:
    import hashlib

    payload = "\n".join(sorted(str(value) for value in groups)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bootstrap_seed_means(
    seed_means: Mapping[int, float], *, replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    values = np.asarray(
        [float(seed_means[seed]) for seed in sorted(seed_means)], dtype=float
    )
    if values.size < 2 or not np.isfinite(values).all():
        raise ValueError("target-veto bootstrap requires finite seed means")
    rng = np.random.default_rng(int(bootstrap_seed))
    draws = rng.choice(values, size=(int(replicates), values.size), replace=True).mean(
        axis=1
    )
    return {
        "unit": "simulator_seed_equal_scenario_mean",
        "replicates": int(replicates),
        "seed": int(bootstrap_seed),
        "observed": float(values.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_nonnegative": float(np.mean(draws >= 0.0)),
    }


def summarize_veto_gate_candidate(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    risk_multiplier: float,
    min_context_trust: float,
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    expected_strata = {
        (int(seed), str(scenario)) for seed in seeds for scenario in scenarios
    }
    observed_strata = {
        (int(row["simulator_seed"]), str(row["scenario"])) for row in records
    }
    if observed_strata != expected_strata:
        raise ValueError("target-veto OOF records do not cover the seed/scenario matrix")
    evaluated = []
    for raw in records:
        row = dict(raw)
        upper = float(row["target_predicted_delta"]) + float(
            risk_multiplier
        ) * float(row["target_uncertainty"])
        accepted = bool(
            row["short_proposed"]
            and float(row["target_context_trust"]) >= float(min_context_trust)
            and upper < 0.0
        )
        evaluated.append(
            {
                **row,
                "target_upper": upper,
                "target_veto_accepted": accepted,
                "deployed_delta": (
                    float(row["actual_group_normalized_delta"]) if accepted else 0.0
                ),
            }
        )
    stratum_means = {
        key: float(
            np.mean(
                [
                    float(row["deployed_delta"])
                    for row in evaluated
                    if (int(row["simulator_seed"]), str(row["scenario"])) == key
                ]
            )
        )
        for key in sorted(expected_strata)
    }
    seed_means = {
        int(seed): float(
            np.mean(
                [stratum_means[(int(seed), str(scenario))] for scenario in scenarios]
            )
        )
        for seed in seeds
    }
    accepted = [row for row in evaluated if bool(row["target_veto_accepted"])]
    retained_seeds = {
        int(row["simulator_seed"]) for row in accepted
    }
    harmful_fraction = float(
        np.mean(
            [float(row["actual_group_normalized_delta"]) > 0.0 for row in accepted]
        )
    ) if accepted else 1.0
    interval = _bootstrap_seed_means(
        seed_means,
        replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    mean_delta = float(np.mean(list(seed_means.values())))
    seed_std = float(np.std(list(seed_means.values())))
    worst_seed = float(max(seed_means.values()))
    gates = {
        "minimum_retained_groups": len(accepted)
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
        "key": f"r{str(float(risk_multiplier)).replace('.', 'p')}_t{str(float(min_context_trust)).replace('.', 'p')}",
        "risk_multiplier": float(risk_multiplier),
        "min_context_trust": float(min_context_trust),
        "decision_count": len(evaluated),
        "short_proposal_count": sum(bool(row["short_proposed"]) for row in evaluated),
        "retained_group_count": len(accepted),
        "retained_seed_count": len(retained_seeds),
        "retained_seed_fraction": len(retained_seeds) / max(len(seeds), 1),
        "harmful_group_fraction": harmful_fraction,
        "mean_delta": mean_delta,
        "seed_delta_std": seed_std,
        "worst_seed_delta": worst_seed,
        "robust_score": mean_delta + 0.5 * seed_std + max(worst_seed, 0.0),
        "bootstrap": interval,
        "stratum_mean_delta": {
            f"seed{key[0]}::{key[1]}": value
            for key, value in stratum_means.items()
        },
        "seed_mean_delta": {str(key): value for key, value in seed_means.items()},
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def select_veto_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    training_spec: Mapping[str, Any],
) -> dict[str, Any]:
    rule = dict(training_spec["selection_rule"])
    bootstrap = dict(training_spec["bootstrap"])
    grid = [
        summarize_veto_gate_candidate(
            records,
            seeds=seeds,
            scenarios=scenarios,
            risk_multiplier=float(candidate["risk_multiplier"]),
            min_context_trust=float(candidate["min_context_trust"]),
            selection_rule=rule,
            bootstrap=bootstrap,
        )
        for candidate in training_spec["gate_candidates"]
    ]
    feasible = [row for row in grid if bool(row["feasible"])]
    selected = min(
        feasible,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_delta"]),
            float(row["harmful_group_fraction"]),
            -int(row["retained_seed_count"]),
            -int(row["retained_group_count"]),
            -float(row["min_context_trust"]),
            -float(row["risk_multiplier"]),
            str(row["key"]),
        ),
        default=None,
    )
    config = LongHorizonTargetVetoConfig(
        enabled=selected is not None,
        risk_multiplier=float(selected["risk_multiplier"]) if selected else 0.0,
        min_context_trust=float(selected["min_context_trust"])
        if selected
        else 1.0,
        rollout_value_horizon_sec=300,
    )
    return {
        "decision": "deploy_target_veto" if selected else "always_veto_to_phase_pressure",
        "selected": selected,
        "config": {
            "enabled": config.enabled,
            "risk_multiplier": config.risk_multiplier,
            "min_context_trust": config.min_context_trust,
            "rollout_value_horizon_sec": config.rollout_value_horizon_sec,
        },
        "feasible_candidate_count": len(feasible),
        "grid": grid,
    }


def fit_city_target_veto(
    *,
    city: str,
    scenarios: Sequence[str],
    bank: Mapping[str, MechanismDataset],
    frozen_model: Mapping[str, Any],
    protocol: Mapping[str, Any],
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = _merge_city_datasets(bank, scenarios, city=city)
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    proposal_records = frozen_proposal_records(
        contrast, frozen_model=frozen_model, scenarios=scenarios
    )
    seeds = tuple(int(value) for value in protocol["long_horizon_collection"]["seeds"])
    observed_seeds = {int(row["simulator_seed"]) for row in proposal_records}
    if observed_seeds != set(seeds):
        raise ValueError(f"v65 city {city} does not cover all collection seeds")
    with ThreadPoolExecutor(max_workers=min(max(int(workers), 1), len(seeds))) as pool:
        folds = list(
            pool.map(
                lambda seed: _fit_oof_fold(
                    heldout_seed=int(seed),
                    contrast=contrast,
                    proposal_records=proposal_records,
                ),
                seeds,
            )
        )
    oof_records = [row for fold in folds for row in fold["records"]]
    if (
        len(oof_records) != len(proposal_records)
        or {str(row["group_id"]) for row in oof_records}
        != {str(row["group_id"]) for row in proposal_records}
    ):
        raise RuntimeError("v65 OOF predictions are not an exact action-group partition")
    gate = select_veto_gate(
        oof_records,
        seeds=seeds,
        scenarios=scenarios,
        training_spec=protocol["target_veto_training"],
    )
    final_model = PairwiseActionAdvantageRegressor(
        causal=True, config=_target_model_config()
    )
    final_fit = final_model.fit(contrast)
    config = LongHorizonTargetVetoConfig(**gate["config"])
    target_veto = LongHorizonTargetVeto(model=final_model, config=config)
    summary = {
        "city": city,
        "scenarios": list(scenarios),
        "seeds": list(seeds),
        "dataset_rows": int(contrast.size),
        "action_group_count": len(proposal_records),
        "short_proposal_count": sum(
            bool(row["short_proposed"]) for row in proposal_records
        ),
        "model_family": MODEL_FAMILY,
        "model_config": {
            key: value for key, value in vars(_target_model_config()).items()
        },
        "oof_folds": folds,
        "oof_records": oof_records,
        "gate_selection": gate,
        "final_fit": final_fit,
        "target_veto_diagnostics": target_veto.diagnostics(),
    }
    artifact = {
        "protocol": MODEL_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": city,
        "classification": "target_simulator_labeled_few_shot_adaptation",
        "zero_shot": False,
        "proposal_role": "immutable_v60_hierarchical_cfcmt",
        "veto_role": "veto_only_never_action_originator",
        "target_veto": target_veto,
        "target_model": final_model,
        "selected_gate": gate,
        "final_fit": final_fit,
    }
    return summary, artifact


def run_target_veto_freeze(
    *,
    cache_root: Path,
    cache_audit_path: Path,
    expected_cache_audit_sha256: str,
    protocol_path: Path,
    parent_protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = json.loads(Path(protocol_path).read_text(encoding="utf-8"))
    parent = json.loads(Path(parent_protocol_path).read_text(encoding="utf-8"))
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    freeze_audit = json.loads(Path(freeze_audit_path).read_text(encoding="utf-8"))
    cache_audit_sha = _sha256(cache_audit_path)
    if (
        protocol.get("protocol") != PROTOCOL
        or parent.get("protocol") != PARENT_PROTOCOL
        or _sha256(parent_protocol_path) != protocol["parent_protocol"]["sha256"]
        or cache_audit_sha != str(expected_cache_audit_sha256)
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision") != CACHE_AUDIT_DECISION
        or not bool(cache_audit.get("integrity_gate", {}).get("passed", False))
        or cache_audit.get("frozen_protocol_sha256") != _sha256(protocol_path)
        or _sha256(freeze_audit_path)
        != parent["hierarchical_freeze_audit"]["sha256"]
        or freeze_audit.get("status") != "PASS"
        or not bool(freeze_audit.get("integrity_gate", {}).get("passed", False))
    ):
        raise ValueError("v65 target-veto training evidence chain changed")
    cache_sha, cache_count = aggregate_cache_sha256(cache_root)
    if (
        cache_sha != cache_audit.get("cache_sha256")
        or cache_count != int(protocol["long_horizon_collection"]["expected_cache_file_count"])
        or cache_count != int(cache_audit.get("cache_file_count", -1))
    ):
        raise ValueError("v65 target-veto cache identity changed")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite target-veto freeze: {output_root}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    manifest = load_traffic_signal_manifest(external_manifest_path)
    collection = dict(protocol["long_horizon_collection"])
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=tuple(int(value) for value in collection["seeds"]),
        collection_shards=int(collection["collection_shards_per_scenario_seed"]),
        workers=max(int(workers), 1),
    )
    if any(
        int(dataset.metadata.get("rollout_value_horizon_sec", -1)) != 300
        for dataset in bank.values()
    ):
        raise ValueError("v65 loaded cache does not have a 300-second value horizon")
    output_root.mkdir(parents=True, exist_ok=True)
    city_results: dict[str, Any] = {}
    city_artifacts: dict[str, Any] = {}
    for city, scenarios in sorted(protocol["city_scenarios"].items()):
        certificate, frozen_model = _load_frozen_city_model(
            city=str(city),
            freeze_root=freeze_root,
            protocol=parent,
            freeze_audit=freeze_audit,
        )
        summary, artifact = fit_city_target_veto(
            city=str(city),
            scenarios=tuple(str(value) for value in scenarios),
            bank=bank,
            frozen_model=frozen_model,
            protocol=protocol,
            workers=workers,
        )
        city_root = output_root / str(city)
        city_root.mkdir(parents=True, exist_ok=False)
        model_path = city_root / "model.pkl"
        model_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
        freeze_payload = {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "city": str(city),
            "proposal_certificate_sha256": _sha256(
                Path(freeze_root) / str(city) / "freeze.json"
            ),
            "proposal_model_sha256": _sha256(
                Path(freeze_root) / str(city) / "model.pkl"
            ),
            "proposal_deployment": certificate["deployment"],
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
            "active": bool(summary["gate_selection"]["config"]["enabled"]),
        }
    active_cities = [city for city, row in city_artifacts.items() if row["active"]]
    integrity_gate = {
        "all_cities_present": set(city_results) == set(protocol["city_scenarios"]),
        "all_city_artifacts_written": all(
            Path(row["model"]["path"]).is_file()
            and Path(row["certificate"]["path"]).is_file()
            for row in city_artifacts.values()
        ),
        "at_least_one_city_has_oof_feasible_veto": bool(active_cities),
        "validation_and_prospective_seeds_absent": not (
            set(collection["seeds"])
            & set(protocol["validation"]["closed_loop_seeds"])
        )
        and not (
            set(collection["seeds"])
            & set(protocol["prospective_confirmation"]["closed_loop_seeds"])
        ),
    }
    integrity_gate["passed"] = all(integrity_gate.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity_gate["passed"] else "FAIL",
        "decision": (
            AUTHORIZATION_DECISION if integrity_gate["passed"] else FAILURE_DECISION
        ),
        "claim_boundary": (
            "Target-simulator-labeled few-shot OOF action veto only. This result "
            "does not establish closed-loop, validation, or prospective efficacy."
        ),
        "frozen_protocol_sha256": _sha256(protocol_path),
        "parent_protocol_sha256": _sha256(parent_protocol_path),
        "cache_audit_sha256": cache_audit_sha,
        "cache_sha256": cache_sha,
        "cache_file_count": cache_count,
        "bank_audit": bank_audit,
        "active_cities": active_cities,
        "city_artifacts": city_artifacts,
        "city_results": city_results,
        "integrity_gate": integrity_gate,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--expected-cache-audit-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--parent-protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite target-veto result: {args.out}")
    payload = run_target_veto_freeze(
        cache_root=args.cache_root,
        cache_audit_path=args.cache_audit,
        expected_cache_audit_sha256=args.expected_cache_audit_sha256,
        protocol_path=args.protocol,
        parent_protocol_path=args.parent_protocol,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "active_cities": payload["active_cities"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
