"""Run one authorized v43 external-network closed-loop confirmation rollout."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
    _local_group_alpha,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    CANDIDATE_KEYS,
    CONFIDENCE_MULTIPLIERS,
    CONSTANT_ALPHAS,
    DISAGREEMENT_CAPS,
    constant_candidate_key,
    local_candidate_key,
)
from cf_h2o.eval.traffic_signal_external_city_baseline_freeze import (
    BENCHMARK_FAMILIES,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    BASELINE_MODEL_PROTOCOL,
    METHOD_MODEL_PROTOCOL,
    PROTOCOL,
    V9_BASELINE_MODEL_PROTOCOL,
    V9_METHOD_MODEL_PROTOCOL,
    V9_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_heldout_evaluation import (
    RESULT_PROTOCOL as OFFLINE_RESULT_PROTOCOL,
    V9_CLOSED_LOOP_DECISION,
    V9_RESULT_PROTOCOL as V9_OFFLINE_RESULT_PROTOCOL,
    _load_full_refit_model_artifact,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardConfig,
    FittedContrastModelsV3,
    PriorRegularizationConfig,
    ResidualExecutionTrustRegionConfig,
    _group_adjusted_scores,
    evaluate_policy_v3,
)
from cf_h2o.eval.traffic_signal_counterfactual_cache import _tree_sha256
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.action_contrast import action_group_ids
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


RESULT_PROTOCOL = "tsc-v43r39-external-closed-loop-rollout-v4"
AUTHORIZATION_DECISION = "authorize_closed_loop_external_confirmation"
JOINT_AUDIT_PROTOCOL = "tsc-v43r39-external-full-budget-joint-freeze-audit-v1"
V9_JOINT_AUDIT_PROTOCOL = (
    "tsc-v54r50-external-v9-full-budget-joint-freeze-audit-v1"
)
METHOD_POLICY = "cfcmt_selected_mpc"
BASELINE_POLICY_TO_RUNTIME = {
    "fixed_time": ("fixed_program", None),
    "phase_pressure": ("phase_pressure", None),
    "max_pressure": ("max_pressure", None),
    "rigid_anchor_mpc": (
        f"{ANCHOR_FAMILY}_contrast_raw",
        ("method", ANCHOR_FAMILY),
    ),
    "simulator_only_mpc": (
        "simulator_contrast_raw",
        ("baseline", "simulator"),
    ),
    "h2oplus_style_dense_residual_mpc": (
        "dense_contrast_raw",
        ("baseline", "dense"),
    ),
}
DIAGNOSTIC_BASELINE_POLICY_TO_RUNTIME = {
    "causal_target_only_mpc": (
        "causal_target_only_contrast_raw",
        ("baseline", "causal_target_only"),
    ),
}
POLICIES = (METHOD_POLICY, *BASELINE_POLICY_TO_RUNTIME)
CONSTANT_CANDIDATES = {
    constant_candidate_key(alpha): float(alpha) for alpha in CONSTANT_ALPHAS
}
LOCAL_CANDIDATES = {
    local_candidate_key(cap, confidence): (float(cap), float(confidence))
    for cap in DISAGREEMENT_CAPS
    for confidence in CONFIDENCE_MULTIPLIERS
}


@dataclass
class AnchoredBlendAudit:
    prediction_calls: int = 0
    action_groups: int = 0
    alpha_sum: float = 0.0
    alpha_min: float = 1.0
    alpha_max: float = 0.0
    nonzero_alpha_groups: int = 0
    full_alpha_groups: int = 0

    def update(self, alpha: float) -> None:
        value = float(alpha)
        self.action_groups += 1
        self.alpha_sum += value
        self.alpha_min = min(self.alpha_min, value)
        self.alpha_max = max(self.alpha_max, value)
        self.nonzero_alpha_groups += int(value > 1e-12)
        self.full_alpha_groups += int(value >= 1.0 - 1e-12)

    def to_dict(self) -> dict[str, Any]:
        count = max(self.action_groups, 1)
        return {
            "protocol": "online-frozen-anchored-blend-audit-v1",
            "prediction_calls": self.prediction_calls,
            "action_groups": self.action_groups,
            "mean_alpha": self.alpha_sum / count,
            "minimum_alpha": self.alpha_min if self.action_groups else None,
            "maximum_alpha": self.alpha_max if self.action_groups else None,
            "nonzero_alpha_fraction": self.nonzero_alpha_groups / count,
            "full_alpha_fraction": self.full_alpha_groups / count,
        }


@dataclass(frozen=True)
class ClosedLoopDiagnosticSpec:
    """A post-freeze deployment diagnostic that reuses frozen models and inputs."""

    name: str
    source_policy: str
    coordination_mode: str
    result_protocol: str
    guard_config: ContrastGuardConfig | None = None
    cooldown_intervals_override: int | None = None
    execution_trust_region: ResidualExecutionTrustRegionConfig | None = None
    allowed_seeds: tuple[int, ...] | None = None
    analysis_status: str = "post_hoc_explanatory_not_preregistered_confirmation"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source_policy": self.source_policy,
            "coordination_mode": self.coordination_mode,
            "result_protocol": self.result_protocol,
            "analysis_status": self.analysis_status,
            "frozen_models_reused_without_refit": True,
            "frozen_evaluation_scenarios_and_seeds_reused": True,
            "guard_config": (
                {
                    "enabled": self.guard_config.enabled,
                    "risk_multiplier": self.guard_config.risk_multiplier,
                    "min_context_trust": self.guard_config.min_context_trust,
                    "margin": self.guard_config.margin,
                    "max_relative_rule_gap": self.guard_config.max_relative_rule_gap,
                }
                if self.guard_config is not None
                else None
            ),
            "cooldown_intervals_override": self.cooldown_intervals_override,
            "execution_trust_region": (
                asdict(self.execution_trust_region)
                if self.execution_trust_region is not None
                else None
            ),
            "allowed_seeds": list(self.allowed_seeds) if self.allowed_seeds is not None else None,
        }


@dataclass(frozen=True)
class ClosedLoopMethodRuntimeOverride:
    """An audited post-v43 method runtime that keeps the frozen environment."""

    runtime_policy: str
    models: FittedContrastModelsV3
    selected_candidate: str
    anchored_blend_model: Any
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_policy": self.runtime_policy,
            "selected_candidate": self.selected_candidate,
            "provenance": dict(self.provenance),
        }


class FrozenAnchoredBlendModel:
    """Reproduce the frozen offline anchor/correction score blend online."""

    def __init__(
        self,
        *,
        anchor_model: Any,
        correction_model: Any,
        candidate: str,
        anchor_objective_mode: str,
        correction_objective_mode: str,
    ) -> None:
        if candidate not in CANDIDATE_KEYS:
            raise ValueError(f"unknown frozen anchored candidate: {candidate}")
        self.anchor_model = anchor_model
        self.correction_model = correction_model
        self.candidate = str(candidate)
        self.anchor_objective_mode = str(anchor_objective_mode)
        self.correction_objective_mode = str(correction_objective_mode)
        self.audit = AnchoredBlendAudit()

    def predict(self, dataset):
        self.audit.prediction_calls += 1
        anchor_score, anchor_uncertainty, anchor_trust, _ = (
            _group_adjusted_scores(
                dataset,
                self.anchor_model.predict(dataset),
                objective_mode=self.anchor_objective_mode,
            )
        )
        correction_score, correction_uncertainty, correction_trust, _ = (
            _group_adjusted_scores(
                dataset,
                self.correction_model.predict(dataset),
                objective_mode=self.correction_objective_mode,
            )
        )
        groups = action_group_ids(dataset)
        alpha_rows = np.zeros(dataset.size, dtype=float)
        if self.candidate in CONSTANT_CANDIDATES:
            alpha_rows.fill(CONSTANT_CANDIDATES[self.candidate])
            for _ in np.unique(groups):
                self.audit.update(CONSTANT_CANDIDATES[self.candidate])
        else:
            cap, confidence = LOCAL_CANDIDATES[self.candidate]
            for group in np.unique(groups):
                rows = np.flatnonzero(groups == group)
                alpha, _ = _local_group_alpha(
                    anchor_score=anchor_score,
                    correction_score=correction_score,
                    anchor_uncertainty=anchor_uncertainty,
                    correction_uncertainty=correction_uncertainty,
                    group_rows=rows,
                    disagreement_cap=cap,
                    confidence_multiplier=confidence,
                )
                alpha_rows[rows] = alpha
                self.audit.update(alpha)

        score = anchor_score + alpha_rows * (correction_score - anchor_score)
        uncertainty = (
            (1.0 - alpha_rows) * anchor_uncertainty
            + alpha_rows * correction_uncertainty
        )
        return {
            "control_cost": {
                "mean": score,
                "uncertainty": uncertainty,
                "context_trust": np.minimum(anchor_trust, correction_trust),
            }
        }


def _fitted_runtime_bundle(
    *,
    offline_payload: Mapping[str, Any],
    family: str,
    model: Any,
    prediction_horizon_sec: int,
    guard_config: ContrastGuardConfig | None = None,
    regularizer_config: PriorRegularizationConfig | None = None,
    target_support: Any | None = None,
) -> FittedContrastModelsV3:
    prior = offline_payload["model_ensemble"][0].prior_spec
    objective_mode = str(
        offline_payload["model_ensemble"][0].objective_modes[family]
    )
    if objective_mode != "control_only":
        raise ValueError(
            "v43 frozen online score wrappers require control_only objectives"
        )
    return FittedContrastModelsV3(
        family_models={family: model},
        prior_policy=str(getattr(prior, "key", "")),
        prior_spec=prior,
        prediction_horizon_sec=int(prediction_horizon_sec),
        objective_modes={family: objective_mode},
        guards={family: guard_config or ContrastGuardConfig()},
        regularizers={family: regularizer_config or PriorRegularizationConfig()},
        target_support=target_support,
        hierarchy_layers={},
        diagnostics={
            "protocol": "v43-frozen-single-refit-online-runtime-bundle-v1",
            "family": family,
            "deployment_guard_installed": guard_config is not None,
            "deployment_regularizer_installed": regularizer_config is not None,
            "target_action_support_installed": target_support is not None,
        },
    )


def _load_city_models(
    *,
    city: str,
    freeze_root: Path,
    joint_audit: Mapping[str, Any],
    method_model_protocol: str = METHOD_MODEL_PROTOCOL,
    baseline_model_protocol: str = BASELINE_MODEL_PROTOCOL,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    city_root = Path(freeze_root) / city
    method_result = json.loads(
        (city_root / "method_freeze.json").read_text(encoding="utf-8")
    )
    baseline_result = json.loads(
        (city_root / "baseline_freeze.json").read_text(encoding="utf-8")
    )
    joint_city = joint_audit["cities"][city]
    if (
        _sha256(city_root / "method_freeze.json")
        != joint_city["method_result"]["sha256"]
        or _sha256(city_root / "baseline_freeze.json")
        != joint_city["baseline_result"]["sha256"]
    ):
        raise ValueError(f"closed-loop freeze result mismatch for {city}")
    method_groups = tuple(method_result["selected_group_ids"])
    baseline_groups = tuple(baseline_result["selected_group_ids"])
    method_payload = _load_full_refit_model_artifact(
        city_root / "method_model.pkl",
        expected_sha256=joint_city["method_model"]["sha256"],
        expected_protocol=method_model_protocol,
        city=city,
        expected_families=(ANCHOR_FAMILY, CORRECTION_FAMILY),
        expected_group_ids=method_groups,
    )
    baseline_payload = _load_full_refit_model_artifact(
        city_root / "baseline_model.pkl",
        expected_sha256=joint_city["baseline_model"]["sha256"],
        expected_protocol=baseline_model_protocol,
        city=city,
        expected_families=BENCHMARK_FAMILIES,
        expected_group_ids=baseline_groups,
    )
    method_prior = str(method_payload["model_ensemble"][0].prior_spec.key)
    baseline_prior = str(baseline_payload["model_ensemble"][0].prior_spec.key)
    if (
        method_groups != baseline_groups
        or method_payload["fold_assignment"]
        != baseline_payload["fold_assignment"]
        or method_prior != baseline_prior
    ):
        raise ValueError(f"closed-loop method/baseline mismatch for {city}")
    return method_result, method_payload, baseline_payload


def run_external_closed_loop_rollout(
    *,
    protocol_spec_path: Path,
    offline_authorization_path: Path,
    expected_offline_authorization_sha256: str,
    joint_freeze_audit_path: Path,
    expected_joint_freeze_sha256: str,
    freeze_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    expected_external_manifest_sha256: str,
    conversion_manifest_path: Path,
    expected_conversion_manifest_sha256: str,
    expected_conversion_tree_sha256: str,
    expected_sumo_version: str,
    scenario: str,
    seed: int,
    policy: str,
    tripinfo_out: Path,
    diagnostic_spec: ClosedLoopDiagnosticSpec | None = None,
    method_runtime_override: ClosedLoopMethodRuntimeOverride | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    protocol = json.loads(Path(protocol_spec_path).read_text(encoding="utf-8"))
    authorization = json.loads(
        Path(offline_authorization_path).read_text(encoding="utf-8")
    )
    joint = json.loads(Path(joint_freeze_audit_path).read_text(encoding="utf-8"))
    protocol_name = str(protocol.get("protocol", ""))
    if protocol_name == PROTOCOL:
        offline_result_protocol = OFFLINE_RESULT_PROTOCOL
        authorization_decision = AUTHORIZATION_DECISION
        joint_audit_protocol = JOINT_AUDIT_PROTOCOL
        method_model_protocol = METHOD_MODEL_PROTOCOL
        baseline_model_protocol = BASELINE_MODEL_PROTOCOL
        generation = "v8"
    elif protocol_name == V9_PROTOCOL:
        offline_result_protocol = V9_OFFLINE_RESULT_PROTOCOL
        authorization_decision = V9_CLOSED_LOOP_DECISION
        joint_audit_protocol = V9_JOINT_AUDIT_PROTOCOL
        method_model_protocol = V9_METHOD_MODEL_PROTOCOL
        baseline_model_protocol = V9_BASELINE_MODEL_PROTOCOL
        generation = "v9"
    else:
        raise ValueError("closed-loop protocol changed")
    parent_spec = dict(protocol["parent_protocol"])
    parent_path = Path(str(parent_spec["path"]))
    if not parent_path.is_absolute():
        parent_path = Path(protocol_spec_path).resolve().parents[2] / parent_path
    if _sha256(parent_path) != str(parent_spec["sha256"]):
        raise ValueError("v43 closed-loop parent protocol changed")
    parent_protocol = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        _sha256(offline_authorization_path)
        != str(expected_offline_authorization_sha256)
        or authorization.get("protocol") != offline_result_protocol
        or authorization.get("decision") != authorization_decision
        or not bool(
            authorization.get("offline_confirmation_gate", {}).get(
                "passed", False
            )
        )
        or _sha256(joint_freeze_audit_path)
        != str(expected_joint_freeze_sha256)
        or authorization.get("joint_freeze_audit_sha256")
        != str(expected_joint_freeze_sha256)
        or joint.get("status") != "PASS"
        or joint.get("protocol") != joint_audit_protocol
    ):
        raise ValueError("closed-loop rollout lacks frozen offline authorization")

    closed_loop = dict(protocol["closed_loop"])
    if not bool(
        closed_loop.get("enabled_only_after_offline_gate", False)
        if generation == "v8"
        else closed_loop.get(
            "algorithm_and_candidate_grid_frozen_before_v9_rollout", False
        )
    ):
        raise ValueError("closed-loop authorization contract changed")
    expected_policies = (METHOD_POLICY, *tuple(closed_loop["baselines"]))
    if tuple(expected_policies) != POLICIES:
        raise ValueError("closed-loop policy set changed")
    if diagnostic_spec is None:
        if policy not in POLICIES:
            raise ValueError("closed-loop policy set changed")
        source_policy = policy
        coordination_mode = "sparse"
        result_protocol = RESULT_PROTOCOL
    else:
        if (
            policy != diagnostic_spec.name
            or diagnostic_spec.source_policy
            not in {
                *POLICIES,
                "selected_source_prior",
                *DIAGNOSTIC_BASELINE_POLICY_TO_RUNTIME,
            }
            or diagnostic_spec.coordination_mode
            not in {"sparse", "spatial_only", "direct"}
            or diagnostic_spec.result_protocol == RESULT_PROTOCOL
            or (
                diagnostic_spec.guard_config is not None
                and diagnostic_spec.source_policy != METHOD_POLICY
            )
            or (
                diagnostic_spec.cooldown_intervals_override is not None
                and int(diagnostic_spec.cooldown_intervals_override) < 0
            )
            or (
                diagnostic_spec.execution_trust_region is not None
                and diagnostic_spec.source_policy != METHOD_POLICY
            )
            or (
                diagnostic_spec.allowed_seeds is not None
                and (
                    not diagnostic_spec.allowed_seeds
                    or len(set(diagnostic_spec.allowed_seeds))
                    != len(diagnostic_spec.allowed_seeds)
                    or any(int(value) < 0 for value in diagnostic_spec.allowed_seeds)
                )
            )
        ):
            raise ValueError("invalid post-hoc closed-loop diagnostic specification")
        source_policy = diagnostic_spec.source_policy
        coordination_mode = diagnostic_spec.coordination_mode
        result_protocol = diagnostic_spec.result_protocol
    if method_runtime_override is not None and (
        diagnostic_spec is None
        or source_policy != METHOD_POLICY
        or not method_runtime_override.runtime_policy.endswith(
            (
                "_contrast_guard",
                "_contrast_raw",
                "_contrast_regularized",
                "_contrast_pressure_gate",
            )
        )
    ):
        raise ValueError("method runtime override requires an explicit method diagnostic")
    target_protocol = dict(protocol["target_protocol"])
    preregistered_seeds = {
        int(value)
        for key in (
            "closed_loop_seeds",
            "closed_loop_development_seeds",
            "closed_loop_validation_seeds",
            "closed_loop_prospective_seeds",
        )
        for value in target_protocol.get(key, ())
    }
    diagnostic_seeds = (
        {int(value) for value in diagnostic_spec.allowed_seeds}
        if diagnostic_spec is not None and diagnostic_spec.allowed_seeds is not None
        else set()
    )
    if int(seed) not in preregistered_seeds | diagnostic_seeds:
        raise ValueError("closed-loop seed is not preregistered")
    city_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in protocol["target_protocol"][
            "external_city_scenarios"
        ].items()
    }
    matching_cities = [
        city for city, scenarios in city_scenarios.items() if scenario in scenarios
    ]
    if len(matching_cities) != 1:
        raise ValueError(f"closed-loop scenario is not uniquely assigned: {scenario}")
    city = matching_cities[0]

    conversion_root = Path(conversion_root).resolve()
    conversion_tree_sha256 = _tree_sha256(conversion_root)
    if (
        _sha256(external_manifest_path)
        != str(expected_external_manifest_sha256)
        or _sha256(conversion_manifest_path)
        != str(expected_conversion_manifest_sha256)
        or conversion_tree_sha256 != str(expected_conversion_tree_sha256)
    ):
        raise ValueError("closed-loop conversion input identity changed")
    parent_cache = dict(parent_protocol["counterfactual_cache"])
    if (
        str(parent_cache["manifest_sha256"])
        != str(expected_external_manifest_sha256)
        or int(parent_cache["control_interval_sec"]) <= 0
        or int(parent_cache["counterfactual_horizon_intervals"]) <= 0
    ):
        raise ValueError("closed-loop counterfactual estimand contract changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(external_manifest_path)
    if scenario not in manifest.sumocfgs or manifest.city_groups[scenario] != city:
        raise ValueError("closed-loop manifest city mapping changed")
    if libsumo_version() != str(expected_sumo_version):
        raise ValueError("closed-loop SUMO version changed")

    method_result, method_payload, baseline_payload = _load_city_models(
        city=city,
        freeze_root=freeze_root,
        joint_audit=joint,
        method_model_protocol=method_model_protocol,
        baseline_model_protocol=baseline_model_protocol,
    )
    selected_candidate = str(method_result["selected_candidate"])
    authorized_candidate = str(
        authorization["city_results"][city]["selected_candidate"]
    )
    if selected_candidate != authorized_candidate:
        raise ValueError("closed-loop selected candidate changed after offline gate")

    wrapper: FrozenAnchoredBlendModel | None = None
    runtime_policy: str
    models: FittedContrastModelsV3 | None
    control_interval_sec = int(parent_cache["control_interval_sec"])
    warmup_sec = float(parent_cache["warmup_sec"])
    prediction_horizon_sec = int(
        control_interval_sec
        * int(parent_cache["counterfactual_horizon_intervals"])
    )
    if source_policy == METHOD_POLICY:
        if method_runtime_override is not None:
            wrapper = method_runtime_override.anchored_blend_model
            runtime_policy = method_runtime_override.runtime_policy
            models = method_runtime_override.models
            runtime_selected_candidate = method_runtime_override.selected_candidate
        else:
            offline = method_payload["model_ensemble"][0]
            wrapper = FrozenAnchoredBlendModel(
                anchor_model=offline.family_models[ANCHOR_FAMILY],
                correction_model=offline.family_models[CORRECTION_FAMILY],
                candidate=selected_candidate,
                anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
                correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
            )
            runtime_policy = (
                f"{ANCHOR_FAMILY}_contrast_guard"
                if diagnostic_spec is not None
                and diagnostic_spec.guard_config is not None
                else f"{ANCHOR_FAMILY}_contrast_raw"
            )
            models = _fitted_runtime_bundle(
                offline_payload=method_payload,
                family=ANCHOR_FAMILY,
                model=wrapper,
                prediction_horizon_sec=prediction_horizon_sec,
                guard_config=(
                    diagnostic_spec.guard_config
                    if diagnostic_spec is not None
                    else None
                ),
            )
            runtime_selected_candidate = selected_candidate
    elif source_policy == "selected_source_prior":
        offline = method_payload["model_ensemble"][0]
        runtime_policy = "selected_source_prior"
        models = _fitted_runtime_bundle(
            offline_payload=method_payload,
            family=ANCHOR_FAMILY,
            model=offline.family_models[ANCHOR_FAMILY],
            prediction_horizon_sec=prediction_horizon_sec,
        )
    else:
        runtime_mapping = (
            BASELINE_POLICY_TO_RUNTIME
            if source_policy in BASELINE_POLICY_TO_RUNTIME
            else DIAGNOSTIC_BASELINE_POLICY_TO_RUNTIME
        )
        runtime_policy, model_source = runtime_mapping[source_policy]
        if model_source is None:
            models = None
        else:
            source, family = model_source
            payload = method_payload if source == "method" else baseline_payload
            model = payload["model_ensemble"][0].family_models[family]
            models = _fitted_runtime_bundle(
                offline_payload=payload,
                family=family,
                model=model,
                prediction_horizon_sec=prediction_horizon_sec,
            )

    temporary_tripinfo = tripinfo_out.with_suffix(
        tripinfo_out.suffix + f".tmp-{os.getpid()}"
    )
    metrics = evaluate_policy_v3(
        sumo_api=load_libsumo(),
        sumocfg=manifest.sumocfgs[scenario],
        scenario=scenario,
        policy=runtime_policy,
        models=models,
        duration_sec=float(closed_loop["horizon_sec"]),
        control_interval_sec=control_interval_sec,
        warmup_sec=warmup_sec,
        seed=int(seed),
        tripinfo_output=temporary_tripinfo,
        residual_coordination_mode=coordination_mode,
        residual_cooldown_intervals_override=(
            diagnostic_spec.cooldown_intervals_override
            if diagnostic_spec is not None
            else None
        ),
        residual_execution_trust_region=(
            diagnostic_spec.execution_trust_region
            if diagnostic_spec is not None
            else None
        ),
    )
    if not bool(metrics.get("ok", False)):
        raise RuntimeError(
            f"closed-loop rollout failed for {scenario}/{seed}/{policy}: "
            f"{metrics.get('error', 'unknown error')}"
        )
    if int(metrics.get("starting_teleports", -1)) != 0 or int(
        metrics.get("ending_teleports", -1)
    ) != 0:
        raise RuntimeError("closed-loop rollout contains teleports")
    if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
        raise RuntimeError("closed-loop rollout did not produce tripinfo evidence")
    temporary_tripinfo.replace(tripinfo_out)
    tripinfo_size_bytes = int(tripinfo_out.stat().st_size)
    tripinfo_sha256 = _sha256(tripinfo_out)
    return {
        "protocol": result_protocol,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "generation": generation,
        "offline_authorization_sha256": _sha256(offline_authorization_path),
        "joint_freeze_audit_sha256": _sha256(joint_freeze_audit_path),
        "external_manifest_sha256": _sha256(external_manifest_path),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": conversion_tree_sha256,
        "sumo_version": libsumo_version(),
        "city": city,
        "scenario": scenario,
        "seed": int(seed),
        "policy": policy,
        "source_policy": source_policy,
        "runtime_policy": runtime_policy,
        "selected_candidate": (
            runtime_selected_candidate if source_policy == METHOD_POLICY else None
        ),
        "prediction_horizon_sec": prediction_horizon_sec,
        "evaluation_horizon_sec": int(closed_loop["horizon_sec"]),
        "control_interval_sec": control_interval_sec,
        "warmup_sec": warmup_sec,
        "model_artifacts": joint["cities"][city],
        "anchored_blend_audit": wrapper.audit.to_dict() if wrapper else None,
        "post_hoc_diagnostic": (
            diagnostic_spec.to_dict() if diagnostic_spec is not None else None
        ),
        "method_runtime_override": (
            method_runtime_override.to_dict()
            if method_runtime_override is not None
            else None
        ),
        "tripinfo_evidence": {
            "filename": tripinfo_out.name,
            "sha256": tripinfo_sha256,
            "size_bytes": tripinfo_size_bytes,
        },
        "metrics": metrics,
        "elapsed_sec": float(time.monotonic() - started),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--expected-offline-authorization-sha256", required=True)
    parser.add_argument("--joint-freeze-audit", type=Path, required=True)
    parser.add_argument("--expected-joint-freeze-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--expected-external-manifest-sha256", required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--expected-conversion-manifest-sha256", required=True)
    parser.add_argument("--expected-conversion-tree-sha256", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--tripinfo-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.tripinfo_out.exists():
        raise FileExistsError("refusing to overwrite closed-loop evidence")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    payload = run_external_closed_loop_rollout(
        protocol_spec_path=args.protocol_spec,
        offline_authorization_path=args.offline_authorization,
        expected_offline_authorization_sha256=(
            args.expected_offline_authorization_sha256
        ),
        joint_freeze_audit_path=args.joint_freeze_audit,
        expected_joint_freeze_sha256=args.expected_joint_freeze_sha256,
        freeze_root=args.freeze_root,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        expected_external_manifest_sha256=(
            args.expected_external_manifest_sha256
        ),
        conversion_manifest_path=args.conversion_manifest,
        expected_conversion_manifest_sha256=(
            args.expected_conversion_manifest_sha256
        ),
        expected_conversion_tree_sha256=args.expected_conversion_tree_sha256,
        expected_sumo_version=args.expected_sumo_version,
        scenario=args.scenario,
        seed=args.seed,
        policy=args.policy,
        tripinfo_out=args.tripinfo_out,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "scenario": args.scenario,
                "seed": args.seed,
                "policy": args.policy,
                "mean_waiting_time": payload["metrics"][
                    "mean_tripinfo_waiting_time"
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
