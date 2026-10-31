"""Run topology-repaired source-weight diagnostics in the frozen v43 environment."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    FrozenCausalSourceWeightModel,
    SOURCE_WEIGHT_GRID,
)
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    ClosedLoopMethodRuntimeOverride,
    FrozenAnchoredBlendModel,
    METHOD_POLICY,
    _fitted_runtime_bundle,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    MODEL_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v92-topology-repaired-source-weight-rollout-v1"
ANALYSIS_PROTOCOL = "tsc-v92-post-v91-source-weight-redevelopment-v1"


def source_weight_key(source_weight: float) -> str:
    weight = float(source_weight)
    if weight not in SOURCE_WEIGHT_GRID:
        raise ValueError(f"source weight is outside the frozen grid: {weight}")
    return str(weight).replace(".", "p")


def load_topology_repaired_runtime_override(
    *,
    model_path: Path,
    expected_model_sha256: str,
    city: str,
    source_weight: float,
    prediction_horizon_sec: int,
    guard_config: ContrastGuardConfig | None = None,
) -> ClosedLoopMethodRuntimeOverride:
    """Load one immutable refit and construct its source-weight runtime."""

    if _sha256(model_path) != str(expected_model_sha256):
        raise ValueError("topology-repaired model identity changed")
    payload = pickle.loads(Path(model_path).read_bytes())
    if (
        payload.get("protocol") != MODEL_PROTOCOL
        or payload.get("city") != city
        or float(source_weight) not in tuple(payload.get("source_weight_grid", ()))
    ):
        raise ValueError("topology-repaired model contract changed")
    offline = payload["model"]
    selected_candidate = str(payload["selected_candidate"])
    source_model = FrozenAnchoredBlendModel(
        anchor_model=offline.family_models[ANCHOR_FAMILY],
        correction_model=offline.family_models[CORRECTION_FAMILY],
        candidate=selected_candidate,
        anchor_objective_mode=offline.objective_modes[ANCHOR_FAMILY],
        correction_objective_mode=offline.objective_modes[CORRECTION_FAMILY],
    )
    weighted_model = FrozenCausalSourceWeightModel(
        source_model=source_model,
        target_only_model=offline.family_models["causal_target_only"],
        source_objective_mode="control_only",
        target_only_objective_mode=offline.objective_modes["causal_target_only"],
        source_weight=float(source_weight),
    )
    guarded = bool(guard_config is not None and guard_config.enabled)
    # evaluate_policy_v3 dispatches on the frozen family policy identifier. The
    # weight and guard remain explicit in the outer policy and provenance.
    runtime_policy = f"{ANCHOR_FAMILY}_contrast_{'guard' if guarded else 'raw'}"
    models = _fitted_runtime_bundle(
        offline_payload={"model_ensemble": [offline]},
        family=ANCHOR_FAMILY,
        model=weighted_model,
        prediction_horizon_sec=int(prediction_horizon_sec),
        guard_config=guard_config if guarded else None,
    )
    return ClosedLoopMethodRuntimeOverride(
        runtime_policy=runtime_policy,
        models=models,
        selected_candidate=selected_candidate,
        anchored_blend_model=weighted_model,
        provenance={
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "model_path": str(Path(model_path).resolve()),
            "model_sha256": str(expected_model_sha256),
            "model_protocol": MODEL_PROTOCOL,
            "city": city,
            "source_weight": float(source_weight),
            "guard_config": (
                {
                    "enabled": guard_config.enabled,
                    "risk_multiplier": guard_config.risk_multiplier,
                    "min_context_trust": guard_config.min_context_trust,
                    "margin": guard_config.margin,
                    "max_relative_rule_gap": guard_config.max_relative_rule_gap,
                }
                if guard_config is not None
                else None
            ),
            "topology_repair_protocol": payload["topology_repair_protocol"],
        },
    )


def run_topology_repaired_source_weight_rollout(
    *,
    model_path: Path,
    expected_model_sha256: str,
    source_weight: float,
    allowed_seeds: tuple[int, ...],
    rollout_kwargs: Mapping[str, Any],
    guard_config: ContrastGuardConfig | None = None,
) -> dict[str, Any]:
    """Evaluate one weight while retaining every frozen v43 rollout input."""

    protocol_path = Path(str(rollout_kwargs["protocol_spec_path"]))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent_path = Path(str(protocol["parent_protocol"]["path"]))
    if not parent_path.is_absolute():
        parent_path = protocol_path.resolve().parents[2] / parent_path
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cache = parent["counterfactual_cache"]
    prediction_horizon_sec = int(cache["control_interval_sec"]) * int(
        cache["counterfactual_horizon_intervals"]
    )
    scenario = str(rollout_kwargs["scenario"])
    city_scenarios = protocol["target_protocol"]["external_city_scenarios"]
    cities = [
        city for city, scenarios in city_scenarios.items() if scenario in scenarios
    ]
    if len(cities) != 1:
        raise ValueError("source-weight scenario is not uniquely assigned")
    city = str(cities[0])
    runtime_override = load_topology_repaired_runtime_override(
        model_path=model_path,
        expected_model_sha256=expected_model_sha256,
        city=city,
        source_weight=source_weight,
        prediction_horizon_sec=prediction_horizon_sec,
        guard_config=guard_config,
    )
    key = source_weight_key(source_weight)
    diagnostic = ClosedLoopDiagnosticSpec(
        name=f"cfcmt_topology_repaired_source_weight_{key}_mpc",
        source_policy=METHOD_POLICY,
        coordination_mode="sparse",
        result_protocol=RESULT_PROTOCOL,
        guard_config=guard_config,
        allowed_seeds=tuple(int(value) for value in allowed_seeds),
        analysis_status="post_v91_redevelopment_diagnostic_not_confirmation",
    )
    result = run_external_closed_loop_rollout(
        **dict(rollout_kwargs),
        policy=diagnostic.name,
        diagnostic_spec=diagnostic,
        method_runtime_override=runtime_override,
    )
    result["analysis_protocol"] = ANALYSIS_PROTOCOL
    result["source_weight"] = float(source_weight)
    result["source_guard_config"] = (
        runtime_override.provenance["guard_config"]
    )
    return result
