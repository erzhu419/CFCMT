"""Run closed-loop rollouts for fixed source-city causal mixtures."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_causal_source_components import (
    MODEL_BUNDLE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    FrozenCausalSourceMixtureModel,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    ClosedLoopMethodRuntimeOverride,
    FrozenAnchoredBlendModel,
    METHOD_POLICY,
    _fitted_runtime_bundle,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig
from cf_h2o.eval.traffic_signal_strict_target_only_fit import (
    MODEL_PROTOCOL as STRICT_TARGET_ONLY_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY,
)
from cf_h2o.eval.traffic_signal_source_only_guard import load_source_only_guard
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    MODEL_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v93-source-city-causal-component-rollout-v1"
ANALYSIS_PROTOCOL = "tsc-v93-source-city-causal-selection-development-v1"


def load_source_city_component_runtime_override(
    *,
    main_model_path: Path,
    expected_main_model_sha256: str,
    component_bundle_path: Path,
    expected_component_bundle_sha256: str,
    city: str,
    source_city_weights: Mapping[str, float],
    prediction_horizon_sec: int,
    guard_config: ContrastGuardConfig | None = None,
    source_only_guard_path: Path | None = None,
    expected_source_only_guard_sha256: str | None = None,
    strict_target_only_model_path: Path | None = None,
    expected_strict_target_only_model_sha256: str | None = None,
) -> ClosedLoopMethodRuntimeOverride:
    strict_values = (
        strict_target_only_model_path,
        expected_strict_target_only_model_sha256,
    )
    if any(value is not None for value in strict_values) and not all(
        value is not None for value in strict_values
    ):
        raise ValueError("strict target-only model arguments are incomplete")
    source_guard_values = (
        source_only_guard_path,
        expected_source_only_guard_sha256,
    )
    if any(value is not None for value in source_guard_values) and not all(
        value is not None for value in source_guard_values
    ):
        raise ValueError("source-only guard arguments are incomplete")
    if guard_config is not None and source_only_guard_path is not None:
        raise ValueError("direct and source-only artifact guards are mutually exclusive")
    if _sha256(main_model_path) != str(expected_main_model_sha256):
        raise ValueError("main topology-repaired model identity changed")
    if _sha256(component_bundle_path) != str(expected_component_bundle_sha256):
        raise ValueError("source-component bundle identity changed")
    main = pickle.loads(Path(main_model_path).read_bytes())
    bundle = pickle.loads(Path(component_bundle_path).read_bytes())
    if (
        main.get("protocol") != MODEL_PROTOCOL
        or main.get("city") != city
        or bundle.get("protocol") != MODEL_BUNDLE_PROTOCOL
        or bundle.get("city") != city
        or bundle.get("main_model_sha256") != str(expected_main_model_sha256)
        or bundle.get("selected_candidate") != main.get("selected_candidate")
    ):
        raise ValueError("source-component runtime contract changed")
    component_payloads = bundle["component_models"]
    unknown = set(str(value) for value in source_city_weights) - set(
        component_payloads
    )
    if unknown:
        raise ValueError(f"unknown source-city components: {sorted(unknown)}")
    selected_candidate = str(main["selected_candidate"])
    source_models = {
        group: FrozenAnchoredBlendModel(
            anchor_model=component_payloads[group].family_models[ANCHOR_FAMILY],
            correction_model=component_payloads[group].family_models[
                CORRECTION_FAMILY
            ],
            candidate=selected_candidate,
            anchor_objective_mode=component_payloads[group].objective_modes[
                ANCHOR_FAMILY
            ],
            correction_objective_mode=component_payloads[group].objective_modes[
                CORRECTION_FAMILY
            ],
        )
        for group, weight in source_city_weights.items()
        if float(weight) > 0.0
    }
    offline = main["model"]
    uncertainty_scale = 1.0
    source_guard_provenance = None
    if source_only_guard_path is not None:
        guard_config, uncertainty_scale, source_guard_payload = (
            load_source_only_guard(
                Path(source_only_guard_path),
                expected_sha256=str(expected_source_only_guard_sha256),
                expected_prior_policy=str(offline.prior_spec.key),
                family=ANCHOR_FAMILY,
            )
        )
        source_guard_provenance = {
            "path": str(Path(source_only_guard_path).resolve()),
            "sha256": str(expected_source_only_guard_sha256),
            "protocol": source_guard_payload["protocol"],
            "source_domains": list(source_guard_payload["source_domains"]),
            "target_data_consumed": False,
            "target_closed_loop_consumed": False,
            "uncertainty_scale": float(uncertainty_scale),
        }
    if strict_target_only_model_path is not None:
        strict_path = Path(strict_target_only_model_path)
        if _sha256(strict_path) != str(expected_strict_target_only_model_sha256):
            raise ValueError("strict target-only model identity changed")
        strict = pickle.loads(strict_path.read_bytes())
        if (
            strict.get("protocol") != STRICT_TARGET_ONLY_MODEL_PROTOCOL
            or strict.get("city") != city
            or strict.get("family") != TARGET_ONLY_FAMILY
            or strict.get("main_model_sha256")
            != str(expected_main_model_sha256)
            or strict.get("prior_policy") != str(offline.prior_spec.key)
            or tuple(strict.get("selected_group_ids", ()))
            != tuple(main.get("selected_group_ids", ()))
            or int(
                strict.get("fit_diagnostics", {}).get(
                    "source_row_count_consumed", -1
                )
            )
            != 0
        ):
            raise ValueError("strict target-only runtime contract changed")
        target_only_model = strict["model"]
        target_only_objective_mode = str(strict["objective_mode"])
        target_anchor = {
            "family": TARGET_ONLY_FAMILY,
            "protocol": STRICT_TARGET_ONLY_MODEL_PROTOCOL,
            "path": str(strict_path.resolve()),
            "sha256": str(expected_strict_target_only_model_sha256),
            "source_rows_consumed": 0,
        }
    else:
        target_only_model = offline.family_models["causal_target_only"]
        target_only_objective_mode = offline.objective_modes[
            "causal_target_only"
        ]
        target_anchor = {
            "family": "causal_target_only",
            "protocol": "legacy-near-target-only-anchor-v1",
            "path": str(Path(main_model_path).resolve()),
            "sha256": str(expected_main_model_sha256),
        }
    mixture = FrozenCausalSourceMixtureModel(
        source_models=source_models,
        target_only_model=target_only_model,
        source_objective_modes={group: "control_only" for group in source_models},
        target_only_objective_mode=target_only_objective_mode,
        source_city_weights={
            str(group): float(weight)
            for group, weight in source_city_weights.items()
            if float(weight) > 0.0
        },
        uncertainty_scale=uncertainty_scale,
    )
    guarded = guard_config is not None
    runtime_policy = f"{ANCHOR_FAMILY}_contrast_{'guard' if guarded else 'raw'}"
    models = _fitted_runtime_bundle(
        offline_payload={"model_ensemble": [offline]},
        family=ANCHOR_FAMILY,
        model=mixture,
        prediction_horizon_sec=int(prediction_horizon_sec),
        guard_config=guard_config if guarded else None,
    )
    return ClosedLoopMethodRuntimeOverride(
        runtime_policy=runtime_policy,
        models=models,
        selected_candidate=selected_candidate,
        anchored_blend_model=mixture,
        provenance={
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "main_model_path": str(Path(main_model_path).resolve()),
            "main_model_sha256": str(expected_main_model_sha256),
            "component_bundle_path": str(Path(component_bundle_path).resolve()),
            "component_bundle_sha256": str(expected_component_bundle_sha256),
            "component_bundle_protocol": MODEL_BUNDLE_PROTOCOL,
            "city": city,
            "source_city_weights": dict(mixture.source_city_weights),
            "source_mass": float(mixture.source_mass),
            "target_only_anchor": target_anchor,
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
            "source_only_guard": source_guard_provenance,
            "uncertainty_scale": float(uncertainty_scale),
            "topology_repair_protocol": bundle["topology_repair_protocol"],
        },
    )


def run_source_city_component_rollout(
    *,
    main_model_path: Path,
    expected_main_model_sha256: str,
    component_bundle_path: Path,
    expected_component_bundle_sha256: str,
    source_city_weights: Mapping[str, float],
    candidate_key: str,
    allowed_seeds: tuple[int, ...],
    rollout_kwargs: Mapping[str, Any],
    guard_config: ContrastGuardConfig | None = None,
    source_only_guard_path: Path | None = None,
    expected_source_only_guard_sha256: str | None = None,
    strict_target_only_model_path: Path | None = None,
    expected_strict_target_only_model_sha256: str | None = None,
) -> dict[str, Any]:
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
        raise ValueError("source-component scenario is not uniquely assigned")
    city = str(cities[0])
    runtime_override = load_source_city_component_runtime_override(
        main_model_path=main_model_path,
        expected_main_model_sha256=expected_main_model_sha256,
        component_bundle_path=component_bundle_path,
        expected_component_bundle_sha256=expected_component_bundle_sha256,
        city=city,
        source_city_weights=source_city_weights,
        prediction_horizon_sec=prediction_horizon_sec,
        guard_config=guard_config,
        source_only_guard_path=source_only_guard_path,
        expected_source_only_guard_sha256=expected_source_only_guard_sha256,
        strict_target_only_model_path=strict_target_only_model_path,
        expected_strict_target_only_model_sha256=(
            expected_strict_target_only_model_sha256
        ),
    )
    guarded = bool(
        guard_config is not None or source_only_guard_path is not None
    )
    diagnostic = ClosedLoopDiagnosticSpec(
        name=f"cfcmt_source_component_{candidate_key}_{'guard' if guarded else 'raw'}",
        source_policy=METHOD_POLICY,
        coordination_mode="sparse",
        result_protocol=RESULT_PROTOCOL,
        guard_config=guard_config,
        allowed_seeds=tuple(int(value) for value in allowed_seeds),
        analysis_status="post-v92-source-city-selection-development",
    )
    result = run_external_closed_loop_rollout(
        **dict(rollout_kwargs),
        policy=diagnostic.name,
        diagnostic_spec=diagnostic,
        method_runtime_override=runtime_override,
    )
    result["analysis_protocol"] = ANALYSIS_PROTOCOL
    result["candidate_key"] = str(candidate_key)
    result["source_city_weights"] = dict(
        runtime_override.provenance["source_city_weights"]
    )
    result["source_mass"] = float(runtime_override.provenance["source_mass"])
    result["source_guard_config"] = runtime_override.provenance["guard_config"]
    result["source_only_guard"] = runtime_override.provenance[
        "source_only_guard"
    ]
    result["target_only_anchor"] = runtime_override.provenance[
        "target_only_anchor"
    ]
    return result
