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
    FrozenCausalSourceSelectorModel,
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
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    FittedContrastModelsV3,
    PriorRegularizationConfig,
)
from cf_h2o.eval.traffic_signal_strict_target_only_fit import (
    MODEL_PROTOCOL as STRICT_TARGET_ONLY_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY,
)
from cf_h2o.eval.traffic_signal_source_only_guard import load_source_only_guard
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    MODEL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    H2OPLUS_FAMILY as PURE_WAITING_H2OPLUS_FAMILY,
    H2OPLUS_MODEL_PROTOCOL as PURE_WAITING_H2OPLUS_MODEL_PROTOCOL,
    MODEL_BUNDLE_PROTOCOL as PURE_WAITING_MODEL_BUNDLE_PROTOCOL,
    STRICT_TARGET_MODEL_PROTOCOL as PURE_WAITING_STRICT_TARGET_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY as PURE_WAITING_TARGET_ONLY_FAMILY,
    validate_architecture_matched_target_contract,
    validate_target_label_free_blend_contract,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    PURE_WAITING_RESULT_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v93-source-city-causal-component-rollout-v1"
ANALYSIS_PROTOCOL = "tsc-v93-source-city-causal-selection-development-v1"
PURE_WAITING_ANALYSIS_PROTOCOL = (
    "tsc-v117-pure-waiting-selector-heldout-city-confirmation-v2"
)
PURE_WAITING_ROLLOUT_PROTOCOL = (
    "tsc-v117-pure-waiting-selector-heldout-city-rollout-v2"
)
PURE_WAITING_ARMS = (
    "target_only_b100",
    "source_selected_b100",
    "source_selected_b0",
)
PURE_WAITING_H2OPLUS_ARMS = (
    "h2oplus_dense_b0",
    "h2oplus_dense_b100",
)


def load_pure_waiting_h2oplus_runtime_override(
    *,
    model_path: Path,
    expected_model_sha256: str,
    expected_adaptation_contract_sha256: str,
    city: str,
    arm: str,
    prediction_horizon_sec: int = 450,
) -> ClosedLoopMethodRuntimeOverride:
    """Load an H2O+-style dense comparator with the V115 information budget."""

    if arm not in PURE_WAITING_H2OPLUS_ARMS:
        raise ValueError(f"unknown pure-waiting H2O+ arm: {arm}")
    if _sha256(model_path) != str(expected_model_sha256):
        raise ValueError("pure-waiting H2O+ model identity changed")
    payload = pickle.loads(Path(model_path).read_bytes())
    budget = 0 if arm.endswith("_b0") else 100
    diagnostics = dict(payload.get("fit_diagnostics", {}))
    fitted_domains = set(str(value) for value in diagnostics.get("source_domains", ()))
    expected_target_presence = budget > 0
    if (
        payload.get("protocol") != PURE_WAITING_H2OPLUS_MODEL_PROTOCOL
        or payload.get("city") != city
        or payload.get("family") != PURE_WAITING_H2OPLUS_FAMILY
        or payload.get("objective_mode") != "control_only"
        or payload.get("target_name") != "prefix_mean_cost_450s"
        or payload.get("prior_policy") != "phase_pressure"
        or str(getattr(payload.get("prior_spec"), "key", ""))
        != "phase_pressure"
        or int(payload.get("target_group_budget", -1)) != budget
        or payload.get("adaptation_contract_sha256")
        != str(expected_adaptation_contract_sha256)
        or len(tuple(payload.get("selected_group_ids", ()))) != budget
        or (city in fitted_domains) != expected_target_presence
        or int(diagnostics.get("target_adaptation_groups", -1)) != budget
        or int(prediction_horizon_sec) != 450
    ):
        raise ValueError("pure-waiting H2O+ runtime contract changed")
    model = payload["model"]
    models = FittedContrastModelsV3(
        family_models={PURE_WAITING_H2OPLUS_FAMILY: model},
        prior_policy="phase_pressure",
        prior_spec=payload["prior_spec"],
        prediction_horizon_sec=450,
        objective_modes={PURE_WAITING_H2OPLUS_FAMILY: "control_only"},
        guards={PURE_WAITING_H2OPLUS_FAMILY: ContrastGuardConfig()},
        regularizers={
            PURE_WAITING_H2OPLUS_FAMILY: PriorRegularizationConfig()
        },
        target_support=None,
        hierarchy_layers={},
        diagnostics={
            "protocol": PURE_WAITING_ANALYSIS_PROTOCOL,
            "arm": arm,
            "city": city,
            "comparator": "h2oplus_style_dense_residual_mpc",
        },
    )
    return ClosedLoopMethodRuntimeOverride(
        runtime_policy=f"{PURE_WAITING_H2OPLUS_FAMILY}_contrast_raw",
        models=models,
        selected_candidate=arm,
        anchored_blend_model=None,
        provenance={
            "analysis_protocol": PURE_WAITING_ANALYSIS_PROTOCOL,
            "arm": arm,
            "city": city,
            "model_path": str(Path(model_path).resolve()),
            "model_sha256": str(expected_model_sha256),
            "adaptation_contract_sha256": str(
                expected_adaptation_contract_sha256
            ),
            "target_transition_label_budget": budget,
            "comparator": "h2oplus_style_dense_residual_mpc",
            "guard_config": None,
        },
    )


def _pure_waiting_profile(
    selector: Mapping[str, Any],
    *,
    arm: str,
) -> tuple[dict[str, Any], str, bool]:
    if arm not in PURE_WAITING_ARMS:
        raise ValueError(f"unknown pure-waiting confirmation arm: {arm}")
    if arm == "source_selected_b0":
        section = selector.get("zero_shot_source_admission")
        if not isinstance(section, Mapping) or not section.get(
            "source_transfer_gate_passed", False
        ):
            raise ValueError("V116 did not authorize the B0 source profile")
        selection = dict(section["full_development_selection"])
        definitions = dict(section["profile_definitions"])
        selected_key = selection.get("selected_profile_key")
        if selected_key is None:
            raise ValueError("V116 B0 selection fell back to phase pressure")
        return dict(definitions[str(selected_key)]), str(selected_key), True

    selection = dict(selector["full_development_selection"])
    definitions = dict(selector["profile_definitions"])
    selected_key = selection.get("selected_profile_key")
    if selected_key is None:
        raise ValueError("V116 B100 selection fell back to phase pressure")
    selected = dict(definitions[str(selected_key)])
    if arm == "source_selected_b100":
        if (
            not selector.get("source_transfer_gate_passed", False)
            or selected.get("source_city_group") is None
        ):
            raise ValueError("V116 did not authorize the B100 source profile")
        return selected, str(selected_key), False
    target_key = (
        selected_key
        if selected.get("source_city_group") is None
        else selected.get("target_profile_key")
    )
    if target_key is None or str(target_key) not in definitions:
        raise ValueError("V116 selected source has no matched target-only profile")
    target = dict(definitions[str(target_key)])
    if target.get("source_city_group") is not None:
        raise ValueError("V116 target-only comparator contains source data")
    return target, str(target_key), False


def load_pure_waiting_selector_runtime_override(
    *,
    component_bundle_path: Path,
    expected_component_bundle_sha256: str,
    selector_result_path: Path,
    expected_selector_result_sha256: str,
    city: str,
    arm: str,
    prediction_horizon_sec: int = 450,
    strict_target_only_model_path: Path | None = None,
    expected_strict_target_only_model_sha256: str | None = None,
) -> ClosedLoopMethodRuntimeOverride:
    """Instantiate the exact V116 selector profile for prospective rollout."""

    if _sha256(component_bundle_path) != str(expected_component_bundle_sha256):
        raise ValueError("pure-waiting component bundle identity changed")
    if _sha256(selector_result_path) != str(expected_selector_result_sha256):
        raise ValueError("pure-waiting selector result identity changed")
    selector = json.loads(Path(selector_result_path).read_text(encoding="utf-8"))
    if (
        selector.get("protocol") != PURE_WAITING_RESULT_PROTOCOL
        or selector.get("selector_cache_audit", {}).get("decision")
        != "authorize_v116_pure_waiting_nested_source_selection"
    ):
        raise ValueError("V116 selector contract changed")
    profile, profile_key, zero_shot = _pure_waiting_profile(selector, arm=arm)
    bundle = pickle.loads(Path(component_bundle_path).read_bytes())
    expected_budget = 0 if zero_shot else 100
    component_payloads = dict(bundle.get("component_models", {}))
    if (
        bundle.get("protocol") != PURE_WAITING_MODEL_BUNDLE_PROTOCOL
        or bundle.get("city") != city
        or bundle.get("target_name") != "prefix_mean_cost_450s"
        or bundle.get("prior_policy") != "phase_pressure"
        or int(bundle.get("target_group_budget", -1)) != expected_budget
        or not component_payloads
        or int(prediction_horizon_sec) != 450
    ):
        raise ValueError("pure-waiting component runtime contract changed")
    validate_target_label_free_blend_contract(bundle)
    prior_specs = [models.prior_spec for models in component_payloads.values()]
    if any(str(spec.key) != "phase_pressure" for spec in prior_specs):
        raise ValueError("pure-waiting component priors changed")
    prior_spec = prior_specs[0]
    selected_candidate = str(bundle["selected_candidate"])
    source_models = {
        str(group): FrozenAnchoredBlendModel(
            anchor_model=models.family_models[ANCHOR_FAMILY],
            correction_model=models.family_models[CORRECTION_FAMILY],
            candidate=selected_candidate,
            anchor_objective_mode=models.objective_modes[ANCHOR_FAMILY],
            correction_objective_mode=models.objective_modes[CORRECTION_FAMILY],
        )
        for group, models in component_payloads.items()
    }

    strict = None
    strict_values = (
        strict_target_only_model_path,
        expected_strict_target_only_model_sha256,
    )
    if any(value is not None for value in strict_values) and not all(
        value is not None for value in strict_values
    ):
        raise ValueError("pure-waiting strict target arguments are incomplete")
    if not zero_shot:
        if strict_target_only_model_path is None:
            raise ValueError("B100 runtime requires the strict target-only model")
        if _sha256(strict_target_only_model_path) != str(
            expected_strict_target_only_model_sha256
        ):
            raise ValueError("pure-waiting strict target identity changed")
        strict = pickle.loads(Path(strict_target_only_model_path).read_bytes())
        if (
            strict.get("protocol") != PURE_WAITING_STRICT_TARGET_MODEL_PROTOCOL
            or strict.get("city") != city
            or strict.get("family") != PURE_WAITING_TARGET_ONLY_FAMILY
            or strict.get("target_name") != "prefix_mean_cost_450s"
            or int(strict.get("target_group_budget", -1)) != 100
            or strict.get("adaptation_contract_sha256")
            != bundle.get("adaptation_contract_sha256")
            or tuple(strict.get("selected_group_ids", ()))
            != tuple(bundle.get("selected_group_ids", ()))
            or int(
                strict.get("fit_diagnostics", {}).get(
                    "source_row_count_consumed", -1
                )
            )
            != 0
        ):
            raise ValueError("pure-waiting strict target runtime contract changed")
        validate_architecture_matched_target_contract(strict)
        if (
            strict.get("selected_candidate") != selected_candidate
            or strict.get("blend_candidate_selection_sha256")
            != bundle.get("candidate_selection_sha256")
        ):
            raise ValueError(
                "pure-waiting target-only/source component architecture differs"
            )

    source_group = profile.get("source_city_group")
    source_weight = float(profile.get("source_weight", 0.0))
    minimum_support = float(profile.get("minimum_source_support", 0.0))
    if arm == "target_only_b100":
        if source_group is not None or source_weight != 0.0 or strict is None:
            raise ValueError("target-only V116 arm changed information budget")
        runtime_model = strict["model"]
    else:
        normalized_group = str(source_group)
        if zero_shot:
            if not normalized_group.startswith("b0_"):
                raise ValueError("B0 selector profile lost its information label")
            normalized_group = normalized_group.removeprefix("b0_")
        runtime_model = FrozenCausalSourceSelectorModel(
            source_models=source_models,
            source_objective_modes={
                group: "control_only" for group in source_models
            },
            source_profile=normalized_group,
            source_weight=source_weight,
            minimum_source_support=minimum_support,
            target_only_model=None if zero_shot else strict["model"],
            target_only_objective_mode="control_only",
        )
    guard = ContrastGuardConfig(
        enabled=True,
        risk_multiplier=float(profile["risk_multiplier"]),
        min_context_trust=float(profile["minimum_context_trust"]),
        margin=0.0,
        max_relative_rule_gap=1.0,
    )
    models = FittedContrastModelsV3(
        family_models={ANCHOR_FAMILY: runtime_model},
        prior_policy=str(prior_spec.key),
        prior_spec=prior_spec,
        prediction_horizon_sec=450,
        objective_modes={ANCHOR_FAMILY: "control_only"},
        guards={ANCHOR_FAMILY: guard},
        regularizers={ANCHOR_FAMILY: PriorRegularizationConfig()},
        target_support=None,
        hierarchy_layers={},
        diagnostics={
            "protocol": PURE_WAITING_ANALYSIS_PROTOCOL,
            "arm": arm,
            "profile_key": profile_key,
            "selector_city": selector.get("city"),
            "deployment_city": city,
        },
    )
    return ClosedLoopMethodRuntimeOverride(
        runtime_policy=f"{ANCHOR_FAMILY}_contrast_guard",
        models=models,
        selected_candidate=selected_candidate,
        anchored_blend_model=runtime_model,
        provenance={
            "analysis_protocol": PURE_WAITING_ANALYSIS_PROTOCOL,
            "arm": arm,
            "city": city,
            "selector_city": selector.get("city"),
            "selector_result_path": str(Path(selector_result_path).resolve()),
            "selector_result_sha256": str(expected_selector_result_sha256),
            "component_bundle_path": str(Path(component_bundle_path).resolve()),
            "component_bundle_sha256": str(expected_component_bundle_sha256),
            "strict_target_only_model_path": (
                str(Path(strict_target_only_model_path).resolve())
                if strict_target_only_model_path is not None
                else None
            ),
            "strict_target_only_model_sha256": (
                str(expected_strict_target_only_model_sha256)
                if strict_target_only_model_path is not None
                else None
            ),
            "profile_key": profile_key,
            "profile": profile,
            "target_transition_label_budget": 0 if zero_shot else 100,
            "source_profile": source_group,
            "source_weight": source_weight,
            "minimum_source_support": minimum_support,
            "guard_config": {
                "enabled": True,
                "risk_multiplier": guard.risk_multiplier,
                "min_context_trust": guard.min_context_trust,
                "margin": guard.margin,
                "max_relative_rule_gap": guard.max_relative_rule_gap,
            },
        },
    )


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


def run_pure_waiting_selector_rollout(
    *,
    component_bundle_path: Path,
    expected_component_bundle_sha256: str,
    selector_result_path: Path,
    expected_selector_result_sha256: str,
    arm: str,
    candidate_key: str,
    allowed_seeds: tuple[int, ...],
    rollout_kwargs: Mapping[str, Any],
    strict_target_only_model_path: Path | None = None,
    expected_strict_target_only_model_sha256: str | None = None,
) -> dict[str, Any]:
    protocol_path = Path(str(rollout_kwargs["protocol_spec_path"]))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    parent_path = Path(str(protocol["parent_protocol"]["path"]))
    if not parent_path.is_absolute():
        parent_path = protocol_path.resolve().parents[2] / parent_path
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cache = dict(parent["counterfactual_cache"])
    prediction_horizon_sec = int(cache["control_interval_sec"]) * int(
        cache["counterfactual_horizon_intervals"]
    )
    if prediction_horizon_sec != 450:
        raise ValueError("V117 requires the frozen 450-second rollout horizon")
    scenario = str(rollout_kwargs["scenario"])
    city_scenarios = protocol["target_protocol"]["external_city_scenarios"]
    cities = [
        city for city, scenarios in city_scenarios.items() if scenario in scenarios
    ]
    if len(cities) != 1:
        raise ValueError("V117 scenario is not uniquely assigned to one city")
    city = str(cities[0])
    selector = json.loads(Path(selector_result_path).read_text(encoding="utf-8"))
    if str(selector.get("city")) == city:
        raise ValueError("V117 deployment city must be held out from V116 selection")
    runtime_override = load_pure_waiting_selector_runtime_override(
        component_bundle_path=component_bundle_path,
        expected_component_bundle_sha256=expected_component_bundle_sha256,
        selector_result_path=selector_result_path,
        expected_selector_result_sha256=expected_selector_result_sha256,
        city=city,
        arm=arm,
        prediction_horizon_sec=prediction_horizon_sec,
        strict_target_only_model_path=strict_target_only_model_path,
        expected_strict_target_only_model_sha256=(
            expected_strict_target_only_model_sha256
        ),
    )
    diagnostic = ClosedLoopDiagnosticSpec(
        name=f"cfcmt_pure_waiting_{candidate_key}",
        source_policy=METHOD_POLICY,
        coordination_mode="sparse",
        result_protocol=PURE_WAITING_ROLLOUT_PROTOCOL,
        guard_config=runtime_override.models.guards[ANCHOR_FAMILY],
        allowed_seeds=tuple(int(value) for value in allowed_seeds),
        analysis_status="prospective-v117-selector-city-heldout-confirmation",
    )
    result = run_external_closed_loop_rollout(
        **dict(rollout_kwargs),
        policy=diagnostic.name,
        diagnostic_spec=diagnostic,
        method_runtime_override=runtime_override,
    )
    result["analysis_protocol"] = PURE_WAITING_ANALYSIS_PROTOCOL
    result["candidate_key"] = str(candidate_key)
    result["selector_arm"] = arm
    result["selector_profile"] = runtime_override.provenance["profile"]
    result["target_transition_label_budget"] = runtime_override.provenance[
        "target_transition_label_budget"
    ]
    result["selector_result_sha256"] = str(expected_selector_result_sha256)
    result["component_bundle_sha256"] = str(
        expected_component_bundle_sha256
    )
    result["strict_target_only_model_sha256"] = (
        str(expected_strict_target_only_model_sha256)
        if strict_target_only_model_path is not None
        else None
    )
    return result
