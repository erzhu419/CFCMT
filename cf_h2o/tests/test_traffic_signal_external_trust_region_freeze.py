from __future__ import annotations

from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    _fitted_runtime_bundle,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    _finite_sample_upper_quantile,
    select_oof_pressure_regularizer,
    select_oof_trust_region,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardConfig,
    PriorRegularizationConfig,
)


def _records(*, actual_delta: float):
    return [
        {
            "fold_index": fold,
            "validation_seed": 5057 + fold,
            "scenario": scenario,
            "group_id": f"{scenario}:fold{fold}:group{index}",
            "learned_differs": index < 4,
            "predicted_delta": -1.0,
            "uncertainty": 0.1,
            "context_trust": 0.8,
            "relative_rule_gap": 0.04,
            "actual_delta": actual_delta,
            "candidate_features": {"total_veh": 4.0},
        }
        for fold in (0, 1)
        for scenario in ("a", "b")
        for index in range(20)
    ]


def test_finite_sample_upper_quantile_uses_conformal_rank():
    assert _finite_sample_upper_quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 3.0


def test_oof_trust_region_enables_only_fold_safe_improvement():
    config, audit = select_oof_trust_region(_records(actual_delta=-0.5))

    assert config.enabled
    assert audit["decision"] == "deploy_oof_conformal_trust_region"
    assert audit["selected"]["worst_fold_mean_delta"] < 0.0
    assert audit["selected"]["accepted_overrides"] > 0


def test_oof_trust_region_falls_back_when_actions_are_harmful():
    config, audit = select_oof_trust_region(_records(actual_delta=0.2))

    assert not config.enabled
    assert audit["decision"] == "deploy_pressure_prior_only"
    assert audit["selected"] is None


def test_oof_trust_region_excludes_empty_local_states():
    records = _records(actual_delta=-0.5)
    for row in records:
        row["candidate_features"]["total_veh"] = 0.0

    config, audit = select_oof_trust_region(records)

    assert not config.enabled
    assert audit["decision"] == "deploy_pressure_prior_only"


def test_oof_pressure_regularizer_selects_fold_safe_operational_actions():
    records = _records(actual_delta=-0.5)
    for row in records:
        row["relative_rule_gap"] = 0.2

    config, audit = select_oof_pressure_regularizer(records)

    assert config.enabled
    assert audit["decision"] == "deploy_oof_pressure_regularized_direct_mpc"
    assert audit["selected"]["every_fold_covered"] is True
    assert audit["selected"]["worst_fold_mean_delta"] < 0.0


def test_oof_pressure_regularizer_falls_back_for_harmful_actions():
    config, audit = select_oof_pressure_regularizer(_records(actual_delta=0.2))

    assert config == PriorRegularizationConfig()
    assert audit["decision"] == "deploy_pressure_prior_only"


def test_oof_pressure_regularizer_excludes_empty_states():
    records = _records(actual_delta=-0.5)
    for row in records:
        row["candidate_features"]["total_veh"] = 0.0

    config, audit = select_oof_pressure_regularizer(records)

    assert not config.enabled
    assert audit["selected"] is None


def test_runtime_bundle_installs_frozen_guard():
    class Prior:
        key = "prior"

    class Offline:
        prior_spec = Prior()
        objective_modes = {"family": "control_only"}

    guard = ContrastGuardConfig(
        enabled=True,
        risk_multiplier=1.5,
        min_context_trust=0.25,
        margin=0.1,
        max_relative_rule_gap=0.05,
    )
    bundle = _fitted_runtime_bundle(
        offline_payload={"model_ensemble": [Offline()]},
        family="family",
        model=object(),
        prediction_horizon_sec=60,
        guard_config=guard,
    )

    assert bundle.guards["family"] == guard
    assert bundle.diagnostics["deployment_guard_installed"] is True


def test_runtime_bundle_installs_frozen_pressure_regularizer():
    class Prior:
        key = "prior"

    class Offline:
        prior_spec = Prior()
        objective_modes = {"family": "control_only"}

    regularizer = PriorRegularizationConfig(
        enabled=True,
        blend_weight=6.0,
        risk_multiplier=0.25,
        min_context_trust=0.5,
    )
    bundle = _fitted_runtime_bundle(
        offline_payload={"model_ensemble": [Offline()]},
        family="family",
        model=object(),
        prediction_horizon_sec=60,
        regularizer_config=regularizer,
    )

    assert bundle.regularizers["family"] == regularizer
    assert bundle.diagnostics["deployment_regularizer_installed"] is True
