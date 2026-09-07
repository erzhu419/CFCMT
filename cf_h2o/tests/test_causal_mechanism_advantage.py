import numpy as np

from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    CausalMechanismAdvantageConfig,
    CausalMechanismAdvantageModel,
    BoostedDirectRolloutValueCausalMechanismAdvantageModel,
    DirectRolloutValueCausalMechanismAdvantageModel,
    FusedCausalMechanismAdvantageModel,
    FusedPhysicalResidualCausalMechanismAdvantageModel,
    FusedRigidActionAdvantageModel,
    LOCAL_PHYSICAL_RESIDUAL_PROTOCOL,
    MECHANISM_OUTCOME_REPLACEMENT_STACK_PROTOCOL,
    PhysicalResidualCausalMechanismAdvantageModel,
    RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL,
    RolloutValueResidualCausalMechanismAdvantageModel,
    SOURCE_EXPERT_GATE_FEATURE_NAMES,
    SOURCE_OOF_RIDGE_LCB_EXPERT_GATE_PROTOCOL,
    _apply_source_expert_gate,
    _local_physical_mechanism_dataset,
    _source_expert_gate_records,
)
from cf_h2o.traffic_signal.mechanism_world_model import (
    MechanismDataset,
    MechanismDefinition,
)


DEFINITIONS = (
    MechanismDefinition(
        name="queue_propagation",
        target_name="next_total_queue",
        parent_variants={
            "minimal": ("total_q", "delta_green_q"),
            "physical": (
                "total_q",
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "delta_switch_indicator",
            ),
        },
    ),
    MechanismDefinition(
        name="control_cost",
        target_name="interval_cost",
        parent_variants={
            "minimal": ("total_q", "delta_green_q"),
            "physical": (
                "total_q",
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "delta_switch_indicator",
            ),
        },
    ),
)


def _dataset(include_target: bool = False) -> MechanismDataset:
    features = []
    contexts = []
    interval_cost = []
    next_total_queue = []
    domains = []
    groups = []
    references = []
    domain_names = ["a", "b", "c", "d"] + (["target"] if include_target else [])
    for domain_index, domain in enumerate(domain_names):
        group_count = 8 if domain == "target" else 12
        for group_index in range(group_count):
            state = (group_index - group_count / 2.0) / max(group_count / 3.0, 1.0)
            for action in (0.0, 1.0):
                action_effect = action * (-0.8 * state + 0.04 * domain_index)
                features.append(
                    [
                        5.0 + state,
                        action * (4.0 + state),
                        0.0,
                        action * (4.0 + state),
                        action,
                    ]
                )
                contexts.append([float(domain_index), float(domain_index % 2)])
                interval_cost.append(action_effect)
                next_total_queue.append(1.4 * action_effect)
                domains.append(domain)
                groups.append(f"{domain}:g{group_index}")
                references.append(action == 0.0)
    interval = np.asarray(interval_cost, dtype=float)
    queue = np.asarray(next_total_queue, dtype=float)
    return MechanismDataset(
        feature_names=(
            "total_q",
            "green_q",
            "reference_green_q",
            "delta_green_q",
            "delta_switch_indicator",
        ),
        features=np.asarray(features, dtype=float),
        context_names=("network_scale", "network_family"),
        context=np.asarray(contexts, dtype=float),
        priors={
            "interval_cost": np.zeros_like(interval),
            "next_total_queue": np.zeros_like(queue),
        },
        targets={"interval_cost": interval, "next_total_queue": queue},
        domains=np.asarray(domains),
        metadata={"action_group_ids": groups, "is_reference": references},
    )


def test_mechanism_stack_uses_cross_city_predictions_and_preserves_reference():
    dataset = _dataset()
    model = CausalMechanismAdvantageModel(DEFINITIONS)
    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]
    reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)

    assert diagnostics["source"]["source_domains"] == ["a", "b", "c", "d"]
    assert diagnostics["source"]["selected_candidate"]["blend_weight"] in {
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    }
    assert np.isfinite(prediction["mean"]).all()
    assert np.all(prediction["uncertainty"] >= 0.0)
    assert np.allclose(prediction["mean"][reference], 0.0)
    assert np.allclose(prediction["uncertainty"][reference], 0.0)


def test_rigid_anchored_stack_adds_a_residual_instead_of_replacing_core():
    source = np.asarray([1.0, -2.0])
    component = np.asarray([0.4, 0.8])
    replacement = CausalMechanismAdvantageModel(DEFINITIONS)
    residual = CausalMechanismAdvantageModel(
        DEFINITIONS,
        config=CausalMechanismAdvantageConfig(
            source_stack_protocol=(
                RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
            )
        ),
    )

    assert replacement.config.source_stack_protocol == (
        MECHANISM_OUTCOME_REPLACEMENT_STACK_PROTOCOL
    )
    assert np.allclose(
        replacement._combine_source_stack(source, component, 0.5),
        [0.7, -0.6],
    )
    assert np.allclose(
        residual._combine_source_stack(source, component, 0.5),
        [1.2, -1.6],
    )
    assert np.allclose(
        residual._stack_disagreement(source, component),
        np.abs(component),
    )


def test_rigid_anchored_stack_uses_pair_excluded_core_residuals():
    model = CausalMechanismAdvantageModel(
        DEFINITIONS,
        config=CausalMechanismAdvantageConfig(
            source_stack_protocol=(
                RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
            )
        ),
    )

    source = model.fit(_dataset())["source"]

    assert source["source_stack_protocol"] == (
        RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
    )
    assert source["pair_excluded_core_fit_count"] == 6
    assert source["pair_excluded_mechanism_fit_count"] == 6
    for row in source["grid"]:
        if row["blend_weight"] == 0.0:
            assert row["domain_regret"] == source["core_domain_regret"]


def test_rollout_value_residual_retains_only_policy_conditioned_terminal_load():
    terminal = MechanismDefinition(
        name="terminal_clearance",
        target_name="terminal_system_load",
        parent_variants={
            "minimal": ("total_q", "delta_green_q"),
            "physical": (
                "total_q",
                "green_q",
                "reference_green_q",
                "delta_green_q",
                "delta_switch_indicator",
            ),
        },
    )
    model = RolloutValueResidualCausalMechanismAdvantageModel(
        (*DEFINITIONS, terminal),
        config=CausalMechanismAdvantageConfig(
            source_stack_protocol=(
                RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
            )
        ),
    )

    assert tuple(definition.name for definition in model.definitions) == (
        "terminal_clearance",
    )
    assert model.config.mechanism_dataset_protocol == (
        "group_invariant_local_physical_simulator_residual_v1"
    )
    assert model.config.mechanism_action_ranking_selection is True
    assert model.config.source_stack_protocol == (
        RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
    )

    direct = DirectRolloutValueCausalMechanismAdvantageModel(
        (*DEFINITIONS, terminal),
        config=CausalMechanismAdvantageConfig(
            source_stack_protocol=(
                RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
            )
        ),
    )
    assert tuple(definition.name for definition in direct.definitions) == (
        "terminal_clearance",
    )
    assert direct.config.mechanism_dataset_protocol == (
        "domain_normalized_outcome_zero_prior_v1"
    )
    assert direct.config.mechanism_action_ranking_selection is True

    boosted = BoostedDirectRolloutValueCausalMechanismAdvantageModel(
        (*DEFINITIONS, terminal),
        config=CausalMechanismAdvantageConfig(
            source_stack_protocol=(
                RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
            )
        ),
    )
    assert boosted.config.mechanism_estimator_protocol == "causal_boosted_v1"
    assert boosted.config.mechanism_dataset_protocol == (
        "domain_normalized_outcome_zero_prior_v1"
    )
    assert boosted.config.mechanism_action_ranking_selection is True


def test_source_oof_expert_gate_uses_group_level_rigid_residual_disagreements():
    dataset = _dataset()
    groups = np.asarray(dataset.metadata["action_group_ids"])
    domains = np.asarray(dataset.domains)
    reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)
    core = np.where(reference, 0.0, 0.15)
    residual = np.where(reference, 0.12, 0.0)
    target = np.zeros(dataset.size, dtype=float)
    for group_index, group in enumerate(np.unique(groups)):
        rows = np.flatnonzero(groups == group)
        target[rows] = [0.0, -0.20 if group_index % 3 else 0.10]

    records = _source_expert_gate_records(
        core,
        residual,
        groups,
        domains=domains,
        target=target,
    )
    assert records["features"].shape == (
        np.unique(groups).size,
        len(SOURCE_EXPERT_GATE_FEATURE_NAMES),
    )
    assert np.unique(records["domains"]).size == 4

    model = CausalMechanismAdvantageModel(
        DEFINITIONS,
        config=CausalMechanismAdvantageConfig(
            source_stack_protocol=(
                RIGID_ANCHORED_MECHANISM_RESIDUAL_STACK_PROTOCOL
            ),
            source_expert_gate_protocol=(
                SOURCE_OOF_RIDGE_LCB_EXPERT_GATE_PROTOCOL
            ),
            source_expert_gate_error_quantile=0.50,
            source_expert_gate_min_disagreement_groups=4,
        ),
    )
    diagnostics, gated = model._fit_source_expert_gate(
        dataset,
        target=target,
        core_score=core,
        residual_score=residual,
        domains=domains,
        groups=groups,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["target_labels_used"] is False
    assert diagnostics["source_city_oof_fold_count"] == 4
    assert diagnostics["disagreement_group_count"] == np.unique(groups).size
    assert model.source_expert_gate_model is not None
    assert np.isfinite(gated).all()
    for group in np.unique(groups):
        rows = groups == group
        assert np.allclose(gated[rows], core[rows]) or np.allclose(
            gated[rows], residual[rows]
        )


def test_source_expert_gate_applies_only_selected_group_expert():
    core = np.asarray([0.0, 0.2, 0.0, 0.3])
    residual = np.asarray([0.1, 0.0, 0.2, 0.0])
    groups = np.asarray(["a", "a", "b", "b"])
    gated = _apply_source_expert_gate(
        core,
        residual,
        groups,
        np.asarray(["a", "b"]),
        np.asarray([True, False]),
    )
    assert np.allclose(gated, [0.1, 0.0, 0.0, 0.3])


def test_target_groups_only_fit_low_capacity_target_head():
    dataset = _dataset(include_target=True)
    model = CausalMechanismAdvantageModel(DEFINITIONS, target_domain="target")
    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]

    assert "target" not in diagnostics["source"]["source_domains"]
    assert diagnostics["target"]["group_count"] == 8
    assert 0.0 <= diagnostics["target"]["target_weight"] <= 0.75
    assert np.isfinite(prediction["mean"]).all()


def test_outer_city_stack_selection_does_not_read_heldout_mechanism_labels():
    dataset = _dataset()
    perturbed_queue = np.asarray(dataset.targets["next_total_queue"], dtype=float).copy()
    perturbed_queue[np.asarray(dataset.domains) == "a"] += 1000.0
    perturbed = MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors=dataset.priors,
        targets={
            **dataset.targets,
            "next_total_queue": perturbed_queue,
        },
        domains=dataset.domains,
        metadata=dict(dataset.metadata),
    )

    original = CausalMechanismAdvantageModel(DEFINITIONS).fit(dataset)["source"]
    changed = CausalMechanismAdvantageModel(DEFINITIONS).fit(perturbed)["source"]

    assert original["source_selection_protocol"] == (
        "nested_outer_city_inner_city_cross_fit_v2"
    )
    assert original["pair_excluded_mechanism_fit_count"] == 6
    original_a = [row["domain_regret"]["a"] for row in original["grid"]]
    changed_a = [row["domain_regret"]["a"] for row in changed["grid"]]
    assert np.allclose(original_a, changed_a, rtol=0.0, atol=1e-12)


def test_fused_target_specialist_is_group_oof_and_sample_capped():
    dataset = _dataset(include_target=True)
    model = FusedCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )

    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]
    target = diagnostics["target"]
    reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)

    assert "target" not in diagnostics["source"]["source_domains"]
    assert target["selection_protocol"] == (
        "target-action-group-oof-joint-mechanism-gate-specialist-fusion-v2"
    )
    assert target["group_count"] == 8
    assert np.isclose(target["empirical_weight_cap"], 0.5)
    assert 0.0 <= target["target_specialist_weight"] <= 0.5
    assert 0.0 <= target["target_mechanism_gate"] <= 1.0
    assert len(target["grid"]) == 6
    assert np.isfinite(prediction["mean"]).all()
    assert np.all(prediction["uncertainty"] >= 0.0)
    assert np.allclose(prediction["mean"][reference], 0.0)
    assert np.allclose(prediction["uncertainty"][reference], 0.0)


def test_fused_target_fit_reuses_exact_frozen_source_stack():
    dataset = _dataset(include_target=True)
    baseline = FusedCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )
    baseline_diagnostics = baseline.fit(dataset)

    source = CausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )
    source.fit(dataset)
    reused = FusedCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )
    reused_diagnostics = reused.fit_from_fitted_source(dataset, source)

    assert reused_diagnostics["source_reuse"] == {
        "protocol": "exact_matched_frozen_source_stack_v1",
        "source_domain_count": 4,
        "target_heads_copied": False,
    }
    assert reused_diagnostics["source"] == baseline_diagnostics["source"]
    assert reused_diagnostics["target"] == baseline_diagnostics["target"]
    baseline_prediction = baseline.predict(dataset)["control_cost"]
    reused_prediction = reused.predict(dataset)["control_cost"]
    assert set(reused_prediction) == set(baseline_prediction)
    for field in baseline_prediction:
        assert np.allclose(
            reused_prediction[field],
            baseline_prediction[field],
            rtol=0.0,
            atol=1e-12,
        )


def test_source_reuse_rejects_estimator_configuration_mismatch():
    dataset = _dataset(include_target=True)
    source = CausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )
    source.fit(dataset)
    fused = FusedCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
        config=CausalMechanismAdvantageConfig(min_source_robust_gain=0.123),
    )

    with np.testing.assert_raises_regex(ValueError, "estimator configuration"):
        fused.fit_from_fitted_source(dataset, source)


def test_physical_residual_dataset_preserves_and_scales_simulator_prior():
    base = _dataset()
    features = np.column_stack(
        [
            base.features,
            np.full(base.size, 0.25, dtype=float),
            np.full(base.size, 0.5, dtype=float),
        ]
    )
    context = np.column_stack(
        [base.context, np.full(base.size, 0.5), np.full(base.size, 0.2)]
    )
    queue_target = np.asarray(base.targets["next_total_queue"], dtype=float)
    queue_prior = 0.4 * queue_target
    occupancy_target = np.full(base.size, 0.30, dtype=float)
    occupancy_prior = np.full(base.size, 0.20, dtype=float)
    occupancy_definition = MechanismDefinition(
        name="spillback",
        target_name="next_downstream_occupancy",
        parent_variants={"minimal": ("mean_occ",)},
    )
    one_step_speed_definition = MechanismDefinition(
        name="mobility",
        target_name="one_step_mean_speed",
        parent_variants={"minimal": ("mean_occ",)},
    )
    one_step_speed_target = np.full(base.size, 6.0, dtype=float)
    one_step_speed_prior = np.full(base.size, 4.0, dtype=float)
    dataset = MechanismDataset(
        feature_names=(*base.feature_names, "mean_occ", "lane_count_norm"),
        features=features,
        context_names=(*base.context_names, "network_lane_count_norm", "tls_count_norm"),
        context=context,
        priors={
            **base.priors,
            "next_total_queue": queue_prior,
            "next_downstream_occupancy": occupancy_prior,
            "one_step_mean_speed": one_step_speed_prior,
        },
        targets={
            **base.targets,
            "next_downstream_occupancy": occupancy_target,
            "one_step_mean_speed": one_step_speed_target,
        },
        domains=base.domains,
        metadata=dict(base.metadata),
    )

    transformed = _local_physical_mechanism_dataset(
        dataset,
        (*DEFINITIONS[:1], occupancy_definition, one_step_speed_definition),
    )

    assert transformed.metadata["mechanism_dataset_protocol"] == (
        LOCAL_PHYSICAL_RESIDUAL_PROTOCOL
    )
    assert np.allclose(
        transformed.priors["next_total_queue"],
        queue_prior / 6.0,
    )
    assert np.allclose(
        transformed.targets["next_total_queue"],
        queue_target / 6.0,
    )
    assert np.allclose(transformed.features[:, 0], base.features[:, 0] / 6.0)
    assert np.allclose(transformed.features[:, -2], 0.25)
    assert np.allclose(
        transformed.priors["next_downstream_occupancy"],
        occupancy_prior,
    )
    assert np.allclose(
        transformed.targets["next_downstream_occupancy"],
        occupancy_target,
    )
    assert np.allclose(
        transformed.priors["one_step_mean_speed"],
        one_step_speed_prior / 13.89,
    )
    assert np.allclose(
        transformed.targets["one_step_mean_speed"],
        one_step_speed_target / 13.89,
    )


def test_physical_residual_model_uses_prior_only_when_prior_is_exact():
    base = _dataset(include_target=True)
    exact_prior = np.asarray(base.targets["next_total_queue"], dtype=float).copy()
    dataset = MechanismDataset(
        feature_names=base.feature_names,
        features=base.features,
        context_names=base.context_names,
        context=base.context,
        priors={**base.priors, "next_total_queue": exact_prior},
        targets=base.targets,
        domains=base.domains,
        metadata=dict(base.metadata),
    )
    mechanism = PhysicalResidualCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )
    diagnostics = mechanism.fit(dataset)
    fused = FusedPhysicalResidualCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )
    reused = fused.fit_from_fitted_source(dataset, mechanism)

    queue_fit = diagnostics["source"]["mechanism_fit"]["queue_propagation"]
    assert queue_fit["variant_name"] == "prior_only"
    assert diagnostics["source"]["mechanism_dataset_protocol"] == (
        LOCAL_PHYSICAL_RESIDUAL_PROTOCOL
    )
    assert reused["source_reuse"]["protocol"] == (
        "exact_matched_frozen_source_stack_v1"
    )


def test_fused_model_selects_intermediate_weight_for_target_mechanism_shift():
    base = _dataset(include_target=True)
    target_mask = np.asarray(base.domains) == "target"
    reference = np.asarray(base.metadata["is_reference"], dtype=bool)
    interval_cost = np.asarray(base.targets["interval_cost"], dtype=float).copy()
    target_candidates = target_mask & ~reference
    target_state = base.features[target_candidates, 0] - 5.0
    interval_cost[target_candidates] = 0.9 * target_state - 0.2
    shifted = MechanismDataset(
        feature_names=base.feature_names,
        features=base.features,
        context_names=base.context_names,
        context=base.context,
        priors=base.priors,
        targets={**base.targets, "interval_cost": interval_cost},
        domains=base.domains,
        metadata=dict(base.metadata),
    )

    first = FusedCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    ).fit(shifted)["target"]
    second = FusedCausalMechanismAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    ).fit(shifted)["target"]

    assert first["enabled"] is True
    assert first["target_specialist_weight"] == 0.5
    assert 0.0 < first["target_specialist_weight"] < 1.0
    assert first["selected_candidate"]["mean_gain_vs_source"] > 0.0
    assert first["selected_candidate"] == second["selected_candidate"]


def test_fused_rigid_ablation_disables_only_source_mechanism_stack():
    dataset = _dataset(include_target=True)
    model = FusedRigidActionAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )

    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]

    assert diagnostics["source"]["stack_enabled"] is False
    assert diagnostics["source"]["stack_weight"] == 0.0
    assert diagnostics["source"]["ablation"] == (
        "source_mechanism_stack_forced_off"
    )
    assert diagnostics["source"]["source_selection_protocol"] == (
        "exact_rigid_core_fast_path_v1"
    )
    assert diagnostics["source"]["full_stack_fit_performed"] is False
    assert diagnostics["target"]["selection_protocol"] == (
        "target-action-group-oof-joint-mechanism-gate-specialist-fusion-v2"
    )
    assert diagnostics["target"]["target_mechanism_gate"] == 0.0
    assert diagnostics["target"]["effective_mechanism_stack_weight"] == 0.0
    assert 0.0 <= diagnostics["target"]["target_specialist_weight"] <= 0.5
    assert np.isfinite(prediction["mean"]).all()
    assert np.all(prediction["uncertainty"] >= 0.0)


def test_target_mechanism_gate_rejects_harmful_source_stack():
    class HarmfulStack:
        @staticmethod
        def predict(features):
            candidate = np.linalg.norm(features, axis=1) > 1e-12
            return np.where(candidate, -100.0, 0.0)

    class HarmfulFused(FusedCausalMechanismAdvantageModel):
        def _fit_source(self, dataset):
            diagnostics = super()._fit_source(dataset)
            self.stack_model = HarmfulStack()
            self.stack_weight = 1.0
            self.stack_calibration_error = 1.0
            return {
                **diagnostics,
                "stack_enabled": True,
                "stack_weight": 1.0,
            }

    model = HarmfulFused(
        DEFINITIONS,
        target_domain="target",
        config=CausalMechanismAdvantageConfig(
            target_specialist_blend_weights=(0.0,),
            target_mechanism_gate_weights=(0.0, 1.0),
        ),
    )

    diagnostics = model.fit(_dataset(include_target=True))["target"]
    prediction = model.predict(_dataset(include_target=True))["control_cost"]

    assert diagnostics["target_mechanism_gate"] == 0.0
    assert diagnostics["effective_mechanism_stack_weight"] == 0.0
    assert diagnostics["mechanism_gate_selected"] is False
    assert np.allclose(prediction["mechanism_stack_weight"], 0.0)
    assert np.allclose(prediction["target_mechanism_gate"], 0.0)


def test_fused_rigid_fast_path_matches_full_fit_then_forced_off():
    class FullFitThenForcedOff(FusedCausalMechanismAdvantageModel):
        def _fit_source(self, dataset):
            diagnostics = super()._fit_source(dataset)
            self.stack_model = None
            self.stack_weight = 0.0
            self.stack_calibration_error = 0.0
            return diagnostics

    dataset = _dataset(include_target=True)
    fast = FusedRigidActionAdvantageModel(
        DEFINITIONS,
        target_domain="target",
    )
    reference = FullFitThenForcedOff(
        DEFINITIONS,
        target_domain="target",
    )

    fast.fit(dataset)
    reference.fit(dataset)
    fast_prediction = fast.predict(dataset)["control_cost"]
    reference_prediction = reference.predict(dataset)["control_cost"]

    for field in (
        "mean",
        "uncertainty",
        "context_trust",
        "context_distance",
        "context_support",
    ):
        assert np.allclose(
            fast_prediction[field],
            reference_prediction[field],
            rtol=0.0,
            atol=1e-12,
        )
    assert set(fast_prediction) == set(reference_prediction)
