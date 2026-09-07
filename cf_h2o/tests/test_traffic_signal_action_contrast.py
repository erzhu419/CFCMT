import numpy as np
import pytest

from cf_h2o.traffic_signal.action_contrast import (
    build_action_contrast_dataset,
    rule_reference_indices,
    select_source_only_reference_policy,
    select_contextual_policy_from_domain_costs,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.generalized_pressure import (
    PressurePolicySpec,
    generalized_pressure_grid,
)
from cf_h2o.traffic_signal.action_ranker import (
    ACTION_TARGET_SCALE_PROTOCOL,
    AntisymmetricPairwiseActionRegressor,
    CAUSAL_CORE_RANKING_PARENTS,
    CAUSAL_RIGID_RANKING_PARENTS,
    ActionAdvantageConfig,
    GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
    PairwiseActionAdvantageRegressor,
    PairwiseActionRanker,
    PairwisePreferenceConfig,
    TargetAdaptedActionAdvantageRegressor,
    TargetOnlyActionAdvantageRegressor,
    domain_normalized_action_target,
    group_normalized_action_target,
)
from cf_h2o.traffic_signal.dynamic_graph_context import SIGNAL_GRAPH_STATE_FEATURES
from cf_h2o.traffic_signal.context_reference import (
    CONTEXT_SUPPORT_PROTOCOL,
    fit_context_reference_geometry,
)


def _dataset():
    names = ("green_q", "green_down_q", "service_pressure")
    features = np.asarray(
        [
            [5.0, 1.0, 3.0],
            [8.0, 7.0, 0.5],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [9.0, 8.0, 0.2],
            [4.0, 0.0, 4.0],
        ]
    )
    values = np.asarray([7.0, 9.0, 8.0, 5.0, 8.0, 6.0])
    return MechanismDataset(
        feature_names=names,
        features=features,
        context_names=("demand",),
        context=np.asarray([[1.0]] * 3 + [[2.0]] * 3),
        priors={"interval_cost": values + 1.0},
        targets={"interval_cost": values},
        domains=np.asarray(["a"] * 3 + ["b"] * 3),
        metadata={"action_group_ids": ["a0"] * 3 + ["b0"] * 3},
    )


def test_rule_reference_indices_are_group_local():
    dataset = _dataset()
    assert rule_reference_indices(dataset, policy="phase_pressure") == {"a0": 1, "b0": 4}
    assert rule_reference_indices(dataset, policy="max_pressure") == {"a0": 0, "b0": 5}
    assert rule_reference_indices(dataset, policy="spillback_pressure") == {"a0": 0, "b0": 5}


def test_action_contrast_has_exact_zero_reference_labels():
    contrast = build_action_contrast_dataset(
        _dataset(),
        reference_policy="phase_pressure",
        contrast_features=("green_q", "green_down_q", "service_pressure"),
    )
    assert contrast.feature_names[-3:] == (
        "delta_green_q",
        "delta_green_down_q",
        "delta_service_pressure",
    )
    assert np.allclose(contrast.targets["interval_cost"][[1, 4]], 0.0)
    assert np.allclose(contrast.priors["interval_cost"][[1, 4]], 0.0)
    assert np.allclose(contrast.features[1, -3:], 0.0)
    assert np.allclose(contrast.features[4, -3:], 0.0)


def test_source_only_selector_uses_equal_domain_normalized_regret():
    selected, diagnostics = select_source_only_reference_policy(_dataset())
    assert selected == "max_pressure"
    assert diagnostics["domain_regret"]["max_pressure"]["a"] == 0.0
    assert diagnostics["domain_regret"]["max_pressure"]["b"] > 0.0
    assert diagnostics["selector"] == "global_no_target_context"


def test_contextual_selector_is_chosen_without_target_labels():
    dataset = _dataset()
    selected, diagnostics = select_source_only_reference_policy(
        dataset,
        target_context=np.asarray([2.1]),
    )
    assert selected in {"max_pressure", "spillback_pressure"}
    assert diagnostics["target_context"] == [2.1]
    assert diagnostics["source_only_selected_selector"] in {"global", "nearest", "idw2", "idw3"}


def test_closed_loop_selector_uses_only_source_domain_costs():
    selected, diagnostics = select_contextual_policy_from_domain_costs(
        {
            "a": {"max_pressure": 1.0, "phase_pressure": 2.0},
            "b": {"max_pressure": 3.0, "phase_pressure": 1.0},
            "c": {"max_pressure": 1.2, "phase_pressure": 2.0},
        },
        {
            "a": np.asarray([0.0]),
            "b": np.asarray([5.0]),
            "c": np.asarray([0.5]),
        },
        target_context=np.asarray([0.2]),
        policies=("max_pressure", "phase_pressure"),
    )
    assert selected == "phase_pressure"
    assert diagnostics["selector"] == "global"
    assert diagnostics["criterion"].startswith("source_only_closed_loop")


def test_context_selector_is_gated_when_source_count_is_small():
    selected, diagnostics = select_contextual_policy_from_domain_costs(
        {
            "a": {"max_pressure": 1.0, "phase_pressure": 4.0},
            "b": {"max_pressure": 4.0, "phase_pressure": 1.0},
            "c": {"max_pressure": 1.0, "phase_pressure": 4.0},
        },
        {
            "a": np.asarray([0.0]),
            "b": np.asarray([10.0]),
            "c": np.asarray([0.1]),
        },
        target_context=np.asarray([10.1]),
        policies=("max_pressure", "phase_pressure"),
    )
    assert selected == "max_pressure"
    assert diagnostics["selector"] == "global"
    assert diagnostics["selector_gate"]["min_context_domains"] == 6


def test_generalized_pressure_reference_can_penalize_switching():
    base = _dataset()
    dataset = MechanismDataset(
        feature_names=(*base.feature_names, "switch_indicator"),
        features=np.column_stack([base.features, [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]]),
        context_names=base.context_names,
        context=base.context,
        priors=base.priors,
        targets=base.targets,
        domains=base.domains,
        metadata=base.metadata,
    )
    spec = PressurePolicySpec(switch_penalty=10.0)
    assert rule_reference_indices(dataset, policy=spec) == {"a0": 0, "b0": 5}
    keys = [item.key for item in generalized_pressure_grid()]
    assert len(keys) == len(set(keys))
    assert {"phase_pressure", "max_pressure", "spillback_pressure"}.issubset(keys)


def test_pairwise_action_ranker_keeps_reference_score_zero():
    contrast = build_action_contrast_dataset(
        _dataset(),
        reference_policy="max_pressure",
        contrast_features=("green_q", "green_down_q", "service_pressure"),
    )
    ranker = PairwiseActionRanker(causal=True)
    diagnostics = ranker.fit(contrast)
    prediction = ranker.predict(contrast)["control_cost"]
    reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    assert diagnostics["candidate_rows"] == 4
    assert np.allclose(prediction["mean"][reference], 0.0)
    assert np.allclose(prediction["uncertainty"][reference], 0.0)
    assert np.all((prediction["improvement_probability"] >= 0.0) & (prediction["improvement_probability"] <= 1.0))

    out_of_support = MechanismDataset(
        feature_names=contrast.feature_names,
        features=contrast.features,
        context_names=contrast.context_names,
        context=np.full_like(contrast.context, 100.0),
        priors=contrast.priors,
        targets=contrast.targets,
        domains=np.asarray(["target"] * contrast.size),
        metadata=contrast.metadata,
    )
    target_prediction = ranker.predict(out_of_support)["control_cost"]
    assert np.max(target_prediction["context_support"]) < 0.10
    assert np.max(target_prediction["context_trust"]) < 0.10


def test_pairwise_advantage_regressor_keeps_reference_score_zero():
    contrast = build_action_contrast_dataset(
        _dataset(),
        reference_policy="max_pressure",
        contrast_features=("green_q", "green_down_q", "service_pressure"),
    )
    model = PairwiseActionAdvantageRegressor(causal=True)
    diagnostics = model.fit(contrast)
    prediction = model.predict(contrast)["control_cost"]
    reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    assert diagnostics["rows"] == contrast.size
    assert set(diagnostics["domain_target_scales"]) == {"a", "b"}
    assert np.allclose(prediction["mean"][reference], 0.0)
    assert np.allclose(prediction["uncertainty"][reference], 0.0)


def test_balanced_advantage_excludes_reference_rows_and_balances_signs():
    contrast = build_action_contrast_dataset(
        _dataset(),
        reference_policy="max_pressure",
        contrast_features=("green_q", "green_down_q", "service_pressure"),
    )
    model = PairwiseActionAdvantageRegressor(
        causal=True,
        config=ActionAdvantageConfig(
            candidate_only=True,
            balance_candidate_signs=True,
        ),
    )
    diagnostics = model.fit(contrast)
    reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)

    assert diagnostics["rows"] == contrast.size
    assert diagnostics["training_rows"] == int(np.sum(~reference))
    assert diagnostics["candidate_only"] is True
    assert diagnostics["sign_balance"]["enabled"] is True
    assert np.allclose(model.predict(contrast)["control_cost"]["mean"][reference], 0.0)


def test_candidate_only_advantage_handles_constant_candidate_targets():
    contrast = build_action_contrast_dataset(
        _dataset(),
        reference_policy="max_pressure",
        contrast_features=("green_q", "green_down_q", "service_pressure"),
    )
    constant = MechanismDataset(
        feature_names=contrast.feature_names,
        features=contrast.features,
        context_names=contrast.context_names,
        context=contrast.context,
        priors={"interval_cost": np.zeros(contrast.size)},
        targets={"interval_cost": np.zeros(contrast.size)},
        domains=contrast.domains,
        metadata=dict(contrast.metadata),
    )
    model = PairwiseActionAdvantageRegressor(
        causal=True,
        config=ActionAdvantageConfig(candidate_only=True),
    )

    diagnostics = model.fit(constant)
    prediction = model.predict(constant)["control_cost"]

    assert diagnostics["estimator"] == "constant"
    assert diagnostics["training_rows"] < diagnostics["rows"]
    assert np.allclose(prediction["mean"], 0.0)


def test_domain_action_normalization_removes_city_scale():
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.arange(6, dtype=float)[:, None],
        context_names=("z",),
        context=np.zeros((6, 1)),
        priors={"interval_cost": np.zeros(6)},
        targets={"interval_cost": np.asarray([0.0, -2.0, 1.0, 0.0, -200.0, 100.0])},
        domains=np.asarray(["small"] * 3 + ["large"] * 3),
        metadata={"action_group_ids": ["small:g"] * 3 + ["large:g"] * 3},
    )

    normalized, scales = domain_normalized_action_target(dataset)

    assert scales == {"large": 300.0, "small": 3.0}
    assert np.allclose(normalized[:3], normalized[3:])


def test_group_action_normalization_matches_group_regret_estimand():
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.arange(6, dtype=float)[:, None],
        context_names=("z",),
        context=np.zeros((6, 1)),
        priors={"interval_cost": np.zeros(6)},
        targets={"interval_cost": np.asarray([0.0, -2.0, 1.0, 0.0, -200.0, 100.0])},
        domains=np.asarray(["city"] * 6),
        metadata={
            "action_group_ids": ["city:g0"] * 3 + ["city:g1"] * 3,
            "is_reference": [True, False, False, True, False, False],
        },
    )

    normalized, summary = group_normalized_action_target(dataset)

    assert np.allclose(normalized[:3], normalized[3:])
    assert summary == {
        "group_count": 2,
        "minimum": 3.0,
        "median": 151.5,
        "maximum": 300.0,
    }
    model = PairwiseActionAdvantageRegressor(
        causal=False,
        config=ActionAdvantageConfig(
            target_normalization_protocol=(
                GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
            )
        ),
    )
    diagnostics = model.fit(dataset)
    assert diagnostics["target_normalization_protocol"] == (
        GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
    )
    assert diagnostics["domain_target_scales"] == {}
    assert diagnostics["group_target_scale_summary"] == summary


def test_antisymmetric_pairwise_model_ranks_all_actions_and_zeroes_reference():
    rows = []
    targets = []
    domains = []
    groups = []
    references = []
    contexts = []
    for group_index in range(24):
        domain = "a" if group_index < 12 else "b"
        state = float(group_index % 6)
        for action_index, green_queue in enumerate((0.0, 1.0, 2.0)):
            rows.append([state, green_queue])
            targets.append(-green_queue + 0.05 * state)
            domains.append(domain)
            groups.append(f"{domain}:{group_index}")
            references.append(action_index == 0)
            contexts.append([float(group_index)])
    dataset = MechanismDataset(
        feature_names=("total_q", "green_q"),
        features=np.asarray(rows, dtype=float),
        context_names=("demand",),
        context=np.asarray(contexts, dtype=float),
        priors={"interval_cost": np.zeros(len(rows))},
        targets={"interval_cost": np.asarray(targets, dtype=float)},
        domains=np.asarray(domains),
        metadata={"action_group_ids": groups, "is_reference": references},
    )
    model = AntisymmetricPairwiseActionRegressor(
        config=PairwisePreferenceConfig(
            max_iter=60,
            max_leaf_nodes=7,
            min_samples_leaf=2,
            l2_regularization=1.0,
        )
    )

    diagnostics = model.fit(dataset)
    prediction = model.predict(dataset)["control_cost"]

    assert diagnostics["preference_protocol"] == (
        "all_action_antisymmetric_pairwise_v1"
    )
    assert diagnostics["unordered_pair_count"] == 72
    assert diagnostics["oriented_pair_count"] == 144
    assert diagnostics["pair_target_minimum"] >= -1.0
    assert diagnostics["pair_target_maximum"] <= 1.0
    assert np.allclose(
        prediction["mean"][np.asarray(references, dtype=bool)], 0.0
    )
    selected = []
    group_values = np.asarray(groups)
    for group in np.unique(group_values):
        group_rows = np.flatnonzero(group_values == group)
        selected.append(int(np.argmin(prediction["mean"][group_rows])))
    assert np.mean(np.asarray(selected) == 2) >= 0.90


def test_antisymmetric_pairwise_model_rejects_action_dependent_state_parent():
    dataset = MechanismDataset(
        feature_names=("total_q", "green_q"),
        features=np.asarray([[1.0, 0.0], [2.0, 1.0]]),
        context_names=("demand",),
        context=np.zeros((2, 1)),
        priors={"interval_cost": np.zeros(2)},
        targets={"interval_cost": np.asarray([0.0, -1.0])},
        domains=np.asarray(["a", "a"]),
        metadata={"action_group_ids": ["a:g", "a:g"], "is_reference": [True, False]},
    )

    with pytest.raises(ValueError, match="state parents vary"):
        AntisymmetricPairwiseActionRegressor().fit(dataset)


def test_domain_action_normalization_preserves_quantized_large_network_signal():
    dataset = MechanismDataset(
        feature_names=("x",),
        features=np.arange(6, dtype=float)[:, None],
        context_names=("z",),
        context=np.zeros((6, 1)),
        priors={"interval_cost": np.zeros(6)},
        targets={
            "interval_cost": np.asarray(
                [0.0, -2e-5, 1e-5, 0.0, -2.0, 1.0]
            )
        },
        domains=np.asarray(["large_network"] * 3 + ["small_network"] * 3),
        metadata={
            "action_group_ids": ["large:g"] * 3 + ["small:g"] * 3,
            "is_reference": [True, False, False, True, False, False],
        },
    )

    normalized, scales = domain_normalized_action_target(dataset)

    assert np.isclose(scales["large_network"], 3e-5)
    assert np.isclose(scales["small_network"], 3.0)
    assert np.allclose(normalized[:3], normalized[3:])

    model = PairwiseActionAdvantageRegressor(
        causal=False,
        config=ActionAdvantageConfig(candidate_only=False),
    )
    diagnostics = model.fit(dataset)
    assert diagnostics["domain_target_scale_protocol"] == (
        ACTION_TARGET_SCALE_PROTOCOL
    )


def test_rigid_causal_parents_exclude_graph_features():
    assert set(CAUSAL_RIGID_RANKING_PARENTS) <= set(CAUSAL_CORE_RANKING_PARENTS)
    assert not set(CAUSAL_RIGID_RANKING_PARENTS) & set(SIGNAL_GRAPH_STATE_FEATURES)
    assert set(SIGNAL_GRAPH_STATE_FEATURES) <= set(CAUSAL_CORE_RANKING_PARENTS)


def test_context_reference_keeps_network_anchors_with_city_balanced_scale():
    context = np.asarray([[0.0], [0.0], [10.0], [10.0], [20.0], [20.0]])
    domains = np.asarray(["city_a"] * 4 + ["city_b"] * 2)
    geometry = fit_context_reference_geometry(context, domains)
    assert geometry.domain_names == ("city_a", "city_b")
    assert np.allclose(geometry.domain_centroids[:, 0], [5.0, 20.0])
    assert np.allclose(geometry.support_anchors[:, 0], [0.0, 10.0, 20.0])
    assert np.allclose(geometry.mean, [12.5])
    assert np.allclose(geometry.scale, [7.5])
    assert geometry.diagnostics() == {
        "protocol": CONTEXT_SUPPORT_PROTOCOL,
        "domain_count": 2,
        "support_anchor_count": 3,
    }


def test_target_advantage_adapter_shrinks_toward_target_head():
    base = _dataset()
    copies = 4
    absolute = MechanismDataset(
        feature_names=base.feature_names,
        features=np.tile(base.features, (copies, 1)),
        context_names=base.context_names,
        context=np.tile(base.context, (copies, 1)),
        priors={name: np.tile(values, copies) for name, values in base.priors.items()},
        targets={name: np.tile(values, copies) for name, values in base.targets.items()},
        domains=np.tile(base.domains, copies),
        metadata={
            "action_group_ids": [
                f"{domain}{copy}"
                for copy in range(copies)
                for domain in ("a", "a", "a", "b", "b", "b")
            ]
        },
    )
    contrast = build_action_contrast_dataset(
        absolute,
        reference_policy="max_pressure",
        contrast_features=("green_q", "green_down_q", "service_pressure"),
    )
    model = TargetAdaptedActionAdvantageRegressor(target_domain="b")
    diagnostics = model.fit(contrast)
    prediction = model.predict(contrast)["control_cost"]
    reference = np.asarray(contrast.metadata["is_reference"], dtype=bool)
    assert diagnostics["target_group_count"] == 4
    assert 0.0 < diagnostics["selected_target_weight"] < 1.0
    assert np.allclose(prediction["mean"][reference], 0.0)


def test_strict_target_only_advantage_consumes_no_source_rows():
    base = _dataset()
    copies = 4
    absolute = MechanismDataset(
        feature_names=base.feature_names,
        features=np.tile(base.features, (copies, 1)),
        context_names=base.context_names,
        context=np.tile(base.context, (copies, 1)),
        priors={name: np.tile(values, copies) for name, values in base.priors.items()},
        targets={name: np.tile(values, copies) for name, values in base.targets.items()},
        domains=np.tile(base.domains, copies),
        metadata={
            "action_group_ids": [
                f"{domain}{copy}"
                for copy in range(copies)
                for domain in ("a", "a", "a", "b", "b", "b")
            ]
        },
    )
    contrast = build_action_contrast_dataset(
        absolute,
        reference_policy="max_pressure",
        contrast_features=("green_q", "green_down_q", "service_pressure"),
    )

    model = TargetOnlyActionAdvantageRegressor(target_domain="b")
    diagnostics = model.fit(contrast)
    prediction = model.predict(contrast)["control_cost"]

    assert diagnostics["target_group_count"] == 4
    assert diagnostics["source_row_count_consumed"] == 0
    assert diagnostics["target_row_count"] == 12
    assert np.all(np.isfinite(prediction["mean"]))
