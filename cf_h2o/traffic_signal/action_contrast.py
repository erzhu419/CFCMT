"""Paired action-contrast datasets for transferable traffic mechanisms."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.generalized_pressure import (
    PressurePolicySpec,
    pressure_scores_from_feature_matrix,
    pressure_spec,
)


DEFAULT_CONTRAST_FEATURES = (
    "green_q",
    "red_q",
    "green_veh",
    "red_veh",
    "green_occ",
    "red_occ",
    "green_down_q",
    "green_down_occ",
    "green_link_ratio",
    "green_lane_ratio",
    "service_pressure",
    "red_pressure",
    "switch_indicator",
    "clearance_fraction",
    "current_phase_overlap",
    "green_q_mean",
    "green_q_max",
    "green_q_cv",
    "red_q_max",
    "red_q_cv",
    "green_speed_mean",
    "red_speed_mean",
    "green_occ_max",
    "red_occ_max",
    "green_down_q_max",
    "green_down_occ_max",
    "green_down_occ_std",
    "green_movement_ratio",
    "green_outgoing_ratio",
    "green_movement_down_q_mean",
    "green_movement_down_occ_mean",
    "green_movement_down_occ_max",
    "movement_service_pressure",
    "queue_concentration",
)


def action_group_ids(dataset: MechanismDataset) -> np.ndarray:
    values = np.asarray(dataset.metadata.get("action_group_ids", ()))
    if values.shape != (dataset.size,):
        raise ValueError("metadata.action_group_ids must contain one value per row")
    return values


def rule_reference_indices(
    dataset: MechanismDataset,
    *,
    policy: str | PressurePolicySpec | Mapping[str, Any],
) -> dict[Any, int]:
    """Choose exactly one reference row per paired state using local rule scores."""

    scores = pressure_scores_from_feature_matrix(
        dataset.features,
        dataset.feature_names,
        policy,
    )
    groups = action_group_ids(dataset)
    result: dict[Any, int] = {}
    for group in _ordered_unique(groups):
        rows = np.flatnonzero(groups == group)
        result[group] = int(rows[int(np.argmax(scores[rows]))])
    return result


def select_source_only_reference_policy(
    dataset: MechanismDataset,
    *,
    policies: Sequence[str] = ("max_pressure", "phase_pressure", "spillback_pressure"),
    target_name: str = "interval_cost",
    target_context: np.ndarray | None = None,
) -> tuple[str, dict[str, Any]]:
    """Select a rule with source-only contextual leave-one-domain validation."""

    groups = action_group_ids(dataset)
    target = np.asarray(dataset.targets[target_name], dtype=float)
    domain_scores: dict[str, dict[str, float]] = {policy: {} for policy in policies}
    references = {policy: rule_reference_indices(dataset, policy=policy) for policy in policies}
    for domain in _ordered_unique(dataset.domains):
        domain_mask = dataset.domains == domain
        domain_groups = _ordered_unique(groups[domain_mask])
        for policy in policies:
            regrets = []
            for group in domain_groups:
                rows = np.flatnonzero(groups == group)
                values = target[rows]
                scale = action_group_range(values)
                reference_row = references[policy][group]
                regrets.append((float(target[reference_row]) - float(np.min(values))) / scale)
            domain_scores[policy][str(domain)] = float(np.mean(regrets)) if regrets else float("inf")
    mean_scores = {
        policy: float(np.mean(list(scores.values()))) if scores else float("inf")
        for policy, scores in domain_scores.items()
    }
    global_selected = min(policies, key=lambda policy: (mean_scores[policy], policies.index(policy)))
    domain_names = list(next(iter(domain_scores.values()))) if domain_scores else []
    domain_context = {
        str(domain): np.mean(dataset.context[dataset.domains == domain], axis=0)
        for domain in _ordered_unique(dataset.domains)
    }
    selector_cv: dict[str, dict[str, Any]] = {}
    selector_names = ("global", "nearest", "idw2", "idw3")
    for selector in selector_names:
        heldout_rows = {}
        for heldout in domain_names:
            train_domains = [name for name in domain_names if name != heldout]
            chosen = _contextual_policy_choice(
                train_domains=train_domains,
                domain_context=domain_context,
                domain_scores=domain_scores,
                query_context=domain_context[heldout],
                policies=policies,
                selector=selector,
            )
            heldout_rows[heldout] = {
                "selected_policy": chosen,
                "actual_normalized_regret": float(domain_scores[chosen][heldout]),
            }
        selector_cv[selector] = {
            "mean_regret": float(
                np.mean([row["actual_normalized_regret"] for row in heldout_rows.values()])
            )
            if heldout_rows
            else float("inf"),
            "heldout": heldout_rows,
        }
    selected_selector = min(
        selector_names,
        key=lambda name: (selector_cv[name]["mean_regret"], selector_names.index(name)),
    )
    if target_context is None:
        selected = global_selected
        applied_selector = "global_no_target_context"
    else:
        selected = _contextual_policy_choice(
            train_domains=domain_names,
            domain_context=domain_context,
            domain_scores=domain_scores,
            query_context=np.asarray(target_context, dtype=float).reshape(-1),
            policies=policies,
            selector=selected_selector,
        )
        applied_selector = selected_selector
    return selected, {
        "selected": selected,
        "global_selected": global_selected,
        "selector": applied_selector,
        "source_only_selected_selector": selected_selector,
        "criterion": "source_leave_one_domain_contextual_normalized_one_step_regret",
        "mean_regret": mean_scores,
        "domain_regret": domain_scores,
        "selector_cv": selector_cv,
        "domain_context": {name: values.tolist() for name, values in domain_context.items()},
        "target_context": None if target_context is None else np.asarray(target_context, dtype=float).tolist(),
    }


def select_contextual_policy_from_domain_costs(
    domain_policy_costs: Mapping[str, Mapping[str, float]],
    domain_context: Mapping[str, np.ndarray],
    *,
    target_context: np.ndarray,
    policies: Sequence[str],
    min_context_domains: int = 6,
    min_context_relative_improvement: float = 0.10,
    min_context_absolute_improvement: float = 0.002,
    min_context_win_fraction: float = 0.60,
) -> tuple[str, dict[str, Any]]:
    """Select a target rule from source-only closed-loop domain costs.

    Every source domain receives equal weight after conversion to relative
    regret.  Contextual selection is enabled only when leave-one-domain-out
    decisions beat a robust global selector by a material and stable margin;
    otherwise the method falls back to the global source-only policy.
    """

    domain_names = sorted(domain_policy_costs)
    if len(domain_names) < 2:
        raise ValueError("closed-loop policy selection requires at least two source domains")
    policy_scores: dict[str, dict[str, float]] = {policy: {} for policy in policies}
    for domain in domain_names:
        costs = domain_policy_costs[domain]
        best = min(float(costs[policy]) for policy in policies)
        denominator = max(abs(best), 1.0)
        for policy in policies:
            policy_scores[policy][domain] = (float(costs[policy]) - best) / denominator

    selector_names = ("global", "nearest", "idw2", "idw3")
    selector_cv = {}
    for selector in selector_names:
        heldout_rows = {}
        for heldout in domain_names:
            train_domains = [domain for domain in domain_names if domain != heldout]
            chosen = _contextual_policy_choice(
                train_domains=train_domains,
                domain_context=domain_context,
                domain_scores=policy_scores,
                query_context=domain_context[heldout],
                policies=policies,
                selector=selector,
            )
            heldout_rows[heldout] = {
                "selected_policy": chosen,
                "actual_relative_regret": float(policy_scores[chosen][heldout]),
            }
        regret_values = np.asarray(
            [row["actual_relative_regret"] for row in heldout_rows.values()],
            dtype=float,
        )
        selector_cv[selector] = {
            "mean_relative_regret": float(np.mean(regret_values)),
            "std_relative_regret": float(np.std(regret_values)),
            "worst_relative_regret": float(np.max(regret_values)),
            "robust_relative_regret": _robust_regret(regret_values),
            "heldout": heldout_rows,
        }
    global_cv = selector_cv["global"]
    contextual_candidates = []
    for selector in selector_names[1:]:
        candidate = selector_cv[selector]
        wins = sum(
            candidate["heldout"][domain]["actual_relative_regret"]
            + 1e-12
            < global_cv["heldout"][domain]["actual_relative_regret"]
            for domain in domain_names
        )
        losses = sum(
            candidate["heldout"][domain]["actual_relative_regret"]
            > global_cv["heldout"][domain]["actual_relative_regret"]
            + 1e-12
            for domain in domain_names
        )
        candidate["wins_vs_global"] = int(wins)
        candidate["losses_vs_global"] = int(losses)
        robust_gain = float(global_cv["robust_relative_regret"] - candidate["robust_relative_regret"])
        required_gain = max(
            float(min_context_absolute_improvement),
            float(min_context_relative_improvement) * float(global_cv["robust_relative_regret"]),
        )
        candidate["robust_gain_vs_global"] = robust_gain
        candidate["required_gain"] = required_gain
        candidate["eligible"] = bool(
            len(domain_names) >= int(min_context_domains)
            and robust_gain >= required_gain
            and wins >= int(np.ceil(float(min_context_win_fraction) * len(domain_names)))
            and losses < wins
            and candidate["worst_relative_regret"] <= global_cv["worst_relative_regret"] + 1e-12
        )
        if candidate["eligible"]:
            contextual_candidates.append(selector)
    selected_selector = min(
        contextual_candidates,
        key=lambda name: (selector_cv[name]["robust_relative_regret"], selector_names.index(name)),
        default="global",
    )
    if selected_selector == "global":
        selected = _robust_global_policy_choice(
            train_domains=domain_names,
            domain_scores=policy_scores,
            policies=policies,
        )
    else:
        selected = _contextual_policy_choice(
            train_domains=domain_names,
            domain_context=domain_context,
            domain_scores=policy_scores,
            query_context=np.asarray(target_context, dtype=float),
            policies=policies,
            selector=selected_selector,
        )
    return selected, {
        "selected": selected,
        "selector": selected_selector,
        "criterion": "source_only_closed_loop_equal_domain_robust_regret_gated_context_selector",
        "source_domain_policy_costs": {
            domain: {policy: float(value) for policy, value in costs.items()}
            for domain, costs in domain_policy_costs.items()
        },
        "source_domain_relative_regret": policy_scores,
        "selector_cv": selector_cv,
        "selector_gate": {
            "min_context_domains": int(min_context_domains),
            "min_context_relative_improvement": float(min_context_relative_improvement),
            "min_context_absolute_improvement": float(min_context_absolute_improvement),
            "min_context_win_fraction": float(min_context_win_fraction),
        },
        "target_context": np.asarray(target_context, dtype=float).tolist(),
    }


def build_action_contrast_dataset(
    dataset: MechanismDataset,
    *,
    reference_policy: str | PressurePolicySpec | Mapping[str, Any],
    contrast_features: Sequence[str] = DEFAULT_CONTRAST_FEATURES,
) -> MechanismDataset:
    """Convert absolute paired outcomes into action-minus-reference mechanisms."""

    groups = action_group_ids(dataset)
    resolved_reference = pressure_spec(reference_policy)
    references = rule_reference_indices(dataset, policy=resolved_reference)
    feature_index = {name: idx for idx, name in enumerate(dataset.feature_names)}
    missing = [name for name in contrast_features if name not in feature_index]
    if missing:
        raise KeyError(f"missing contrast features: {missing}")
    selected = np.asarray([feature_index[name] for name in contrast_features], dtype=int)
    reference_rows = np.asarray([references[group] for group in groups], dtype=int)
    reference_features = np.asarray(dataset.features[reference_rows][:, selected], dtype=float)
    candidate_features = np.asarray(dataset.features[:, selected], dtype=float)
    augmented = np.concatenate(
        [dataset.features, reference_features, candidate_features - reference_features],
        axis=1,
    )
    feature_names = (
        *dataset.feature_names,
        *(f"reference_{name}" for name in contrast_features),
        *(f"delta_{name}" for name in contrast_features),
    )
    priors = {
        name: np.asarray(values, dtype=float) - np.asarray(values, dtype=float)[reference_rows]
        for name, values in dataset.priors.items()
    }
    targets = {
        name: np.asarray(values, dtype=float) - np.asarray(values, dtype=float)[reference_rows]
        for name, values in dataset.targets.items()
    }
    is_reference = np.arange(dataset.size) == reference_rows
    return MechanismDataset(
        feature_names=tuple(feature_names),
        features=augmented,
        context_names=dataset.context_names,
        context=dataset.context.copy(),
        priors=priors,
        targets=targets,
        domains=dataset.domains.copy(),
        metadata={
            **dict(dataset.metadata),
            "action_group_ids": groups.tolist(),
            "reference_policy": resolved_reference.key,
            "reference_policy_spec": resolved_reference.to_dict(),
            "reference_rows": reference_rows.tolist(),
            "is_reference": is_reference.tolist(),
            "contrast_features": list(contrast_features),
            "label_semantics": "candidate_action_minus_rule_reference_same_state",
        },
    )


def make_target_action_contrasts(
    absolute_candidates: MechanismDataset,
    *,
    reference_policy: str | PressurePolicySpec | Mapping[str, Any],
    contrast_features: Sequence[str] = DEFAULT_CONTRAST_FEATURES,
) -> tuple[MechanismDataset, int]:
    """Create one target-state contrast group and return its reference index."""

    if np.unique(action_group_ids(absolute_candidates)).size != 1:
        raise ValueError("target action contrast expects exactly one state group")
    references = rule_reference_indices(absolute_candidates, policy=reference_policy)
    reference_index = next(iter(references.values()))
    return (
        build_action_contrast_dataset(
            absolute_candidates,
            reference_policy=reference_policy,
            contrast_features=contrast_features,
        ),
        int(reference_index),
    )


def _ordered_unique(values: np.ndarray) -> list[Any]:
    result = []
    seen = set()
    for value in np.asarray(values).tolist():
        key = value.item() if hasattr(value, "item") else value
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result


def _contextual_policy_choice(
    *,
    train_domains: Sequence[str],
    domain_context: Mapping[str, np.ndarray],
    domain_scores: Mapping[str, Mapping[str, float]],
    query_context: np.ndarray,
    policies: Sequence[str],
    selector: str,
) -> str:
    if not train_domains:
        raise ValueError("contextual policy selection requires at least one source domain")
    values = np.vstack([domain_context[name] for name in train_domains])
    query = np.asarray(query_context, dtype=float).reshape(-1)
    scale = np.std(values, axis=0)
    scale = np.maximum(scale, 0.05)
    distance = np.sqrt(np.sum(((values - query[None, :]) / scale[None, :]) ** 2, axis=1))
    if selector == "global":
        return _robust_global_policy_choice(
            train_domains=train_domains,
            domain_scores=domain_scores,
            policies=policies,
        )
    elif selector == "nearest":
        weights = np.zeros(len(train_domains), dtype=float)
        weights[int(np.argmin(distance))] = 1.0
    elif selector in {"idw2", "idw3"}:
        k = min(int(selector[-1]), len(train_domains))
        nearest = np.argsort(distance)[:k]
        weights = np.zeros(len(train_domains), dtype=float)
        weights[nearest] = 1.0 / np.maximum(distance[nearest], 0.25)
    else:
        raise ValueError(f"unknown contextual source selector: {selector}")
    weights /= max(float(weights.sum()), 1e-12)
    predicted = {
        policy: float(
            sum(weights[idx] * float(domain_scores[policy][name]) for idx, name in enumerate(train_domains))
        )
        for policy in policies
    }
    return min(policies, key=lambda policy: (predicted[policy], policies.index(policy)))


def _robust_global_policy_choice(
    *,
    train_domains: Sequence[str],
    domain_scores: Mapping[str, Mapping[str, float]],
    policies: Sequence[str],
) -> str:
    objectives = {
        policy: _robust_regret(
            np.asarray([float(domain_scores[policy][domain]) for domain in train_domains])
        )
        for policy in policies
    }
    return min(policies, key=lambda policy: (objectives[policy], policies.index(policy)))


def _robust_regret(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return float("inf")
    return float(np.mean(values) + 0.25 * np.std(values) + 0.25 * np.max(values))
