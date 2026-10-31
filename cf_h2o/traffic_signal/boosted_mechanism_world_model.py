"""Nonlinear causal mechanism residuals with cross-fitted context adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from cf_h2o.traffic_signal.action_scaling import action_group_range
from cf_h2o.traffic_signal.mechanism_world_model import (
    InvariantMechanismWorldModel,
    MechanismDataset,
    MechanismDefinition,
    MechanismFitConfig,
)
from cf_h2o.traffic_signal.context_reference import (
    CONTEXT_SUPPORT_PROTOCOL,
    fit_context_reference_geometry,
)


@dataclass(frozen=True)
class BoostedFitConfig:
    learning_rate: float = 0.06
    max_iter: int = 40
    max_leaf_nodes: int = 15
    min_samples_leaf: int = 20
    l2_regularization: float = 5.0
    uncertainty_quantile: float = 0.90
    complexity_penalty: float = 0.002
    winsor_quantile: float = 0.01
    random_state: int = 20260803
    action_ranking_targets: tuple[str, ...] = ("interval_cost",)
    select_parent_variants: bool = False


@dataclass
class _BoostedComponent:
    definition: MechanismDefinition
    variant_name: str
    parent_names: tuple[str, ...]
    feature_indices: np.ndarray
    include_context: bool
    target_scale: float
    model: HistGradientBoostingRegressor | None
    calibration_error: float
    source_contexts: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray
    cv_score: float
    cv_domain_errors: dict[str, float]


class BoostedMechanismWorldModel:
    """Matched-capacity nonlinear residual model in causal or dense mode."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        causal: bool,
        config: BoostedFitConfig | None = None,
    ) -> None:
        self.definitions = tuple(definitions)
        self.causal = bool(causal)
        self.config = config or BoostedFitConfig()
        self.components: dict[str, _BoostedComponent] = {}
        for definition in self.definitions:
            definition.validate()

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        if np.unique(dataset.domains).size < 2:
            raise ValueError("boosted mechanism transfer requires at least two source domains")
        self.components = {}
        diagnostics = {}
        for definition in self.definitions:
            if self.causal:
                if self.config.select_parent_variants:
                    selection = self._select_variant(dataset, definition)
                else:
                    variant_name = "physical" if "physical" in definition.parent_variants else next(iter(definition.parent_variants))
                    parent_names = tuple(definition.parent_variants[variant_name])
                    selection = {
                        "variant_name": variant_name,
                        "parent_names": parent_names,
                        "cv_score": float("nan"),
                        "cv_domain_errors": {},
                        "selection_score": float("nan"),
                    }
                variant_name = str(selection["variant_name"])
                parent_names = tuple(selection["parent_names"])
            else:
                variant_name = "dense_all_local_plus_context"
                parent_names = tuple(dataset.feature_names)
                selection = {
                    "cv_score": float("nan"),
                    "cv_domain_errors": {},
                    "selection_score": float("nan"),
                }
            component = self._fit_component(
                dataset,
                definition,
                variant_name=variant_name,
                parent_names=parent_names,
            )
            component.cv_score = float(selection["cv_score"])
            component.cv_domain_errors = dict(selection["cv_domain_errors"])
            self.components[definition.name] = component
            diagnostics[definition.name] = {
                "target_name": definition.target_name,
                "variant_name": variant_name,
                "parent_names": list(parent_names),
                "feature_count": int(len(parent_names) + (dataset.context.shape[1] if not self.causal else 0)),
                "enabled": component.model is not None,
                "cv_score": float(component.cv_score),
                "cv_domain_errors": dict(component.cv_domain_errors),
                "selection_score": float(selection["selection_score"]),
                "calibration_error": float(component.calibration_error),
                "target_scale": float(component.target_scale),
                "estimator": "HistGradientBoostingRegressor",
                "context_support_protocol": CONTEXT_SUPPORT_PROTOCOL,
                "context_support_anchor_count": int(component.source_contexts.shape[0]),
            }
        return diagnostics

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.components:
            raise RuntimeError("fit must be called before predict")
        return {
            name: self._predict_component(component, dataset)
            for name, component in self.components.items()
        }

    def cross_fitted_means(self, dataset: MechanismDataset) -> dict[str, np.ndarray]:
        """Predict every source domain using an estimator that excluded it."""

        if not self.components:
            raise RuntimeError("fit must be called before cross_fitted_means")
        result = {definition.target_name: np.zeros(dataset.size, dtype=float) for definition in self.definitions}
        for heldout in np.unique(dataset.domains):
            train_mask = dataset.domains != heldout
            valid_mask = ~train_mask
            train = _subset_rows(dataset, train_mask)
            valid = _subset_rows(dataset, valid_mask)
            for definition in self.definitions:
                template = self.components[definition.name]
                component = self._fit_component(
                    train,
                    definition,
                    variant_name=template.variant_name,
                    parent_names=template.parent_names,
                )
                result[definition.target_name][valid_mask] = self._predict_component(component, valid)["mean"]
        return result

    def _select_variant(
        self,
        dataset: MechanismDataset,
        definition: MechanismDefinition,
    ) -> dict[str, Any]:
        prior_domain_errors = {}
        for domain in np.unique(dataset.domains):
            valid = _subset_rows(dataset, dataset.domains == domain)
            prior = np.asarray(valid.priors[definition.target_name], dtype=float)
            prior_domain_errors[str(domain)] = _validation_error(
                valid,
                prior,
                definition.target_name,
                action_ranking=definition.target_name in self.config.action_ranking_targets,
            )
        candidates = [
            {
                "variant_name": "prior_only",
                "parent_names": (),
                "cv_score": float(np.mean(list(prior_domain_errors.values()))),
                "cv_domain_errors": prior_domain_errors,
                "selection_score": float(np.mean(list(prior_domain_errors.values()))),
            }
        ]
        for variant_name, parent_names in definition.parent_variants.items():
            domain_errors = {}
            for heldout in np.unique(dataset.domains):
                train_mask = dataset.domains != heldout
                if np.unique(dataset.domains[train_mask]).size < 2:
                    continue
                train = _subset_rows(dataset, train_mask)
                valid = _subset_rows(dataset, ~train_mask)
                component = self._fit_component(
                    train,
                    definition,
                    variant_name=variant_name,
                    parent_names=tuple(parent_names),
                )
                prediction = self._predict_component(component, valid)["mean"]
                domain_errors[str(heldout)] = _validation_error(
                    valid,
                    prediction,
                    definition.target_name,
                    action_ranking=definition.target_name in self.config.action_ranking_targets,
                )
            cv_score = float(np.mean(list(domain_errors.values()))) if domain_errors else float("inf")
            candidates.append(
                {
                    "variant_name": variant_name,
                    "parent_names": tuple(parent_names),
                    "cv_score": cv_score,
                    "cv_domain_errors": domain_errors,
                    "selection_score": cv_score + self.config.complexity_penalty * len(parent_names),
                }
            )
        return min(
            candidates,
            key=lambda item: (item["selection_score"], len(item["parent_names"])),
        )

    def _fit_component(
        self,
        dataset: MechanismDataset,
        definition: MechanismDefinition,
        *,
        variant_name: str,
        parent_names: tuple[str, ...],
    ) -> _BoostedComponent:
        context_geometry = fit_context_reference_geometry(
            dataset.context, dataset.domains
        )
        source_contexts = context_geometry.support_anchors
        context_mean = context_geometry.mean
        context_scale = context_geometry.scale
        target = np.asarray(dataset.targets[definition.target_name], dtype=float)
        prior = np.asarray(dataset.priors[definition.target_name], dtype=float)
        residual = target - prior
        weights = _domain_balanced_weights(dataset.domains)
        target_scale = _weighted_scale(residual, weights)
        if variant_name == "prior_only":
            model = None
            prediction = prior
            indices = np.zeros(0, dtype=int)
        else:
            indices = _feature_indices(dataset.feature_names, parent_names)
            features = dataset.features[:, indices]
            if not self.causal:
                features = np.concatenate([features, dataset.context], axis=1)
            lower = float(np.quantile(residual, self.config.winsor_quantile))
            upper = float(np.quantile(residual, 1.0 - self.config.winsor_quantile))
            y = np.clip(residual, lower, upper) / target_scale
            model = self._new_estimator()
            model.fit(features, y, sample_weight=weights)
            prediction = prior + model.predict(features) * target_scale
        calibration = max(
            float(np.quantile(np.abs(target - prediction), self.config.uncertainty_quantile)),
            1e-8,
        )
        return _BoostedComponent(
            definition=definition,
            variant_name=variant_name,
            parent_names=parent_names,
            feature_indices=indices,
            include_context=not self.causal,
            target_scale=target_scale,
            model=model,
            calibration_error=calibration,
            source_contexts=source_contexts,
            context_mean=context_mean,
            context_scale=context_scale,
            cv_score=float("nan"),
            cv_domain_errors={},
        )

    def _predict_component(
        self,
        component: _BoostedComponent,
        dataset: MechanismDataset,
    ) -> dict[str, np.ndarray]:
        prior = np.asarray(dataset.priors[component.definition.target_name], dtype=float)
        if component.model is None:
            residual = np.zeros(dataset.size, dtype=float)
        else:
            features = dataset.features[:, component.feature_indices]
            if component.include_context:
                features = np.concatenate([features, dataset.context], axis=1)
            residual = component.model.predict(features) * component.target_scale
        mean = prior + residual
        if component.definition.clip_min is not None or component.definition.clip_max is not None:
            mean = np.clip(
                mean,
                -np.inf if component.definition.clip_min is None else component.definition.clip_min,
                np.inf if component.definition.clip_max is None else component.definition.clip_max,
            )
        distance = _context_distance(
            dataset.context,
            component.source_contexts,
            component.context_mean,
            component.context_scale,
        )
        trust = np.exp(-distance)
        support = _context_support(dataset.context, component.source_contexts, component.context_scale)
        trust = np.minimum(trust, support)
        if component.include_context:
            trust = np.ones(dataset.size, dtype=float)
            distance = np.zeros(dataset.size, dtype=float)
            support = np.ones(dataset.size, dtype=float)
        uncertainty = component.calibration_error * (1.0 + 0.25 * distance)
        return {
            "prior": prior,
            "global_residual": residual,
            "latent_residual": np.zeros(dataset.size, dtype=float),
            "mean": mean,
            "uncertainty": np.maximum(uncertainty, 0.0),
            "context_trust": np.clip(trust, 0.0, 1.0),
            "context_distance": distance,
            "context_support": support,
        }

    def _new_estimator(self) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            learning_rate=self.config.learning_rate,
            max_iter=self.config.max_iter,
            max_leaf_nodes=self.config.max_leaf_nodes,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            early_stopping=False,
            random_state=self.config.random_state,
        )


class BoostedCausalTransferWorldModel:
    """Nonlinear invariant mechanisms plus cross-fitted low-rank adaptation."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        boosted_config: BoostedFitConfig | None = None,
        adaptation_config: MechanismFitConfig | None = None,
    ) -> None:
        self.definitions = tuple(definitions)
        self.boosted_config = boosted_config or BoostedFitConfig()
        self.adaptation_config = adaptation_config or MechanismFitConfig()
        self.global_model = BoostedMechanismWorldModel(
            definitions,
            causal=True,
            config=self.boosted_config,
        )
        self.adaptation_model = InvariantMechanismWorldModel(definitions, self.adaptation_config)

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        global_diagnostics = self.global_model.fit(dataset)
        cross_fitted = self.global_model.cross_fitted_means(dataset)
        adaptation_data = _replace_priors(dataset, cross_fitted)
        adaptation_diagnostics = self.adaptation_model.fit(adaptation_data)
        return {
            name: {
                "global": global_diagnostics[name],
                "adaptation": adaptation_diagnostics[name],
                "enabled": bool(
                    global_diagnostics[name]["enabled"]
                    or adaptation_diagnostics[name]["enabled"]
                ),
            }
            for name in global_diagnostics
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        global_prediction = self.global_model.predict(dataset)
        global_means = {
            definition.target_name: global_prediction[definition.name]["mean"]
            for definition in self.definitions
        }
        adaptation_prediction = self.adaptation_model.predict(_replace_priors(dataset, global_means))
        result = {}
        for definition in self.definitions:
            global_row = global_prediction[definition.name]
            adaptation_row = adaptation_prediction[definition.name]
            adaptation_component = self.adaptation_model.components[definition.name]
            if not adaptation_component.enabled:
                result[definition.name] = {
                    **global_row,
                    "latent_residual": np.zeros(dataset.size, dtype=float),
                }
                continue
            result[definition.name] = {
                "prior": global_row["prior"],
                "global_residual": global_row["global_residual"],
                "latent_residual": adaptation_row["mean"] - global_row["mean"],
                "mean": adaptation_row["mean"],
                "uncertainty": global_row["uncertainty"] + adaptation_row["uncertainty"],
                "context_trust": np.minimum(
                    global_row["context_trust"],
                    adaptation_row["context_trust"],
                ),
                "context_distance": np.maximum(
                    global_row["context_distance"],
                    adaptation_row["context_distance"],
                ),
                "context_support": np.minimum(
                    global_row.get("context_support", global_row["context_trust"]),
                    adaptation_row.get("context_support", adaptation_row["context_trust"]),
                ),
            }
        return result


def _replace_priors(
    dataset: MechanismDataset,
    priors: Mapping[str, np.ndarray],
) -> MechanismDataset:
    return MechanismDataset(
        feature_names=dataset.feature_names,
        features=dataset.features,
        context_names=dataset.context_names,
        context=dataset.context,
        priors={name: np.asarray(values, dtype=float) for name, values in priors.items()},
        targets=dataset.targets,
        domains=dataset.domains,
        metadata=dict(dataset.metadata),
    )


def _validation_error(
    dataset: MechanismDataset,
    prediction: np.ndarray,
    target_name: str,
    *,
    action_ranking: bool,
) -> float:
    if not action_ranking:
        target = np.asarray(dataset.targets[target_name], dtype=float)
        prior = np.asarray(dataset.priors[target_name], dtype=float)
        denominator = max(float(np.mean(np.abs(target - prior))), 1e-8)
        return float(np.mean(np.abs(target - prediction)) / denominator)
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
    if groups.shape != (dataset.size,):
        raise ValueError("action ranking requires row-aligned action_group_ids")
    target = np.asarray(dataset.targets[target_name], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    regrets = []
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        if rows.size <= 1:
            continue
        selected = int(rows[int(np.argmin(prediction[rows]))])
        values = target[rows]
        scale = action_group_range(values)
        regrets.append((float(target[selected]) - float(np.min(values))) / scale)
    return float(np.mean(regrets)) if regrets else 0.0


def _subset_rows(dataset: MechanismDataset, mask: np.ndarray) -> MechanismDataset:
    mask = np.asarray(mask, dtype=bool)
    subset = dataset.subset(mask)
    metadata = dict(subset.metadata)
    for key, value in tuple(metadata.items()):
        array = np.asarray(value)
        if array.shape == (dataset.size,):
            metadata[key] = array[mask].tolist()
    return MechanismDataset(
        feature_names=subset.feature_names,
        features=subset.features,
        context_names=subset.context_names,
        context=subset.context,
        priors=subset.priors,
        targets=subset.targets,
        domains=subset.domains,
        metadata=metadata,
    )


def _feature_indices(all_names: Sequence[str], selected_names: Sequence[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(all_names)}
    missing = [name for name in selected_names if name not in index]
    if missing:
        raise KeyError(f"missing boosted mechanism parents: {missing}")
    return np.asarray([index[name] for name in selected_names], dtype=int)


def _domain_balanced_weights(domains: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(domains, return_counts=True)
    count_by_domain = {domain: count for domain, count in zip(unique, counts)}
    weights = np.asarray([1.0 / count_by_domain[domain] for domain in domains], dtype=float)
    return weights * (weights.size / max(float(weights.sum()), 1e-12))


def _weighted_scale(values: np.ndarray, weights: np.ndarray) -> float:
    normalized = weights / max(float(np.sum(weights)), 1e-12)
    mean = float(np.sum(values * normalized))
    variance = float(np.sum((values - mean) ** 2 * normalized))
    return max(np.sqrt(max(variance, 0.0)), 1e-6)


def _context_distance(
    context: np.ndarray,
    source_contexts: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    target = (np.asarray(context, dtype=float) - mean[None, :]) / scale[None, :]
    source = (np.asarray(source_contexts, dtype=float) - mean[None, :]) / scale[None, :]
    return np.min(
        np.sqrt(np.sum((target[:, None, :] - source[None, :, :]) ** 2, axis=2)),
        axis=1,
    )


def _context_support(
    context: np.ndarray,
    source_contexts: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    values = np.asarray(context, dtype=float)
    sources = np.asarray(source_contexts, dtype=float)
    lower = np.min(sources, axis=0)
    upper = np.max(sources, axis=0)
    below = np.maximum((lower[None, :] - values) / scale[None, :], 0.0)
    above = np.maximum((values - upper[None, :]) / scale[None, :], 0.0)
    extrapolation = np.max(np.maximum(below, above), axis=1)
    return np.exp(-extrapolation)
