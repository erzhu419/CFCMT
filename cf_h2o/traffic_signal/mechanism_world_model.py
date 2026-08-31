"""Invariant mechanism residual models for cross-network signal control.

The model deliberately uses a transparent hierarchical construction:

1. an uncalibrated mechanism prior supplies each next-state quantity;
2. a source-balanced residual model uses only admissible local parents;
3. source-specific coefficient offsets are factorized into a low-rank mechanism
   basis;
4. a context encoder predicts the factor score from static, label-free network
   descriptors;
5. leave-one-source validation selects the parent family, latent rank, and
   adaptation shrinkage independently for every mechanism.

Network context never enters a residual head directly.  It can only modulate a
mechanism's local coefficients through the low-rank factor, which makes the
claimed separation between mechanism parents and target instantiation
executable and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.traffic_signal.action_scaling import action_group_range

from cf_h2o.traffic_signal.context_reference import (
    CONTEXT_SUPPORT_PROTOCOL,
    fit_context_reference_geometry,
)


FORBIDDEN_LOCAL_MARKERS = ("domain", "scenario", "network_id", "city_id", "tls_id", "phase_id")


@dataclass(frozen=True)
class MechanismDefinition:
    """Output and admissible parent families for one transition mechanism."""

    name: str
    target_name: str
    parent_variants: Mapping[str, tuple[str, ...]]
    clip_min: float | None = None
    clip_max: float | None = None

    def validate(self) -> None:
        if not self.parent_variants:
            raise ValueError(f"{self.name}: at least one parent variant is required")
        for variant, parents in self.parent_variants.items():
            if not parents:
                raise ValueError(f"{self.name}/{variant}: empty parent set")
            for parent in parents:
                lowered = parent.lower()
                if any(marker in lowered for marker in FORBIDDEN_LOCAL_MARKERS):
                    raise ValueError(f"{self.name}/{variant}: forbidden local parent {parent!r}")


@dataclass(frozen=True)
class MechanismDataset:
    """Aligned local features, label-free context, priors, and mechanism labels."""

    feature_names: tuple[str, ...]
    features: np.ndarray
    context_names: tuple[str, ...]
    context: np.ndarray
    priors: Mapping[str, np.ndarray]
    targets: Mapping[str, np.ndarray]
    domains: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        features = np.asarray(self.features, dtype=float)
        context = np.asarray(self.context, dtype=float)
        domains = np.asarray(self.domains)
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise ValueError("features must be [N, len(feature_names)]")
        if context.ndim != 2 or context.shape[1] != len(self.context_names):
            raise ValueError("context must be [N, len(context_names)]")
        if context.shape[0] != features.shape[0] or domains.shape != (features.shape[0],):
            raise ValueError("features, context, and domains must have the same row count")
        for collection_name, collection in (("priors", self.priors), ("targets", self.targets)):
            for name, values in collection.items():
                if np.asarray(values).reshape(-1).shape != (features.shape[0],):
                    raise ValueError(f"{collection_name}[{name!r}] must have N values")

    @property
    def size(self) -> int:
        return int(self.features.shape[0])

    def subset(self, mask: np.ndarray) -> "MechanismDataset":
        mask = np.asarray(mask)
        return MechanismDataset(
            feature_names=self.feature_names,
            features=self.features[mask],
            context_names=self.context_names,
            context=self.context[mask],
            priors={name: np.asarray(values)[mask] for name, values in self.priors.items()},
            targets={name: np.asarray(values)[mask] for name, values in self.targets.items()},
            domains=self.domains[mask],
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class MechanismFitConfig:
    ridge_l2: float = 3.0
    domain_offset_l2: float = 12.0
    context_l2: float = 8.0
    huber_delta: float = 1.5
    huber_iterations: int = 3
    latent_ranks: tuple[int, ...] = (0, 1, 2)
    adaptation_shrinkages: tuple[float, ...] = (0.0, 0.5, 1.0)
    uncertainty_quantile: float = 0.90
    min_domain_rows: int = 12
    complexity_penalty: float = 0.002
    action_ranking_targets: tuple[str, ...] = ()


@dataclass
class _Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, weights: np.ndarray) -> "_Standardizer":
        weights = np.asarray(weights, dtype=float).reshape(-1)
        weights = weights / max(float(weights.sum()), 1e-12)
        mean = np.sum(values * weights[:, None], axis=0)
        variance = np.sum((values - mean[None, :]) ** 2 * weights[:, None], axis=0)
        return cls(mean=mean, scale=np.sqrt(np.maximum(variance, 1e-8)))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean[None, :]) / self.scale[None, :]


@dataclass
class _MechanismComponent:
    definition: MechanismDefinition
    variant_name: str
    parent_names: tuple[str, ...]
    feature_indices: np.ndarray
    feature_scaler: _Standardizer
    target_scale: float
    global_coef: np.ndarray
    latent_rank: int
    latent_basis: np.ndarray
    context_scaler: _Standardizer
    context_encoder_coef: np.ndarray
    adaptation_shrinkage: float
    source_contexts: np.ndarray
    support_contexts: np.ndarray
    domain_offset_coef: np.ndarray
    calibration_error: float
    cv_ratio: float
    cv_domain_errors: dict[str, float]
    enabled: bool = True

    def predict(self, dataset: MechanismDataset) -> dict[str, np.ndarray]:
        x = dataset.features[:, self.feature_indices]
        x_aug = _with_intercept(self.feature_scaler.transform(x))
        global_residual = (x_aug @ self.global_coef) * self.target_scale
        latent_residual = np.zeros(dataset.size, dtype=float)
        context_distance = _nearest_context_distance(
            dataset.context,
            self.support_contexts,
            self.context_scaler,
        )
        context_trust = np.exp(-context_distance / max(np.sqrt(dataset.context.shape[1]), 1.0))
        context_trust = np.clip(context_trust, 0.0, 1.0)
        if self.latent_rank > 0 and self.adaptation_shrinkage > 0.0:
            context_aug = _with_intercept(self.context_scaler.transform(dataset.context))
            theta = context_aug @ self.context_encoder_coef
            local_basis = x_aug @ self.latent_basis.T
            latent_residual = (
                np.sum(local_basis * theta, axis=1)
                * self.target_scale
                * self.adaptation_shrinkage
                * context_trust
            )

        prior = np.asarray(dataset.priors[self.definition.target_name], dtype=float).reshape(-1)
        mean = prior + global_residual + latent_residual
        if self.definition.clip_min is not None or self.definition.clip_max is not None:
            mean = np.clip(
                mean,
                -np.inf if self.definition.clip_min is None else self.definition.clip_min,
                np.inf if self.definition.clip_max is None else self.definition.clip_max,
            )

        source_predictions = x_aug @ self.domain_offset_coef.T if self.domain_offset_coef.size else np.zeros((dataset.size, 1))
        disagreement = np.std(source_predictions, axis=1) * self.target_scale
        uncertainty = self.calibration_error * (1.0 + 0.35 * context_distance) + disagreement
        return {
            "prior": prior,
            "global_residual": global_residual,
            "latent_residual": latent_residual,
            "mean": mean,
            "uncertainty": np.maximum(uncertainty, 0.0),
            "context_trust": context_trust,
            "context_distance": context_distance,
            "context_support": context_trust,
        }


class InvariantMechanismWorldModel:
    """Source-validated, mechanism-factored simulator residual model."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        config: MechanismFitConfig | None = None,
    ) -> None:
        self.definitions = tuple(definitions)
        self.config = config or MechanismFitConfig()
        self.components: dict[str, _MechanismComponent] = {}
        for definition in self.definitions:
            definition.validate()

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        _validate_dataset_outputs(dataset, self.definitions)
        if np.unique(dataset.domains).size < 2:
            raise ValueError("mechanism transfer requires at least two source domains")
        diagnostics: dict[str, Any] = {}
        self.components = {}
        for definition in self.definitions:
            selection = self._select_component(dataset, definition)
            if selection["variant_name"] == "prior_only":
                component = _fit_prior_component(dataset, definition, self.config)
            else:
                component = _fit_component(
                    dataset,
                    definition,
                    variant_name=selection["variant_name"],
                    parent_names=selection["parent_names"],
                    latent_rank=selection["latent_rank"],
                    adaptation_shrinkage=selection["adaptation_shrinkage"],
                    config=self.config,
                )
            component.cv_ratio = float(selection["cv_ratio"])
            component.cv_domain_errors = dict(selection["cv_domain_errors"])
            self.components[definition.name] = component
            diagnostics[definition.name] = self._component_diagnostics(component, selection)
        return diagnostics

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.components:
            raise RuntimeError("fit must be called before predict")
        return {name: component.predict(dataset) for name, component in self.components.items()}

    def _select_component(self, dataset: MechanismDataset, definition: MechanismDefinition) -> dict[str, Any]:
        prior_errors = {}
        use_action_ranking = definition.target_name in self.config.action_ranking_targets
        for domain in np.unique(dataset.domains):
            mask = dataset.domains == domain
            domain_data = _subset_for_cv(dataset, mask)
            if use_action_ranking:
                reference_prediction = np.asarray(
                    domain_data.priors[definition.target_name],
                    dtype=float,
                )
                prior_errors[str(domain)] = _grouped_action_regret(
                    domain_data,
                    reference_prediction,
                    definition.target_name,
                )
            else:
                target = np.asarray(domain_data.targets[definition.target_name], dtype=float)
                prior = np.asarray(domain_data.priors[definition.target_name], dtype=float)
                prior_errors[str(domain)] = float(np.mean(np.abs(target - prior)))
        prior_score = float(np.mean(list(prior_errors.values()))) if prior_errors else float("inf")
        candidates: list[dict[str, Any]] = [
            {
                "variant_name": "prior_only",
                "parent_names": (),
                "latent_rank": 0,
                "adaptation_shrinkage": 0.0,
                "cv_ratio": 1.0,
                "cv_domain_errors": prior_errors,
                "selection_score": prior_score if use_action_ranking else 1.0,
            }
        ]
        for variant_name, parent_names in definition.parent_variants.items():
            _feature_indices(dataset.feature_names, parent_names)
            for latent_rank in self.config.latent_ranks:
                shrinkages = (0.0,) if int(latent_rank) <= 0 else self.config.adaptation_shrinkages
                for shrinkage in shrinkages:
                    cv = _leave_one_domain_score(
                        dataset,
                        definition,
                        variant_name=variant_name,
                        parent_names=tuple(parent_names),
                        latent_rank=int(latent_rank),
                        adaptation_shrinkage=float(shrinkage),
                        config=self.config,
                    )
                    parent_penalty = self.config.complexity_penalty * len(parent_names)
                    rank_penalty = self.config.complexity_penalty * 2.0 * int(latent_rank)
                    candidates.append(
                        {
                            "variant_name": variant_name,
                            "parent_names": tuple(parent_names),
                            "latent_rank": int(latent_rank),
                            "adaptation_shrinkage": float(shrinkage),
                            "cv_ratio": float(cv["ratio"]),
                            "cv_domain_errors": cv["domain_errors"],
                            "selection_score": float(
                                (cv["score"] if use_action_ranking else cv["ratio"])
                                + parent_penalty
                                + rank_penalty
                            ),
                        }
                    )
        return min(candidates, key=lambda item: (item["selection_score"], len(item["parent_names"]), item["latent_rank"]))

    @staticmethod
    def _component_diagnostics(component: _MechanismComponent, selection: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "target_name": component.definition.target_name,
            "enabled": bool(component.enabled),
            "variant_name": component.variant_name,
            "parent_names": list(component.parent_names),
            "latent_rank": int(component.latent_rank),
            "adaptation_shrinkage": float(component.adaptation_shrinkage),
            "cv_ratio": float(component.cv_ratio),
            "cv_domain_errors": dict(component.cv_domain_errors),
            "selection_score": float(selection["selection_score"]),
            "calibration_error": float(component.calibration_error),
            "target_scale": float(component.target_scale),
            "context_support_protocol": CONTEXT_SUPPORT_PROTOCOL,
            "context_support_anchor_count": int(component.support_contexts.shape[0]),
        }


class DenseResidualWorldModel:
    """Same-output dense baseline with direct access to local and context features."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        ridge_l2: float = 3.0,
        huber_delta: float = 1.5,
        huber_iterations: int = 3,
    ) -> None:
        self.definitions = tuple(definitions)
        self.ridge_l2 = float(ridge_l2)
        self.huber_delta = float(huber_delta)
        self.huber_iterations = int(huber_iterations)
        self.models: dict[str, dict[str, Any]] = {}

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        features = np.concatenate([dataset.features, dataset.context], axis=1)
        weights = _domain_balanced_weights(dataset.domains)
        scaler = _Standardizer.fit(features, weights)
        x_aug = _with_intercept(scaler.transform(features))
        self.models = {}
        diagnostics = {}
        for definition in self.definitions:
            target = np.asarray(dataset.targets[definition.target_name], dtype=float)
            prior = np.asarray(dataset.priors[definition.target_name], dtype=float)
            residual = target - prior
            target_scale = _weighted_scale(residual, weights)
            coef = _robust_ridge(
                x_aug,
                residual / target_scale,
                weights,
                l2=self.ridge_l2,
                huber_delta=self.huber_delta,
                iterations=self.huber_iterations,
            )
            pred = prior + (x_aug @ coef) * target_scale
            errors = np.abs(target - pred)
            calibration_error = float(np.quantile(errors, 0.90))
            self.models[definition.name] = {
                "definition": definition,
                "scaler": scaler,
                "coef": coef,
                "target_scale": target_scale,
                "calibration_error": calibration_error,
            }
            diagnostics[definition.name] = {
                "feature_count": int(features.shape[1]),
                "calibration_error": calibration_error,
            }
        return diagnostics

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        features = np.concatenate([dataset.features, dataset.context], axis=1)
        result = {}
        for name, model in self.models.items():
            x_aug = _with_intercept(model["scaler"].transform(features))
            residual = (x_aug @ model["coef"]) * model["target_scale"]
            prior = np.asarray(dataset.priors[model["definition"].target_name], dtype=float)
            mean = prior + residual
            definition = model["definition"]
            if definition.clip_min is not None or definition.clip_max is not None:
                mean = np.clip(
                    mean,
                    -np.inf if definition.clip_min is None else definition.clip_min,
                    np.inf if definition.clip_max is None else definition.clip_max,
                )
            result[name] = {
                "prior": prior,
                "global_residual": residual,
                "latent_residual": np.zeros(dataset.size, dtype=float),
                "mean": mean,
                "uncertainty": np.full(dataset.size, model["calibration_error"], dtype=float),
                "context_trust": np.ones(dataset.size, dtype=float),
                "context_distance": np.zeros(dataset.size, dtype=float),
            }
        return result


class PriorMechanismWorldModel:
    """Uncalibrated-prior baseline with source-only error calibration."""

    def __init__(
        self,
        definitions: Sequence[MechanismDefinition],
        *,
        uncertainty_quantile: float = 0.90,
    ) -> None:
        self.definitions = tuple(definitions)
        self.uncertainty_quantile = float(uncertainty_quantile)
        self.calibration: dict[str, float] = {}

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        diagnostics = {}
        self.calibration = {}
        for definition in self.definitions:
            target = np.asarray(dataset.targets[definition.target_name], dtype=float)
            prior = np.asarray(dataset.priors[definition.target_name], dtype=float)
            error = np.abs(target - prior)
            calibration = max(float(np.quantile(error, self.uncertainty_quantile)), 1e-8)
            self.calibration[definition.name] = calibration
            diagnostics[definition.name] = {"calibration_error": calibration}
        return diagnostics

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        if not self.calibration:
            raise RuntimeError("fit must be called before predict")
        result = {}
        for definition in self.definitions:
            prior = np.asarray(dataset.priors[definition.target_name], dtype=float).reshape(-1)
            result[definition.name] = {
                "prior": prior,
                "global_residual": np.zeros(dataset.size, dtype=float),
                "latent_residual": np.zeros(dataset.size, dtype=float),
                "mean": prior.copy(),
                "uncertainty": np.full(dataset.size, self.calibration[definition.name], dtype=float),
                "context_trust": np.ones(dataset.size, dtype=float),
                "context_distance": np.zeros(dataset.size, dtype=float),
            }
        return result


def _fit_component(
    dataset: MechanismDataset,
    definition: MechanismDefinition,
    *,
    variant_name: str,
    parent_names: tuple[str, ...],
    latent_rank: int,
    adaptation_shrinkage: float,
    config: MechanismFitConfig,
) -> _MechanismComponent:
    feature_indices = _feature_indices(dataset.feature_names, parent_names)
    x = np.asarray(dataset.features[:, feature_indices], dtype=float)
    weights = _domain_balanced_weights(dataset.domains)
    feature_scaler = _Standardizer.fit(x, weights)
    x_aug = _with_intercept(feature_scaler.transform(x))
    target = np.asarray(dataset.targets[definition.target_name], dtype=float).reshape(-1)
    prior = np.asarray(dataset.priors[definition.target_name], dtype=float).reshape(-1)
    residual = target - prior
    target_scale = _weighted_scale(residual, weights)
    y = residual / target_scale
    global_coef = _robust_ridge(
        x_aug,
        y,
        weights,
        l2=config.ridge_l2,
        huber_delta=config.huber_delta,
        iterations=config.huber_iterations,
    )

    domains = np.unique(dataset.domains)
    context_geometry = fit_context_reference_geometry(
        dataset.context, dataset.domains, min_scale=1e-4
    )
    source_contexts = context_geometry.domain_centroids
    context_weights = np.ones(domains.size, dtype=float)
    context_scaler = _Standardizer.fit(source_contexts, context_weights)
    context_aug = _with_intercept(context_scaler.transform(source_contexts))

    offsets = []
    for domain in domains:
        mask = dataset.domains == domain
        if int(mask.sum()) < config.min_domain_rows:
            offsets.append(np.zeros_like(global_coef))
            continue
        domain_residual = y[mask] - x_aug[mask] @ global_coef
        domain_weights = np.ones(int(mask.sum()), dtype=float)
        offsets.append(
            _robust_ridge(
                x_aug[mask],
                domain_residual,
                domain_weights,
                l2=config.domain_offset_l2,
                huber_delta=config.huber_delta,
                iterations=config.huber_iterations,
                regularize_intercept=True,
            )
        )
    domain_offset_coef = np.vstack(offsets)
    max_rank = min(int(latent_rank), max(domains.size - 1, 0), x_aug.shape[1])
    if max_rank > 0 and np.any(np.abs(domain_offset_coef) > 1e-12):
        _, _, vt = np.linalg.svd(domain_offset_coef, full_matrices=False)
        latent_basis = vt[:max_rank]
        theta_targets = domain_offset_coef @ latent_basis.T
        context_encoder_coef = _ridge_solve(
            context_aug,
            theta_targets,
            np.ones(domains.size, dtype=float),
            l2=config.context_l2,
            regularize_intercept=False,
        )
    else:
        max_rank = 0
        latent_basis = np.zeros((0, x_aug.shape[1]), dtype=float)
        context_encoder_coef = np.zeros((context_aug.shape[1], 0), dtype=float)

    fitted = prior + (x_aug @ global_coef) * target_scale
    calibration_error = float(np.quantile(np.abs(target - fitted), config.uncertainty_quantile))
    return _MechanismComponent(
        definition=definition,
        variant_name=variant_name,
        parent_names=parent_names,
        feature_indices=feature_indices,
        feature_scaler=feature_scaler,
        target_scale=target_scale,
        global_coef=global_coef,
        latent_rank=max_rank,
        latent_basis=latent_basis,
        context_scaler=context_scaler,
        context_encoder_coef=context_encoder_coef,
        adaptation_shrinkage=float(adaptation_shrinkage if max_rank > 0 else 0.0),
        source_contexts=source_contexts,
        support_contexts=context_geometry.support_anchors,
        domain_offset_coef=domain_offset_coef,
        calibration_error=max(calibration_error, 1e-8),
        cv_ratio=float("nan"),
        cv_domain_errors={},
        enabled=True,
    )


def _fit_prior_component(
    dataset: MechanismDataset,
    definition: MechanismDefinition,
    config: MechanismFitConfig,
) -> _MechanismComponent:
    weights = _domain_balanced_weights(dataset.domains)
    empty_features = np.zeros((dataset.size, 0), dtype=float)
    feature_scaler = _Standardizer.fit(empty_features, weights)
    context_domains = np.unique(dataset.domains)
    context_geometry = fit_context_reference_geometry(
        dataset.context, dataset.domains, min_scale=1e-4
    )
    source_contexts = context_geometry.domain_centroids
    context_scaler = _Standardizer.fit(source_contexts, np.ones(context_domains.size, dtype=float))
    target = np.asarray(dataset.targets[definition.target_name], dtype=float)
    prior = np.asarray(dataset.priors[definition.target_name], dtype=float)
    errors = np.abs(target - prior)
    return _MechanismComponent(
        definition=definition,
        variant_name="prior_only",
        parent_names=(),
        feature_indices=np.zeros(0, dtype=int),
        feature_scaler=feature_scaler,
        target_scale=1.0,
        global_coef=np.zeros(1, dtype=float),
        latent_rank=0,
        latent_basis=np.zeros((0, 1), dtype=float),
        context_scaler=context_scaler,
        context_encoder_coef=np.zeros((dataset.context.shape[1] + 1, 0), dtype=float),
        adaptation_shrinkage=0.0,
        source_contexts=source_contexts,
        support_contexts=context_geometry.support_anchors,
        domain_offset_coef=np.zeros((context_domains.size, 1), dtype=float),
        calibration_error=max(float(np.quantile(errors, config.uncertainty_quantile)), 1e-8),
        cv_ratio=1.0,
        cv_domain_errors={},
        enabled=False,
    )


def _leave_one_domain_score(
    dataset: MechanismDataset,
    definition: MechanismDefinition,
    *,
    variant_name: str,
    parent_names: tuple[str, ...],
    latent_rank: int,
    adaptation_shrinkage: float,
    config: MechanismFitConfig,
) -> dict[str, Any]:
    domain_errors: dict[str, float] = {}
    baseline_errors: dict[str, float] = {}
    for domain in np.unique(dataset.domains):
        train_mask = dataset.domains != domain
        valid_mask = ~train_mask
        if np.unique(dataset.domains[train_mask]).size < 1:
            continue
        component = _fit_component(
            _subset_for_cv(dataset, train_mask),
            definition,
            variant_name=variant_name,
            parent_names=parent_names,
            latent_rank=latent_rank,
            adaptation_shrinkage=adaptation_shrinkage,
            config=config,
        )
        valid = _subset_for_cv(dataset, valid_mask)
        pred = component.predict(valid)["mean"]
        target = np.asarray(valid.targets[definition.target_name], dtype=float)
        prior = np.asarray(valid.priors[definition.target_name], dtype=float)
        if definition.target_name in config.action_ranking_targets:
            domain_errors[str(domain)] = _grouped_action_regret(valid, pred, definition.target_name)
            baseline_errors[str(domain)] = _grouped_action_regret(
                valid,
                prior,
                definition.target_name,
            )
        else:
            domain_errors[str(domain)] = float(np.mean(np.abs(target - pred)))
            baseline_errors[str(domain)] = float(np.mean(np.abs(target - prior)))
    if not domain_errors:
        return {"ratio": float("inf"), "score": float("inf"), "domain_errors": {}}
    ratios = [domain_errors[name] / max(baseline_errors[name], 1e-8) for name in domain_errors]
    return {
        "ratio": float(np.mean(ratios)),
        "score": float(np.mean(list(domain_errors.values()))),
        "domain_errors": domain_errors,
    }


def _subset_for_cv(dataset: MechanismDataset, mask: np.ndarray) -> MechanismDataset:
    """Subset row-aligned metadata used by grouped action validation."""

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


def _grouped_action_regret(
    dataset: MechanismDataset,
    prediction: np.ndarray,
    target_name: str,
) -> float:
    groups = np.asarray(dataset.metadata.get("action_group_ids", ()))
    is_reference = np.asarray(dataset.metadata.get("is_reference", ()), dtype=bool)
    if groups.shape != (dataset.size,) or is_reference.shape != (dataset.size,):
        raise ValueError("action-ranking selection requires row-aligned action_group_ids and is_reference")
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    actual = np.asarray(dataset.targets[target_name], dtype=float).reshape(-1)
    # Integer group codes make reductions linear in the number of rows. Stable
    # integer sorting is used only for the small per-group quantile needed by
    # action_group_range; it also preserves np.argmin's first-row tie rule.
    group_names, group_index = np.unique(groups, return_inverse=True)
    group_count = int(group_names.size)
    counts = np.bincount(group_index, minlength=group_count)
    reference_counts = np.bincount(
        group_index,
        weights=is_reference.astype(int),
        minlength=group_count,
    )
    eligible = counts > 1
    invalid = np.flatnonzero(eligible & (reference_counts != 1))
    if invalid.size:
        group = group_names[int(invalid[0])]
        raise ValueError(f"action group {group!r} must contain exactly one reference row")
    if not np.any(eligible):
        return 0.0

    minimum_prediction = np.full(group_count, np.inf, dtype=float)
    np.minimum.at(minimum_prediction, group_index, prediction)
    minimizing_rows = prediction == minimum_prediction[group_index]
    selected_rows = np.full(group_count, dataset.size, dtype=int)
    row_indices = np.arange(dataset.size, dtype=int)
    np.minimum.at(
        selected_rows,
        group_index[minimizing_rows],
        row_indices[minimizing_rows],
    )

    minimum_actual = np.full(group_count, np.inf, dtype=float)
    maximum_actual = np.full(group_count, -np.inf, dtype=float)
    np.minimum.at(minimum_actual, group_index, actual)
    np.maximum.at(maximum_actual, group_index, actual)
    order = np.argsort(group_index, kind="stable")
    boundaries = np.concatenate(
        [np.asarray([0], dtype=int), np.cumsum(counts, dtype=int)]
    )
    scales = maximum_actual - minimum_actual
    for group_id in np.flatnonzero(eligible):
        rows = order[boundaries[group_id] : boundaries[group_id + 1]]
        scales[group_id] = action_group_range(actual[rows])
    regrets = (
        actual[selected_rows[eligible]] - minimum_actual[eligible]
    ) / scales[eligible]
    return float(np.mean(regrets))


def _domain_balanced_weights(domains: np.ndarray) -> np.ndarray:
    domains = np.asarray(domains)
    unique, counts = np.unique(domains, return_counts=True)
    count_by_domain = {domain: count for domain, count in zip(unique, counts)}
    weights = np.asarray([1.0 / count_by_domain[domain] for domain in domains], dtype=float)
    return weights * (weights.size / max(float(weights.sum()), 1e-12))


def _weighted_scale(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = weights / max(float(weights.sum()), 1e-12)
    mean = float(np.sum(values * weights))
    variance = float(np.sum((values - mean) ** 2 * weights))
    return max(np.sqrt(max(variance, 0.0)), 1e-6)


def _robust_ridge(
    x: np.ndarray,
    y: np.ndarray,
    base_weights: np.ndarray,
    *,
    l2: float,
    huber_delta: float,
    iterations: int,
    regularize_intercept: bool = False,
) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    weights = np.asarray(base_weights, dtype=float).reshape(-1)
    coef = _ridge_solve(x, y, weights, l2=l2, regularize_intercept=regularize_intercept)
    for _ in range(max(int(iterations) - 1, 0)):
        residual = y - x @ coef
        row_error = np.sqrt(np.mean(residual**2, axis=1))
        scale = max(float(np.median(np.abs(row_error - np.median(row_error)))) * 1.4826, 1e-6)
        threshold = max(float(huber_delta), 1e-6) * scale
        huber_weight = np.minimum(1.0, threshold / np.maximum(row_error, 1e-12))
        coef = _ridge_solve(
            x,
            y,
            weights * huber_weight,
            l2=l2,
            regularize_intercept=regularize_intercept,
        )
    return coef.reshape(x.shape[1], -1).squeeze(-1) if coef.shape[1] == 1 else coef


def _ridge_solve(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    l2: float,
    regularize_intercept: bool,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    weights = np.maximum(np.asarray(weights, dtype=float).reshape(-1), 0.0)
    sqrt_weight = np.sqrt(weights)[:, None]
    xw = x * sqrt_weight
    yw = y * sqrt_weight
    penalty = np.eye(x.shape[1], dtype=float) * float(l2)
    if not regularize_intercept and penalty.size:
        penalty[0, 0] = 0.0
    lhs = xw.T @ xw + penalty
    rhs = xw.T @ yw
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(lhs) @ rhs


def _feature_indices(all_names: Sequence[str], selected_names: Sequence[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(all_names)}
    missing = [name for name in selected_names if name not in index]
    if missing:
        raise KeyError(f"missing mechanism parent features: {missing}")
    return np.asarray([index[name] for name in selected_names], dtype=int)


def _with_intercept(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.concatenate([np.ones((values.shape[0], 1), dtype=float), values], axis=1)


def _nearest_context_distance(
    context: np.ndarray,
    source_contexts: np.ndarray,
    scaler: _Standardizer,
) -> np.ndarray:
    target = scaler.transform(np.asarray(context, dtype=float))
    source = scaler.transform(np.asarray(source_contexts, dtype=float))
    distances = np.sqrt(np.sum((target[:, None, :] - source[None, :, :]) ** 2, axis=2))
    return np.min(distances, axis=1)


def _validate_dataset_outputs(dataset: MechanismDataset, definitions: Sequence[MechanismDefinition]) -> None:
    for definition in definitions:
        if definition.target_name not in dataset.targets:
            raise KeyError(f"missing target for {definition.name}: {definition.target_name}")
        if definition.target_name not in dataset.priors:
            raise KeyError(f"missing prior for {definition.name}: {definition.target_name}")
        for parent_names in definition.parent_variants.values():
            _feature_indices(dataset.feature_names, parent_names)
