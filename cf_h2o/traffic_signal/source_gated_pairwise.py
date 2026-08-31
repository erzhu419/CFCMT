"""Source-city validated gate between rigid and all-action pairwise experts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from cf_h2o.traffic_signal.action_ranker import (
    CAUSAL_RIGID_RANKING_PARENTS,
    ActionAdvantageConfig,
    AntisymmetricPairwiseActionRegressor,
    PairwiseActionAdvantageRegressor,
    PairwisePreferenceConfig,
    group_normalized_action_target,
)
from cf_h2o.traffic_signal.causal_mechanism_advantage import (
    SOURCE_EXPERT_GATE_FEATURE_NAMES,
    _apply_source_expert_gate,
    _domain_action_regret,
    _equal_domain_record_weights,
    _robust_regret,
    _source_expert_gate_records,
    _weighted_standardizer,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


SOURCE_GATED_PAIRWISE_PROTOCOL = "source_city_oof_pairwise_lcb_gate_v1"


@dataclass(frozen=True)
class SourceGatedPairwiseConfig:
    gate_ridge_alpha: float = 10.0
    gate_error_quantile: float = 0.75
    minimum_disagreement_groups: int = 24
    minimum_robust_gain: float = 0.002
    worst_regret_tolerance: float = 0.01
    fit_workers: int = 5
    pairwise_config: PairwisePreferenceConfig = PairwisePreferenceConfig()


def _new_rigid_model() -> PairwiseActionAdvantageRegressor:
    return PairwiseActionAdvantageRegressor(
        causal=True,
        config=ActionAdvantageConfig(
            candidate_only=True,
            balance_candidate_signs=True,
        ),
        causal_feature_names=CAUSAL_RIGID_RANKING_PARENTS,
    )


def _subset(dataset: MechanismDataset, mask: np.ndarray) -> MechanismDataset:
    mask = np.asarray(mask, dtype=bool)
    subset = dataset.subset(mask)
    metadata = dict(dataset.metadata)
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


class SourceGatedPairwiseActionRegressor:
    """Use the pairwise expert only where source-city OOF evidence supports it."""

    def __init__(self, *, config: SourceGatedPairwiseConfig | None = None) -> None:
        self.config = config or SourceGatedPairwiseConfig()
        if not 0.0 < float(self.config.gate_error_quantile) < 1.0:
            raise ValueError("pairwise gate error quantile must be in (0, 1)")
        self.rigid_model = _new_rigid_model()
        self.pairwise_model = AntisymmetricPairwiseActionRegressor(
            config=self.config.pairwise_config
        )
        self.gate_model: Ridge | None = None
        self.gate_feature_mean = np.zeros(
            len(SOURCE_EXPERT_GATE_FEATURE_NAMES), dtype=float
        )
        self.gate_feature_scale = np.ones(
            len(SOURCE_EXPERT_GATE_FEATURE_NAMES), dtype=float
        )
        self.gate_error_margin = 0.0
        self.source_diagnostics: dict[str, Any] = {}

    def fit(self, dataset: MechanismDataset) -> dict[str, Any]:
        domains = np.asarray(dataset.domains, dtype=str)
        source_domains = tuple(str(value) for value in np.unique(domains))
        if len(source_domains) < 4:
            raise ValueError("source-gated pairwise model requires four source domains")
        rigid_fit = self.rigid_model.fit(dataset)
        pairwise_fit = self.pairwise_model.fit(dataset)
        rigid_oof = np.zeros(dataset.size, dtype=float)
        pairwise_oof = np.zeros(dataset.size, dtype=float)
        covered = np.zeros(dataset.size, dtype=bool)

        def fit_fold(heldout: str) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
            valid_mask = domains == heldout
            train = _subset(dataset, ~valid_mask)
            valid = _subset(dataset, valid_mask)
            rigid = _new_rigid_model()
            pairwise = AntisymmetricPairwiseActionRegressor(
                config=self.config.pairwise_config
            )
            rigid.fit(train)
            pairwise.fit(train)
            return (
                heldout,
                valid_mask,
                np.asarray(rigid.predict(valid)["control_cost"]["mean"], dtype=float),
                np.asarray(pairwise.predict(valid)["control_cost"]["mean"], dtype=float),
            )

        with ThreadPoolExecutor(
            max_workers=min(max(int(self.config.fit_workers), 1), len(source_domains))
        ) as pool:
            fold_rows = list(pool.map(fit_fold, source_domains))
        for _, valid_mask, rigid_prediction, pairwise_prediction in fold_rows:
            rigid_oof[valid_mask] = rigid_prediction
            pairwise_oof[valid_mask] = pairwise_prediction
            covered[valid_mask] = True
        if not np.all(covered):
            raise RuntimeError("source pairwise gate produced incomplete city OOF scores")

        target, scale_summary = group_normalized_action_target(
            dataset, "interval_cost"
        )
        groups = np.asarray(dataset.metadata["action_group_ids"])
        records = _source_expert_gate_records(
            rigid_oof,
            pairwise_oof,
            groups,
            domains=domains,
            target=target,
        )
        features = np.asarray(records["features"], dtype=float)
        record_domains = np.asarray(records["domains"], dtype=str)
        cost_delta = np.asarray(records["cost_delta"], dtype=float)
        record_count = int(features.shape[0])
        record_domain_names = tuple(str(value) for value in np.unique(record_domains))
        diagnostics: dict[str, Any] = {
            "protocol": SOURCE_GATED_PAIRWISE_PROTOCOL,
            "source_domains": list(source_domains),
            "source_domain_count": len(source_domains),
            "source_city_oof_fold_count": len(fold_rows),
            "rigid_fit": rigid_fit,
            "pairwise_fit": pairwise_fit,
            "group_target_scale_summary": scale_summary,
            "feature_names": list(SOURCE_EXPERT_GATE_FEATURE_NAMES),
            "disagreement_group_count": record_count,
            "gate_enabled": False,
            "reason": "insufficient_source_oof_disagreements",
            "target_labels_used": False,
        }
        if (
            record_count < int(self.config.minimum_disagreement_groups)
            or len(record_domain_names) < 3
        ):
            self.source_diagnostics = diagnostics
            return {
                "estimator": "source_gated_antisymmetric_pairwise_advantage",
                "config": asdict(self.config),
                "source": diagnostics,
            }

        weights = _equal_domain_record_weights(record_domains)
        gate_oof = np.zeros(record_count, dtype=float)
        gate_covered = np.zeros(record_count, dtype=bool)
        for heldout in record_domain_names:
            valid_mask = record_domains == heldout
            train_mask = ~valid_mask
            mean, scale = _weighted_standardizer(
                features[train_mask], weights[train_mask]
            )
            model = Ridge(
                alpha=float(self.config.gate_ridge_alpha), fit_intercept=True
            )
            model.fit(
                (features[train_mask] - mean) / scale,
                cost_delta[train_mask],
                sample_weight=weights[train_mask],
            )
            gate_oof[valid_mask] = model.predict(
                (features[valid_mask] - mean) / scale
            )
            gate_covered[valid_mask] = True
        if not np.all(gate_covered):
            raise RuntimeError("pairwise expert gate produced incomplete record OOF")
        one_sided_error = cost_delta - gate_oof
        error_margin = max(
            float(
                np.quantile(
                    one_sided_error, float(self.config.gate_error_quantile)
                )
            ),
            0.0,
        )
        gate_choice = gate_oof + error_margin < 0.0
        gated_oof = _apply_source_expert_gate(
            rigid_oof,
            pairwise_oof,
            groups,
            records["group_ids"],
            gate_choice,
        )
        rigid_regret = _domain_action_regret(target, rigid_oof, groups, domains)
        pairwise_regret = _domain_action_regret(target, pairwise_oof, groups, domains)
        gated_regret = _domain_action_regret(target, gated_oof, groups, domains)
        oracle_oof = _apply_source_expert_gate(
            rigid_oof,
            pairwise_oof,
            groups,
            records["group_ids"],
            cost_delta < 0.0,
        )
        oracle_regret = _domain_action_regret(target, oracle_oof, groups, domains)
        robust_gain = float(_robust_regret(rigid_regret) - _robust_regret(gated_regret))
        improving_domains = sum(
            gated_regret[name] + 1e-12 < rigid_regret[name]
            for name in rigid_regret
        )
        required_improving = max(1, int(np.ceil(len(source_domains) / 3.0)))
        selected = bool(
            np.any(gate_choice)
            and robust_gain >= float(self.config.minimum_robust_gain)
            and max(gated_regret.values())
            <= max(rigid_regret.values()) + float(self.config.worst_regret_tolerance)
            and improving_domains >= required_improving
        )
        if selected:
            self.gate_feature_mean, self.gate_feature_scale = _weighted_standardizer(
                features, weights
            )
            self.gate_model = Ridge(
                alpha=float(self.config.gate_ridge_alpha), fit_intercept=True
            )
            self.gate_model.fit(
                (features - self.gate_feature_mean) / self.gate_feature_scale,
                cost_delta,
                sample_weight=weights,
            )
            self.gate_error_margin = float(error_margin)
            reason = "selected_by_nested_source_city_oof_regret"
        else:
            self.gate_model = None
            self.gate_error_margin = 0.0
            reason = "pairwise_gate_failed_source_city_oof_regret"
        diagnostics.update(
            {
                "gate_enabled": selected,
                "reason": reason,
                "gate_ridge_alpha": float(self.config.gate_ridge_alpha),
                "gate_error_quantile": float(self.config.gate_error_quantile),
                "one_sided_error_margin": float(error_margin),
                "source_city_oof_mae": float(np.mean(np.abs(cost_delta - gate_oof))),
                "source_city_oof_one_sided_coverage": float(
                    np.mean(cost_delta <= gate_oof + error_margin)
                ),
                "selected_pairwise_group_count": int(np.sum(gate_choice)),
                "selected_pairwise_fraction_of_disagreements": float(
                    np.mean(gate_choice)
                ),
                "rigid_domain_regret": rigid_regret,
                "pairwise_domain_regret": pairwise_regret,
                "gated_domain_regret": gated_regret,
                "oracle_domain_regret": oracle_regret,
                "robust_gain_vs_rigid": robust_gain,
                "improving_domains": int(improving_domains),
                "required_improving_domains": int(required_improving),
            }
        )
        self.source_diagnostics = diagnostics
        return {
            "estimator": "source_gated_antisymmetric_pairwise_advantage",
            "config": asdict(self.config),
            "source": diagnostics,
        }

    def predict(self, dataset: MechanismDataset) -> dict[str, dict[str, np.ndarray]]:
        rigid = self.rigid_model.predict(dataset)["control_cost"]
        pairwise = self.pairwise_model.predict(dataset)["control_cost"]
        rigid_score = np.asarray(rigid["mean"], dtype=float)
        pairwise_score = np.asarray(pairwise["mean"], dtype=float)
        groups = np.asarray(dataset.metadata["action_group_ids"])
        gate_rows = np.zeros(dataset.size, dtype=float)
        predicted_delta_rows = np.zeros(dataset.size, dtype=float)
        score = rigid_score.copy()
        if self.gate_model is not None:
            records = _source_expert_gate_records(
                rigid_score, pairwise_score, groups
            )
            features = np.asarray(records["features"], dtype=float)
            if features.shape[0]:
                predicted_delta = np.asarray(
                    self.gate_model.predict(
                        (features - self.gate_feature_mean)
                        / self.gate_feature_scale
                    ),
                    dtype=float,
                )
                gate_choice = (
                    predicted_delta + self.gate_error_margin < 0.0
                )
                score = _apply_source_expert_gate(
                    rigid_score,
                    pairwise_score,
                    groups,
                    records["group_ids"],
                    gate_choice,
                )
                for group, choice, delta in zip(
                    records["group_ids"], gate_choice, predicted_delta
                ):
                    rows = np.flatnonzero(groups == group)
                    gate_rows[rows] = float(choice)
                    predicted_delta_rows[rows] = float(delta)
        uncertainty = (
            (1.0 - gate_rows) * np.asarray(rigid["uncertainty"], dtype=float)
            + gate_rows * np.asarray(pairwise["uncertainty"], dtype=float)
        )
        trust = (
            (1.0 - gate_rows) * np.asarray(rigid["context_trust"], dtype=float)
            + gate_rows * np.asarray(pairwise["context_trust"], dtype=float)
        )
        distance = np.maximum(
            np.asarray(rigid["context_distance"], dtype=float),
            np.asarray(pairwise["context_distance"], dtype=float),
        )
        support = np.minimum(
            np.asarray(rigid.get("context_support", rigid["context_trust"]), dtype=float),
            np.asarray(pairwise.get("context_support", pairwise["context_trust"]), dtype=float),
        )
        reference = np.asarray(dataset.metadata["is_reference"], dtype=bool)
        score[reference] = 0.0
        uncertainty[reference] = 0.0
        return {
            "control_cost": {
                "prior": np.zeros(dataset.size, dtype=float),
                "global_residual": score,
                "latent_residual": gate_rows * (pairwise_score - rigid_score),
                "mean": score,
                "uncertainty": np.maximum(uncertainty, 0.0),
                "context_trust": np.clip(trust, 0.0, 1.0),
                "context_distance": distance,
                "context_support": support,
                "source_pairwise_gate": gate_rows,
                "source_pairwise_gate_predicted_cost_delta": predicted_delta_rows,
            }
        }
