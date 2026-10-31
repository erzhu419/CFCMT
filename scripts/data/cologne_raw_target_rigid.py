"""Fit the frozen raw-delta rigid anchor with either approved sample weighting."""

import numpy as np

from cf_h2o.traffic_signal import action_ranker as ranker


RAW_TARGET_PROTOCOL = 'raw_action_delta_v1'
SCORE_UNITS = 'raw_native_halted_per_lane_cost_difference'
DOMAIN_GROUP_WEIGHT_PROTOCOL = 'domain_group_balanced_v1'


def fit_raw_target_anchor(contrast, *, sample_weight_protocol=ranker.GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL,
                          feature_names=None):
    """Fit raw targets with fixed weights and the default or explicitly ordered features."""
    if sample_weight_protocol not in (ranker.GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL, DOMAIN_GROUP_WEIGHT_PROTOCOL):
        raise ValueError('Use the frozen group-normalized or base domain/group sample weights')
    original_weights = sample_weight_protocol == ranker.GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
    selected_features = ranker.CAUSAL_RIGID_RANKING_PARENTS if feature_names is None else tuple(feature_names)
    if feature_names is not None:
        missing = [name for name in selected_features if name not in contrast.feature_names]
        if not selected_features or missing:
            raise ValueError(f'Explicit fit features must be nonempty and present in contrast: missing={missing}')
    config = ranker.ActionAdvantageConfig(
        candidate_only=True, balance_candidate_signs=original_weights,
        target_normalization_protocol=RAW_TARGET_PROTOCOL,
    )
    model = ranker.PairwiseActionAdvantageRegressor(
        causal=True, config=config, causal_feature_names=selected_features,
    )
    model.feature_names = tuple(name for name in model.causal_feature_names if name in contrast.feature_names)
    model.feature_indices = ranker._feature_indices(contrast.feature_names, model.feature_names)
    features_all = model._features(contrast)
    raw_all = np.asarray(contrast.targets['interval_cost'], dtype=float)
    reference = ranker._reference_mask(contrast)
    if np.any(raw_all[reference] != 0.):
        raise ValueError('Use the action-minus-PP contrast, whose reference targets are zero')
    raw_max_abs = float(np.max(np.abs(raw_all)))
    if raw_max_abs > config.target_clip:
        raise ValueError('The fixed raw B100 target must leave the original clip inactive')
    fit_mask = ~reference
    features, raw = features_all[fit_mask], raw_all[fit_mask]
    domains = np.asarray(contrast.domains)
    groups = np.asarray(contrast.metadata['action_group_ids'])

    weights = ranker._domain_group_balanced_weights(domains[fit_mask], groups[fit_mask])
    group_scale_summary = None
    sign_balance = {'enabled': False, 'negative_weight_multiplier': 1.0,
                    'nonnegative_weight_multiplier': 1.0}
    if original_weights:
        # Preserve A's original per-fit operations exactly.
        normalized_all, group_scale_summary = ranker.group_normalized_action_target(
            contrast, 'interval_cost', target_clip=float(config.target_clip),
        )
        normalized = normalized_all[fit_mask]
        weights *= np.clip(np.abs(normalized), 0.25, 4.0)
        negative = normalized < -1e-8
        if np.any(negative) and np.any(~negative):
            total = float(np.sum(weights))
            cap = max(float(config.max_sign_weight_multiplier), 1.0)
            negative_multiplier = float(np.clip(0.5 * total / float(np.sum(weights[negative])), 1.0 / cap, cap))
            nonnegative_multiplier = float(np.clip(0.5 * total / float(np.sum(weights[~negative])), 1.0 / cap, cap))
            weights *= np.where(negative, negative_multiplier, nonnegative_multiplier)
            sign_balance = {'enabled': True, 'negative_weight_multiplier': negative_multiplier,
                            'nonnegative_weight_multiplier': nonnegative_multiplier}
    weights *= weights.size / max(float(np.sum(weights)), 1e-12)

    model.constant_score = float(np.average(raw, weights=weights))
    if float(np.std(raw)) < 1e-8:
        fitted = np.full(raw.shape, model.constant_score, dtype=float)
    else:
        model.model = ranker.HistGradientBoostingRegressor(
            loss='squared_error', learning_rate=config.learning_rate, max_iter=config.max_iter,
            max_leaf_nodes=config.max_leaf_nodes, min_samples_leaf=config.min_samples_leaf,
            l2_regularization=config.l2_regularization, early_stopping=False, random_state=config.random_state,
        )
        model.model.fit(features, raw, sample_weight=weights)
        fitted = np.asarray(model.model.predict(features), dtype=float)
    model.calibration_error = max(float(np.quantile(np.abs(raw-fitted), config.uncertainty_quantile)), 1e-6)
    geometry = ranker.fit_context_reference_geometry(contrast.context, contrast.domains)
    model.source_contexts, model.context_mean, model.context_scale = geometry.support_anchors, geometry.mean, geometry.scale
    model.score_units = SCORE_UNITS
    model.sample_weight_source_protocol = sample_weight_protocol

    fitted_all = (np.full(contrast.size, model.constant_score, dtype=float) if model.model is None
                  else np.asarray(model.model.predict(features_all), dtype=float))
    sign_mask = ~reference & (np.abs(raw_all) > 1e-6)
    diagnostics = {
        'target_name': 'interval_cost', 'target_definition': 'C_action - C_phase_pressure',
        'target_normalization_protocol': RAW_TARGET_PROTOCOL, 'score_units': SCORE_UNITS,
        'estimator': 'HistGradientBoostingRegressor' if model.model is not None else 'constant',
        'model_class': f'{type(model).__module__}.{type(model).__name__}',
        'causal': True, 'candidate_only': True, 'feature_names': list(model.feature_names),
        'feature_count': int(features.shape[1]), 'rows': int(contrast.size),
        'actual_group_count': int(len(np.unique(groups))), 'training_rows': int(features.shape[0]),
        'sample_weight_source_protocol': model.sample_weight_source_protocol,
        'sample_weight_contract': ('Original group-normalized target magnitude, domain/group balance, sign balance, then mean one; recomputed only from this fit subset.'
                                   if original_weights else 'Base domain/group balance, then mean one; recomputed only from this fit subset.'),
        'weight_group_scale_summary': group_scale_summary, 'sign_balance': sign_balance,
        'sample_weight_mean': float(np.mean(weights)), 'sample_weight_min': float(np.min(weights)),
        'sample_weight_max': float(np.max(weights)),
        'raw_target_max_abs': raw_max_abs, 'raw_target_clipped_count': 0,
        'candidate_negative_rate': float(np.mean(raw < -1e-8)),
        'training_weighted_mae': float(np.average(np.abs(raw-fitted), weights=weights)),
        'training_nonzero_sign_accuracy': float(np.mean((fitted_all[sign_mask] < 0.) == (raw_all[sign_mask] < 0.))) if np.any(sign_mask) else 1.,
        'calibration_error': float(model.calibration_error), 'context_reference': geometry.diagnostics(),
        'server_only_sample_weights': weights.tolist(),
        'server_only_fit_row_indices': np.flatnonzero(fit_mask).tolist(),
    }
    return model, diagnostics
