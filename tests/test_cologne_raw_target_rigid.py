from dataclasses import asdict, replace
import importlib
from pathlib import Path
import pickle

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from cf_h2o.traffic_signal import action_ranker as ranker
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


@pytest.fixture
def raw_fit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    return importlib.import_module('cologne_raw_target_rigid')


def contrast_case(*, uniform_eight=False):
    groups, domains, raw, references, context = [], [], [], [], []
    for group in range(24):
        domain = 'a' if uniform_eight or group < 10 else 'b' if group < 15 else 'c'
        values = ([0., -1., .5, 0., 1.8, -.25, .75, -.5] if uniform_eight
                  else [0., -1., .5, 0., 1.8, -.25][:4 + group % 3])
        action_deltas = np.asarray(values) * (.005 * (group+1))
        for action, target in enumerate(action_deltas):
            groups.append(f'{domain}:group{group}')
            domains.append(domain)
            raw.append(target)
            references.append(action == 0)
            context.append([ord(domain)-ord('a'), group/24])
    names = ('unused_extra', *reversed(ranker.CAUSAL_RIGID_RANKING_PARENTS))
    features = np.random.default_rng(73).normal(size=(len(raw), len(names)))
    features[:, names.index('delta_service_pressure')] = np.asarray(raw) * 10.
    return MechanismDataset(
        feature_names=names, features=features, context_names=('lanes', 'density'),
        context=np.asarray(context), priors={}, targets={'interval_cost': np.asarray(raw)},
        domains=np.asarray(domains), metadata={'action_group_ids': groups, 'is_reference': references},
    )


@pytest.mark.parametrize('heldout_domain', [None, 'b'])
def test_raw_fit_preserves_old_features_hgb_parameters_and_exact_fold_weights(
        raw_fit, monkeypatch, heldout_domain):
    contrast = contrast_case()
    if heldout_domain:
        contrast = ranker._row_subset(contrast, contrast.domains != heldout_domain)
    captured = []
    original_fit = ranker.HistGradientBoostingRegressor.fit

    def spy(estimator, features, target, sample_weight=None):
        captured.append({'features': features.copy(), 'target': target.copy(),
                         'weights': sample_weight.copy(), 'params': estimator.get_params()})
        return original_fit(estimator, features, target, sample_weight=sample_weight)

    monkeypatch.setattr(ranker.HistGradientBoostingRegressor, 'fit', spy)
    legacy = ranker.PairwiseActionAdvantageRegressor(
        causal=True, config=ranker.ActionAdvantageConfig(
            candidate_only=True, balance_candidate_signs=True,
            target_normalization_protocol=ranker.GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL),
        causal_feature_names=ranker.CAUSAL_RIGID_RANKING_PARENTS)
    with threadpool_limits(limits=1):
        legacy_diagnostics = legacy.fit(contrast)
        model, diagnostics = raw_fit.fit_raw_target_anchor(contrast)
    assert type(model) is type(legacy) is ranker.PairwiseActionAdvantageRegressor
    before, after = captured
    np.testing.assert_array_equal(after['features'], before['features'])
    np.testing.assert_array_equal(after['weights'], before['weights'])
    assert after['params'] == before['params']
    candidates = ~np.asarray(contrast.metadata['is_reference'])
    np.testing.assert_array_equal(after['target'], contrast.targets['interval_cost'][candidates])
    assert not np.array_equal(before['target'], after['target'])
    assert model.feature_names == legacy.feature_names == ranker.CAUSAL_RIGID_RANKING_PARENTS
    assert asdict(model.config) == {**asdict(legacy.config),
                                  'target_normalization_protocol': raw_fit.RAW_TARGET_PROTOCOL}
    for attribute in ('source_contexts', 'context_mean', 'context_scale'):
        np.testing.assert_array_equal(getattr(model, attribute), getattr(legacy, attribute))
    assert diagnostics['sign_balance'] == legacy_diagnostics['sign_balance']
    assert diagnostics['actual_group_count'] == len(set(contrast.metadata['action_group_ids']))
    assert diagnostics['training_rows'] == int(candidates.sum())
    np.testing.assert_array_equal(diagnostics['server_only_sample_weights'], before['weights'])
    assert diagnostics['raw_target_clipped_count'] == 0


def test_original_class_pickle_preserves_raw_scores_reference_zero_and_raw_calibration(raw_fit):
    contrast = contrast_case()
    with threadpool_limits(limits=1):
        model, diagnostics = raw_fit.fit_raw_target_anchor(contrast)
        restored = pickle.loads(pickle.dumps(model))
        prediction = restored.predict(contrast)['control_cost']
        estimator_scores = model.model.predict(model._features(contrast))
    assert type(restored) is ranker.PairwiseActionAdvantageRegressor
    assert restored.config.target_normalization_protocol == raw_fit.RAW_TARGET_PROTOCOL
    assert restored.sample_weight_source_protocol == ranker.GROUP_RANGE_ACTION_TARGET_NORMALIZATION_PROTOCOL
    assert restored.score_units == diagnostics['score_units'] == raw_fit.SCORE_UNITS
    assert restored.score_units == 'raw_native_halted_per_lane_cost_difference'
    reference = np.asarray(contrast.metadata['is_reference'])
    expected = estimator_scores.copy()
    expected[reference] = 0.
    np.testing.assert_array_equal(prediction['mean'], expected)
    np.testing.assert_array_equal(prediction['uncertainty'][reference], 0.)
    error = np.abs(contrast.targets['interval_cost'][~reference] - estimator_scores[~reference])
    assert model.calibration_error == restored.calibration_error == max(float(np.quantile(error, .9)), 1e-6)
    assert diagnostics['training_weighted_mae'] == np.average(error, weights=diagnostics['server_only_sample_weights'])
    np.testing.assert_array_equal(restored.context_mean, model.context_mean)


def test_domain_group_protocol_changes_only_hgb_sample_weights(raw_fit, monkeypatch):
    contrast = contrast_case()
    captured = []
    original_fit = ranker.HistGradientBoostingRegressor.fit

    def spy(estimator, features, target, sample_weight=None):
        captured.append((features.copy(), target.copy(), sample_weight.copy(), estimator.get_params()))
        return original_fit(estimator, features, target, sample_weight=sample_weight)

    monkeypatch.setattr(ranker.HistGradientBoostingRegressor, 'fit', spy)
    with threadpool_limits(limits=1):
        original, _ = raw_fit.fit_raw_target_anchor(contrast)
        base, diagnostics = raw_fit.fit_raw_target_anchor(
            contrast, sample_weight_protocol=raw_fit.DOMAIN_GROUP_WEIGHT_PROTOCOL)
    old_call, new_call = captured
    np.testing.assert_array_equal(new_call[0], old_call[0])
    np.testing.assert_array_equal(new_call[1], old_call[1])
    assert new_call[3] == old_call[3]
    assert not np.array_equal(new_call[2], old_call[2])
    assert asdict(base.config) == {**asdict(original.config), 'balance_candidate_signs': False}
    assert base.feature_names == original.feature_names
    for attribute in ('source_contexts', 'context_mean', 'context_scale'):
        np.testing.assert_array_equal(getattr(base, attribute), getattr(original, attribute))
    candidates = ~np.asarray(contrast.metadata['is_reference'])
    groups = np.asarray(contrast.metadata['action_group_ids'])[candidates]
    domains = contrast.domains[candidates]
    # Equal domain mass and equal group mass within each domain, independent of target signs/magnitudes.
    for domain in np.unique(domains):
        domain_groups = np.unique(groups[domains == domain])
        for group in domain_groups:
            expected_mass = len(groups) / (len(np.unique(domains)) * len(domain_groups))
            assert np.sum(new_call[2][groups == group]) == pytest.approx(expected_mass)
    assert diagnostics['sample_weight_source_protocol'] == 'domain_group_balanced_v1'
    assert diagnostics['weight_group_scale_summary'] is None
    assert not diagnostics['sign_balance']['enabled']


def test_one_city_seven_candidate_rows_per_group_get_equal_base_weights(raw_fit):
    contrast = contrast_case(uniform_eight=True)
    with threadpool_limits(limits=1):
        model, diagnostics = raw_fit.fit_raw_target_anchor(
            contrast, sample_weight_protocol=raw_fit.DOMAIN_GROUP_WEIGHT_PROTOCOL)
    assert diagnostics['training_rows'] == 24 * 7
    np.testing.assert_allclose(diagnostics['server_only_sample_weights'], 1., atol=1e-15, rtol=0.)
    restored = pickle.loads(pickle.dumps(model))
    assert type(restored) is ranker.PairwiseActionAdvantageRegressor
    assert restored.sample_weight_source_protocol == 'domain_group_balanced_v1'
    assert restored.config.target_normalization_protocol == raw_fit.RAW_TARGET_PROTOCOL
    assert not restored.config.balance_candidate_signs


def test_explicit_42_features_enter_hgb_in_order_without_changing_labels_or_weights(raw_fit, monkeypatch):
    omitted = importlib.import_module('cologne_omitted_feature_audit').OMITTED_FEATURE_NAMES
    contrast = contrast_case(uniform_eight=True)
    extra_names = tuple(reversed(omitted))
    extra_values = np.arange(contrast.size * len(extra_names), dtype=float).reshape(contrast.size, -1) / 7.
    contrast = replace(contrast, feature_names=(*contrast.feature_names, *extra_names),
                       features=np.column_stack((contrast.features, extra_values)))
    requested = (*ranker.CAUSAL_RIGID_RANKING_PARENTS, *omitted)
    captured = []
    original_fit = ranker.HistGradientBoostingRegressor.fit

    def spy(estimator, features, target, sample_weight=None):
        captured.append((features.copy(), target.copy(), sample_weight.copy(), estimator.get_params()))
        return original_fit(estimator, features, target, sample_weight=sample_weight)

    monkeypatch.setattr(ranker.HistGradientBoostingRegressor, 'fit', spy)
    with threadpool_limits(limits=1):
        original, original_diagnostics = raw_fit.fit_raw_target_anchor(contrast)
        expanded, diagnostics = raw_fit.fit_raw_target_anchor(contrast, feature_names=requested)
    old_call, new_call = captured
    assert old_call[0].shape == (24 * 7, 29)
    assert new_call[0].shape == (24 * 7, 42)
    candidates = ~np.asarray(contrast.metadata['is_reference'])
    indices = [contrast.feature_names.index(name) for name in requested]
    np.testing.assert_array_equal(new_call[0], contrast.features[candidates][:, indices])
    np.testing.assert_array_equal(new_call[0][:, :29], old_call[0])
    np.testing.assert_array_equal(new_call[1], old_call[1])
    np.testing.assert_array_equal(new_call[2], old_call[2])
    assert new_call[3] == old_call[3]
    assert type(expanded) is type(original) is ranker.PairwiseActionAdvantageRegressor
    assert expanded.feature_names == expanded.causal_feature_names == requested
    assert diagnostics['feature_names'] == list(requested)
    assert diagnostics['feature_count'] == 42
    assert asdict(expanded.config) == asdict(original.config)
    for key in ('server_only_fit_row_indices', 'server_only_sample_weights', 'score_units',
                'target_normalization_protocol', 'sample_weight_source_protocol'):
        assert diagnostics[key] == original_diagnostics[key]
    for attribute in ('source_contexts', 'context_mean', 'context_scale'):
        np.testing.assert_array_equal(getattr(expanded, attribute), getattr(original, attribute))


def test_explicit_missing_feature_is_rejected_instead_of_silently_dropped(raw_fit):
    requested = (*ranker.CAUSAL_RIGID_RANKING_PARENTS, 'current_green_elapsed_norm')
    with pytest.raises(ValueError, match='current_green_elapsed_norm'):
        raw_fit.fit_raw_target_anchor(contrast_case(), feature_names=requested)
