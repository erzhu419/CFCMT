"""Small zero-physical checks of directional permission and discrimination."""

from math import log

import numpy as np
import pytest
from scipy.special import expit, softmax

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw
from experiments.tsc_packet_conditional_cache_20261110.verify import physical_evaluation_forbidden
from experiments.tsc_packet_lc_discrimination_20261111.readout import (
    direction_readout, regression_summary, training_summary, weighted_auc,
)
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import extended
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw


WIDTH = 38
PHI = np.asarray([[1., 2.]+[0.]*(WIDTH-2), [1., -1.]+[0.]*(WIDTH-2)])


def coefficients():
    theta = np.zeros((5, WIDTH))
    theta[0, :2] = [-.4, .65]
    theta[3, :2] = [.2, .55]
    theta[4, :2] = [-.3, -.35]
    return theta


def outcome(label, survived=True):
    return dict(survived=survived, lane_change=label if survived else None,
                physical_speed_mps=1. if survived else None, started=None, entry_event=0)


def exact_two_goal_law(*, left_permitted=True):
    """Two goal branches have different features, survival and legal masks."""
    factors, prefixes, leaves = [], [dict(parent=-1, factor_id=-1, log_prior=0.)], []
    outcomes = [outcome('left'), outcome('stay'), outcome('right'), outcome(None, False)]
    for feature, prior in enumerate((.3, .7)):
        goal = len(prefixes)
        prefixes.append(dict(parent=0, factor_id=-1, log_prior=log(prior)))
        for bit in (0, 1):
            factor_id = len(factors)
            factors.append(dict(kind='bernoulli', actor='ego', feature_id=feature, block=0, bit=bit))
            prefix_id = len(prefixes)
            prefixes.append(dict(parent=goal, factor_id=factor_id, log_prior=0.))
            if bit:
                leaves.append(dict(prefix_id=prefix_id, outcome_id=3))
                continue
            allowed = [0, 1, 2] if feature == 0 and left_permitted else [1, 2]
            for choice in allowed:
                focal_id = len(factors)
                factors.append(dict(kind='categorical', actor='ego', feature_id=feature,
                                    allowed=allowed, choice=choice))
                leaves.append(dict(prefix_id=len(prefixes), outcome_id=choice))
                prefixes.append(dict(parent=prefix_id, factor_id=focal_id, log_prior=0.))
    return CachedLaw(dict(focal_id='ego', features=PHI.tolist(), factors=factors,
        prefixes=prefixes, leaves=leaves, outcomes=outcomes, stats={}, old_stopped=False,
        eligibility=dict(lane_change=True, motion=True, start=False, yield_entry=False)))


def fixed_mixture_law():
    anchor = np.zeros((5, WIDTH))
    anchor[0, :2] = [.7, -.45]
    data = dict(focal_id='ego', features=PHI.tolist(), stats={}, old_stopped=False,
        factors=[dict(kind='bernoulli', actor='ego', feature_id=feature, block=0, bit=bit)
                 for feature, bit in ((0, 0), (1, 0), (0, 1))],
        focal_groups=[dict(feature_id=0, allowed=[0, 1, 2]), dict(feature_id=1, allowed=[1, 2])],
        particles=[dict(factor_ids=[factor], focal_group=group, outcome=outcome(label, survived))
                   for factor, group, label, survived in
                   ((0, 0, 'stay', True), (1, 1, 'stay', True),
                    (2, -1, None, False), (0, 0, 'stay', True))],
        eligibility=dict(lane_change=True, motion=True, start=False, yield_entry=False))
    return MixtureParticleLaw(data, anchor), anchor


def expected_readout(theta, direction, survivor_mass):
    scores = np.column_stack((PHI @ theta[3], np.zeros(2), PHI @ theta[4]))
    categorical = np.vstack((softmax(scores[0]), np.r_[0., softmax(scores[1, 1:])]))
    weights = np.asarray(survivor_mass)/sum(survivor_mass)
    allowed = np.asarray([True, direction == 2])
    allowed_mass = float(weights @ allowed)
    row = 3 if direction == 0 else 4
    terms = PHI*theta[row]
    allowed_terms = (weights[allowed] @ terms[allowed])/allowed_mass
    return dict(direction_probability=float(weights @ categorical[:, direction]),
                allowed_mass=allowed_mass,
                conditional_on_allowed_probability=float(weights @ categorical[:, direction])/allowed_mass,
                latent_score_terms=allowed_terms, survivor_score_terms=weights @ terms,
                mean_category_score=float((weights @ terms).sum()),
                mean_allowed_category_score=float(allowed_terms.sum()))


@pytest.mark.parametrize('direction', [0, 2])
def test_exact_permission_and_scores_use_survivor_not_goal_prior_or_class_posterior(direction):
    theta = coefficients()
    mass = np.asarray([.3, .7])*expit(-(PHI @ theta[0]))
    expected = expected_readout(theta, direction, mass)
    with physical_evaluation_forbidden():
        law = exact_two_goal_law()
        actual = direction_readout(law, theta, direction)
        native = law.marginal(theta, 'lane_change', direction)
        stay = law.marginal(theta, 'lane_change', 1)
    assert actual['survivor_probability'] == pytest.approx(float(mass.sum()), abs=1e-13)
    for key in ('direction_probability', 'allowed_mass', 'conditional_on_allowed_probability',
                'mean_category_score', 'mean_allowed_category_score'):
        assert actual[key] == pytest.approx(expected[key], abs=1e-13)
    for key in ('latent_score_terms', 'survivor_score_terms'):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-13, rtol=0.)
    assert actual['direction_probability'] == pytest.approx(native['likelihood'], abs=1e-13)
    assert actual['direction_probability'] == pytest.approx(
        actual['allowed_mass']*actual['conditional_on_allowed_probability'], abs=1e-13)
    # These are latent category scores; their weighted mean is not marginal log odds.
    assert not np.isclose(actual['mean_category_score'], log(native['likelihood']/stay['likelihood']))


@pytest.mark.parametrize('direction', [0, 2])
def test_fixed_mixture_permission_and_score_weights_use_corrected_surviving_particles(direction):
    theta = coefficients()
    law, anchor = fixed_mixture_law()
    bits = np.asarray([0, 0, 1, 0])
    features = PHI[[0, 1, 0, 0]]
    target_bit_probability = np.where(bits, expit(features @ theta[0]), expit(-(features @ theta[0])))
    anchor_bit_probability = np.where(bits, expit(features @ anchor[0]), expit(-(features @ anchor[0])))
    weights = 2*target_bit_probability/(.5+anchor_bit_probability)
    expected = expected_readout(theta, direction, [weights[0]+weights[3], weights[1]])
    offset = law.log_offset.copy()
    with physical_evaluation_forbidden():
        actual = direction_readout(law, theta, direction)
        native = law.marginal(theta, 'lane_change', direction)
    assert actual['survivor_probability'] == pytest.approx(
        float(weights[[0, 1, 3]].sum()/weights.sum()), abs=1e-13)
    for key in ('direction_probability', 'allowed_mass', 'conditional_on_allowed_probability',
                'mean_category_score', 'mean_allowed_category_score'):
        assert actual[key] == pytest.approx(expected[key], abs=1e-13)
    for key in ('latent_score_terms', 'survivor_score_terms'):
        np.testing.assert_allclose(actual[key], expected[key], atol=1e-13, rtol=0.)
    assert actual['direction_probability'] == pytest.approx(native['likelihood'], abs=1e-13)
    np.testing.assert_array_equal(law.log_offset, offset)


def test_zero_permission_remains_zero_probability_and_keeps_survivor_score_probe():
    theta = coefficients()
    law = exact_two_goal_law(left_permitted=False)
    mass = np.asarray([.3, .7])*expit(-(PHI @ theta[0]))
    with physical_evaluation_forbidden():
        left = direction_readout(law, theta, 0)
        # The same law also supports a right probe, irrespective of its observed label.
        right = direction_readout(law, theta, 2)
    assert left['allowed_mass'] == left['direction_probability'] == 0.
    assert left['conditional_on_allowed_probability'] is None
    assert left['mean_allowed_category_score'] is None
    assert left['latent_score_terms'] is None
    np.testing.assert_allclose(left['survivor_score_terms'],
        (mass/mass.sum()) @ (PHI*theta[3]), atol=1e-13, rtol=0.)
    assert right['allowed_mass'] == pytest.approx(1., abs=1e-13)
    assert right['direction_probability'] > 0.


def test_weighted_auc_preserves_natural_weights_zero_scores_and_half_credit_ties():
    # Uniform AUC would be .625; dropping the zero-permission positive changes it too.
    positive, stay = [-float('inf'), log(.8)], [-float('inf'), log(.5)]
    actual = weighted_auc(positive, [1., 3.], stay, [2., 6.])
    assert actual == pytest.approx(25/32, abs=1e-14)
    assert weighted_auc(positive, [7., 21.], stay, [10., 30.]) == pytest.approx(actual, abs=1e-14)


def probability_row(probability, *, permitted=True):
    log_probability = log(probability) if probability else -float('inf')
    terms = [log_probability if probability else -.4]+[0.]*(WIDTH-1)
    return dict(direction_probability=probability, allowed_mass=float(permitted),
        direction_log_probability=extended(log_probability),
        allowed_log_mass=extended(0. if permitted else -float('inf')),
        conditional_on_allowed_probability=probability if permitted else None,
        conditional_log_probability=extended(log_probability if permitted else None),
        mean_category_score=terms[0], mean_allowed_category_score=terms[0] if permitted else None,
        survivor_score_terms=terms, latent_score_terms=terms if permitted else None)


def test_training_uses_registered_weights_and_same_stays_for_both_direction_probes():
    records = []
    rows = ((0, 1., 0., .2), (0, 3., .8, .1), (2, 5., .6, .4),
            (1, 2., 0., .6), (1, 6., .5, .2))
    for index, (label, weight, left, right) in enumerate(rows):
        records.append(dict(case_index=index, y=label, objective_weight=weight/17.,
            reference=dict(seed=76116901), points={'frozen': {
                '0':probability_row(left, permitted=bool(left)), '2':probability_row(right)}}))
    actual = training_summary(records, 'frozen')
    assert actual['left']['all']['positive']['count'] == 2
    assert actual['right']['all']['positive']['count'] == 1
    for direction in ('left', 'right'):
        assert actual[direction]['all']['stay']['count'] == 2
        assert actual[direction]['all']['stay']['weight'] == pytest.approx(8/17)
    assert actual['left']['all']['weighted_auc']['direction_log_probability'] == pytest.approx(25/32)
    assert actual['right']['all']['weighted_auc']['direction_log_probability'] == pytest.approx(.75)
    assert actual['left']['all']['positive']['zero_permission'] == 1
    assert actual['left']['all']['weighted_auc']['conditional_log_probability'] is None
    assert actual['left']['all']['stay']['conditional_on_allowed_probability']['missing'] == 1


def test_regression_keeps_original_pairs_and_probes_the_stay_in_pair_direction():
    records, coverage = [], {}
    # Each actual stay has both directional probes; the paired score uses the positive's direction.
    for pair, direction, time, positive, stay in ((0, 0, 15., .3, .2), (1, 2, 315., .1, .4)):
        for role, probability, label in (('positive', positive, direction), ('stay', stay, 1)):
            index = 2*pair+int(role == 'stay')
            records.append(dict(case_index=index, pair_index=pair, direction=direction, role=role,
                y=label, reference=dict(time_sec=time), points={'frozen': {
                    str(direction):probability_row(probability),
                    str(2-direction):probability_row(.9-probability)}}))
            coverage[('regression', index, direction)] = dict(matching_train_stay_count=int(pair == 0))
    actual = regression_summary(records, 'frozen', coverage)
    left, right = actual['left'], actual['right']
    assert left['pairs'] == right['pairs'] == 1
    assert left['ranking']['direction_log_probability']['all']['wins'] == 1
    assert right['ranking']['direction_log_probability']['all']['losses'] == 1
    assert left['ranking']['direction_log_probability']['time_block_0']['score'] == 1.
    assert right['ranking']['direction_log_probability']['time_block_1']['score'] == 0.
    assert left['ranking']['direction_log_probability']['positive_covered']['wins'] == 1
    assert right['ranking']['direction_log_probability']['positive_uncovered']['losses'] == 1


def test_zero_permission_negative_is_retained_and_conditional_subset_is_explicitly_renormalized():
    records = []
    for index, (label, weight, probability) in enumerate(
            ((0, 1., .2), (0, 3., .8), (1, 10., 0.), (1, 2., .4), (1, 6., .9))):
        records.append(dict(case_index=index, y=label, objective_weight=weight/22.,
            reference=dict(seed=76116901), points={'frozen': {
                '0':probability_row(probability, permitted=bool(probability)),
                '2':probability_row(.1)}}))
    actual = training_summary(records, 'frozen')['left']['all']
    assert actual['stay']['count'] == 3
    assert actual['stay']['zero_permission'] == 1
    assert actual['weighted_auc']['direction_log_probability'] == pytest.approx(23/36)
    assert actual['weighted_auc']['conditional_log_probability'] is None
    conditional = actual['explicitly_defined_subset_auc']['conditional_log_probability']
    assert conditional['positive_count'] == 2
    assert conditional['stay_count'] == 2
    assert conditional['excluded_positive_count'] == 0
    assert conditional['excluded_stay_count'] == 1
    assert conditional['positive_weight'] == pytest.approx(4/22)
    assert conditional['stay_weight'] == pytest.approx(8/22)
    assert conditional['excluded_positive_weight'] == 0.
    assert conditional['excluded_stay_weight'] == pytest.approx(10/22)
    assert conditional['auc'] == pytest.approx(3/16)
    description = actual['stay']['conditional_on_allowed_probability']
    assert description['weighted_mean'] is None
    assert description['defined_weight'] == pytest.approx(8/22)
    assert description['defined_weighted_mean'] == pytest.approx(.775)
