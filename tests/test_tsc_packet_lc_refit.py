"""Pure cached-law checks for the fixed 76-variable LC fit."""

from copy import deepcopy
import json

import numpy as np
import pytest

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw
from experiments.tsc_packet_conditional_cache_20261110.verify import physical_evaluation_forbidden
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_lc_refit_20261111.objective import LCObjective
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw


WIDTH = 38


def _outcome(label, survived=True):
    return dict(survived=survived, lane_change=label if survived else None,
                physical_speed_mps=1. if survived else None, started=None, entry_event=0)


def exact_law():
    """The earlier actor can forbid focal left; the brake bit controls survival."""
    features = [[1., .3]+[0.]*(WIDTH-2), [1., -.4]+[0.]*(WIDTH-2)]
    factors = [
        dict(kind='bernoulli', actor='ego', feature_id=0, block=0, bit=0),
        dict(kind='bernoulli', actor='ego', feature_id=0, block=0, bit=1),
        dict(kind='categorical', actor='other', feature_id=1, allowed=[0, 1], choice=0),
        dict(kind='categorical', actor='ego', feature_id=0, allowed=[1, 2], choice=1),
        dict(kind='categorical', actor='ego', feature_id=0, allowed=[1, 2], choice=2),
        dict(kind='categorical', actor='other', feature_id=1, allowed=[0, 1], choice=1),
        dict(kind='categorical', actor='ego', feature_id=0, allowed=[0, 1, 2], choice=0),
        dict(kind='categorical', actor='ego', feature_id=0, allowed=[0, 1, 2], choice=1),
        dict(kind='categorical', actor='ego', feature_id=0, allowed=[0, 1, 2], choice=2),
    ]
    parents_factors = [(-1, -1), (0, 0), (1, 2), (2, 3), (2, 4),
                       (1, 5), (5, 6), (5, 7), (5, 8), (0, 1)]
    prefixes = [dict(parent=parent, factor_id=factor, log_prior=0.)
                for parent, factor in parents_factors]
    outcomes = [_outcome('left'), _outcome('stay'), _outcome('right'), _outcome(None, False)]
    leaves = [dict(prefix_id=prefix, outcome_id=outcome) for prefix, outcome in
              ((3, 1), (4, 2), (6, 0), (7, 1), (8, 2), (9, 3))]
    return CachedLaw(dict(focal_id='ego', stats={}, features=features, factors=factors,
        prefixes=prefixes, outcomes=outcomes, leaves=leaves, old_stopped=False,
        eligibility=dict(lane_change=True, motion=True, start=False, yield_entry=False)))


def mixed_law(anchor):
    """A fixed pool with two survivor masks, prefocal LC and one ended particle."""
    data = dict(focal_id='ego', stats={},
        features=[[1., .3]+[0.]*(WIDTH-2), [1., -.4]+[0.]*(WIDTH-2)],
        factors=[
            dict(kind='bernoulli', actor='ego', feature_id=0, block=0, bit=0),
            dict(kind='bernoulli', actor='ego', feature_id=0, block=0, bit=1),
            dict(kind='categorical', actor='other', feature_id=1, allowed=[0, 1], choice=0),
            dict(kind='categorical', actor='other', feature_id=1, allowed=[0, 1], choice=1)],
        focal_groups=[dict(feature_id=0, allowed=[0, 1, 2]),
                      dict(feature_id=0, allowed=[1, 2])],
        particles=[
            dict(factor_ids=[0, 3], focal_group=0, outcome=_outcome('stay')),
            dict(factor_ids=[0, 2], focal_group=1, outcome=_outcome('stay')),
            dict(factor_ids=[1], focal_group=-1, outcome=_outcome(None, False)),
            dict(factor_ids=[0, 3], focal_group=0, outcome=_outcome('stay'))],
        old_stopped=False, eligibility=dict(lane_change=True, motion=True, start=False, yield_entry=False))
    ratios = ParticleLaw(data)._evaluate(anchor)[0]
    for i, row in enumerate(data['particles']):
        row.update(proposal_component=i % 2, proposal_log_ratio=float(ratios[i]) if i % 2 else 0.)
    return MixtureParticleLaw(data, anchor)


def constant_stay_law(ended=False):
    return CachedLaw(dict(focal_id='ego', stats={}, features=[[1.]+[0.]*(WIDTH-1)],
        factors=[dict(kind='categorical', actor='ego', feature_id=0, allowed=[1], choice=1)],
        prefixes=[dict(parent=-1, factor_id=0, log_prior=0.)],
        leaves=[dict(prefix_id=0, outcome_id=0)], outcomes=[_outcome('stay', not ended)],
        old_stopped=False, eligibility=dict(lane_change=True, motion=True, start=False, yield_entry=False)))


def weighted_objective():
    anchor = .06*np.cos(np.arange(5*WIDTH)).reshape(5, WIDTH)
    laws = {0: exact_law(), 1: mixed_law(anchor), 2: exact_law()}
    references = [dict(case_index=i, seed=76116901+i, y=y, objective_weight=weight)
                  for i, (y, weight) in enumerate(((0, .07), (1, .83), (2, .10)))]
    return LCObjective(laws, references, anchor), laws, references, anchor


def test_natural_weight_loss_gradient_and_fixed_rows_match_direct_laws():
    objective, laws, references, anchor = weighted_objective()
    value = objective.initial_vector + .03*np.sin(np.arange(2*WIDTH))
    theta = objective.coefficients(value)
    with physical_evaluation_forbidden():
        row = objective.evaluate(value)
        native = [laws[r['case_index']].marginal(theta, 'lane_change', r['y']) for r in references]
    expected_loss = sum(r['objective_weight']*a['nll'] for r, a in zip(references, native))
    gradient = sum((r['objective_weight']*a['gradient'][3:5] for r, a in zip(references, native)),
                   np.zeros((2, WIDTH)))
    expected_penalty = .001*np.sum(theta[3:5, 1:]**2)
    gradient[:, 1:] += .002*theta[3:5, 1:]
    assert row['unpenalized'] == pytest.approx(expected_loss, abs=1e-14)
    assert row['regularization'] == pytest.approx(expected_penalty, abs=1e-14)
    assert row['objective'] == pytest.approx(expected_loss+expected_penalty, abs=1e-14)
    np.testing.assert_allclose(row['gradient'], gradient.reshape(-1), rtol=0., atol=1e-13)
    np.testing.assert_array_equal(theta[:3], anchor[:3])
    np.testing.assert_array_equal(anchor, .06*np.cos(np.arange(5*WIDTH)).reshape(5, WIDTH))
    assert row['all_references_supported']


def test_lc_gradient_matches_fixed_direction_differences_with_mixed_anchor_frozen():
    objective, laws, _, _ = weighted_objective()
    value = objective.initial_vector + .04*np.cos(np.arange(2*WIDTH))
    original_offset = laws[1].log_offset.copy()
    rng = np.random.default_rng(20261127)
    directions = np.concatenate((rng.normal(size=(6, 2*WIDTH)), np.eye(2*WIDTH)[[0, 7, WIDTH, WIDTH+7]]))
    with physical_evaluation_forbidden():
        analytical = objective.evaluate(value)['gradient']
        for direction in directions:
            direction /= np.linalg.norm(direction)
            h = 1e-5
            finite = (objective.evaluate(value+h*direction)['objective']-
                      objective.evaluate(value-h*direction)['objective'])/(2*h)
            assert finite == pytest.approx(float(analytical @ direction), abs=2e-8)
    np.testing.assert_array_equal(laws[1].log_offset, original_offset)


def test_constant_stay_law_has_unpenalized_lc_biases_and_only_lc_slope_penalty():
    anchor = np.zeros((5, WIDTH))
    anchor[:3, :] = .7
    refs = [dict(case_index=0, seed=76116901, y=1, objective_weight=1.)]
    objective = LCObjective({0: constant_stay_law()}, refs, anchor)
    value = objective.initial_vector.copy()
    value[[0, WIDTH]] = [.3, -.4]
    with physical_evaluation_forbidden():
        row = objective.evaluate(value)
    assert row['unpenalized'] == row['regularization'] == row['objective'] == 0.
    np.testing.assert_allclose(row['gradient'], np.zeros(2*WIDTH), rtol=0., atol=1e-14)
    value[7] = .6
    with physical_evaluation_forbidden():
        penalized = objective.evaluate(value)
    expected = np.zeros(2*WIDTH)
    expected[7] = .0012
    assert penalized['regularization'] == pytest.approx(.00036, abs=1e-14)
    np.testing.assert_allclose(penalized['gradient'], expected, rtol=0., atol=1e-14)
    np.testing.assert_array_equal(objective.coefficients(value)[:3], anchor[:3])


@pytest.mark.parametrize('label,ended', [(0, False), (1, True)])
def test_unsupported_or_undefined_label_stops_without_a_finite_substitute(label, ended):
    refs = [dict(case_index=0, seed=76116901, y=label, objective_weight=1.)]
    objective = LCObjective({0: constant_stay_law(ended=ended)}, refs, np.zeros((5, WIDTH)))
    with physical_evaluation_forbidden(), pytest.raises(ValueError):
        objective.fun(objective.initial_vector)
    assert len(objective.references) == 1


def test_nonfinite_lc_parameter_and_bad_weights_are_rejected():
    objective, _, _, _ = weighted_objective()
    value = objective.initial_vector.copy()
    value[4] = np.nan
    with physical_evaluation_forbidden(), pytest.raises(ValueError):
        objective.fun(value)
    refs = [dict(case_index=0, seed=76116901, y=1, objective_weight=.5)]
    with pytest.raises(ValueError):
        LCObjective({0: constant_stay_law()}, refs, np.zeros((5, WIDTH)))


def test_fun_records_one_json_scalar_receipt_and_returns_the_same_weighted_objective():
    receipts = []
    objective, laws, refs, anchor = weighted_objective()
    recorded = LCObjective(laws, refs, anchor, record=receipts.append)
    with physical_evaluation_forbidden():
        expected = objective.evaluate(objective.initial_vector)
        value, gradient = recorded.fun(recorded.initial_vector)
    assert value == pytest.approx(expected['objective'], abs=1e-14)
    np.testing.assert_allclose(gradient, expected['gradient'], rtol=0., atol=1e-13)
    assert len(receipts) == 1
    json.dumps(receipts[0], allow_nan=False)
