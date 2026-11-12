"""Frozen synthetic cache, independent enumeration and analytic-score checks."""
import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from math import fsum, log
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from experiments.tsc_packet_conditional_law_20261110.law import enumerate_conditional
from experiments.tsc_packet_conditional_panel_20261110.synthetic import cases
from experiments.tsc_packet_conditional_panel_20261110.stream import _observation_masses
from experiments.tsc_packet_discrete_braking_20261110 import conditional_features, motion
from experiments.tsc_packet_joint_law_definition_20261110.law import PARAMETERS, CHOICES, marginal
from experiments.tsc_packet_joint_law_definition_20261110.observation import speed_likelihood
from experiments.tsc_packet_shared_motion_20261110.api import canonical_features
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from . import cache as implementation


STEP = 1e-5
ABS_TOL = 1e-7
REL_TOL = 1e-5
MASS_TOL = 1e-10


def parameter_probes():
    """The two parameter settings fixed before this verification."""
    b, j = np.indices((len(PARAMETERS), len(FEATURE_NAMES)))
    return (np.zeros_like(b, dtype=float), .1*(-1.)**(b+j)/(j+1))


def gradient_directions():
    p = len(FEATURE_NAMES)
    row = (-1.)**np.arange(p)/(np.arange(p)+1)
    result = []
    for block in range(len(PARAMETERS)):
        direction = np.zeros((len(PARAMETERS), p))
        direction[block] = row/np.linalg.norm(row)
        result.append(direction)
    b, j = np.indices((len(PARAMETERS), p))
    mixed = (-1.)**(b+j)/(j+1)
    mixed[:, 0] = 0.
    result.append(mixed/np.linalg.norm(mixed))
    return result


def _rng_state(kernel):
    return {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
            for key, row in kernel.noise._packets.items()}


@contextmanager
def count_physical_preparation():
    """Distinguish enumeration branches from extra feature preparation."""
    counts = dict(resolver_constructions=0, resolver_resolve_method_calls=0)
    original_init = motion.MotionResolver.__init__
    original_resolve = motion.MotionResolver.resolve

    def initialize(resolver, *args, **kwargs):
        counts['resolver_constructions'] += 1
        return original_init(resolver, *args, **kwargs)

    def resolve(resolver, *args, **kwargs):
        counts['resolver_resolve_method_calls'] += 1
        return original_resolve(resolver, *args, **kwargs)

    with patch.object(motion.MotionResolver, '__init__', initialize), \
            patch.object(motion.MotionResolver, 'resolve', resolve):
        yield counts


@contextmanager
def physical_evaluation_forbidden():
    """A cached readout must have no constructor, forecast or feature path."""
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('A repeated cached evaluation attempted physical preparation/resolution')

    with ExitStack() as stack:
        stack.enter_context(patch.object(motion.MotionResolver, '__init__', forbidden))
        stack.enter_context(patch.object(motion.MotionResolver, 'resolve', forbidden))
        for module, names in ((motion, ('_priority_order', 'build_context')),
                (conditional_features, ('prepare_receiving_context', 'prepare_feature_kernel')),
                (implementation, ('_priority_order', 'prepare_receiving_context', 'canonical_features', 'build_context'))):
            for name in names:
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, forbidden))
        yield calls
    if calls:
        raise AssertionError('Physical calls occurred during cache reuse')


def _fold(law, focal):
    result = defaultdict(float)
    for atom in law['atoms']:
        result[tuple(atom['actors'][focal].items())] += atom['probability']
    return result


def _joint_reference(law, focal, label, speed):
    survivor = fsum(atom['probability'] for atom in law['atoms'] if atom['actors'][focal]['survived'])
    return fsum(atom['probability']*speed_likelihood(speed,
        atom['actors'][focal]['physical_speed_mps'], law['old_stopped'][focal])
        for atom in law['atoms'] if atom['actors'][focal]['survived'] and
        atom['actors'][focal]['lane_change'] == CHOICES[label])/survivor


def _queries(law, focal):
    for y in (0., .05, .1, 1.):
        yield ('motion', y), marginal(law, focal, 'motion', y)['likelihood']
    for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2))):
        if law['eligibility'][focal][head]:
            for y in labels:
                yield (head, y), marginal(law, focal, head, y)['likelihood']
    if law['eligibility'][focal]['lane_change']:
        for label in range(3):
            for y in (0., 1.):
                yield ('class_speed', label, y), _joint_reference(law, focal, label, y)


def _read(cache, theta, query):
    if query[0] == 'class_speed':
        return cache.class_speed(theta, query[1], query[2])
    return cache.marginal(theta, query[0], query[1])


def _observation_total(law, focal):
    survivors = [atom for atom in law['atoms'] if atom['actors'][focal]['survived']]
    denominator = fsum(atom['probability'] for atom in survivors)
    stopped = law['old_stopped'][focal]
    zero = fsum(atom['probability']*speed_likelihood(0., atom['actors'][focal]['physical_speed_mps'], stopped)
                for atom in survivors)/denominator
    density = lambda y: fsum(atom['probability']*speed_likelihood(y,
        atom['actors'][focal]['physical_speed_mps'], stopped) for atom in survivors)/denominator
    lower, error0 = quad(density, 0., .1, epsabs=1e-10, epsrel=1e-10)
    upper, error1 = quad(density, .1, float('inf'), epsabs=1e-10, epsrel=1e-10)
    analytic = fsum(atom['probability']*sum(_observation_masses(
        atom['actors'][focal]['physical_speed_mps'], stopped)) for atom in survivors)/denominator
    assert abs(zero+lower+upper-1.) < MASS_TOL
    assert abs(analytic-1.) < MASS_TOL
    if stopped:
        assert abs(upper-marginal(law, focal, 'start', 1)['likelihood']) < MASS_TOL
    return abs(zero+lower+upper-1.), error0+error1


def _check_compact_payload(cache, focal):
    payload = cache.to_dict()
    encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).encode()
    def check(value):
        if isinstance(value, dict):
            assert not {'packets', 'projected_packets', 'kernel', 'full_frame'} & value.keys()
            for child in value.values():
                check(child)
        elif isinstance(value, list):
            for child in value:
                check(child)
    check(payload)
    return len(encoded)


def check_case(kernel, focal):
    began = time.monotonic()
    state, connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    context, rng = deepcopy(kernel._mean_alignment_context), _rng_state(kernel)
    started = time.monotonic()
    with count_physical_preparation() as preparation:
        cache = implementation.build_cache(kernel, focal, max_stochastic=4)
    construction_sec = time.monotonic()-started
    payload_bytes = _check_compact_payload(cache, focal)
    references = [enumerate_conditional(kernel, focal, dict(zip(PARAMETERS, theta)),
                    max_stochastic=4) for theta in parameter_probes()]
    maximum_atom_error = maximum_likelihood_error = maximum_log_error = 0.
    maximum_gradient_error = maximum_score_error = maximum_observation_error = 0.
    comparisons = gradient_checks = zero_support_checks = 0
    directions = gradient_directions()
    reused_at = time.monotonic()
    with physical_evaluation_forbidden() as forbidden_calls:
        for theta, reference in zip(parameter_probes(), references):
            law, summary = cache.law(theta), cache.summary(theta)
            assert set(law['old_stopped']) == {focal} and set(law['eligibility']) == {focal}
            assert all(set(atom['actors']) == {focal} for atom in law['atoms'])
            expected, actual = _fold(reference, focal), _fold(law, focal)
            assert expected.keys() == actual.keys()
            error = max(abs(actual[key]-expected[key]) for key in expected)
            maximum_atom_error = max(maximum_atom_error, error)
            assert error < MASS_TOL
            assert abs(summary['total_probability']-1.) < MASS_TOL
            score_error = float(np.max(np.abs(summary['weighted_score'])))
            maximum_score_error = max(maximum_score_error, score_error)
            assert score_error < MASS_TOL
            observation_error, unused_bound = _observation_total(law, focal)
            maximum_observation_error = max(maximum_observation_error, observation_error)
            survivor = fsum(atom['probability'] for atom in law['atoms'] if atom['actors'][focal]['survived'])
            end = fsum(atom['probability'] for atom in law['atoms'] if not atom['actors'][focal]['survived'])
            assert abs(survivor+end-1.) < MASS_TOL
            for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2))):
                if law['eligibility'][focal][head]:
                    assert abs(fsum(cache.marginal(theta, head, y)['likelihood'] for y in labels)-1.) < MASS_TOL
            if law['eligibility'][focal]['lane_change']:
                for y in (0., 1.):
                    assert abs(fsum(cache.class_speed(theta, label, y)['likelihood'] for label in range(3))-
                        cache.marginal(theta, 'motion', y)['likelihood']) < MASS_TOL
            for query, likelihood in _queries(reference, focal):
                result = _read(cache, theta, query)
                assert result['defined']
                error = abs(result['likelihood']-likelihood)
                maximum_likelihood_error = max(maximum_likelihood_error, error)
                assert error < MASS_TOL
                comparisons += 1
                if likelihood == 0.:
                    assert result['likelihood'] == 0. and np.isneginf(result['log_likelihood'])
                    assert np.isposinf(result['nll'])
                    zero_support_checks += 1
                    continue
                log_error = abs(result['log_likelihood']-log(likelihood))
                maximum_log_error = max(maximum_log_error, log_error)
                assert log_error < MASS_TOL
                gradient = np.asarray(result['gradient'])
                assert gradient.shape == theta.shape and np.isfinite(gradient).all()
                for direction in directions:
                    positive = _read(cache, theta+STEP*direction, query)['nll']
                    negative = _read(cache, theta-STEP*direction, query)['nll']
                    difference = (positive-negative)/(2*STEP)
                    analytic = float(np.sum(gradient*direction))
                    error = abs(difference-analytic)
                    maximum_gradient_error = max(maximum_gradient_error, error)
                    assert error <= ABS_TOL+REL_TOL*max(abs(difference), abs(analytic)), (query, difference, analytic)
                    gradient_checks += 1
    assert not forbidden_calls
    assert kernel.state == state and kernel._connections == connections
    assert kernel._mean_alignment_context == context and _rng_state(kernel) == rng
    stats = dict(cache.stats)
    stats['cache_internal_elapsed_sec'] = stats.pop('elapsed_sec')
    assert stats['physical_calls'] == references[0]['diagnostics']['physical_branches']
    assert stats['leaf_visits'] == references[0]['diagnostics']['enumerated_leaf_branches']
    return cache, dict(frame_actor_count=len(state['packets']),
        stochastic_actor_count=len(references[0]['stochastic_participants']),
        cache_constructions=1, parameter_probes=2, **stats,
        construction_resolver_constructions=preparation['resolver_constructions'],
        construction_resolver_resolve_method_calls=preparation['resolver_resolve_method_calls'],
        nonenumeration_resolver_constructions=preparation['resolver_constructions']-stats['physical_calls'],
        cache_json_bytes=payload_bytes, construction_sec=construction_sec,
        cached_verification_sec=time.monotonic()-reused_at,
        maximum_atom_probability_error=maximum_atom_error, maximum_likelihood_error=maximum_likelihood_error,
        maximum_log_likelihood_error=maximum_log_error, maximum_gradient_direction_error=maximum_gradient_error,
        maximum_weighted_score_error=maximum_score_error, maximum_observation_mass_error=maximum_observation_error,
        marginal_comparisons=comparisons, gradient_direction_checks=gradient_checks,
        independent_reference_physical_calls=sum(int(ref['diagnostics']['physical_branches']) for ref in references),
        zero_support_checks=zero_support_checks, reuse_physical_calls=0,
        input_state_context_and_rng_unchanged=True, elapsed_sec=time.monotonic()-began)


def check_closed_forms(caches, kernels):
    zero, nonzero = parameter_probes()
    stopped_kernel, stopped_focal = kernels['stopped_open']
    stopped_phi = canonical_features(stopped_kernel)[stopped_focal]
    start_errors = []
    with physical_evaluation_forbidden():
        for theta in (zero, nonzero):
            p_brake = expit(theta[PARAMETERS.index('brake')] @ stopped_phi)
            p_start = expit(theta[PARAMETERS.index('start_permission')] @ stopped_phi)
            expected = p_start*(1-p_brake)
            actual = caches['stopped_open'].marginal(theta, 'start', 1)['likelihood']
            start_errors.append(abs(actual-expected))
            assert start_errors[-1] < MASS_TOL
        survival = caches['intent_and_survival'].marginal(zero, 'motion', 1.)
        p = .5
        denominator = .4+.6*p
        numerator = .4*(1-p)*norm.pdf(3.)+p*norm.pdf(1.)
        expected = numerator/denominator
        bias = FEATURE_NAMES.index('bias')
        brake = PARAMETERS.index('brake')
        survival_score = float(np.asarray(survival['denominator_score'])[brake, bias])
        numerator_score = float(np.asarray(survival['numerator_score'])[brake, bias])
        expected_score = .6*p*(1-p)/denominator
        assert abs(survival['survivor_probability']-denominator) < MASS_TOL
        assert abs(survival['likelihood']-expected) < MASS_TOL
        assert abs(survival_score-3/14) < MASS_TOL
        assert abs(survival_score-expected_score) < MASS_TOL
        gradient = np.asarray(survival['gradient'])
        assert abs(gradient[brake, bias]-(-numerator_score+3/14)) < MASS_TOL
        direction = np.zeros_like(zero)
        direction[brake, bias] = 1.
        closed_errors = []
        for beta in (-STEP, STEP):
            p = expit(beta)
            expected = (.4*(1-p)*norm.pdf(3.)+p*norm.pdf(1.))/(.4+.6*p)
            actual = caches['intent_and_survival'].marginal(beta*direction, 'motion', 1.)['likelihood']
            closed_errors.append(abs(actual-expected))
            assert closed_errors[-1] < MASS_TOL
    return dict(start_maximum_error=max(start_errors), survival_probability_at_zero=denominator,
        speed_one_likelihood_at_zero=survival['likelihood'], survival_denominator_nll_gradient=survival_score,
        expected_survival_denominator_nll_gradient=3/14, brake_bias_nll_gradient=float(gradient[brake, bias]),
        survival_perturbation_maximum_error=max(closed_errors), cached_physical_calls=0)


def run(output=None):
    began = time.monotonic()
    kernels = cases()
    caches, rows = {}, {}
    for name, (kernel, focal) in kernels.items():
        caches[name], rows[name] = check_case(kernel, focal)
    closed = check_closed_forms(caches, kernels)
    result = dict(status='PASS_SYNTHETIC_BRANCH_CACHE_GRADIENT', synthetic_scenarios=len(rows),
        parameter_probes=2, gradient_directions=6, cases=rows, closed_forms=closed,
        cache_constructions=len(rows), total_gradient_direction_checks=sum(row['gradient_direction_checks'] for row in rows.values()),
        total_enumeration_physical_calls=sum(row['physical_calls'] for row in rows.values()),
        total_independent_reference_physical_calls=sum(row['independent_reference_physical_calls'] for row in rows.values()),
        total_construction_resolver_constructions=sum(row['construction_resolver_constructions'] for row in rows.values()),
        total_cache_json_bytes=sum(row['cache_json_bytes'] for row in rows.values()),
        total_reuse_physical_calls=0, real_record_response_queries=0, new_native_starts=0,
        new_fits=0, optimizer_calls=0, elapsed_sec=time.monotonic()-began,
        scope='Synthetic implementation verification only; no real cache reconstruction, fitting, prediction or control result.')
    if output is not None:
        Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'synthetic_scenarios',
        'total_gradient_direction_checks', 'total_enumeration_physical_calls', 'total_reuse_physical_calls', 'elapsed_sec')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
