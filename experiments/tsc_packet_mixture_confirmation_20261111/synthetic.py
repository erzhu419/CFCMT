"""Frozen synthetic gate for the two-component prefix proposal."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from math import fsum, log
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.cache import build_cache
from experiments.tsc_packet_conditional_cache_20261110.verify import (
    _fold, parameter_probes, physical_evaluation_forbidden,
)
from experiments.tsc_packet_conditional_law_20261110.law import enumerate_conditional
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import (
    consistency, directional_check,
)
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver
from experiments.tsc_packet_joint_law_definition_20261110.law import CHOICES, PARAMETERS
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import cases
from experiments.tsc_packet_real_particle_pilot_20261110.sampler import make_sampler
from experiments.tsc_packet_real_particle_pilot_20261110.synthetic import (
    _input_state, _same_input, numeric_memo_only,
)
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from .readout import MixtureParticleLaw, evaluate_pool, mixture_log_offset
from .sampler import MixtureSampler


# Each exact cost is goal configurations times applicable Bernoulli combinations.
FIXTURES = (
    ('two_actor_lateral', 'front', 2, 4),  # Reuse geometry with the later actor focal.
    ('stopped_open', 'ego', 1, 4),
    ('yielding_entry', 'ego', 1, 4),
    ('intent_and_survival', 'ego', 1, 4),
    ('receiving_intent', 'ego', 1, 2),
)
Q0_STREAM_SIZE = 16
MIXTURE_SIZE = 128
ANCHOR_BIAS_ADJUSTMENTS = (.75, -.85, .65, .95, -1.10)
MASS_TOL = 1e-10
DECLARED_CALLS = sum(2*Q0_STREAM_SIZE + MIXTURE_SIZE + 3*exact
                     for _, _, _, exact in FIXTURES)


def targets():
    zero, anchor = parameter_probes()
    anchor[:, FEATURE_NAMES.index('bias')] += ANCHOR_BIAS_ADJUSTMENTS
    return zero, anchor


def _fixture_receipt():
    return [dict(fixture=name, focal_id=focal, stochastic_actors=actors,
        old_q0_calls=Q0_STREAM_SIZE, new_q0_calls=Q0_STREAM_SIZE,
        new_mixture_calls=MIXTURE_SIZE, exact_cache_calls=exact,
        independent_enumeration_calls_per_target=exact,
        seed_zero=20261115+100000*i, seed_fitted=20261116+100000*i)
        for i, (name, focal, actors, exact) in enumerate(FIXTURES)]


def _pure_payload():
    def outcome(survived, speed):
        return dict(survived=survived, physical_speed_mps=speed, started=None,
            entry_event=0, lane_change='stay' if survived else None,
            lane_id='a' if survived else None, position_m=speed)

    phi = [1., .3]+[0.]*(len(FEATURE_NAMES)-2)
    factors = [dict(kind='bernoulli', actor='ego', feature_id=0, block=0, bit=bit)
               for bit in (0, 1)]
    factors += [dict(kind='categorical', actor='front', feature_id=0,
                     allowed=[0, 1], choice=choice) for choice in (0, 1)]
    rows = [dict(factor_ids=[0, 2], focal_group=0, focal_targets={'stay': 'a'},
                 outcome=outcome(True, 2.)),
            dict(factor_ids=[1, 3], focal_group=1, focal_targets={'left': 'l', 'stay': 'a'},
                 outcome=outcome(True, .5)),
            dict(factor_ids=[0], focal_group=-1, focal_targets={}, outcome=outcome(False, None))]
    return dict(focal_id='ego', old_stopped=False, stats={},
        eligibility=dict(motion=True, lane_change=True, start=False, yield_entry=False),
        features=[phi], factors=factors, particles=rows,
        focal_groups=[dict(feature_id=0, allowed=[1]), dict(feature_id=0, allowed=[0, 1])])


def pure_checks():
    zero, anchor = targets()
    data = _pure_payload()
    base, mixed = ParticleLaw(data), MixtureParticleLaw(data, anchor)
    with physical_evaluation_forbidden():
        ratios = base._evaluate(anchor)[0]
        offset = np.log(.5+.5*np.exp(ratios))
        np.testing.assert_allclose(mixture_log_offset(ratios), offset, rtol=0, atol=1e-14)
        zero_corrected = MixtureParticleLaw(data, zero)
        for point in (zero, anchor):
            lw, score, prob, focal_score = base._evaluate(point)
            corrected = mixed._evaluate(point)
            np.testing.assert_allclose(corrected[0], lw-offset, rtol=0, atol=1e-14)
            for actual, expected in zip(corrected[1:], (score, prob, focal_score)):
                np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(zero_corrected._evaluate(point)[0], lw)
        extremes = np.array([-np.inf, -1000., 0., 1000., np.inf])
        extreme_offsets = mixture_log_offset(extremes)
        assert extreme_offsets[0] == log(.5) and np.isposinf(extreme_offsets[-1])
        np.testing.assert_allclose(np.exp(-extreme_offsets[1:-1]) +
            np.exp(extremes[1:-1]-extreme_offsets[1:-1]), 2., rtol=0, atol=2e-13)
        weights = [np.exp(mixed._evaluate(point)[0]) for point in (zero, anchor)]
        np.testing.assert_allclose(weights[0]+weights[1], 2., rtol=0, atol=1e-14)
        assert max(weight.max() for weight in weights) <= 2.+1e-14
        maximum_error = max(consistency(mixed, point, data['eligibility'], n)[0]
                            for point in (zero, anchor) for n in (1, 2, 3))
        directions, gradient_error = 0, 0.
        for point in (zero, anchor):
            for query in (('motion', 1.), ('lane_change', 0), ('class_speed', 0, 1.)):
                n, error = directional_check(mixed, point, query, 3)
                assert n == 6
                directions += n
                gradient_error = max(gradient_error, error)
        absent = mixed.marginal(anchor, 'lane_change', 0, n=1)
        assert absent['defined'] and not absent['supported'] and absent['likelihood'] == 0.
        assert absent['nll'] == float('inf') and absent['gradient'] is None
        no_survivor = dict(data, particles=data['particles'][-1:])
        undefined = MixtureParticleLaw(no_survivor, anchor).marginal(anchor, 'motion', 1.)
        assert not undefined['defined'] and undefined['likelihood'] is None
        assert undefined['nll'] is None and undefined['posterior_ess'] == 0.
        balanced = dict(data, particles=[dict(data['particles'][source],
            proposal_component=i % 2, proposal_log_ratio=float(ratios[source]) if i % 2 else 0.)
            for i, source in enumerate((0, 0, 1, 2))])
        balanced_pool = MixtureParticleLaw(balanced, anchor)
        checked = evaluate_pool(balanced_pool, anchor, [{'head': 'lane_change', 'y': 0}],
                                sizes=(2, 4))
        assert checked['2']['component_counts'] == {'0': 1, 'f': 1}
        assert checked['4']['component_counts'] == {'0': 2, 'f': 2}
        assert all(not row['supported'] for row in checked['2']['lane_change_combinations'].values())
        assert all(row['supported'] for row in checked['4']['lane_change_combinations'].values())
    return dict(status='PASS_PURE_MIXTURE_ARITHMETIC', full_forward_calls=0,
        zero_anchor_reduces_to_original=True, endpoint_bound_and_weight_sum_pass=True,
        both_balanced_prefixes_verified=True,
        unsupported_infinity_and_empty_survivor_preserved=True,
        maximum_normalization_error=maximum_error, directional_derivative_checks=directions,
        maximum_directional_derivative_error=gradient_error)


def _path_key(data, row):
    factors = []
    for index in row['factor_ids']:
        factor = data['factors'][index]
        factors.append((factor['kind'], factor['actor'], tuple(data['features'][factor['feature_id']]),
                        factor.get('block'), factor.get('bit'), tuple(factor.get('allowed', ())),
                        factor.get('choice')))
    return (tuple(row['current_goals']), tuple(tuple(pair) for pair in row['receiving_goals']),
            tuple(factors))


def _enumerated_prefixes(cache, theta):
    """Collapse the analytically integrated focal and later choices in the oracle."""
    data, focal = cache.data, cache.focal_id
    probability = np.exp(np.column_stack([cache._evaluate(point)[0] for point in theta]))
    paths = []
    for node in data['prefixes']:
        previous = paths[node['parent']] if node['parent'] >= 0 else ()
        paths.append(previous + ((node['factor_id'],) if node['factor_id'] >= 0 else ()))
    rows, groups, group_ids, masses, keys = [], [], {}, [], {}
    for leaf, p in zip(data['leaves'], probability):
        receiving = data['receiving_configurations'][leaf['receiving_configuration_id']]
        physical = data['physical_draws'][receiving['physical_draw_id']]
        goal = data['goal_configurations'][physical['goal_configuration_id']]
        outcome = data['outcomes'][leaf['outcome_id']]
        factors = []
        group_id = -1
        for index in paths[leaf['prefix_id']]:
            factor = data['factors'][index]
            if factor['kind'] == 'categorical':
                if not outcome['survived']:
                    break
                if factor['actor'] == focal:
                    key = (factor['feature_id'], tuple(factor['allowed']))
                    if key not in group_ids:
                        group_ids[key] = len(groups)
                        groups.append(dict(feature_id=key[0], allowed=list(key[1])))
                    group_id = group_ids[key]
                    break
            factors.append(index)
        assert (group_id >= 0) == outcome['survived']
        row = dict(factor_ids=factors, focal_group=group_id, focal_targets={}, outcome=None,
            current_goals=[x['goal'] for x in goal['goals']],
            receiving_goals=[(x['actor'], x['goal']) for x in receiving['goals']])
        path_key = _path_key(data, row)
        if path_key not in keys:
            keys[path_key] = len(rows)
            rows.append(row)
            masses.append(np.zeros(2))
        index = keys[path_key]
        stored = rows[index]
        assert stored['focal_group'] == group_id
        if outcome['survived']:
            stored['focal_targets'][outcome['lane_change']] = outcome['lane_id']
        if outcome['lane_change'] in ('stay', None):
            assert stored['outcome'] in (None, outcome)
            stored['outcome'] = outcome
        masses[index] += p
    assert all(row['outcome'] is not None for row in rows)
    for row in rows:
        if row['outcome']['survived']:
            assert set(row['focal_targets']) == {CHOICES[i] for i in groups[row['focal_group']]['allowed']}
    return dict(data, particles=rows, focal_groups=groups), np.asarray(masses), keys


def _mass_map(data, mass, class_prob):
    atoms = defaultdict(list)
    for weight, row in zip(mass, data['particles']):
        if not row['outcome']['survived']:
            atoms[tuple(row['outcome'].items())].append(float(weight))
            continue
        for label, lane in row['focal_targets'].items():
            outcome = dict(row['outcome'], lane_change=label, lane_id=lane)
            atoms[tuple(outcome.items())].append(float(weight*class_prob[row['focal_group'], CHOICES.index(label)]))
    return {key: fsum(values) for key, values in atoms.items()}


def _maximum_atom_error(actual, expected):
    return max(abs(actual.get(key, 0.)-expected.get(key, 0.))
               for key in actual.keys() | expected.keys())


def _case_readout(old, q0, sampled, cache, references, theta):
    zero, anchor = theta
    for key in ('focal_id', 'old_stopped', 'eligibility', 'stochastic_participants',
                'frame_actor_count', 'parameter_names', 'feature_names', 'seed',
                'features', 'factors', 'focal_groups', 'particles'):
        assert old.data[key] == q0.data[key], key
    with physical_evaluation_forbidden():
        exact_data, masses, keys = _enumerated_prefixes(cache, theta)
        exact_pool = MixtureParticleLaw(exact_data, anchor)
        pool = MixtureParticleLaw(sampled.data, anchor)
        np.testing.assert_allclose(masses.sum(axis=0), np.ones(2), rtol=0, atol=MASS_TOL)
        components = [row['proposal_component'] for row in sampled.rows]
        assert components == [0, 1]*(MIXTURE_SIZE//2)
        for original, mixture in zip(old.rows, sampled.rows[::2]):
            assert _path_key(old.data, original) == _path_key(sampled.data, mixture)
            for key in ('outcome', 'focal_targets'):
                assert original[key] == mixture[key]
        fitted_factor_coverage = Counter()
        maximum_path_ratio_error = 0.
        for row, component in zip(sampled.rows, components):
            index = keys[_path_key(sampled.data, row)]
            expected = exact_data['particles'][index]
            assert row['outcome'] == expected['outcome'] and row['focal_targets'] == expected['focal_targets']
            log_ratio = log(masses[index, component])-log(masses[index, 0])
            maximum_path_ratio_error = max(maximum_path_ratio_error,
                abs(log_ratio-row['proposal_log_ratio']))
            if component == 1:
                for i in row['factor_ids']:
                    factor = sampled.data['factors'][i]
                    if factor['kind'] == 'bernoulli':
                        fitted_factor_coverage[f"bernoulli_block_{factor['block']}"] += 1
                    elif len(factor['allowed']) > 1:
                        fitted_factor_coverage['multichoice_prefocal_lc'] += 1
                fitted_factor_coverage['receiving_intents'] += len(row['receiving_goals'])
        assert maximum_path_ratio_error <= MASS_TOL
        mixture_mass = masses.mean(axis=1)
        endpoints = []
        for point, reference in zip(theta, references):
            expected = _fold(reference, cache.focal_id)
            np.testing.assert_allclose(sum(expected.values()), 1., rtol=0, atol=MASS_TOL)
            lw, _, focal_prob, _ = exact_pool._evaluate(point)
            quadrature = _mass_map(exact_data, mixture_mass*np.exp(lw), focal_prob)
            oracle = _fold(cache.law(point), cache.focal_id)
            error = max(_maximum_atom_error(quadrature, expected), _maximum_atom_error(quadrature, oracle))
            assert error <= MASS_TOL
            wmax = float(np.exp(pool._evaluate(point)[0]).max())
            assert wmax <= 2.+MASS_TOL
            normalization = max(consistency(pool, point, pool.data['eligibility'], n)[0]
                                for n in (MIXTURE_SIZE//2, MIXTURE_SIZE))
            empirical = _fold(pool.law(point), cache.focal_id)
            assert all(key in expected for key, mass in empirical.items() if mass > 0.)
            endpoints.append(dict(exact_corrected_maximum_atom_error=error,
                sample_atom_total_variation=.5*fsum(abs(empirical.get(key, 0.)-expected.get(key, 0.))
                    for key in empirical.keys() | expected.keys()),
                max_raw_weight=wmax, maximum_normalization_error=normalization))
    return dict(exact_prefix_count=len(keys), q0_same_seed_payload_identical=True,
        interleaved_components_verified=True, maximum_sampled_path_log_ratio_error=maximum_path_ratio_error,
        fitted_factor_coverage=dict(fitted_factor_coverage), zero=endpoints[0], fitted=endpoints[1])


def run(output, *, pure_only=False):
    start = time.monotonic()
    receipts = _fixture_receipt()
    result = dict(status='RUNNING_SYNTHETIC_MIXTURE_GATE', fixtures=receipts,
        planned_full_forward_calls=0 if pure_only else DECLARED_CALLS,
        anchor_bias_adjustments=ANCHOR_BIAS_ADJUSTMENTS,
        exact_mass_tolerance=MASS_TOL, full_forward_calls=0, cases={},
        real_record_response_queries=0, new_native_starts=0, new_fits=0, optimizer_calls=0)
    limit = 0

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')

    def spend(count, operation):
        nonlocal limit
        before = result['full_forward_calls']
        limit = before+count
        assert limit <= DECLARED_CALLS <= 1024
        value = operation()
        assert result['full_forward_calls'] == limit
        return value

    original_init = MotionResolver.__init__

    def counted_init(resolver, *args, **kwargs):
        if result['full_forward_calls'] >= limit:
            raise AssertionError('Synthetic full-forward call would exceed the frozen allowance')
        result['full_forward_calls'] += 1
        return original_init(resolver, *args, **kwargs)

    save()  # Freeze cases, seeds, target and symbolic costs before response queries.
    try:
        result['pure'] = pure_checks()
        if not pure_only:
            theta = targets()
            with patch.object(MotionResolver, '__init__', counted_init):
                fixtures = cases()
                for receipt, (_, focal, actors, exact_calls) in zip(receipts, FIXTURES):
                    name = receipt['fixture']
                    result['active_fixture'] = name
                    kernel, _ = fixtures[name]
                    before = _input_state(kernel)
                    old = make_sampler(kernel, focal, physical_limit=Q0_STREAM_SIZE)
                    sampler = MixtureSampler(kernel, focal, theta[1],
                        physical_limit=Q0_STREAM_SIZE+MIXTURE_SIZE)
                    assert len(old.participants) == len(sampler.participants) == actors
                    assert old.participants == sampler.participants
                    seed, seed_fitted = receipt['seed_zero'], receipt['seed_fitted']
                    old_q0 = spend(Q0_STREAM_SIZE, lambda: old.draw(Q0_STREAM_SIZE, seed))
                    new_q0 = spend(Q0_STREAM_SIZE, lambda: sampler.draw(Q0_STREAM_SIZE, seed))
                    sampled = spend(MIXTURE_SIZE, lambda: sampler.draw_mixture(MIXTURE_SIZE, seed, seed_fitted))
                    cache = spend(exact_calls, lambda: build_cache(kernel, focal))
                    references = [spend(exact_calls, lambda: enumerate_conditional(kernel, focal,
                        dict(zip(PARAMETERS, point)))) for point in theta]
                    _same_input(kernel, before)
                    numeric_memo_only(sampler)
                    row = _case_readout(old_q0, new_q0, sampled, cache, references, theta)
                    coverage = row['fitted_factor_coverage']
                    requirement = {'two_actor_lateral': 'multichoice_prefocal_lc',
                        'stopped_open': 'bernoulli_block_1', 'yielding_entry': 'bernoulli_block_2',
                        'receiving_intent': 'receiving_intents'}.get(name)
                    if requirement:
                        assert coverage.get(requirement, 0) > 0, (name, requirement, coverage)
                    result['cases'][name] = row
                    save()
            assert result['full_forward_calls'] == DECLARED_CALLS
            result['active_fixture'] = None
        result['status'] = ('PASS_PURE_MIXTURE_ARITHMETIC' if pure_only
                            else 'PASS_SYNTHETIC_MIXTURE_GATE')
    except Exception as error:
        result.update(status='FAILED_SYNTHETIC_MIXTURE_GATE', error=repr(error))
        raise
    finally:
        result['elapsed_sec'] = time.monotonic()-start
        save()
    print(json.dumps({key: result[key] for key in ('status', 'full_forward_calls', 'elapsed_sec')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pure-only', action='store_true')
    args = parser.parse_args()
    run(args.output, pure_only=args.pure_only)
