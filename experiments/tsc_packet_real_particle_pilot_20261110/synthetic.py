"""One small same-seed equivalence gate; no real records or full MC panel."""
import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.verify import parameter_probes, _rng_state
from experiments.tsc_packet_conditional_panel_20261110.initialize import prepare_zero_kernel
from experiments.tsc_packet_conditional_particles_20261110.fixtures import CASE_NAMES, cases
from experiments.tsc_packet_conditional_particles_20261110 import particles as frozen
from experiments.tsc_packet_conditional_particles_20261110.verify import all_queries, consistency, read
from experiments.tsc_packet_discrete_braking_20261110 import motion
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import executor_seed
from . import sampler as lightweight


PARTICLES = 16


@contextmanager
def feature_only():
    """Geometry/context creation is allowed; full forward calls are not."""
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('Feature preparation attempted a full physical response')
    with ExitStack() as stack:
        stack.enter_context(patch.object(motion.MotionResolver, '__init__', forbidden))
        stack.enter_context(patch.object(motion.MotionResolver, 'resolve', forbidden))
        stack.enter_context(patch.object(frozen, 'prepare_receiving_context', forbidden))
        if hasattr(lightweight, 'prepare_receiving_context'):
            stack.enter_context(patch.object(lightweight, 'prepare_receiving_context', forbidden))
        yield calls
    assert not calls


@contextmanager
def reuse_only():
    with feature_only(), ExitStack() as stack:
        def forbidden(*args, **kwargs):
            raise AssertionError('A cached likelihood readout attempted physical preparation')
        for module, names in ((frozen, ('_priority_order',)),
                              (lightweight, ('build_context', '_priority_order'))):
            for name in names:
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, forbidden))
        yield


def numeric_memo_only(sampler):
    """The configuration table must hold only the declared numeric payload."""
    for value in sampler.configurations.values():
        assert set(value) == {'phi', 'applicability'}
        assert set(value['phi']) == set(sampler.participants)
        assert set(value['applicability']) == set(sampler.participants)
        for phi in value['phi'].values():
            assert isinstance(phi, np.ndarray) and phi.ndim == 1
            assert np.issubdtype(phi.dtype, np.number) and np.isfinite(phi).all()
        for flags in value['applicability'].values():
            assert set(flags) == {'start', 'yielding'}
            assert all(isinstance(flag, (bool, np.bool_)) for flag in flags.values())
    assert sampler.stats['base_kernel_deepcopies'] == 1
    assert sampler.stats['peak_live_goal_views'] <= 1
    assert sampler.stats['live_goal_views'] <= 1


def manual_unprepared():
    prepared, focal = cases()['responsive_background']
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    packets = deepcopy(prepared.state['packets'])
    frame = dict(time_s=prepared.state['time_s'], packets=packets,
        native_lanes=prepared.sensor_operator._reconstruct(packets), native_pending_ids=[],
        executors={key: dict(snapshot=deepcopy(seed['snapshot'])) for key, seed in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))
    action = dict(requests={}, applied_states=dict(prepared.signals))
    kernel = prepare_zero_kernel(prepared.static, frame, seeds, prepared.laws, action, 23,
                                 prepared._spatial_geometry)
    assert not any(hasattr(kernel, name) for name in
        ('_mean_alignment_context', '_conditional_lane_packets', '_conditional_tls_packets'))
    return kernel, focal


def _input_state(kernel):
    return dict(state=deepcopy(kernel.state), connections=deepcopy(kernel._connections),
        rng=_rng_state(kernel), has_context=hasattr(kernel, '_mean_alignment_context'),
        context=deepcopy(getattr(kernel, '_mean_alignment_context', None)))


def _same_input(kernel, before):
    assert kernel.state == before['state'] and kernel._connections == before['connections']
    assert _rng_state(kernel) == before['rng']
    assert hasattr(kernel, '_mean_alignment_context') == before['has_context']
    assert getattr(kernel, '_mean_alignment_context', None) == before['context']


def _compare_row(old, new):
    assert old['defined'] == new['defined'] and old['supported'] == new['supported']
    if not old['defined']:
        assert new['likelihood'] is None and new['gradient'] is None
        return 0., 0.
    probability_error = abs(old['likelihood']-new['likelihood'])
    assert probability_error <= 1e-12
    if old['gradient'] is None:
        assert new['gradient'] is None and old['likelihood'] == new['likelihood'] == 0.
        return probability_error, 0.
    gradient_error = float(np.max(np.abs(old['gradient']-new['gradient'])))
    assert gradient_error <= 1e-12
    assert abs(old['log_likelihood']-new['log_likelihood']) <= 1e-12
    return probability_error, gradient_error


def check_case(kernel, focal, seed):
    before = _input_state(kernel)
    old_sampler = frozen.make_sampler(kernel, focal, max_stochastic=10)
    progress = []
    new_sampler = lightweight.make_sampler(kernel, focal, max_stochastic=10,
        progress=progress.append, physical_limit=PARTICLES,
        feature_limit=len(old_sampler.participants)*(PARTICLES+1))
    goals = [prior[0][0] for prior in new_sampler.goals]
    with feature_only():
        new_sampler._configuration(goals)
    numeric_memo_only(new_sampler)
    old = old_sampler.draw(PARTICLES, seed)
    new = new_sampler.draw(PARTICLES, seed)
    assert old.stats['physical_calls'] == new.stats['physical_calls'] == PARTICLES
    for key in ('focal_id', 'old_stopped', 'eligibility', 'stochastic_participants',
                'frame_actor_count', 'parameter_names', 'feature_names', 'seed',
                'features', 'factors', 'focal_groups', 'particles'):
        assert old.data[key] == new.data[key], key
    probability_error = gradient_error = consistency_error = 0.
    comparisons = 0
    with reuse_only():
        for theta in parameter_probes():
            error, unused = consistency(new, theta, new.data['eligibility'], PARTICLES)
            consistency_error = max(consistency_error, error)
            for query in all_queries(new.data['eligibility']):
                pe, ge = _compare_row(read(old, theta, query), read(new, theta, query))
                probability_error, gradient_error = max(probability_error, pe), max(gradient_error, ge)
                comparisons += 1
    numeric_memo_only(new_sampler)
    assert progress and progress[-1]['physical_calls'] == PARTICLES
    assert max(row['physical_calls'] for row in progress) <= PARTICLES
    _same_input(kernel, before)
    return dict(frame_actor_count=len(kernel.state['packets']),
        stochastic_actor_count=len(new_sampler.participants),
        old_full_forward_calls=PARTICLES, new_full_forward_calls=PARTICLES,
        canonical_feature_evaluations=new_sampler.stats['canonical_feature_evaluations'],
        prepared_goal_configurations=new_sampler.stats['prepared_goal_configurations'],
        numeric_memo_entries=len(new_sampler.configurations),
        peak_live_goal_views=new_sampler.stats['peak_live_goal_views'],
        base_kernel_deepcopies=new_sampler.stats['base_kernel_deepcopies'],
        maximum_probability_difference=probability_error,
        maximum_gradient_difference=gradient_error,
        maximum_consistency_error=consistency_error, readout_comparisons=comparisons,
        feature_stage_full_forward_calls=0, cached_readout_full_forward_calls=0,
        features_factors_masks_outcomes_identical=True, input_and_rng_unchanged=True)


def run(output):
    began = time.monotonic()
    kernels = cases()
    rows = {name: check_case(*kernels[name], 20261111+100000*index)
            for index, name in enumerate(CASE_NAMES)}
    rows['unprepared_responsive_background'] = check_case(*manual_unprepared(), 21961111)
    calls = sum(row['old_full_forward_calls']+row['new_full_forward_calls'] for row in rows.values())
    assert calls == 512
    result = dict(status='PASS_LIGHT_SAMPLER_EQUIVALENCE', cases=rows,
        declared_synthetic_cases=15, unprepared_cases=1,
        explicit_full_forward_calls=calls, real_record_response_queries=0,
        feature_stage_full_forward_calls=0, cached_readout_full_forward_calls=0,
        new_native_starts=0, new_fits=0, optimizer_calls=0,
        maximum_probability_difference=max(row['maximum_probability_difference'] for row in rows.values()),
        maximum_gradient_difference=max(row['maximum_gradient_difference'] for row in rows.values()),
        elapsed_sec=time.monotonic()-began)
    Path(output).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'explicit_full_forward_calls',
        'maximum_probability_difference', 'maximum_gradient_difference', 'elapsed_sec')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
