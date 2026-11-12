"""Four-random-actor and retained synthetic streaming checks; no native data."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
from math import fsum
from pathlib import Path
import time

from scipy.integrate import quad

from experiments.tsc_packet_conditional_law_20261110.law import enumerate_conditional
from experiments.tsc_packet_conditional_law_20261110.synthetic import cases as larger_cases
from experiments.tsc_packet_joint_law_definition_20261110.law import marginal
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import cases as tiny_cases, make_kernel, vehicle
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import executor_seed, lane, road_graph
from .initialize import prepare_zero_kernel
from .stream import stream_zero_law


def four_actor_case(*, stopped=False, earlier_neighbour=False):
    static = road_graph([lane('a'+str(i), 200, edge='a', index=i) for i in range(5)], [])
    speed = 0 if stopped else 5
    rows = [vehicle('focal', 'a2', 40, speed), vehicle('right', 'a1', 44, speed),
            vehicle('left', 'a3', 45, speed), vehicle('outer', 'a4', 49 if stopped else 52, speed)]
    if earlier_neighbour:
        rows = [rows[1], rows[0], rows[2], rows[3]]
    return make_kernel(static, rows), 'focal'


def cases():
    result = dict(cap4_lateral=four_actor_case(),
                  cap4_lateral_earlier_neighbour=four_actor_case(earlier_neighbour=True),
                  cap4_start_permissions=four_actor_case(stopped=True))
    tiny = tiny_cases()
    for name in ('stopped_open', 'blocked_start', 'yielding_entry', 'blocked_entry',
                 'two_actor_lateral', 'intent_and_survival', 'receiving_intent'):
        result[name] = tiny[name]
    larger = larger_cases()
    for name in ('distant_background', 'binding_background', 'responsive_background'):
        result[name] = larger[name]
    return result


def focal_atoms(law, focal):
    folded = defaultdict(float)
    for atom in law['atoms']:
        folded[tuple(atom['actors'][focal].items())] += atom['probability']
    return dict(folded)


def check_manual_initializer():
    prepared, focal = larger_cases()['responsive_background']
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    packets = deepcopy(prepared.state['packets'])
    frame = dict(time_s=prepared.state['time_s'], packets=packets,
        native_lanes=prepared.sensor_operator._reconstruct(packets), native_pending_ids=[],
        executors={key: dict(snapshot=deepcopy(seed['snapshot'])) for key, seed in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))
    action = dict(requests={}, applied_states=dict(prepared.signals))
    manual = prepare_zero_kernel(prepared.static, frame, seeds, prepared.laws, action, 23,
                                prepared._spatial_geometry)
    attrs = ('_conditional_lane_packets', '_conditional_tls_packets', '_mean_alignment_context')
    if any(hasattr(manual, name) for name in attrs):
        raise ArithmeticError('The manual constructor unexpectedly prepared response features')
    state, connections = deepcopy(manual.state), deepcopy(manual._connections)
    rng = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
           for key, row in manual.noise._packets.items()}
    before = stream_zero_law(prepared, focal)
    after = stream_zero_law(manual, focal)
    old, new = focal_atoms(before, focal), focal_atoms(after, focal)
    if old.keys() != new.keys() or max(abs(old[key]-new[key]) for key in old) > 1e-12:
        raise ArithmeticError('Manual physical initialization changed the focal law')
    if before['background_response_ranges'] != after['background_response_ranges']:
        raise ArithmeticError('Manual initialization changed the responsive background')
    if manual.state != state or manual._connections != connections or rng != {
            key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
            for key, row in manual.noise._packets.items()}:
        raise ArithmeticError('The manual input or its random streams changed')
    return dict(no_prepared_conditional_attributes=True, matches_prepared_focal_atoms=True,
        matches_background_response_ranges=True, input_and_rng_unchanged=True,
        frame_actor_count=after['frame_actor_count'], stochastic_actor_count=len(after['stochastic_participants']))


def check_case(kernel, focal):
    began = time.monotonic()
    state, connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    context = deepcopy(kernel._mean_alignment_context)
    rng = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
           for key, row in kernel.noise._packets.items()}
    complete = enumerate_conditional(kernel, focal, max_stochastic=4)
    streamed = stream_zero_law(kernel, focal)
    old_atoms, new_atoms = focal_atoms(complete, focal), focal_atoms(streamed, focal)
    if old_atoms.keys() != new_atoms.keys():
        raise ArithmeticError('Streaming lost or altered a focal observable atom')
    error = max((abs(old_atoms[key]-new_atoms[key]) for key in old_atoms), default=0.)
    if error > 1e-12:
        raise ArithmeticError('Streaming focal atom weights differ from full materialization')
    if any(set(atom['actors']) != {focal} for atom in streamed['atoms']):
        raise ArithmeticError('Streaming retained a duplicated complete-frame atom')
    if any(streamed['diagnostics'][name] != complete['diagnostics'][name]
           for name in ('physical_branches', 'enumerated_leaf_branches')):
        raise ArithmeticError('Streaming and full enumeration did not visit the same branches')
    if kernel.state != state or kernel._connections != connections or kernel._mean_alignment_context != context:
        raise ArithmeticError('Streaming or materialization changed the frozen input')
    if rng != {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
               for key, row in kernel.noise._packets.items()}:
        raise ArithmeticError('Streaming or materialization advanced input random streams')
    zero = marginal(streamed, focal, 'motion', 0.)['likelihood']
    density = lambda y: marginal(streamed, focal, 'motion', y)['likelihood']
    low, err0 = quad(density, 0., .1, epsabs=1e-10, epsrel=1e-10)
    high, err1 = quad(density, .1, float('inf'), epsabs=1e-10, epsrel=1e-10)
    if abs(zero+low+high-1.) > 1e-9:
        raise ArithmeticError('Streaming survivor speed law did not normalize')
    events, maximum_marginal_error = {}, 0.
    for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2))):
        if streamed['eligibility'][focal][head]:
            values = {str(label): marginal(streamed, focal, head, label)['likelihood'] for label in labels}
            for label in labels:
                maximum_marginal_error = max(maximum_marginal_error,
                    abs(values[str(label)]-marginal(complete, focal, head, label)['likelihood']))
            if abs(fsum(values.values())-1.) > 1e-10:
                raise ArithmeticError('Streaming selected event marginal did not normalize')
            events[head] = values
    if maximum_marginal_error > 1e-12:
        raise ArithmeticError('A streaming head marginal differs from full enumeration')
    if streamed['old_stopped'][focal] and abs(high-events['start']['1']) > 1e-9:
        raise ArithmeticError('Streaming start event disagrees with observed speed band')
    survivor = fsum(atom['probability'] for atom in streamed['atoms'] if atom['actors'][focal]['survived'])
    exits = fsum(atom['probability'] for atom in streamed['atoms'] if not atom['actors'][focal]['survived'])
    return dict(frame_actor_count=streamed['frame_actor_count'],
        stochastic_actor_count=len(streamed['stochastic_participants']),
        stochastic_participants=streamed['stochastic_participants'],
        full_materialized_atoms=len(complete['atoms']), retained_focal_atoms=len(streamed['atoms']),
        maximum_atom_probability_difference=error, maximum_event_probability_difference=maximum_marginal_error,
        total_probability=streamed['total_probability'], survivor_probability=survivor, exit_no_speed_probability=exits,
        survivor_observation_total=zero+low+high, quadrature_error_bound=err0+err1,
        event_probabilities=events, physical_branches=int(streamed['diagnostics']['physical_branches']),
        lateral_leaves=int(streamed['diagnostics']['enumerated_leaf_branches']),
        input_state_context_and_rng_unchanged=True,
        background_response_ranges=streamed.get('background_response_ranges', {}),
        elapsed_sec=time.monotonic()-began)


def run(output):
    rows = {name: check_case(kernel, focal) for name, (kernel, focal) in cases().items()}
    four = [name for name, row in rows.items() if row['stochastic_actor_count'] == 4]
    if len(four) != 3:
        raise ArithmeticError('The registered tests must actually exercise four random actors')
    result = dict(status='PASS_SYNTHETIC_CAP4_STREAMING', cases=rows,
        manual_zero_response_initializer=check_manual_initializer(),
        synthetic_scenarios=len(rows), four_stochastic_scenarios=four,
        copy_on_write_matches_full_materialization=True,
        real_record_response_queries=0, new_native_starts=0, new_fits=0, optimizer_calls=0,
        scope='Fixed zero-coefficient synthetic checks only; no real support, fit, transfer or control result.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], synthetic_scenarios=len(rows), four_stochastic_scenarios=len(four))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
