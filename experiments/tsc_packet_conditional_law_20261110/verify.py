"""Larger synthetic frames only; preserve all background and branch mass."""
import argparse
from copy import deepcopy
import json
from math import fsum
from pathlib import Path
import time

from scipy.integrate import quad

from experiments.tsc_packet_joint_law_definition_20261110.law import OutsidePrototype, marginal
from .law import enumerate_conditional
from .synthetic import cases


def check_case(kernel, focal):
    began = time.monotonic()
    state, connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    rng = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate()) for key, row in kernel.noise._packets.items()}
    try:
        law = enumerate_conditional(kernel, focal)
    except OutsidePrototype as error:
        if not hasattr(error, 'selection') or error.selection['status'] != 'OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP':
            raise
        return dict(status='EXPECTED_WHOLE_INPUT_OBSTRUCTION', selection=error.selection,
            frame_actor_count=len(kernel.state['packets']), stochastic_actor_count=None,
            total_probability=None, physical_branches=0, elapsed_sec=time.monotonic()-began)
    if kernel.state != state or kernel._connections != connections or rng != {
            key: (row['intent_rng'].getstate(), row['decision_rng'].getstate()) for key, row in kernel.noise._packets.items()}:
        raise ArithmeticError('A conditional branch mutated the input or its random streams')
    current_ids = set(kernel.state['packets'])
    if any(set(atom['actors']) != current_ids for atom in law['atoms']):
        raise ArithmeticError('An original background actor disappeared from the physical outcome')
    background = set(law['background_conditions'])
    if any(actor['survived'] and actor['lane_change'] != 'stay' for atom in law['atoms']
            for key, actor in atom['actors'].items() if key in background):
        raise ArithmeticError('A conditioned background actor acquired a random LC action')
    survivor = fsum(atom['probability'] for atom in law['atoms'] if atom['actors'][focal]['survived'])
    exits = fsum(atom['probability'] for atom in law['atoms'] if not atom['actors'][focal]['survived'])
    zero = marginal(law, focal, 'motion', 0.)['likelihood']
    density = lambda y: marginal(law, focal, 'motion', y)['likelihood']
    low, error0 = quad(density, 0., .1, epsabs=1e-10, epsrel=1e-10)
    high, error1 = quad(density, .1, float('inf'), epsabs=1e-10, epsrel=1e-10)
    if abs(zero+low+high-1.) > 1e-9:
        raise ArithmeticError('Conditional survivor observation law did not normalize')
    events, totals = {}, {}
    for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2))):
        if law['eligibility'][focal][head]:
            events[head] = {str(y): marginal(law, focal, head, y)['likelihood'] for y in labels}
            totals[head] = fsum(events[head].values())
            if abs(totals[head]-1.) > 1e-10:
                raise ArithmeticError('An event marginal lost conditional branch mass')
    if law['old_stopped'][focal] and abs(high-events['start']['1']) > 1e-9:
        raise ArithmeticError('Start label is inconsistent with the same observed speed')
    return dict(status='PASS_CONDITIONAL_LAW', frame_actor_count=law['frame_actor_count'],
        stochastic_actor_count=law['stochastic_actor_count'], stochastic_participants=law['stochastic_participants'],
        background_actor_count=len(background), all_input_actors_retained=True, input_and_rng_unchanged=True,
        total_probability=law['total_probability'], physical_branches=int(law['diagnostics']['physical_branches']),
        latent_leaf_branches=int(law['diagnostics']['enumerated_leaf_branches']), distinct_atoms=len(law['atoms']),
        survivor_probability=survivor, exit_no_speed_probability=exits,
        survivor_observation_total=zero+low+high, quadrature_error_bound=error0+error1,
        event_probabilities=events, event_totals=totals,
        background_conditions=law['background_conditions'], background_response_ranges=law['background_response_ranges'],
        elapsed_sec=time.monotonic()-began)


def run(output):
    rows = {name: check_case(kernel, focal) for name, (kernel, focal) in cases().items()}
    if rows['overflow']['status'] != 'EXPECTED_WHOLE_INPUT_OBSTRUCTION' or any(
            row['status'] != 'PASS_CONDITIONAL_LAW' for name, row in rows.items() if name != 'overflow'):
        raise ArithmeticError('A declared synthetic input did not follow its registered domain outcome')
    result = dict(status='PASS_SYNTHETIC_CONDITIONAL_BACKGROUND_LAW', cases=rows,
        synthetic_scenarios=len(rows), defined_scenarios=len(rows)-1, expected_domain_obstructions=1,
        scope='Focal-specific conditional working law. BG current goals/R0/permissionsTrue/LCstay fixed; all BG physics re-resolved perbranch. Not a global random policy or full reservoir likelihood.',
        coefficient_initialization='All new local coefficients zero; uniform residual nodes, not old zero-residual baseline.',
        real_record_response_queries=0, new_native_starts=0, new_fits=0, optimizer_calls=0,
        calibration_data_used=False, model_admitted=False, full_reservoir_likelihood_defined=False)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'synthetic_scenarios', 'defined_scenarios', 'expected_domain_obstructions')}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args().output)
