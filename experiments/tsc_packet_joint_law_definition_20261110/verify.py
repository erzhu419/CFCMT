"""Synthetic normalization evidence only; never reads a real-data reference."""
import argparse
import json
from math import fsum
from pathlib import Path

from scipy.integrate import quad

from .law import enumerate_law, marginal
from .synthetic import cases


def check_case(kernel, focal):
    law = enumerate_law(kernel)
    survival = fsum(atom['probability'] for atom in law['atoms'] if atom['actors'][focal]['survived'])
    exit_mass = fsum(atom['probability'] for atom in law['atoms'] if not atom['actors'][focal]['survived'])
    zero = marginal(law, focal, 'motion', 0.)['likelihood']
    density = lambda value: marginal(law, focal, 'motion', value)['likelihood']
    lower, err0 = quad(density, 0., .1, epsabs=1e-10, epsrel=1e-10)
    upper, err1 = quad(density, .1, float('inf'), epsabs=1e-10, epsrel=1e-10)
    mass = zero+lower+upper
    if abs(mass-1.) > 1e-9:
        raise ArithmeticError('Survivor speed mass+density did not normalize')
    event_totals, events = {}, {}
    for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2))):
        if law['eligibility'][focal][head]:
            values = {str(label): marginal(law, focal, head, label)['likelihood'] for label in labels}
            event_totals[head], events[head] = fsum(values.values()), values
            if abs(event_totals[head]-1.) > 1e-10:
                raise ArithmeticError('Selected event marginal did not normalize')
    if law['old_stopped'][focal] and abs(upper-events['start']['1']) > 1e-9:
        raise ArithmeticError('Start label and observed speed band disagree')
    return dict(total_physical_probability=law['total_probability'], distinct_physical_atoms=len(law['atoms']),
        latent_leaf_branches=int(law['diagnostics']['enumerated_leaf_branches']), survivor_probability=survival,
        exit_no_speed_probability=exit_mass, survivor_observation_zero_mass=zero,
        survivor_observation_positive_mass=lower+upper, survivor_observation_total=mass,
        quadrature_error_bound=err0+err1, event_totals=event_totals, event_probabilities=events,
        start_band_matches_observed_speed=(abs(upper-events['start']['1']) <= 1e-9) if law['old_stopped'][focal] else None,
        diagnostics=law['diagnostics'])


def run(output):
    rows = {name: check_case(kernel, focal) for name, (kernel, focal) in cases().items()}
    result = dict(status='PASS_SYNTHETIC_LOCAL_OBSERVABLE_LAW', cases=rows,
        synthetic_scenarios=len(rows), coefficient_initialization='All new coefficient vectors zero; residual nodes uniform, not old zero-residual baseline.',
        real_record_response_queries=0, new_native_starts=0, new_fits=0, optimizer_calls=0,
        calibration_data_used=False, model_admitted=False, full_reservoir_likelihood_defined=False,
        scope='New tiny complete-cohort physical law plus unvalidated censored/start-band velocity observation kernel. No real-data support or control claim.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], synthetic_scenarios=len(rows))))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
