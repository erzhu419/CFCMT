"""The frozen synthetic cohort and an independent ten-actor motion oracle."""
from copy import deepcopy
from itertools import product
from math import log

import numpy as np
from scipy.special import expit

from experiments.tsc_packet_conditional_panel_20261110.synthetic import cases as old_cases
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_receiving_context
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver, _priority_order
from experiments.tsc_packet_joint_law_definition_20261110.observation import speed_likelihood
from experiments.tsc_packet_joint_law_definition_20261110.synthetic import make_kernel, vehicle
from experiments.tsc_packet_shared_motion_20261110.api import canonical_features
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import lane, road_graph


CASE_NAMES = (
    'cap4_lateral', 'cap4_lateral_earlier_neighbour', 'cap4_start_permissions',
    'stopped_open', 'blocked_start', 'yielding_entry', 'blocked_entry',
    'two_actor_lateral', 'intent_and_survival', 'receiving_intent',
    'distant_background', 'binding_background', 'responsive_background',
    'cap5_focal_last', 'cap10_motion_oracle',
)


def pressure_case(actors):
    graph = road_graph([lane('a'+str(i), 200, edge='a', index=i) for i in range(5)], [])
    if actors == 5:
        rows = [vehicle('right', 'a1', 44, 5), vehicle('left', 'a3', 45, 5),
                vehicle('outer', 'a4', 52, 5), vehicle('farleft', 'a0', 46, 5)]
    elif actors == 10:
        rows = [vehicle('side'+str(index), lane_id, position, 5)
                for index, (lane_id, position) in enumerate((
                    ('a0', 30), ('a0', 38), ('a0', 46), ('a1', 32), ('a1', 40),
                    ('a3', 30), ('a3', 38), ('a4', 32), ('a4', 40)))]
    else:
        raise ValueError('Only the two frozen pressure cases are defined')
    rows.append(vehicle('focal', 'a2', 40, 5))
    kernel = make_kernel(graph, rows)
    selection = select_stochastic(kernel, 'focal', max_stochastic=actors)
    if selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE' or len(selection['participants']) != actors:
        raise ArithmeticError('The predeclared pressure geometry did not select all actors')
    return kernel, 'focal'


def cases():
    result = old_cases()
    result['cap5_focal_last'] = pressure_case(5)
    result['cap10_motion_oracle'] = pressure_case(10)
    if tuple(result) != CASE_NAMES:
        raise ArithmeticError('The synthetic cohort order differs from the fixed protocol')
    return result


class MotionOracle:
    """Only focal motion: all ten longitudinal bits checked, no LC oracle."""
    def __init__(self, kernel, focal):
        k = deepcopy(kernel)
        context = prepare_receiving_context(k)
        packets, ids = k.state['packets'], list(k.state['packets'])
        self.phi = canonical_features(k)[focal]
        speeds = {0: [], 1: []}
        calls = 0
        for bits in product((0, 1), repeat=len(ids)):
            decisions = {key: dict(motion_residual_mps=-packets[key]['parameters']['decel_mps2']*bit)
                         for key, bit in zip(ids, bits)}
            permissions = {key: dict(start=True, **{'yield': True}) for key in ids}
            resolver = MotionResolver(context, k, decisions, permissions)
            order = _priority_order(resolver)
            resolver.priority_index = {key: index for index, key in enumerate(order)}
            calls += 1
            for key in order:
                resolver.resolve(key)
            outcome = resolver.results[focal]
            if outcome['ended']:
                raise ArithmeticError('The interior motion oracle unexpectedly ended')
            speeds[bits[ids.index(focal)]].append(outcome['travel_m'])
        if calls != 1024 or any(max(values)-min(values) > 1e-12 for values in speeds.values()):
            raise ArithmeticError('Focal longitudinal motion depends on side-actor residuals')
        self.speeds = [speeds[index][0] for index in (0, 1)]
        self.stats = dict(physical_calls=calls, longitudinal_draws=calls,
                          focal_speeds_mps=self.speeds, lateral_oracle_available=False,
                          nuisance_residual_independence_max_error=max(
                              max(values)-min(values) for values in speeds.values()))

    def marginal(self, theta, head, observed):
        if head != 'motion':
            raise ValueError('The ten-actor oracle covers motion only')
        p = float(expit(np.asarray(theta)[0] @ self.phi))
        a, b = [speed_likelihood(observed, speed, False) for speed in self.speeds]
        likelihood = (1-p)*a+p*b
        gradient = np.zeros((5, len(FEATURE_NAMES)))
        gradient[0] = -p*(1-p)*(b-a)/likelihood*self.phi
        return dict(defined=True, supported=True, likelihood=likelihood,
                    log_likelihood=log(likelihood), nll=-log(likelihood), gradient=gradient,
                    survivor_probability=1.)
