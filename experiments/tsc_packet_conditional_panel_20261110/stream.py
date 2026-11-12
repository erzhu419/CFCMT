"""Exact focal marginals for one fixed zero-coefficient conditional law.

The complete physical frame is resolved per branch. Only focal observable
atoms persist; LC uses copy-on-write actor packets with immutable background.
There is no fitting, node pruning or physical sampling approximation here.
"""
from copy import copy, deepcopy
from itertools import product
from math import fsum, prod
import time

from scipy.special import ndtr

from experiments.tsc_packet_conditional_law_20261110.law import _actor_options
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver, _priority_order, build_context
from experiments.tsc_packet_joint_law_definition_20261110.law import (
    PARAMETERS, CHOICES, OutsidePrototype, _prior, _eligibility, marginal,
)
from experiments.tsc_packet_joint_law_definition_20261110.observation import (
    _parameters, speed_likelihood,
)
from experiments.tsc_packet_lateral_conditioning_20261110.order import _targets


def _local_lateral_leaves(kernel, projected, participants, order, visit_leaf):
    """Enumerate zero-logit LC, retaining the complete gap-testing frame."""
    ids = sorted((key for key in participants if key in projected), key=order.__getitem__)
    def walk(index, working, actions, probability):
        if index == len(ids):
            visit_leaf()
            yield working, actions, probability
            return
        key = ids[index]
        packet = working[key]
        targets = ({'stay': packet['lane_id']} if kernel.topology.lanes[packet['lane_id']]['internal']
                   else _targets(kernel, packet, working))
        factor = 1./len(targets)
        for choice, target in targets.items():
            after = working
            if choice != 'stay':
                after = dict(working)
                moved = deepcopy(packet)
                unresolved = []
                trail, complete = kernel.sensor_operator._lane_change_trail(
                    packet['body_trail'], target, packet['parameters']['length_m'],
                    packet['position_m'], key, unresolved)
                if not complete or unresolved:
                    error = OutsidePrototype('A positive streaming LC branch has unresolved body remap')
                    error.lateral_action_context = dict(packet_id=key, choice=choice,
                        target_lane_id=target, prefix_actions=actions, unresolved=unresolved)
                    raise error
                moved['lane_id'] = target
                kernel._trim_body(moved, trail)
                after[key] = moved
            yield from walk(index+1, after, dict(actions, **{key: choice}), probability*factor)
    yield from walk(0, projected, {}, 1.)


def _focal_outcome(kernel, packet, result, after, actions, focal_id):
    survives = not result['ended']
    trace = result['trace']
    entered = int(any(kernel.topology.lanes[lane]['internal'] for lane in trace[1:]) or
        any(kernel.topology.lanes[a]['edge_id'] != kernel.topology.lanes[b]['edge_id']
            for a, b in zip(trace, trace[1:])))
    return dict(survived=survives, physical_speed_mps=result['travel_m'] if survives else None,
        started=int(result['travel_m'] >= .1) if survives and packet['speed_mps'] < .1 else None,
        entry_event=entered, lane_change=actions[focal_id] if survives else None,
        lane_id=after[focal_id]['lane_id'] if survives else None,
        position_m=after[focal_id]['position_m'] if survives else None)


def _observation_masses(mu, stopped):
    atom, lower, upper, normalizer = _parameters(mu, stopped)
    positive = float(ndtr(upper-mu)-ndtr(lower-mu))/normalizer
    return atom, positive


def stream_zero_law(kernel, focal_id, *, observed_speed=None, physical_limit=10000,
                    leaf_limit=1000000, progress=None):
    """Evaluate one zero-coefficient setting, with hard counted branch limits.

    Limits count full forward calls and LC leaves, not cheap current indexing.
    A failure carries consumed counters and makes the whole case undefined.
    Observation noise is integrated afterward and never changes physical LC.
    """
    began = time.monotonic()
    physical_calls = leaf_visits = 0
    selection = select_stochastic(kernel, focal_id, max_stochastic=4)
    branch_context = dict(phase='selection')
    def visit_leaf():
        nonlocal leaf_visits
        if leaf_visits >= leaf_limit:
            raise OutsidePrototype('Registered LC leaf limit reached')
        leaf_visits += 1
    try:
        if selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE':
            raise OutsidePrototype(selection['status']+' required='+repr(selection['required_participants']))
        participants = list(selection['participants'])
        old = kernel.state['packets']
        ids = list(old)
        background = [key for key in ids if key not in participants]
        # Clone the input once; subsequent branches share immutable static data.
        base = deepcopy(kernel)
        base_intents = dict(base.state['intents'])
        base_connections = dict(base._connections)
        goals = []
        for key in participants:
            geometry = base.topology.lanes[old[key]['lane_id']]
            goals.append([(base_connections[key][0]['to_edge'], 1.)] if geometry['internal']
                         else _prior(base, geometry['edge_id']))
        zero_scores = dict.fromkeys(PARAMETERS, 0.)
        atoms = {}
        # Only min/max are retained, never a per-background/per-leaf trace list.
        bg_ranges = {key: dict(min_travel_m=None, max_travel_m=None,
            min_survivor_speed_mps=None, max_survivor_speed_mps=None,
            survived_physical_branches=0, physical_branches=0) for key in background}
        for goal_index, goal_tuple in enumerate(product(*goals)):
            k = copy(base)
            k.state = dict(base.state, intents=dict(base_intents))
            k._connections = dict(base_connections)
            for key, (goal, unused) in zip(participants, goal_tuple):
                geometry = k.topology.lanes[old[key]['lane_id']]
                if not geometry['internal']:
                    k.state['intents'][key] = dict(road_id=geometry['edge_id'], choice=goal)
                k._connections[key] = k._connection(k.state['packets'][key])
            # No phi/old lane-mask query is needed at exactly zero coefficients.
            # Full-forward decision dictionaries do not use zero-cache identity.
            context = build_context(k)
            options = [_actor_options(k, key, zero_scores) for key in participants]
            for draw in product(*options):
                weight = prod(p for unused, p in goal_tuple)*prod(item[3] for item in draw)
                if weight == 0.:
                    continue
                branch_context = dict(phase='physical', goal_index=goal_index,
                    current_goals={key: item[0] for key, item in zip(participants, goal_tuple)},
                    draws={key: dict(residual_mps=item[0], start_permission=item[1], yield_permission=item[2])
                           for key, item in zip(participants, draw)})
                decisions = {key: dict(motion_residual_mps=0.) for key in ids}
                permissions = {key: dict(start=True, **{'yield': True}) for key in ids}
                for key, item in zip(participants, draw):
                    decisions[key] = dict(motion_residual_mps=item[0])
                    permissions[key] = dict(start=item[1], **{'yield': item[2]})
                if physical_calls >= physical_limit:
                    raise OutsidePrototype('Registered full-forward evaluation limit reached')
                physical_calls += 1  # _priority_order already performs forecasts.
                resolver = MotionResolver(context, k, decisions, permissions)
                ordered = _priority_order(resolver)
                resolver.priority_index = {key: index for index, key in enumerate(ordered)}
                for key in ordered:
                    resolver.resolve(key)
                results = resolver.results
                for key in background:
                    row, response = bg_ranges[key], results[key]
                    travel = response['travel_m']
                    row['physical_branches'] += 1
                    row['min_travel_m'] = travel if row['min_travel_m'] is None else min(row['min_travel_m'], travel)
                    row['max_travel_m'] = travel if row['max_travel_m'] is None else max(row['max_travel_m'], travel)
                    if not response['ended']:
                        row['survived_physical_branches'] += 1
                        row['min_survivor_speed_mps'] = travel if row['min_survivor_speed_mps'] is None else min(row['min_survivor_speed_mps'], travel)
                        row['max_survivor_speed_mps'] = travel if row['max_survivor_speed_mps'] is None else max(row['max_survivor_speed_mps'], travel)
                projected = {key: result['projected_packet'] for key, result in results.items() if not result['ended']}
                receiving = [key for key in participants if key in projected
                    and not k.topology.lanes[projected[key]['lane_id']]['internal']
                    and k.topology.lanes[old[key]['lane_id']]['edge_id'] != k.topology.lanes[projected[key]['lane_id']]['edge_id']
                    and len(k.topology.road_lanes(k.topology.lanes[projected[key]['lane_id']]['edge_id'])) > 1]
                new_priors = [_prior(k, k.topology.lanes[projected[key]['lane_id']]['edge_id']) for key in receiving]
                for new_goals in product(*new_priors):
                    lc_kernel = copy(k)
                    lc_kernel.state = dict(k.state, intents=dict(k.state['intents']))
                    for key, (goal, unused) in zip(receiving, new_goals):
                        lc_kernel.state['intents'][key] = dict(road_id=k.topology.lanes[projected[key]['lane_id']]['edge_id'], choice=goal)
                    new_weight = prod(p for unused, p in new_goals)
                    branch_context['phase'] = 'lateral'
                    branch_context['receiving_goals'] = {key: item[0] for key, item in zip(receiving, new_goals)}
                    for after, actions, q in _local_lateral_leaves(lc_kernel, projected, participants, context['packet_order'], visit_leaf):
                        actor = _focal_outcome(k, old[focal_id], results[focal_id], after, actions, focal_id)
                        atom_key = tuple(actor.items())
                        if atom_key not in atoms:
                            atoms[atom_key] = dict(actors={focal_id: actor}, probability=0., latent_branches=0)
                        atoms[atom_key]['probability'] += weight*new_weight*q
                        atoms[atom_key]['latent_branches'] += 1
                if progress is not None and physical_calls % 128 == 0:
                    progress(dict(physical_calls=physical_calls, leaf_visits=leaf_visits,
                        focal_atoms=len(atoms), elapsed_sec=time.monotonic()-began))
        atom_rows = list(atoms.values())
        total = fsum(atom['probability'] for atom in atom_rows)
        if abs(total-1.) > 1e-10:
            raise ArithmeticError('Streaming conditional mass did not normalize')
        old_stopped = old[focal_id]['speed_mps'] < .1
        law = dict(status='DEFINED_STREAMING_ZERO_CONDITIONAL_LAW', atoms=atom_rows,
            total_probability=total, focal_id=focal_id, selection=selection,
            frame_actor_count=len(ids), stochastic_actor_count=len(participants),
            stochastic_participants=participants, background_actor_count=len(background),
            background_response_ranges=bg_ranges,
            old_stopped={focal_id: old_stopped}, eligibility={focal_id: _eligibility(kernel, old[focal_id])},
            diagnostics=dict(physical_branches=physical_calls, enumerated_leaf_branches=leaf_visits),
            physical_calls=physical_calls, leaf_visits=leaf_visits,
            fixed_coefficients='All five new local vectors zero; no feature scores or parameters fitted.',
            full_background_retained_during_physics=True, full_background_atoms_materialized=False,
            scope='Focal-specific conditional intervention law, fixed-zero candidate; not old190/full-network policy.')
        survives = [atom for atom in atom_rows if atom['actors'][focal_id]['survived']]
        survivor = fsum(atom['probability'] for atom in survives)
        end = fsum(atom['probability'] for atom in atom_rows if not atom['actors'][focal_id]['survived'])
        raw_lc = [fsum(atom['probability'] for atom in survives if atom['actors'][focal_id]['lane_change'] == choice) for choice in CHOICES]
        zeros, positive, kernel_errors = [], [], []
        for atom in survives:
            actor, weight = atom['actors'][focal_id], atom['probability']
            zero, cont = _observation_masses(actor['physical_speed_mps'], old_stopped)
            zeros.append(weight*zero)
            positive.append(weight*cont)
            kernel_errors.append(abs(zero+cont-1.))
        law.update(survivor_probability=survivor, end_probability=end,
            lc_class_mass=raw_lc, conditional_lc_probabilities=[value/survivor for value in raw_lc] if survivor else None,
            survivor_observation_zero_mass=fsum(zeros)/survivor if survivor else None,
            survivor_observation_positive_mass=fsum(positive)/survivor if survivor else None,
            survivor_observation_total=(fsum(zeros)+fsum(positive))/survivor if survivor else None,
            maximum_observation_kernel_normalization_error=max(kernel_errors, default=0.))
        law['event_marginals'] = {head: {str(y): marginal(law, focal_id, head, y)['likelihood'] for y in labels}
            for head, labels in (('start', (0, 1)), ('yield_entry', (0, 1)), ('lane_change', (0, 1, 2)))
            if law['eligibility'][focal_id][head]}
        law['observed_speed_likelihood'] = (marginal(law, focal_id, 'motion', observed_speed)
                                            if observed_speed is not None else None)
        if observed_speed is not None and survivor:
            law['conditional_class_speed_likelihoods'] = [fsum(atom['probability']*
                speed_likelihood(observed_speed, atom['actors'][focal_id]['physical_speed_mps'], old_stopped)
                for atom in survives if atom['actors'][focal_id]['lane_change'] == choice)/survivor for choice in CHOICES]
        else:
            law['conditional_class_speed_likelihoods'] = None
        law['elapsed_sec'] = time.monotonic()-began
        if progress is not None:
            progress(dict(physical_calls=physical_calls, leaf_visits=leaf_visits,
                focal_atoms=len(atoms), elapsed_sec=law['elapsed_sec'], complete=True))
        return law
    except Exception as error:
        failure = error if isinstance(error, OutsidePrototype) else OutsidePrototype(f'{type(error).__name__}: {error}')
        failure.physical_calls, failure.leaf_visits = physical_calls, leaf_visits
        failure.selection = selection
        failure.branch_context = branch_context
        failure.elapsed_sec = time.monotonic()-began
        if failure is error:
            raise
        raise failure from error
