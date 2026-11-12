"""Exact local randomness with an explicit, physically responsive background.

Outside J, current goals are fixed, residuals are0, both permissions True and
LC stay. This is a conditional intervention law, not whole-world randomness.
Every actor remains in each full forward physical resolution and LC gap view.
"""
from collections import defaultdict
from copy import deepcopy
from itertools import product
from math import fsum

import numpy as np

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_receiving_context
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver, _priority_order
from experiments.tsc_packet_joint_law_definition_20261110.law import (
    PARAMETERS, CHOICES, OutsidePrototype, _prior, _eligibility, marginal,
)
from experiments.tsc_packet_lateral_conditioning_20261110.order import _targets, forced_lateral_order
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES, _sigmoid, masked_lane_probabilities
from .selection import select_stochastic


def _lateral_branches(kernel, projected, participants, order, logits):
    ids = sorted((key for key in participants if key in projected), key=order.__getitem__)
    # Background stay is an intervention, so do not query its goal/targets.
    background_actions = {key: 'stay' for key in projected if key not in participants}
    def walk(index, working, actions, probability):
        if index == len(ids):
            yield working, actions, probability
            return
        key = ids[index]
        packet = working[key]
        targets = ({'stay': packet['lane_id']} if kernel.topology.lanes[packet['lane_id']]['internal']
                   else _targets(kernel, packet, working))
        q = masked_lane_probabilities(logits[key], {choice: choice in targets for choice in CHOICES})
        for choice in targets:
            if q[choice] == 0.:
                continue
            result = forced_lateral_order(kernel, working, {key: choice}, order)
            if not result['all_requested_actions_accepted']:
                raise OutsidePrototype('A positive local LC branch failed its frozen body remap')
            yield from walk(index+1, result['packets'], dict(actions, **{key: choice}), probability*q[choice])
    yield from walk(0, projected, background_actions, 1.)


def _actor_options(kernel, key, scores):
    packet = kernel.state['packets'][key]
    connection, compatible = kernel._connections[key]
    brake = _sigmoid(scores['brake'])
    p_start, p_yield = _sigmoid(scores['start_permission']), _sigmoid(scores['yield_permission'])
    start = [(True, 1.)] if packet['speed_mps'] >= .1 else [(False, 1.-p_start), (True, p_start)]
    yielding = (not kernel.topology.lanes[packet['lane_id']]['internal'] and connection is not None
                and compatible and kernel._signal(connection, compatible) == 'g')
    permit = [(False, 1.-p_yield), (True, p_yield)] if yielding else [(True, 1.)]
    return [(r, a, b, pr*pa*pb) for (r, pr), (a, pa), (b, pb) in
            product(((0., 1.-brake), (-packet['parameters']['decel_mps2'], brake)), start, permit)]


def _actor_outcomes(kernel, old, results, after, actions):
    out = {}
    for key, packet in old.items():
        result = results[key]
        survives = not result['ended']
        trace = result['trace']
        entry = int(any(kernel.topology.lanes[lane]['internal'] for lane in trace[1:]) or
            any(kernel.topology.lanes[a]['edge_id'] != kernel.topology.lanes[b]['edge_id'] for a, b in zip(trace, trace[1:])))
        out[key] = dict(survived=survives, physical_speed_mps=result['travel_m'] if survives else None,
            started=int(result['travel_m'] >= .1) if survives and packet['speed_mps'] < .1 else None,
            entry_event=entry, lane_change=actions[key] if survives else None,
            lane_id=after[key]['lane_id'] if survives else None,
            position_m=after[key]['position_m'] if survives else None)
    return out


def enumerate_conditional(kernel, focal_id, coefficient_vectors=None, *, max_stochastic=3):
    """Integrate J's latent branches while resolving the complete frame afresh.

    Current-state geometric selection is fixed before any outcome. All original
    actors occur in returned atoms, including background whose motion responds
    to J. Only J has stochastic LC; background never gains an implicit intent
    draw on a new receiving road merely to maintain its fixed stay intervention.
    """
    selection = select_stochastic(kernel, focal_id, max_stochastic=max_stochastic)
    if selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE':
        error = OutsidePrototype(selection['status']+' required='+repr(selection['required_participants']))
        error.selection = selection
        raise error
    participants = list(selection['participants'])
    old = kernel.state['packets']
    ids = list(old)
    background = [key for key in ids if key not in participants]
    vectors = ({name: np.zeros(len(FEATURE_NAMES)) for name in PARAMETERS} if coefficient_vectors is None
               else {name: np.asarray(coefficient_vectors[name]) for name in PARAMETERS})
    goals = []
    for key in participants:
        geometry = kernel.topology.lanes[old[key]['lane_id']]
        goals.append([(kernel._connections[key][0]['to_edge'], 1.)] if geometry['internal']
                     else _prior(kernel, geometry['edge_id']))
    background_conditions = {key: dict(current_intent=deepcopy(kernel.state['intents'].get(key)),
        current_connection_key=None if kernel._connections[key][0] is None else kernel._connections[key][0]['key'],
        residual_mps=0., start_permission=True, yield_permission=True, lane_change='stay') for key in background}
    response_values = {key: [] for key in background}
    survivor_values = {key: [] for key in background}
    atoms, counts = {}, defaultdict(float)
    for goal_tuple in product(*goals):
        k = deepcopy(kernel)
        for key, (goal, unused) in zip(participants, goal_tuple):
            geometry = k.topology.lanes[old[key]['lane_id']]
            if not geometry['internal']:
                k.state['intents'][key] = dict(road_id=geometry['edge_id'], choice=goal)
            k._connections[key] = k._connection(k.state['packets'][key])
        context = prepare_receiving_context(k)
        packets = k.state['packets']
        phi = {key: np.asarray(k.model.features(k._context(packets[key], packets, *k._connections[key])), dtype=float)
               for key in participants}
        scores = {key: {name: float(vectors[name] @ phi[key]) for name in PARAMETERS} for key in participants}
        options = [_actor_options(k, key, scores[key]) for key in participants]
        for draw in product(*options):
            weight = float(np.prod([p for unused, p in goal_tuple]))*float(np.prod([item[3] for item in draw]))
            if weight == 0.:
                continue
            decisions = {key: dict(motion_residual_mps=0.) for key in ids}
            permissions = {key: dict(start=True, **{'yield': True}) for key in ids}
            for key, item in zip(participants, draw):
                decisions[key] = dict(motion_residual_mps=item[0])
                permissions[key] = dict(start=item[1], **{'yield': item[2]})
            resolver = MotionResolver(context, k, decisions, permissions)
            ordered = _priority_order(resolver)
            resolver.priority_index = {key: index for index, key in enumerate(ordered)}
            for key in ordered:
                resolver.resolve(key)
            results = resolver.results
            counts['physical_branches'] += 1
            for key in background:
                response_values[key].append(results[key]['travel_m'])
                if not results[key]['ended']:
                    survivor_values[key].append(results[key]['travel_m'])
            projected = {key: result['projected_packet'] for key, result in results.items() if not result['ended']}
            receiving = [key for key in participants if key in projected
                and not k.topology.lanes[projected[key]['lane_id']]['internal']
                and k.topology.lanes[old[key]['lane_id']]['edge_id'] != k.topology.lanes[projected[key]['lane_id']]['edge_id']
                and len(k.topology.road_lanes(k.topology.lanes[projected[key]['lane_id']]['edge_id'])) > 1]
            new_priors = [_prior(k, k.topology.lanes[projected[key]['lane_id']]['edge_id']) for key in receiving]
            for new_goals in product(*new_priors):
                lc_kernel = deepcopy(k)
                for key, (goal, unused) in zip(receiving, new_goals):
                    lc_kernel.state['intents'][key] = dict(road_id=k.topology.lanes[projected[key]['lane_id']]['edge_id'], choice=goal)
                new_weight = float(np.prod([p for unused, p in new_goals]))
                logits = {key: dict(left=scores[key]['lane_left'], stay=0., right=scores[key]['lane_right']) for key in participants}
                for after, actions, q in _lateral_branches(lc_kernel, projected, participants, context['packet_order'], logits):
                    actor_outcomes = _actor_outcomes(k, old, results, after, actions)
                    for key in participants:
                        if permissions[key]['start'] and actor_outcomes[key]['started'] == 0:
                            counts['expected_actor_count_start_permitted_but_not_started'] += weight*new_weight*q
                        if permissions[key]['yield'] and _eligibility(k, old[key])['yield_entry'] and not actor_outcomes[key]['entry_event']:
                            counts['expected_actor_count_yield_permitted_but_not_entered'] += weight*new_weight*q
                    atom_key = tuple((key, tuple(actor_outcomes[key].items())) for key in ids)
                    if atom_key not in atoms:
                        atoms[atom_key] = dict(actors=actor_outcomes, probability=0., latent_branches=0)
                    atoms[atom_key]['probability'] += weight*new_weight*q
                    atoms[atom_key]['latent_branches'] += 1
                    counts['enumerated_leaf_branches'] += 1
    total = fsum(atom['probability'] for atom in atoms.values())
    if abs(total-1.) > 1e-10:
        raise ArithmeticError('The complete conditional pushforward did not normalize')
    ranges = {key: dict(min_travel_m=min(response_values[key]), max_travel_m=max(response_values[key]),
        min_survivor_speed_mps=min(survivor_values[key]) if survivor_values[key] else None,
        max_survivor_speed_mps=max(survivor_values[key]) if survivor_values[key] else None,
        survived_physical_branches=len(survivor_values[key]), physical_branches=len(response_values[key])) for key in background}
    return dict(status='DEFINED_CONDITIONAL_LOCAL_OBSERVABLE_LAW', atoms=list(atoms.values()), total_probability=total,
        focal_id=focal_id, selection=selection, frame_actor_count=len(ids), stochastic_actor_count=len(participants),
        stochastic_participants=participants, background_conditions=background_conditions,
        background_response_ranges=ranges, old_stopped={key: old[key]['speed_mps'] < .1 for key in ids},
        eligibility={key: _eligibility(kernel, old[key]) for key in ids}, diagnostics=dict(counts),
        scope='Local conditional intervention: outside J, current goals/residual0/permissionsTrue/LCstay fixed; all actors physically re-resolved in each branch. Not whole-world randomness or old190 likelihood.')
