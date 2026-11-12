"""Exact finite-node local observable law; a new, untrained model proposal.

All current actors in a tiny complete cohort are stochastic. Physical motion
uses the frozen forward resolver; velocity observation error stays outside it.
An unsupported branch rejects the whole input, never a renormalized subset.
"""
from collections import defaultdict
from copy import deepcopy
from itertools import product
from math import fsum

import numpy as np

from experiments.tsc_packet_discrete_braking_20261110.conditional_features import prepare_receiving_context
from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver, _priority_order
from experiments.tsc_packet_lateral_conditioning_20261110.order import _targets, forced_lateral_order
from experiments.tsc_packet_shared_motion_20261110.api import canonical_features
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES, _sigmoid, masked_lane_probabilities
from .actors import select_component
from .observation import speed_likelihood


PARAMETERS = ('brake', 'start_permission', 'yield_permission', 'lane_left', 'lane_right')
CHOICES = ('left', 'stay', 'right')


class OutsidePrototype(ValueError):
    """The complete input lacks a law under the declared enumeration boundary."""


def _prior(kernel, road):
    if road in kernel.laws['intent_probs']:
        return [(goal, p) for goal, p in kernel.laws['intent_probs'][road].items() if p > 0]
    if any(kernel.topology.connections_from(lane) for lane in kernel.topology.road_lanes(road)):
        raise OutsidePrototype('A current/receiving road lacks its frozen intent prior')
    return [('END', 1.)]


def _eligibility(kernel, packet):
    topology = kernel.topology
    geometry = topology.lanes[packet['lane_id']]
    reach = packet['speed_mps']+packet['parameters']['accel_mps2']
    connections = topology.connections_from(packet['lane_id'])
    entry = (not geometry['internal'] and reach >= geometry['length_m']-packet['position_m'] and
        any(c['tls_id'] is not None and kernel.signals[c['tls_id']][c['link_index']] == 'g' for c in connections))
    lateral = not geometry['internal'] and len(topology.road_lanes(geometry['edge_id'])) > 1
    if not lateral:
        possible = topology.internal_connections[packet['lane_id']] if geometry['internal'] else connections
        lateral = any(reach >= topology.path_distance(c, packet['lane_id'], packet['position_m'], c['to_lane'], 0.)
            and len(topology.road_lanes(c['to_edge'])) > 1 for c in possible)
    return dict(motion=True, start=packet['speed_mps'] < .1, yield_entry=bool(entry), lane_change=bool(lateral))


def _lateral_branches(kernel, projected, order, logits):
    ids = sorted(projected, key=order.__getitem__)
    def walk(index, working, actions, probability):
        if index == len(ids):
            yield working, actions, probability
            return
        key = ids[index]
        packet = working[key]
        targets = {'stay': packet['lane_id']} if kernel.topology.lanes[packet['lane_id']]['internal'] else _targets(kernel, packet, working)
        q = masked_lane_probabilities(logits[key], {choice: choice in targets for choice in CHOICES})
        for choice in targets:
            result = forced_lateral_order(kernel, working, {key: choice}, order)
            if not result['all_requested_actions_accepted']:
                raise OutsidePrototype('A supported lateral branch failed its frozen body remap')
            yield from walk(index+1, result['packets'], dict(actions, **{key: choice}), probability*q[choice])
    yield from walk(0, projected, {}, 1.)


def enumerate_law(kernel, coefficient_vectors=None, *, max_actors=3):
    """Enumerate a complete tiny cohort, including current and receiving goals.

    R_j in {0,-b_j}, P(R_j=-b_j)=sigmoid(beta_brake phi_j). Zero
    coefficients give half hard-braking, not the old zero-residual baseline.
    Start/yield permissions precede physical motion; actual events follow it.
    No new boundary insertions are included in this conditional cohort law.
    """
    ids = list(kernel.state['packets'])
    if len(ids) > max_actors:
        raise OutsidePrototype('The complete stochastic cohort exceeds the exact-enumeration cap')
    vectors = ({name: np.zeros(len(FEATURE_NAMES)) for name in PARAMETERS} if coefficient_vectors is None
               else {name: np.asarray(coefficient_vectors[name]) for name in PARAMETERS})
    old = kernel.state['packets']
    goals = []
    for key in ids:
        packet = old[key]
        geometry = kernel.topology.lanes[packet['lane_id']]
        goals.append([(kernel._connections[key][0]['to_edge'], 1.)] if geometry['internal']
                     else _prior(kernel, geometry['edge_id']))
    atoms, counts = {}, defaultdict(float)
    for goal_tuple in product(*goals):
        k = deepcopy(kernel)
        for key, (goal, unused) in zip(ids, goal_tuple):
            geometry = k.topology.lanes[old[key]['lane_id']]
            if not geometry['internal']:
                k.state['intents'][key] = dict(road_id=geometry['edge_id'], choice=goal)
        k._connections = {key: k._connection(k.state['packets'][key]) for key in ids}
        context = prepare_receiving_context(k)
        phi = canonical_features(k)
        scores = {key: {name: float(vectors[name] @ phi[key]) for name in PARAMETERS} for key in ids}
        actor_options = []
        for key in ids:
            p = old[key]
            connection, compatible = k._connections[key]
            brake = _sigmoid(scores[key]['brake'])
            start = [(True, 1.)] if p['speed_mps'] >= .1 else [(False, 1.-_sigmoid(scores[key]['start_permission'])), (True, _sigmoid(scores[key]['start_permission']))]
            yielding = (not k.topology.lanes[p['lane_id']]['internal'] and connection is not None and compatible
                        and k._signal(connection, compatible) == 'g')
            permit = [(False, 1.-_sigmoid(scores[key]['yield_permission'])), (True, _sigmoid(scores[key]['yield_permission']))] if yielding else [(True, 1.)]
            actor_options.append([(r, a, b, pr*pa*pb) for (r, pr), (a, pa), (b, pb) in
                product(((0., 1.-brake), (-p['parameters']['decel_mps2'], brake)), start, permit)])
        for draw in product(*actor_options):
            weight = float(np.prod([p for unused, p in goal_tuple]))*float(np.prod([item[3] for item in draw]))
            decisions = {key: dict(motion_residual_mps=item[0]) for key, item in zip(ids, draw)}
            permissions = {key: dict(start=item[1], **{'yield': item[2]}) for key, item in zip(ids, draw)}
            resolver = MotionResolver(context, k, decisions, permissions)
            ordered = _priority_order(resolver)
            resolver.priority_index = {key: index for index, key in enumerate(ordered)}
            for key in ordered:
                resolver.resolve(key)
            results = resolver.results
            projected = {key: r['projected_packet'] for key, r in results.items() if not r['ended']}
            ended = [key for key, r in results.items() if r['ended']]
            receiving = set()
            for key in ids:
                component = select_component(k.topology, old, projected, context['packet_order'], key,
                    ended_ids=ended, max_actors=max_actors)
                if component['status'].startswith('OBSTRUCTION'):
                    raise OutsidePrototype(component['status'])
                receiving.update(component['receiving_intent_ids'])
            receiving = sorted(receiving, key=context['packet_order'].__getitem__)
            new_priors = [_prior(k, k.topology.lanes[projected[key]['lane_id']]['edge_id']) for key in receiving]
            for new_goals in product(*new_priors):
                lc_kernel = deepcopy(k)
                for key, (goal, unused) in zip(receiving, new_goals):
                    lc_kernel.state['intents'][key] = dict(road_id=k.topology.lanes[projected[key]['lane_id']]['edge_id'], choice=goal)
                new_weight = float(np.prod([p for unused, p in new_goals]))
                logits = {key: dict(left=scores[key]['lane_left'], stay=0., right=scores[key]['lane_right']) for key in projected}
                for after, actions, q in _lateral_branches(lc_kernel, projected, context['packet_order'], logits):
                    actor_outcomes = {}
                    for key in ids:
                        r, p = results[key], old[key]
                        survives = not r['ended']
                        trace = r['trace']
                        entry = int(any(k.topology.lanes[lane]['internal'] for lane in trace[1:]) or
                            any(k.topology.lanes[a]['edge_id'] != k.topology.lanes[b]['edge_id'] for a, b in zip(trace, trace[1:])))
                        actor_outcomes[key] = dict(survived=survives, physical_speed_mps=r['travel_m'] if survives else None,
                            started=int(r['travel_m'] >= .1) if survives and p['speed_mps'] < .1 else None,
                            entry_event=entry, lane_change=actions[key] if survives else None,
                            lane_id=after[key]['lane_id'] if survives else None,
                            position_m=after[key]['position_m'] if survives else None)
                        if permissions[key]['start'] and survives and p['speed_mps'] < .1 and r['travel_m'] < .1:
                            counts['expected_actor_count_start_permitted_but_not_started'] += weight*new_weight*q
                        if permissions[key]['yield'] and _eligibility(k, p)['yield_entry'] and not entry:
                            counts['expected_actor_count_yield_permitted_but_not_entered'] += weight*new_weight*q
                    atom_key = tuple((key, tuple(actor_outcomes[key].items())) for key in ids)
                    if atom_key not in atoms:
                        atoms[atom_key] = dict(actors=actor_outcomes, probability=0., latent_branches=0)
                    atoms[atom_key]['probability'] += weight*new_weight*q
                    atoms[atom_key]['latent_branches'] += 1
                    counts['enumerated_leaf_branches'] += 1
    total = fsum(row['probability'] for row in atoms.values())
    if abs(total-1.) > 1e-10:
        raise ArithmeticError('The complete pushforward probability did not normalize')
    return dict(status='DEFINED_TINY_LOCAL_OBSERVABLE_LAW', atoms=list(atoms.values()), total_probability=total,
        old_stopped={key: old[key]['speed_mps'] < .1 for key in ids},
        eligibility={key: _eligibility(kernel, old[key]) for key in ids}, diagnostics=dict(counts),
        scope='New finite residual/permission model plus a separate censored observation law; complete cohort<=3 only, not old190 or a full-network objective.')


def marginal(law, packet_id, head, observed):
    """Marginal on each original head's selection, under its proper measure."""
    if not law['eligibility'][packet_id][head]:
        raise ValueError('The current-state reference is outside this head selector')
    survivor = fsum(atom['probability'] for atom in law['atoms'] if atom['actors'][packet_id]['survived'])
    conditioned = head in ('motion', 'start', 'lane_change')
    denominator = survivor if conditioned else 1.
    if denominator == 0.:
        return dict(defined=False, likelihood=None, survivor_probability=survivor)
    terms = []
    for atom in law['atoms']:
        actor = atom['actors'][packet_id]
        if conditioned and not actor['survived']:
            continue
        if head == 'motion':
            value = speed_likelihood(observed, actor['physical_speed_mps'], law['old_stopped'][packet_id])
        else:
            field = {'start': 'started', 'yield_entry': 'entry_event', 'lane_change': 'lane_change'}[head]
            label = CHOICES[int(observed)] if head == 'lane_change' else observed
            value = float(actor[field] == label)
        terms.append(atom['probability']*value)
    return dict(defined=True, likelihood=fsum(terms)/denominator, survivor_probability=survivor,
        reference_measure='delta0_plus_positive_Lebesgue' if head == 'motion' else 'counting',
        survivor_conditioned=conditioned)
