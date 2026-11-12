"""Supplied shared beta on all current packets; no coefficient estimation."""
from copy import deepcopy
from itertools import product

import numpy as np

from experiments.tsc_packet_discrete_braking_20261110.motion import MotionResolver
from experiments.tsc_packet_discrete_braking_20261110.conditional_features import _lane_mask
from experiments.tsc_packet_lateral_conditioning_20261110.order import forced_lateral_order
from experiments.tsc_packet_lateral_conditioning_20261110.feasibility import _endpoint_gaps
from .categorical import ordered_tuple_probability


CHOICES = ('left', 'stay', 'right')


def canonical_features(kernel):
    """Read canonical current fields under the installed goal configuration.

    The complete current packet set avoids stale conditional feature indices.
    Context's physical landing features use its declared zero-residual base.
    """
    packets = kernel.state['packets']
    return {key: np.asarray(kernel.model.features(kernel._context(packet, packets,
                *kernel._connections[key])), dtype=float) for key, packet in packets.items()}


def prepare_plan(kernel, focal_id, participants):
    context, packets = kernel._mean_alignment_context, kernel.state['packets']
    resolver = MotionResolver(context, kernel, dict(context['zero_decisions']), context['permit_all'], focal_id=focal_id)
    road = kernel.topology.lanes[packets[focal_id]['lane_id']]['edge_id']
    on_road = [key for key, packet in packets.items() if kernel.topology.lanes[packet['lane_id']]['edge_id'] == road]
    P = {key: resolver.raw(key) for key in on_road}
    local = []
    for key in on_road:
        other = packets[key]
        for actor in participants:
            own = packets[actor]
            margin = max(own['parameters']['length_m']+other['parameters']['min_gap_m'],
                         other['parameters']['length_m']+own['parameters']['min_gap_m'])
            if (other['position_m'] <= own['position_m']+P[actor]+margin and
                    own['position_m'] <= other['position_m']+P[key]+margin):
                local.append(key)
                break
    for key in local:
        packet = packets[key]
        if not (packet['position_m'] > packet['parameters']['length_m'] and
                packet['position_m']+P[key] < kernel.topology.lanes[packet['lane_id']]['length_m']):
            raise ValueError('The registered local actors must remain inside their ordinary road')
    return dict(focal_id=focal_id, participants=participants, local=sorted(local, key=context['packet_order'].__getitem__),
                road=road, P=P, phi=canonical_features(kernel))


def shared_decisions(kernel, phi, beta):
    """The same38 beta scores every actor, including all background dependencies."""
    packets = kernel.state['packets']
    return {key: dict(motion_residual_mps=float(np.clip(np.dot(beta, phi[key]),
        -packet['parameters']['decel_mps2'], packet['parameters']['accel_mps2']))) for key, packet in packets.items()}


def evaluate_shared(kernel, plan, beta):
    context, packets = kernel._mean_alignment_context, kernel.state['packets']
    focal, participants, phi = plan['focal_id'], plan['participants'], plan['phi']
    decisions = shared_decisions(kernel, phi, beta)
    resolver = MotionResolver(context, kernel, decisions, context['permit_all'], focal_id=focal)
    resolver.resolve(focal)
    options = {}
    for key in participants:
        result = resolver.resolve(key)
        # Preserve the preceding readout's causal query order and full side
        # neighbourhood resolution before assembling joint landing states.
        _lane_mask(kernel, packets[key], kernel._intent(packets[key]), result, resolver)
        projected = result['projected_packet']
        options[key] = [('stay', projected['lane_id'])]
        for choice, delta in (('left', 1), ('right', -1)):
            target = kernel.topology.adjacent(projected['lane_id'], delta)
            if kernel._lateral_legal(projected, target) and projected['position_m'] <= kernel.topology.lanes[target]['length_m']:
                options[key].append((choice, target))
    for key in plan['local']:
        result = resolver.resolve(key)
        if result['ended'] or kernel.topology.lanes[result['projected_packet']['lane_id']]['edge_id'] != plan['road']:
            raise ValueError('An audited local actor left its interior road')
    projected = {key: resolver.results[key]['projected_packet'] for key in plan['local']}
    mass, endpoint_mass = np.zeros(3), np.zeros(3)
    grad_left, grad_right = np.zeros((3, 38)), np.zeros((3, 38))
    signature = []
    for choices in product(*(options[key] for key in participants)):
        actions = {key: pair[0] for key, pair in zip(participants, choices)}
        ordered = forced_lateral_order(kernel, projected, actions, context['packet_order'])
        probability = ordered_tuple_probability(ordered, phi, np.zeros(38), np.zeros(38))
        after = deepcopy(projected)
        for key, pair in zip(participants, choices):
            after[key]['lane_id'] = pair[1]
        endpoints = [dict(packet_id=key, lane_id=row['lane_id'], position_m=row['position_m'],
            length_m=row['parameters']['length_m'], min_gap_m=row['parameters']['min_gap_m']) for key, row in after.items()]
        unused, violations = _endpoint_gaps(endpoints)
        index = CHOICES.index(actions[focal])
        mass[index] += probability['q']
        endpoint_mass[index] += probability['q']*(not bool(violations))
        grad_left[index] += probability['dq_left']
        grad_right[index] += probability['dq_right']
        signature.append((tuple(actions.values()), tuple(tuple(step['allowed_choices']) for step in ordered['steps'])))
    if abs(float(mass.sum())-1.) > 1e-10 or max(np.abs(grad_left.sum(axis=0)).max(), np.abs(grad_right.sum(axis=0)).max()) > 1e-8:
        raise ValueError('The conditional LC law or its logit derivative failed normalization')
    motion = {}
    for key in participants:
        response, raw_r = resolver.results[key], decisions[key]['motion_residual_mps']
        expected = max(plan['P'][key]+min(raw_r, 0.), 0.)
        motion[key] = dict(mean_mps=response['travel_m'], physical_proposal_mps=response['physical_proposal_mps'],
            baseline_P_mps=plan['P'][key], residual_mps=raw_r,
            mean_matches_proposal=abs(response['travel_m']-expected) <= 1e-8,
            body_slack_m=response['same_step_path_bound_m']-response['physical_proposal_mps'],
            spatial_slack_m=response['spatial_limit_m']-response['physical_proposal_mps'],
            same_step_path_bound_m=response['same_step_path_bound_m'], spatial_limit_m=response['spatial_limit_m'])
    return dict(class_mass=mass, endpoint_class_mass=endpoint_mass, grad_mass_left=grad_left, grad_mass_right=grad_right,
        lateral_signature=signature, motion=motion, decision_count=len(decisions),
        negative_residual_background_count=sum(decision['motion_residual_mps'] < 0 for key, decision in decisions.items() if key not in participants),
        resolved_background=sum(key not in participants for key in resolver.results))
