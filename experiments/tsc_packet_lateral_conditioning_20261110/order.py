"""Forced-action check of the frozen LC order, without sampling or model edits."""
from copy import deepcopy


def _targets(kernel, packet, working):
    targets = {'stay': packet['lane_id']}
    for choice, delta in (('left', 1), ('right', -1)):
        target = kernel.topology.adjacent(packet['lane_id'], delta)
        if not kernel._lateral_legal(packet, target):
            continue
        if packet['position_m'] > kernel.topology.lanes[target]['length_m']:
            continue
        front, rear, unused = kernel._side_gaps(packet, target, working)
        ahead = [other for key, other in working.items() if key != packet['packet_id'] and
                 other['lane_id'] == target and other['position_m'] >= packet['position_m']]
        behind = [other for key, other in working.items() if key != packet['packet_id'] and
                  other['lane_id'] == target and other['position_m'] < packet['position_m']]
        needed = max(behind, key=lambda row: row['position_m'])['parameters']['min_gap_m'] if behind else 0.
        if not ((ahead and front < packet['parameters']['min_gap_m']) or (behind and rear < needed)):
            targets[choice] = target
    return targets


def forced_lateral_order(kernel, packets, actions, order):
    """Check one desired tuple using the production goal/gap/body rules.

    This is conditional on supplied longitudinal endpoints. Unselected actors
    stay. It establishes an execution-order witness, not a swept lateral path
    or a probability for external motion-residual interventions.
    """
    working, steps, probability = deepcopy(packets), [], 1.
    for pid in sorted(actions, key=lambda key: order[key]):
        choice = actions[pid]
        packet = working[pid]
        targets = _targets(kernel, packet, working)
        reason = None if choice in targets else 'not_allowed_at_actor_execution'
        factor = 1./len(targets) if reason is None else 0.
        probability *= factor
        if reason is None and choice != 'stay':
            target = targets[choice]
            unresolved = []
            trail, complete = kernel.sensor_operator._lane_change_trail(packet['body_trail'], target,
                packet['parameters']['length_m'], packet['position_m'], pid, unresolved)
            if not complete or unresolved:
                raise ValueError('A forced lateral body remap is unresolved')
            packet['lane_id'] = target
            kernel._trim_body(packet, trail)
        steps.append(dict(packet_id=pid, action=choice, accepted=reason is None, reason=reason,
                          allowed_choices=list(targets), zero_logit_probability=factor))
    return dict(all_requested_actions_accepted=all(row['accepted'] for row in steps),
                steps=steps, packets=working, zero_logit_tuple_mass=probability)
