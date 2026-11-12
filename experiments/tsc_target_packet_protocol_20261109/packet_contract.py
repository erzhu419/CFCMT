"""Causal initialization and persistent latent inventories for the next model.

These are packet-model primitives, not a complete traffic transition kernel.
They have no native simulator dependency and do not fit any real data.
"""
from copy import deepcopy
from collections import Counter
from math import sqrt


PACKET_FIELDS = (
    'packet_id', 'lane_id', 'position_m', 'speed_mps', 'acceleration_mps2',
    'lateral_position_m', 'lateral_speed_mps', 'allowed_speed_mps',
    'is_stopped', 'type_id', 'parameters', 'body_trail', 'body_path_complete',
)
PARAMETER_FIELDS = (
    'length_m', 'min_gap_m', 'accel_mps2', 'decel_mps2', 'tau_s',
    'imperfection', 'max_speed_mps', 'speed_factor',
)
EXECUTOR_FIELDS = (
    'mode', 'current_state', 'current_green_state', 'target_green_state',
    'green_elapsed_sec', 'remaining_sec', 'pending_all_red_sec',
)


def initialize_observation(frame, anchor_sec):
    """Copy only observed state at the anchor, never a physical audit's route.

    Pending entry allocation is unknown. The later causal boundary filter
    assigns persistent latent entries; a global empty or nonempty queue is not
    replaced by zero entry queues. Executor static timings come separately
    from the fixed graph/configuration.
    """
    if frame['time_s'] != anchor_sec:
        raise ValueError('Initialize from the anchor observation only')
    if frame['unresolved'] or not all(frame['checks'].values()):
        raise ValueError('The initial observation is unresolved')
    packets = {}
    for key, row in frame['packets'].items():
        packet = {name: deepcopy(row[name]) for name in PACKET_FIELDS if name != 'parameters'}
        packet['parameters'] = {name: row['parameters'][name] for name in PARAMETER_FIELDS}
        if packet['packet_id'] != key or not packet['body_path_complete']:
            raise ValueError('Expected identified packets with resolved body trails')
        packets[key] = packet
    pending = list(frame['native_pending_ids'])
    if set(packets) & set(pending) or len(set(pending)) != len(pending):
        raise ValueError('Active and pending identity inventories must be disjoint')
    return dict(
        time_s=anchor_sec, packets=packets,
        lane_sensors=deepcopy(frame['native_lanes']),
        executors={tls: {key: deepcopy(row['snapshot'][key]) for key in EXECUTOR_FIELDS}
                   for tls, row in frame['executors'].items()},
        pending=[dict(packet_id=key, entry_lane_id=None) for key in pending],
        intents={},
    )


def bounded_speed_proposal(packet, residual_mps, *, net_gap_m=None,
                           leader_speed_mps=0., leader_decel_mps2=None,
                           admissible_travel_m=None):
    """One-second longitudinal proposal, before the future network resolver.

    Nominal deceleration is a proposal/braking scale, not an emergency lower
    bound. Geometry can require stronger braking. All upper bounds are reapplied
    after the learned residual. No longitudinal position is advanced here.
    """
    speed = packet['speed_mps']
    params = packet['parameters']
    accel, decel, tau = (params[name] for name in ('accel_mps2', 'decel_mps2', 'tau_s'))
    upper = min(packet['allowed_speed_mps'], speed+accel)
    if net_gap_m is not None:
        lead_decel = decel if leader_decel_mps2 is None else leader_decel_mps2
        safe = max(0., -decel*tau + sqrt((decel*tau)**2
                    + 2*decel*max(net_gap_m, 0.) + decel/lead_decel*leader_speed_mps**2))
        upper = min(upper, safe)
    if admissible_travel_m is not None:
        upper = min(upper, max(admissible_travel_m, 0.))
    upper = max(upper, 0.)
    proposal = upper + max(-decel, min(residual_mps, accel))
    return min(upper, max(0., proposal))


def categorical_choice(probabilities, uniform):
    """Sample a frozen ordered categorical law from an explicit scenario draw."""
    if not probabilities or not 0 <= uniform < 1:
        raise ValueError('Expected a nonempty law and a uniform draw in [0,1)')
    if any(value < 0 for value in probabilities.values()):
        raise ValueError('Categorical probabilities must be nonnegative')
    total = sum(probabilities.values())
    if abs(total-1.) > 1e-12:
        raise ValueError('Categorical probabilities must sum to one')
    cumulative = 0.
    for choice, probability in probabilities.items():
        cumulative += probability
        if uniform < cumulative:
            return choice
    # The sum can differ from one by roundoff. Retain the last positive bin.
    return next(key for key in reversed(probabilities) if probabilities[key] > 0)


def persistent_intent(inventory, packet_id, road_id, probabilities, uniform):
    """Keep the road-entry intent across lane changes and repeated seconds.

    probabilities includes END on nonsink ordinary roads. It is a learned
    target-history law over road-level legal movements, not a future route.
    """
    prior = inventory.get(packet_id)
    if prior is None or prior['road_id'] != road_id:
        prior = dict(road_id=road_id, choice=categorical_choice(probabilities, uniform))
        inventory[packet_id] = prior
    return prior['choice']


def assign_pending_entries(pending, entry_probabilities, uniforms):
    """Allocate unknown entries once; do not redraw a blocked packet's entry."""
    result = deepcopy(pending)
    for packet in result:
        if packet['entry_lane_id'] is None:
            packet['entry_lane_id'] = categorical_choice(entry_probabilities, uniforms[packet['packet_id']])
    return result


def partition_insertions(pending, inserted_ids):
    """Move admissible packets out of pending, retaining all blocked packets."""
    requested = list(inserted_ids)
    if len(set(requested)) != len(requested):
        raise ValueError('A pending identity can be inserted once')
    selected = set(requested)
    if not selected <= {row['packet_id'] for row in pending}:
        raise ValueError('Insertion requires an existing pending packet')
    return ([deepcopy(row) for row in pending if row['packet_id'] in selected],
            [deepcopy(row) for row in pending if row['packet_id'] not in selected])


def terminate_at_road_end(packet, intent, lane_length_m):
    """END can terminate on a nonsink edge; a turn cannot silently disappear.

    The caller has already propagated the front and verifies its actual
    model position. Signal and motion integration belong to the future kernel.
    """
    return intent == 'END' and packet['position_m'] >= lane_length_m


def verify_identity_step(active_before, pending_before, due, inserted, exited,
                         active_after, pending_after):
    """Close the model's own identity ledgers, including exits on nonsinks."""
    groups = (active_before, pending_before, due, inserted, exited, active_after, pending_after)
    if any(len(set(group)) != len(group) for group in groups):
        raise ValueError('Repeated identity in a single inventory/event')
    before_active, before_pending = set(active_before), set(pending_before)
    if before_active & before_pending or (before_active | before_pending) & set(due):
        raise ValueError('An identity cannot be active, pending or newly due twice')
    if not set(inserted) <= before_pending | set(due):
        raise ValueError('An insertion must be pending or newly due')
    if not set(exited) <= before_active | set(inserted):
        raise ValueError('Only active or same-second inserted identities can exit')
    active = Counter(active_before); active.update(inserted); active.subtract(exited)
    pending = Counter(pending_before); pending.update(due); pending.subtract(inserted)
    if (active != Counter(active_after) or pending != Counter(pending_after)
            or set(active_after) & set(pending_after)):
        raise ValueError('Packet mass/identity ledger does not close')
    return True
