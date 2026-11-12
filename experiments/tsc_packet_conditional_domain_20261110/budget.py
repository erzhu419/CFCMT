"""Static branch-count bounds; no physical response or label is evaluated."""
from math import prod


def _goals(view, road):
    prior = view.laws['intent_probs'].get(road)
    if prior is None:
        if any(view.topology.connections_from(lane) for lane in view.topology.road_lanes(road)):
            raise ValueError('A reachable ordinary road lacks its frozen intent prior')
        return ['END']
    goals = [goal for goal, probability in prior.items() if probability > 0]
    if not goals:
        raise ValueError('A reachable ordinary road has no positive intent support')
    return goals


def _actor_budget(view, packet_id, bounds):
    topology, packet = view.topology, view.state['packets'][packet_id]
    geometry = topology.lanes[packet['lane_id']]
    start_factor = 2 if packet['speed_mps'] < .1 else 1
    if geometry['internal']:
        current = [(topology.by_key[packet['_movement_key']]['to_edge'], 1)]
    else:
        current = []
        for goal in _goals(view, geometry['edge_id']):
            matches = topology.compatible(packet['lane_id'], goal) if goal != 'END' else ()
            connection = matches[0] if matches else None
            yielding = (connection is not None and connection['tls_id'] is not None
                and view.signals[connection['tls_id']][connection['link_index']] == 'g')
            current.append((goal, 2 if yielding else 1))
    per_goal = [dict(goal=goal, residual_nodes=2, start_permission_factor=start_factor,
        yield_permission_factor=yield_factor,
        physical_branches=2*start_factor*yield_factor) for goal, yield_factor in current]
    # Each physical branch ends on at most one ordinary road. Take the maximum
    # future-prior size, rather than multiplying mutually exclusive new roads.
    receiving = {}
    for lane in bounds['longitudinal_fronts']:
        road = topology.lanes[lane]['edge_id']
        if road != geometry['edge_id'] and len(topology.road_lanes(road)) > 1:
            receiving[road] = len(_goals(view, road))
    return dict(packet_id=packet_id, current_positive_goals=len(current),
        current_goal_factors=per_goal,
        physical_branches_upper_bound=sum(row['physical_branches'] for row in per_goal),
        possible_new_receiving_road_goal_counts=receiving,
        receiving_goals_upper_bound=max([1, *receiving.values()]),
        lateral_choices_upper_bound=3)


def symbolic_budget(view, selection):
    """Count full-resolver and LC leaves from the registered geometric domain.

    The physical count is the Cartesian latent count before endpoint merging:
    product_j sum_goal (2 residuals * start factor * compatible-g yield factor).
    It is an upper bound because it does not remove numerically zero weights,
    ended actors, impossible side choices or equivalent physical outcomes.
    No mean, endpoint, lateral probability or native future is queried here.
    """
    result = dict(status='UNDEFINED_SELECTOR_DOMAIN', selector_status=selection['status'],
        required_participants=list(selection['required_participants']),
        participants=[], per_actor={}, physical_resolver_calls_upper_bound=None,
        receiving_goal_product_upper_bound=None, lateral_choice_product_upper_bound=None,
        lateral_leaf_branches_upper_bound=None, real_record_response_queries=0)
    if selection['status'] != 'SELECTED_GEOMETRIC_ENVELOPE':
        return result
    participants = list(selection['participants'])
    try:
        per_actor = {key: _actor_budget(view, key, selection['bounds'][key]) for key in participants}
    except ValueError as error:
        result.update(status='UNDEFINED_INTENT_DOMAIN', reason=str(error))
        return result
    physical = prod(row['physical_branches_upper_bound'] for row in per_actor.values())
    receiving = prod(row['receiving_goals_upper_bound'] for row in per_actor.values())
    lateral = 3**len(participants)
    result.update(status='REGISTERED_SYMBOLIC_UPPER_BOUND', participants=participants, per_actor=per_actor,
        physical_resolver_calls_upper_bound=physical,
        receiving_goal_product_upper_bound=receiving,
        lateral_choice_product_upper_bound=lateral,
        lateral_leaf_branches_upper_bound=physical*receiving*lateral,
        cost_scope='Each physical branch still resolves the entire current frame; no wall-time estimate or real response has been computed.')
    return result
