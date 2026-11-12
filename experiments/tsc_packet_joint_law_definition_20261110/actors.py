"""Label-independent LC components of one realized model-motion branch."""


def select_component(topology, current_packets, projected_packets, order, focal_id,
                     *, ended_ids=(), max_actors=3):
    """Select the whole projected ordinary road in the frozen LC order.

    ``projected_packets`` is the complete surviving CURRENT cohort after a
    model-generated longitudinal branch, never a native next observation.
    New boundary insertions occur after LC in the frozen operator. Including
    every projected front on this road covers all its front-gap interactions;
    lateral actions cannot move a front between roads. This is closure for the
    existing LC operator, not a new swept-body collision guarantee.

    Same-road goals remain persistent. Actors listed in receiving_intent_ids
    need their new road's frozen intent prior before LC; this selector neither
    samples that prior nor calls the kernel's potentially sampling _intent.

    An obstruction applies to the entire input branch. Its probability mass
    must remain unresolved: callers must not drop it and renormalize the rest.
    The cap is a declared exact-enumeration boundary, never a truncation rule.
    """
    current_ids, projected_ids, ended = set(current_packets), set(projected_packets), set(ended_ids)
    result = dict(status=None, road_id=None, participants=[], required_participants=[],
                  receiving_intent_ids=[], focal_stay=False, max_actors=max_actors)
    missing = current_ids-projected_ids-ended
    unexpected = (projected_ids | ended)-current_ids
    overlap = projected_ids & ended
    if missing or unexpected or overlap:
        result.update(status='OBSTRUCTION_INCOMPLETE_LONGITUDINAL_FRAME',
                      missing_ids=sorted(missing), unexpected_ids=sorted(unexpected),
                      projected_and_ended_ids=sorted(overlap))
        return result
    if focal_id in ended:
        result.update(status='CLOSED_NO_LC', focal_stay=True, reason='focal_ended')
        return result
    projected = projected_packets[focal_id]
    geometry = topology.lanes[projected['lane_id']]
    road = geometry['edge_id']
    result['road_id'] = road
    if geometry['internal'] or len(topology.road_lanes(road)) == 1:
        result.update(status='CLOSED_NO_LC', focal_stay=True,
                      reason='focal_internal' if geometry['internal'] else 'single_lane_road')
        return result
    required = sorted((key for key, row in projected_packets.items()
                       if not topology.lanes[row['lane_id']]['internal']
                       and topology.lanes[row['lane_id']]['edge_id'] == road),
                      key=order.__getitem__)
    receiving = [key for key in required
                 if topology.lanes[current_packets[key]['lane_id']]['edge_id'] != road]
    result.update(required_participants=required, receiving_intent_ids=receiving)
    if len(required) > max_actors:
        result.update(status='OBSTRUCTION_COMPONENT_EXCEEDS_CAP', required_actor_count=len(required))
        return result
    result.update(status='CLOSED_ORDINARY_ROAD', participants=required,
                  required_actor_count=len(required))
    return result
