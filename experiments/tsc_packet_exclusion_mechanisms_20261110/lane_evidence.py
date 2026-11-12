"""Observation-only explanation of a frozen candidate's lateral denial."""
from math import isclose

from experiments.tsc_packet_receiving_path_20261110.receiving import _body_intervals
from experiments.tsc_packet_residual_truth_20261110.receiving_trace import _observed_motion, _snapshot


CHOICES = {0: 'left', 1: 'stay', 2: 'right'}


def _same_ordinary_road(topology, first, second):
    a, b = topology.lanes[first], topology.lanes[second]
    return not a['internal'] and not b['internal'] and a['edge_id'] == b['edge_id']


def _receiver_entry_distance(topology, before, after, motion):
    """Distance to the observed final ordinary road; posthoc geometry only."""
    if after is None or topology.lanes[after['lane_id']]['internal']:
        return dict(status='unresolved', reason='no_observed_final_ordinary_road', distance_m=None)
    if _same_ordinary_road(topology, before['lane_id'], after['lane_id']):
        return dict(status='resolved', distance_m=0., already_on_observed_final_road=True)
    if motion['status'] != 'resolved':
        return dict(status='unresolved', reason='observed_front_path_unresolved', distance_m=None)
    path = motion['observed_front_path']
    final_edge = topology.lanes[after['lane_id']]['edge_id']
    groups = []
    for lane in path:
        if groups and _same_ordinary_road(topology, groups[-1][-1], lane):
            groups[-1].append(lane)
        else:
            groups.append([lane])
    distance = -before['position_m']
    for group in groups:
        lane = group[-1]
        if not topology.lanes[lane]['internal'] and topology.lanes[lane]['edge_id'] == final_edge:
            return dict(status='resolved', distance_m=max(float(distance), 0.),
                already_on_observed_final_road=False, observed_receiving_lane=group[0])
        distance += topology.lanes[lane]['length_m']
    return dict(status='unresolved', reason='observed_final_road_not_in_path', distance_m=None)


def _gap(topology, focal, other, target):
    """A physical end-gap needs both fronts on the same actual target lane."""
    if focal is None or other is None:
        return dict(status='not_in_next_observation', role=None, gap_m=None)
    if focal['lane_id'] != target:
        return dict(status='focal_on_different_lane', role=None, gap_m=None)
    if other['lane_id'] != target:
        return dict(status='neighbour_front_on_different_lane', role=None, gap_m=None)
    if other['position_m'] >= focal['position_m']:
        return dict(status='same_actual_lane', role='front',
            gap_m=other['position_m']-other['parameters']['length_m']-focal['position_m'],
            required_gap_m=focal['parameters']['min_gap_m'])
    return dict(status='same_actual_lane', role='rear',
        gap_m=focal['position_m']-focal['parameters']['length_m']-other['position_m'],
        required_gap_m=other['parameters']['min_gap_m'])


def explain_lane(topology, packets, next_packets, focal_before, focal_after, lane_detail, lane_choice, *,
                 physical_proposal_mps, mean0_mps, observed_focal_path=None, observed_neighbour_paths=None,
                 candidate_connection=None):
    """Retain every current denial reason, then append independent observed truth.

    Only the supplied frozen lane_detail determines model-denying neighbours.
    Next packets never change that mask or supply a deployable replacement.
    P and mu are one-second distances numerically; speeds remain snapshots.
    """
    choice = CHOICES[int(lane_choice)]
    motion = _observed_motion(topology, focal_before, focal_after, observed_focal_path)
    receiver = _receiver_entry_distance(topology, focal_before, focal_after, motion)
    projected_lane = lane_detail.get('projected_lane_id')
    projected_position = lane_detail.get('projected_position_m')
    actual_lane = None if focal_after is None else focal_after['lane_id']
    same_final_road = (projected_lane is not None and actual_lane is not None
                      and _same_ordinary_road(topology, projected_lane, actual_lane))
    distance = receiver['distance_m']
    path_contains_final_road = None
    if candidate_connection is not None and actual_lane is not None:
        edge = topology.lanes[actual_lane]['edge_id']
        path_contains_final_road = any(not topology.lanes[lane]['internal']
            and topology.lanes[lane]['edge_id'] == edge for lane in candidate_connection['path'])
    arrival = dict(observed_receiver=receiver, physical_proposal_mps=float(physical_proposal_mps),
        mean0_mps=float(mean0_mps), model_projected_same_ordinary_road_as_actual=bool(same_final_road),
        candidate_path_contains_observed_final_road=path_contains_final_road,
        physical_distance_reaches_observed_receiver_entry=None if distance is None else physical_proposal_mps >= distance-1e-8,
        mean_distance_reaches_observed_receiver_entry=None if distance is None else mean0_mps >= distance-1e-8,
        actual_step_displacement_m=motion.get('displacement_m'),
        observed_focal_motion=motion, next_speed_snapshot_mps=None if focal_after is None else focal_after['speed_mps'])
    displacement = motion.get('displacement_m')
    arrival['next_speed_times_dt_minus_displacement_m'] = (None if focal_after is None or displacement is None
                                                         else focal_after['speed_mps']-displacement)
    reasons = []
    if not same_final_road:
        reasons.append('model_mean_does_not_end_on_observed_final_ordinary_road')
    if distance is not None and physical_proposal_mps < distance-1e-8:
        reasons.append('physical_free_distance_short_of_observed_receiver_entry')
    if distance is not None and mean0_mps < distance-1e-8:
        reasons.append('model_mean_distance_short_of_observed_receiver_entry')
    if path_contains_final_road is False:
        reasons.append('candidate_path_does_not_contain_observed_final_road')
    if choice == 'stay':
        return dict(choice=choice, current_denial_reasons=reasons, arrival_evidence=arrival,
                    denying_neighbours=[], future_used_to_change_mask=False)
    side = lane_detail.get('sides', {}).get(choice, {})
    target = side.get('target_lane_id')
    if target is None:
        reasons.append('no_adjacent_model_target_lane')
    elif not side.get('potential_legal', False):
        reasons.append('candidate_legality_denies_observed_lane_choice')
    states = side.get('neighbour_states', [])
    denying = []
    if projected_position is not None:
        ahead = [other for other in states if other['position_m'] >= projected_position]
        behind = [other for other in states if other['position_m'] < projected_position]
        for other in ahead:
            gap = other['position_m']-other['length_m']-projected_position
            if gap < focal_before['parameters']['min_gap_m']:
                denying.append((other, 'front', gap, focal_before['parameters']['min_gap_m']))
        if behind:
            nearest_position = max(other['position_m'] for other in behind)
            for other in behind:
                gap = projected_position-focal_before['parameters']['length_m']-other['position_m']
                if isclose(other['position_m'], nearest_position, rel_tol=0, abs_tol=1e-8) and gap < other['min_gap_m']:
                    denying.append((other, 'rear', gap, other['min_gap_m']))
    for unused, role, unused_gap, unused_required in denying:
        if role+'_gap_denies_observed_lane_choice' not in reasons:
            reasons.append(role+'_gap_denies_observed_lane_choice')
    paths = observed_neighbour_paths or {}
    neighbours = []
    for model, role, gap, required in denying:
        pid = model['packet_id']
        before, after = packets[pid], next_packets.get(pid)
        observed_motion = _observed_motion(topology, before, after, paths.get(pid))
        actual_gap = _gap(topology, focal_after, after, target)
        current_gap = (_gap(topology, dict(focal_before, lane_id=target), before, target)
            if target is not None and _same_ordinary_road(topology, focal_before['lane_id'], target)
            else dict(status='current_focal_not_on_target_road', role=None, gap_m=None))
        segments = [] if after is None else [dict(lane_id=lane, rear_m=rear, front_m=front)
            for lane, rear, front in _body_intervals(topology, after) if lane == target and front > rear]
        same_road_change = (after is not None and before['lane_id'] != after['lane_id']
                           and _same_ordinary_road(topology, before['lane_id'], after['lane_id']))
        body_gap = None
        if focal_after is not None and focal_after['lane_id'] == target:
            candidates = [segment['rear_m']-focal_after['position_m'] for segment in segments
                          if segment['front_m'] >= focal_after['position_m']]
            body_gap = min(candidates) if candidates else None
        neighbour_displacement = observed_motion.get('displacement_m')
        neighbours.append(dict(packet_id=pid, model_role=role, model_gap_m=float(gap),
            model_required_gap_m=float(required), model_accepted_tplus1=model,
            current=_snapshot(before), next=_snapshot(after), observed_motion=observed_motion,
            observed_same_road_lane_change=bool(same_road_change),
            current_front_on_model_target_lane=before['lane_id'] == target,
            next_front_on_model_target_lane=None if after is None else after['lane_id'] == target,
            next_front_left_model_target_lane=None if after is None else after['lane_id'] != target,
            next_body_still_on_model_target_lane=bool(segments), next_body_segments_on_model_target_lane=segments,
            current_side_gap_at_focal_longitudinal_coordinate=current_gap,
            actual_terminal_gap=actual_gap,
            actual_role_matches_model=None if actual_gap['role'] is None else actual_gap['role'] == role,
            actual_terminal_ahead_body_gap_m=body_gap,
            next_speed_snapshot_mps=None if after is None else after['speed_mps'],
            actual_step_displacement_m=neighbour_displacement,
            next_speed_times_dt_minus_displacement_m=None if after is None or neighbour_displacement is None
                else after['speed_mps']-neighbour_displacement))
    return dict(choice=choice, model_mask_allows_choice=bool(side.get('allowed', False)),
        model_target_lane=target, actual_final_lane=actual_lane,
        model_target_equals_actual_final_lane=None if target is None or actual_lane is None else target == actual_lane,
        current_denial_reasons=reasons, arrival_evidence=arrival, denying_neighbours=neighbours,
        future_used_to_change_mask=False,
        scope='Each intent retains all model arrival, legality and gap reasons. Observed next states explain the same denying actors only; they never repair masks. Native speeds and one-step path displacement are distinct. End gaps do not establish the within-step lane-change order or safety at the lateral event.')
