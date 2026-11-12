"""Current-state endpoint geometry only; no probability or trajectory claims.

API screen_endpoint_boxes(focal, neighbours, *, target_lane, branches=()).
Each actor: packet_id, current_lane, current_x_m, length_m, min_gap_m,
physical_P_mps, b_mps2, a_mps2. All are ordinary-road longitudinal coordinates.
Neighbours include every target-lane actor and all actors in lateral landing
lanes, not only the baseline denier. The proven travel outer box is [0,P].
Each supplied joint branch has branch_id, residuals_mps {packet_id: residual},
projected [{packet_id,lane_id,position_m,length_m,min_gap_m}],
topology_goal_legal bool, pre_lateral_mask_legal bool or None. Projected states
must come from one fresh shared-resolver query; all relevant end-lane actors
must be included. Residual maps contain actors with supplied current metadata.
No observed future or native outcome is an input to this helper.
"""


def _remove_open(intervals, lower, upper):
    if lower >= upper:
        return intervals
    remaining = []
    for left, right in intervals:
        if right <= lower or left >= upper:
            remaining.append((left, right))
            continue
        if left <= lower <= right:
            remaining.append((left, lower))
        if left <= upper <= right:
            remaining.append((upper, right))
    return remaining


def _endpoint_gaps(projected):
    """Check every same-lane adjacent pair, including neighbours' landing lanes."""
    by_lane = {}
    for actor in projected:
        by_lane.setdefault(actor['lane_id'], []).append(actor)
    gaps, violations = [], []
    for lane, actors in sorted(by_lane.items()):
        actors = sorted(actors, key=lambda actor: (actor['position_m'], actor['packet_id']))
        for rear, front in zip(actors, actors[1:]):
            gap = front['position_m']-front['length_m']-rear['position_m']
            row = dict(lane_id=lane, rear_id=rear['packet_id'], front_id=front['packet_id'],
                       gap_m=gap, required_gap_m=rear['min_gap_m'])
            gaps.append(row)
            if gap < rear['min_gap_m']:
                violations.append(row)
    return gaps, violations


def _check_branch(focal, actors, target_lane, branch):
    by_id = {actor['packet_id']: actor for actor in branch['projected']}
    missing = sorted(set(actors)-set(by_id))
    residual_violations = []
    for pid, residual in branch['residuals_mps'].items():
        actor = actors[pid]
        if not -actor['b_mps2'] <= residual <= actor['a_mps2']:
            residual_violations.append(dict(packet_id=pid, residual_mps=residual,
                                           allowed_mps=[-actor['b_mps2'], actor['a_mps2']]))
    gaps, violations = _endpoint_gaps(branch['projected'])
    focal_on_target = by_id[focal['packet_id']]['lane_id'] == target_lane
    geometric = focal_on_target and not violations and not residual_violations and not missing
    return dict(branch_id=branch['branch_id'], endpoint_gaps_feasible=bool(geometric),
        focal_on_target_lane=bool(focal_on_target), missing_actor_ids=missing, gap_checks=gaps, gap_violations=violations,
        residual_domain_violations=residual_violations,
        topology_goal_legal=branch['topology_goal_legal'],
        pre_lateral_mask_legal=branch.get('pre_lateral_mask_legal'),
        topology_legal_endpoint_witness=bool(geometric and branch['topology_goal_legal']),
        endpoints_and_declared_masks_pass=bool(geometric and branch['topology_goal_legal']
                                               and branch.get('pre_lateral_mask_legal') is True))


def screen_endpoint_boxes(focal, neighbours, *, target_lane, branches=()):
    """Separate a relaxed outer-box screen from actually supplied joint endpoints.

    Box feasibility checks focal-to-neighbour gaps under independent choices.
    It supplies only a necessary condition for a global all-stay endpoint.
    Supplied branch checks instead include every same-lane adjacent pair.
    Neither result establishes one-second path feasibility or model probability.
    """
    actors = {actor['packet_id']: actor for actor in [focal, *neighbours]}
    lower, upper = focal['current_x_m'], focal['current_x_m']+focal['physical_P_mps']
    intervals, constraints = [(lower, upper)], []
    target = [actor for actor in neighbours if actor['current_lane'] == target_lane
              and actor['packet_id'] != focal['packet_id']]
    for other in target:
        low, high = other['current_x_m'], other['current_x_m']+other['physical_P_mps']
        front_required = other['length_m']+focal['min_gap_m']
        rear_required = focal['length_m']+other['min_gap_m']
        # Some neighbour endpoint must be ahead by front_required or behind
        # by rear_required. The intersection of both failures is open.
        forbidden = [high-front_required, low+rear_required]
        intervals = _remove_open(intervals, *forbidden)
        constraints.append(dict(packet_id=other['packet_id'], endpoint_outer_m=[low, high],
                                forbidden_focal_open_interval_m=forbidden))
    box_witness = None
    if intervals:
        own = sum(intervals[0])/2
        selected = []
        for other in target:
            low, high = other['current_x_m'], other['current_x_m']+other['physical_P_mps']
            rear_limit = own-focal['length_m']-other['min_gap_m']
            end = low if low <= rear_limit else high
            selected.append(dict(packet_id=other['packet_id'], endpoint_m=end,
                travel_m=end-other['current_x_m'], relative_role='rear' if end < own else 'front'))
        box_witness = dict(focal_endpoint_m=own, focal_travel_m=own-focal['current_x_m'],
                           independent_neighbour_endpoints=selected)
    branch_checks = [_check_branch(focal, actors, target_lane, branch) for branch in branches]
    return dict(proven_travel_outer_rule='Every fixed-connection residual response lies within [0,P]; no monotonicity or filled sample range is assumed.',
        box_condition='Focal ends on target lane and target neighbours retain their current ordinary lane at the endpoint; one common road coordinate is required.',
        relaxed_independent_box_possible=bool(intervals), feasible_focal_outer_intervals_m=[list(row) for row in intervals],
        outer_box_excludes_all_stay_endpoints=not bool(intervals), outer_constraints=constraints,
        relaxed_focal_gap_only_witness=box_witness,
        supplied_joint_branch_checks=branch_checks,
        provided_topology_legal_endpoint_witness=any(row['topology_legal_endpoint_witness'] for row in branch_checks),
        provided_endpoint_and_declared_mask_witness=any(row['endpoints_and_declared_masks_pass'] for row in branch_checks),
        future_used=False, changed_mask=False,
        scope='Outer boxes ignore joint motion coupling and neighbour-neighbour compatibility. Supplied branch geometry checks all included end-lane actors; topology legality and pre-lateral mask acceptance remain separate. Endpoint witnesses do not prove within-step paths, execution order or normalized probability.')
