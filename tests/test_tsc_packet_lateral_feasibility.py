"""Outer interval and supplied joint endpoint diagnostics; no simulator."""
from copy import deepcopy

from experiments.tsc_packet_lateral_conditioning_20261110.feasibility import screen_endpoint_boxes


def actor(pid, lane, x, P=0):
    return dict(packet_id=pid, current_lane=lane, current_x_m=x, length_m=5., min_gap_m=2.,
                physical_P_mps=P, b_mps2=4.5, a_mps2=2.)


def endpoint(pid, lane, x):
    return dict(packet_id=pid, lane_id=lane, position_m=x, length_m=5., min_gap_m=2.)


def branch(points, *, mask=True):
    return dict(branch_id='joint', residuals_mps={'ego': 0.}, projected=points,
                topology_goal_legal=True, pre_lateral_mask_legal=mask)


def test_outer_box_can_rule_out_all_stay_endpoints_without_claiming_sample_completeness():
    ego, front = actor('ego', 'A', 10, 2), actor('front', 'T', 16)
    result = screen_endpoint_boxes(ego, [front], target_lane='T')
    assert result['outer_box_excludes_all_stay_endpoints']
    assert not result['relaxed_independent_box_possible']
    assert result['relaxed_focal_gap_only_witness'] is None


def test_equal_min_gap_is_legal_but_endpoint_witness_does_not_override_pre_lateral_mask():
    ego, front = actor('ego', 'A', 10), actor('front', 'T', 17)
    example = branch([endpoint('ego', 'T', 10), endpoint('front', 'T', 17)], mask=False)
    result = screen_endpoint_boxes(ego, [front], target_lane='T', branches=[example])
    assert result['feasible_focal_outer_intervals_m'] == [[10, 10]]
    assert result['provided_topology_legal_endpoint_witness']
    assert not result['provided_endpoint_and_declared_mask_witness']
    assert result['supplied_joint_branch_checks'][0]['gap_checks'][0]['gap_m'] == 2
    assert not result['future_used'] and not result['changed_mask']


def test_deniers_departure_checks_its_landing_and_all_included_actors():
    ego, denier, landing = actor('ego', 'A', 10, 2), actor('denier', 'T', 16, 4), actor('landing', 'B', 23)
    bad = branch([endpoint('ego', 'T', 12), endpoint('denier', 'B', 20), endpoint('landing', 'B', 23)])
    result = screen_endpoint_boxes(ego, [denier, landing], target_lane='T', branches=[bad])
    checked = result['supplied_joint_branch_checks'][0]
    assert not checked['endpoint_gaps_feasible']
    assert checked['gap_violations'][0]['lane_id'] == 'B'
    assert checked['gap_violations'][0]['rear_id'] == 'denier'
    absent = deepcopy(bad); absent['projected'].pop()
    result = screen_endpoint_boxes(ego, [denier, landing], target_lane='T', branches=[absent])
    assert result['supplied_joint_branch_checks'][0]['missing_actor_ids'] == ['landing']
    assert not result['provided_topology_legal_endpoint_witness']
