"""Reproduce the observed same-road mask clock error without real fitting."""
from copy import deepcopy

from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    lane,road_graph,laws,packet,initial_observation,
)
from experiments.tsc_target_packet_training_20261109.conditional_features import (
    prepare_feature_kernel,candidate_features,
)


def test_focal_and_front_neighbour_use_the_same_predicted_clock():
    static=road_graph([lane('r0',800,edge='r',index=0),lane('r1',800,edge='r',index=1)],[],cost_lanes=['r0','r1'])
    ego=packet('ego','r1',181.2818,6.6457)
    front=packet('front','r0',195.1769,10.2138)
    state=initial_observation(static,[ego,front])
    frame=dict(time_s=0,packets=state['packets'],native_lanes=state['lane_sensors'],native_pending_ids=[],
               executors={},unresolved=[],checks={'synthetic':True})
    original=deepcopy(frame)
    kernel=prepare_feature_kernel(static,frame,{},laws(static),{},123)
    row=candidate_features(kernel,'ego','END')
    projected_ego=ego['position_m']+row['upper']
    assert front['position_m']-front['parameters']['length_m']-projected_ego<0
    predicted_front=kernel._conditional_next_packets['front']
    assert predicted_front['position_m']-front['parameters']['length_m']-projected_ego>ego['parameters']['min_gap_m']
    assert row['lane_mask']['right']
    assert frame==original  # Future observations were neither supplied nor injected.
