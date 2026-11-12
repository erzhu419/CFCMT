"""The diagnostic's one-kernel evaluation matches the full new feature path."""
from copy import deepcopy

import pytest

from test_tsc_packet_spatial_forward_review import crossing_kernel
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet, executor_seed
from experiments.tsc_packet_spatial_conflict_20261110 import conditional_features as candidate
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import paired_admission, seed_context
from experiments.tsc_packet_spatial_conflict_20261110.conflict import configure_geometry
from experiments.tsc_packet_spatial_conflict_20261110.geometry import build_conflict_geometry


@pytest.mark.parametrize('ego_lane,ego_position,foe_lane,foe_position', [
    ('b', 19, 'Ia', 15), ('Ib', 10, 'Ia', 20), ('b', 19, 'out_a', 6),
])
def test_fast_pair_equals_full_conditional(ego_lane, ego_position, foe_lane, foe_position):
    proto = crossing_kernel([packet('primary', foe_lane, foe_position, 0),
                             packet('secondary', ego_lane, ego_position, 5)])
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    frame = dict(time_s=0, packets=deepcopy(proto.state['packets']),
        native_lanes=deepcopy(proto.state['lane_sensors']), native_pending_ids=[],
        executors={tls: dict(snapshot=deepcopy(row)) for tls, row in proto.state['executors'].items()},
        unresolved=[], checks=dict(synthetic=True))
    old = seed_context.prepare_feature_kernel(proto.static, frame, seeds, proto.laws, {'J': 'Gg'}, 0)
    table = build_conflict_geometry(old.topology)
    configure_geometry(old, table)
    before = deepcopy(old.state)
    intent = old._connections['secondary'][0]['to_edge']
    result = paired_admission(old, 'secondary', intent)
    full = candidate.prepare_feature_kernel(proto.static, frame, seeds, proto.laws,
                                           {'J': 'Gg'}, 0, conflict_geometry=table)
    focal = full.state['packets']['secondary']
    connection, compatible = full._connections['secondary']
    view = candidate._context_view(full, focal, connection, compatible)
    detail = candidate._admission_detail(full, focal, connection, compatible, view)
    fast = dict(result['candidate'])
    fast.pop('yield_risk')
    assert fast == detail
    assert old.state == before
