"""Entity comparison data ownership and same-code table-swap checks."""
from copy import deepcopy
import json

import pytest

from experiments.tsc_packet_entity_footprint_20261110.diagnose import (
    SEEDS, load_saved_probes, paired_admission, verify_entity_preflight,
)
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_packet_spatial_conflict_20261110 import conditional_features as shared
from experiments.tsc_packet_spatial_conflict_20261110.conflict import configure_geometry
from experiments.tsc_packet_spatial_conflict_20261110.diagnose import seed_context
from experiments.tsc_packet_spatial_conflict_20261110.geometry import build_conflict_geometry
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet, executor_seed
from test_tsc_packet_spatial_forward_review import crossing_kernel


def test_entity_width_is_derived_from_configured_xml_not_a_hardcoded_constant(tmp_path):
    route = tmp_path/'scenario.rou.xml'
    route.write_text('<routes><vType id="car" width="1.85"/></routes>')
    cfg = tmp_path/'scenario.sumocfg'
    cfg.write_text('<configuration><input><route-files value="scenario.rou.xml"/></input></configuration>')
    proof = dict(status='PASS_ENTITY_INPUT_PREFLIGHT', training_seeds=SEEDS,
        centerline_observed=True, maximum_abs_lateral_position_m=0., maximum_abs_lateral_speed_mps=0.,
        sumocfg=str(cfg), vtype_definitions=[dict(source=str(route), id='car', width='1.85')],
        observed_type_counts={'car': 10}, vehicle_width_by_type={'car': 1.85}, single_vehicle_width_m=1.85)
    path = tmp_path/'preflight.json'
    path.write_text(json.dumps(proof))
    assert verify_entity_preflight(path)[0] == 1.85
    route.write_text('<routes><vType id="car" width="2.1"/></routes>')
    with pytest.raises(ValueError, match='differs'):
        verify_entity_preflight(path)


def test_saved_probe_loader_preserves_the_exact_twenty_old_bad_references(tmp_path):
    output = tmp_path/'output'
    output.mkdir()
    rows = []
    for index in range(253):
        spatial = 1. if index < 21 else 2.
        actual = 1.5 if index < 20 else .5
        rows.append(dict(seed=1, time_sec=index, packet_id=f'p{index}',
            actual_next_speed_mps=actual, paired=dict(baseline={'upper': 2.}, candidate={'upper': spatial})))
    path = output/'seed1_internal_probes_server_only.json'
    original = json.dumps(rows)
    path.write_text(original)
    saved = load_saved_probes(tmp_path)
    bad = {key for key, row in saved[1].items() if row['previously_bad']}
    assert bad == {(index, f'p{index}') for index in range(20)}
    assert len(saved[1]) == 253 and path.read_text() == original


def test_same_admission_code_table_swap_matches_full_entity_context_and_restores_state():
    proto = crossing_kernel([packet('primary', 'Ia', 20, 0), packet('secondary', 'Ib', 10, 5)])
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    frame = dict(time_s=0, packets=deepcopy(proto.state['packets']),
        native_lanes=deepcopy(proto.state['lane_sensors']), native_pending_ids=[],
        executors={tls: dict(snapshot=deepcopy(row)) for tls, row in proto.state['executors'].items()},
        unresolved=[], checks=dict(synthetic=True))
    kernel = seed_context.prepare_feature_kernel(proto.static, frame, seeds, proto.laws, {'J': 'Gg'}, 0)
    lane_table = build_conflict_geometry(kernel.topology)
    entity_table = build_entity_geometry(kernel.topology, 1.85)
    configure_geometry(kernel, lane_table)
    before = deepcopy(kernel.state)
    raw_cache = deepcopy(kernel._conditional_zero_speeds)
    rng = {key: (value['intent_rng'].getstate(), value['decision_rng'].getstate())
           for key, value in kernel.noise._packets.items()}
    same = paired_admission(kernel, 'secondary', 'out_b', lane_table, lane_table)
    assert same['baseline'] == same['candidate']
    pair = paired_admission(kernel, 'secondary', 'out_b', lane_table, entity_table)
    full = shared.prepare_feature_kernel(proto.static, frame, seeds, proto.laws, {'J': 'Gg'}, 0,
                                         conflict_geometry=entity_table)
    focal = full.state['packets']['secondary']
    connection, compatible = full._connections['secondary']
    view = shared._context_view(full, focal, connection, compatible)
    detail = shared._admission_detail(full, focal, connection, compatible, view)
    fast = dict(pair['candidate'])
    fast.pop('yield_risk')
    assert fast == detail
    assert kernel.state == before and kernel._spatial_geometry is lane_table
    assert all(kernel._conditional_zero_speeds[key] == value for key, value in raw_cache.items())
    assert rng == {key: (value['intent_rng'].getstate(), value['decision_rng'].getstate())
                   for key, value in kernel.noise._packets.items()}
