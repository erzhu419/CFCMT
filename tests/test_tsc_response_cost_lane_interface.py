"""Local topology joins, causal feature boundaries and exact lane-change labels."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tsc_response_cost_readout_20261109 import lane_interface as r


def example(tmp_path):
    (tmp_path/'net.xml').write_text('''<net>
      <edge id="a"><lane id="a_0" index="0" length="150" speed="10"/></edge>
      <edge id="b"><lane id="b_0" index="0" length="75" speed="10"/></edge>
      <edge id="r"><lane id="r_0" index="0" length="150" speed="10"/></edge>
      <edge id="q"><lane id="q_0" index="0" length="150" speed="10"/></edge>
      <edge id=":inside"><lane id=":inside_0" index="0" length="30" speed="10"/></edge>
      <connection from="a" fromLane="0" to="r" toLane="0" tl="T" linkIndex="0" dir="s"/>
      <connection from="a" fromLane="0" to="q" toLane="0" tl="T" linkIndex="1" dir="r"/>
      <connection from="b" fromLane="0" to="r" toLane="0" tl="T" linkIndex="2" dir="l"/>
    </net>''')
    cfg = tmp_path/'network.sumocfg'
    cfg.write_text('<configuration><input><net-file value="net.xml"/></input></configuration>')
    conflicts = {'T': dict(foes=[[0, 2], [2, 0]], response=[[1, 2]], full_context_complete=True)}
    static = r.read_lane_static(cfg, conflicts)
    lanes = {'a_0': dict(vehicle_count=2, halted_count=0, occupancy_raw=.1, mean_speed_mps=12.),
             'b_0': dict(vehicle_count=0, halted_count=0, occupancy_raw=0., mean_speed_mps=10.),
             'r_0': dict(vehicle_count=3, halted_count=1, occupancy_raw=.15, mean_speed_mps=5.),
             'q_0': dict(vehicle_count=0, halted_count=0, occupancy_raw=0., mean_speed_mps=10.),
             ':inside_0': dict(vehicle_count=2, halted_count=1, occupancy_raw=1/3, mean_speed_mps=2.)}
    endpoint = dict(offset_sec=0, time_sec=600, lanes=lanes,
                    executors={'T': dict(snapshot={'mode': 'green', 'green_elapsed_sec': 34})})
    frame = dict(offset_sec=0, time_sec=600, vehicles={
        'first': dict(lane_id='a_0', position_m=140., speed_mps=12.),
        'second': dict(lane_id='a_0', position_m=100., speed_mps=12.)})
    catalog = {'movement_a': dict(tls_id='T', incoming_lanes=['a_0'], outgoing_lanes=['r_0', 'q_0'], internal_lanes=[':inside_0'], link_indices=[0, 1]),
               'movement_b': dict(tls_id='T', incoming_lanes=['b_0'], outgoing_lanes=['r_0'], internal_lanes=[':inside_0'], link_indices=[2])}
    return endpoint, frame, catalog, static, {'T': ['Ggr']*10}


def parameters():
    return {vid: dict(lane_id='a_0', type_id='ACTUAL_DEFAULT', speed_mps=12.,
        length_m=5., min_gap_m=2.5, accel_mps2=2.6, decel_mps2=4.5, tau_s=1., imperfection=.5,
        max_speed_mps=55., speed_factor=1.3, allowed_speed_mps=13.) for vid in ('first', 'second')}


def test_exact_mixed_joint_signals_foes_yielding_and_shared_receiving(tmp_path):
    endpoint, frame, catalog, static, nominal = example(tmp_path)
    saved = deepcopy((endpoint, frame, catalog, static, nominal))
    result = r.build_inputs(endpoint, frame, catalog, static, nominal, 'T', parameters())
    nodes = {node['lane_id']: node for node in result['lane_nodes']}
    assert result['lane_to_links'] == {'a_0': [0, 1], 'b_0': [2]}
    assert result['relations']['foes'] == [[0, 2], [2, 0]]
    assert result['relations']['yields_to'] == [[1, 2]]
    assert result['relations']['shared_to_lane'] == [
        dict(from_link=0, to_link=2, receiving_lane='r_0'),
        dict(from_link=2, to_link=0, receiving_lane='r_0')]
    assert nodes['a_0']['nominal_signal_fractions'][0][:2] == [.5, .5]
    assert result['full_nominal_states'] == ['Ggr']*10
    assert nodes['a_0']['to_lanes'] == ['q_0', 'r_0']
    assert result['receiving_nodes']['r_0']['H'] == 1
    assert result['receiving_nodes']['r_0']['M'] == 1.5
    assert result['internal_nodes'][':inside_0']['H'] == 1
    assert result['internal_nodes'][':inside_0']['M'] == .4
    assert result['internal_vehicle_count'] == 2  # Shared movement membership counts this lane once.
    assert result['can_full_fit'] and result['relation_context_complete']
    assert (endpoint, frame, catalog, static, nominal) == saved


def test_lane_ids_are_join_keys_and_targets_have_a_separate_future_interface(tmp_path):
    endpoint, frame, catalog, static, nominal = example(tmp_path)
    before = r.build_inputs(endpoint, frame, catalog, static, nominal, 'T')
    renames = dict(a_0='z_lane', b_0='a_lane', r_0='receive_9', q_0='receive_1')
    renames[':inside_0'] = ':new_inside_0'
    changed_endpoint, changed_frame, changed_catalog, changed_static = deepcopy((endpoint, frame, catalog, static))
    changed_endpoint['lanes'] = {renames[key]: value for key, value in changed_endpoint['lanes'].items()}
    for vehicle in changed_frame['vehicles'].values():
        vehicle['lane_id'] = renames[vehicle['lane_id']]
    for movement in changed_catalog.values():
        for role in ('incoming_lanes', 'outgoing_lanes', 'internal_lanes'):
            movement[role] = [renames[lane] for lane in movement[role]]
    for name in ('lane_lengths', 'lane_speed_limits'):
        changed_static[name] = {renames[key]: value for key, value in changed_static[name].items()}
    for connection in changed_static['connections_by_tls']['T']:
        connection['from_lane'], connection['to_lane'] = renames[connection['from_lane']], renames[connection['to_lane']]
    after = r.build_inputs(changed_endpoint, changed_frame, changed_catalog, changed_static, nominal, 'T')
    for old in before['lane_nodes']:
        new = next(node for node in after['lane_nodes'] if node['lane_id'] == renames[old['lane_id']])
        for key in ('N', 'H', 'M', 'occupancy', 'storage', 'heads', 'nominal_signal_fractions'):
            assert new[key] == old[key]
    end = deepcopy(endpoint)
    end.update(offset_sec=10, time_sec=610)
    end['lanes']['a_0'].update(vehicle_count=1, halted_count=1, mean_speed_mps=2., occupancy_raw=.05)
    end['lanes']['b_0']['mean_speed_mps'] = 99.  # Empty-lane conventions carry no speed moment.
    targets = r.extract_targets(endpoint, end, ['a_0', 'b_0'], static)
    np.testing.assert_allclose(targets['y'], [[-.05, .05, -.11, -.05], [0, 0, 0, 0]])
    assert 'join keys' in before['identifier_semantics'] and 'candidate_index' not in before


def test_empty_lanes_and_missing_parameters_or_vehicle_speeds_are_not_fabricated(tmp_path):
    endpoint, frame, catalog, static, nominal = example(tmp_path)
    result = r.build_inputs(endpoint, frame, catalog, static, nominal, 'T')
    occupied, empty = result['lane_nodes']
    assert occupied['M'] == 2.4 and occupied['heads'][0]['speed_ratio'] == 1.2
    assert [vehicle['distance_m'] for vehicle in occupied['vehicles']] == [10., 50.]
    assert occupied['near_count'] == 2
    assert empty['M'] == 0 and empty['mean_speed_mps'] is None and empty['vehicles'] == []
    assert empty['heads'] == [dict(present=False, distance_ratio=None, speed_ratio=None)]*2
    assert empty['parameters']['reason'] == 'empty_lane_no_parameter_samples'
    assert result['param_observation_status']['reason'] == 'vehicle_params0_not_recorded'
    assert not result['can_full_fit']
    del frame['vehicles']['first']['speed_mps']
    observed = r.build_inputs(endpoint, frame, catalog, static, nominal, 'T', parameters())
    assert observed['lane_nodes'][0]['heads'][0]['speed_ratio'] is None
    assert not observed['lane_nodes'][0]['spatial_complete'] and not observed['can_full_fit']


def test_actual_parameter_getters_and_future_payloads_do_not_enter_inputs(tmp_path):
    endpoint, frame, catalog, static, nominal = example(tmp_path)
    values = dict(getLength=5., getMinGap=2.5, getAccel=2.6, getDecel=4.5, getTau=1.,
                  getImperfection=.5, getMaxSpeed=55., getSpeedFactor=1.3, getAllowedSpeed=13.)
    calls = []
    class Vehicles:
        def getIDList(self): return ['first', 'second', 'outside']
        def getLaneID(self, vid): return 'unobserved_lane' if vid == 'outside' else 'a_0'
        def getTypeID(self, vid): return 'ACTUAL_DEFAULT'
        def getSpeed(self, vid): return 12.
        def __getattr__(self, name):
            def get(vid):
                calls.append((name, vid))
                return values[name]
            return get
    api = SimpleNamespace(vehicle=Vehicles())
    records = r.record_vehicle_params(api, lane_ids=['a_0', 'b_0'])
    assert set(records) == {'first', 'second'} and records['first']['accel_mps2'] == 2.6
    assert records['first']['type_id'] == 'ACTUAL_DEFAULT'
    assert {name for name, _ in calls} == set(values)
    assert set(r.record_vehicle_params(api, seen_ids=['first'])) == {'first'}
    inputs = r.build_inputs(endpoint, frame, catalog, static, nominal, 'T', records)
    endpoint.update(cost=999, actual_signals=['rrr']*450, endpoint10={'future': True})
    frame['future_response'] = [999]
    assert r.build_inputs(endpoint, frame, catalog, static, nominal, 'T', records) == inputs
    assert inputs['lane_nodes'][0]['parameters']['mean']['accel_mps2'] == 2.6
    assert inputs['can_full_fit']
    with pytest.raises(ValueError, match='offset zero'):
        r.build_inputs(dict(endpoint, offset_sec=10), frame, catalog, static, nominal, 'T', records)


def test_identical_parameter_mean_does_not_change_with_lane_population(tmp_path):
    endpoint, frame, catalog, static, nominal = example(tmp_path)
    template = parameters()['first']
    for n in range(1, 13):
        endpoint['lanes']['a_0'].update(vehicle_count=n, occupancy_raw=n*5/150)
        frame['vehicles'] = {str(i): dict(lane_id='a_0', position_m=150-7.5*(i+1), speed_mps=12.)
                             for i in range(n)}
        records = {str(i): template.copy() for i in range(n)}
        result = r.build_inputs(endpoint, frame, catalog, static, nominal, 'T', records)
        assert result['lane_nodes'][0]['parameters']['mean']['accel_mps2'] == 2.6
    records['0']['accel_mps2'] = 5.0
    changed = r.build_inputs(endpoint, frame, catalog, static, nominal, 'T', records)
    assert changed['lane_nodes'][0]['parameters']['mean']['accel_mps2'] == pytest.approx(2.8)
