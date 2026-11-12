"""Native-free tests of causal tails, instantaneous lane changes and getters."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from experiments.tsc_network_dense_measurement_20261109.packet_observer import (
    PARAM_GETTERS, PacketObserver, read_static)


def static(tmp_path, ambiguous=False):
    alternative = ('<edge id=":j" function="internal"><lane id=":j_0" index="0" length="2" speed="10"/></edge>'
        '<connection from="a" to="b" fromLane="0" toLane="0" via=":j_0" tl="tls" linkIndex="1"/>'
        '<connection from=":j" to="b" fromLane="0" toLane="0"/>') if ambiguous else ''
    (tmp_path/'net.xml').write_text('<net>'
        '<edge id="a"><lane id="a0" index="0" length="100" speed="10"/>'
        '<lane id="a1" index="1" length="100" speed="10"/></edge>'
        '<edge id=":i" function="internal"><lane id=":i_0" index="0" length="2" speed="10"/></edge>'
        '<edge id="b"><lane id="b0" index="0" length="100" speed="10"/></edge>'
        '<connection from="a" to="b" fromLane="0" toLane="0" via=":i_0" tl="tls" linkIndex="0"/>'
        '<connection from=":i" to="b" fromLane="0" toLane="0"/>'+alternative+'</net>')
    cfg = tmp_path/'case.sumocfg'
    cfg.write_text('<configuration><input><net-file value="net.xml"/></input></configuration>')
    return read_static(cfg)


def packet(lane, position, speed=1, stopped=False, length=5.):
    return dict(lane=lane, position=position, speed=speed, stopped=stopped,
        parameters=dict(length_m=length, min_gap_m=2.5, accel_mps2=2.6, decel_mps2=4.5,
                        tau_s=1., imperfection=.5, max_speed_mps=10., speed_factor=1.))


class FakeVehicle:
    def __init__(self, api):
        self.api = api

    def getIDList(self):
        return list(self.api.vehicles)

    def __getattr__(self, name):
        assert name not in ('getRoute', 'getRouteIndex', 'getNextTLS', 'getNextLinks')
        fields = dict(getLaneID='lane', getLanePosition='position', getSpeed='speed', isStopped='stopped')
        if name in fields:
            return lambda vid: self.api.vehicles[vid][fields[name]]
        if name in PARAM_GETTERS.values():
            key = next(key for key, getter in PARAM_GETTERS.items() if getter == name)
            return lambda vid: self.api.vehicles[vid]['parameters'][key]
        values = dict(getAcceleration=0., getLateralLanePosition=0., getLateralSpeed=0.,
                      getAllowedSpeed=10., getTypeID='actual_type')
        return lambda vid: values[name]


class FakeAPI:
    def __init__(self, geometry):
        self.geometry = geometry
        self.time, self.vehicles, self.departed, self.native = 0, {}, [], {}
        self.vehicle = FakeVehicle(self)
        self.simulation = SimpleNamespace(getTime=lambda: self.time,
            getDepartedIDList=lambda: self.departed, getPendingVehicles=lambda: [],
            getArrivedIDList=lambda: [], getStartingTeleportIDList=lambda: [],
            getEndingTeleportIDList=lambda: [], getCollidingVehiclesIDList=lambda: [])
        self.lane = SimpleNamespace(**{getter: (lambda lane, key=key: self.native[lane][key])
            for getter, key in [('getLastStepVehicleIDs', 'ids'), ('getLastStepVehicleNumber', 'N'),
                ('getLastStepHaltingNumber', 'H'), ('getLastStepOccupancy', 'O'),
                ('getLastStepMeanSpeed', 'mean_speed_mps')]})

    def frame(self, time, vehicles, occupancy, departed=()):
        self.time, self.vehicles, self.departed = time, deepcopy(vehicles), list(departed)
        self.native = {}
        for lane, geometry in self.geometry['lanes'].items():
            ids = [vid for vid, row in vehicles.items() if row['lane'] == lane]
            speeds = [vehicles[vid]['speed'] for vid in ids
                      if not vehicles[vid]['stopped'] or not geometry['has_lane_changer']]
            self.native[lane] = dict(ids=ids, N=len(ids), H=sum(vehicles[vid]['speed'] < .1 for vid in ids),
                O=occupancy.get(lane, 0.), mean_speed_mps=sum(speeds)/len(speeds) if speeds else geometry['speed_limit_mps'])


def test_xml_preserves_physical_internal_chain_and_exact_joins(tmp_path):
    graph = static(tmp_path)
    assert graph['successors']['a0'] == [':i_0']
    assert graph['successors'][':i_0'] == ['b0']
    assert graph['predecessors']['b0'] == [':i_0']
    assert graph['controlled_lane_ids'] == ['a0']
    assert graph['internal_lane_ids'] == [':i_0']
    assert graph['lane_to_tls'][':i_0'] == ['tls']


def test_short_internal_and_stopped_tail_match_native_without_counting_partial_packets(tmp_path):
    graph = static(tmp_path)
    observer, api = PacketObserver(graph), FakeAPI(graph)
    api.frame(0, {'v': packet('a0', 99)}, {'a0': .05})
    assert all(observer.capture(api, 0, 'prefix')['checks'].values())
    api.frame(1, {'v': packet(':i_0', .5)}, {'a0': .045, ':i_0': .25})
    middle = observer.capture(api, 1, 'prefix')
    assert all(middle['checks'].values())
    assert middle['reconstructed_lanes']['a0']['N'] == middle['reconstructed_lanes']['a0']['H'] == 0
    # SUMO's native partial-length formula is not clipped to short lane length.
    api.frame(2, {'v': packet('b0', .5, speed=0)}, {'a0': .025, ':i_0': 2.25, 'b0': .005})
    entry = observer.capture(api, 2, 'measurement')
    assert all(entry['checks'].values())
    assert entry['packets']['v']['body_trail'] == ['a0', ':i_0', 'b0']
    api.frame(3, {'v': packet('b0', .5, speed=0)}, {'a0': .025, ':i_0': 2.25, 'b0': .005})
    stopped = observer.capture(api, 3, 'measurement')
    assert all(stopped['checks'].values())
    assert stopped['packets']['v']['body_trail'] == entry['packets']['v']['body_trail']
    api.frame(4, {'v': packet('b0', 6)}, {'b0': .05})
    cleared = observer.capture(api, 4, 'measurement')
    assert all(cleared['checks'].values())
    assert cleared['packets']['v']['body_trail'] == ['b0']


def test_instantaneous_lane_change_drops_parallel_ghost_tail_and_tracks_params(tmp_path):
    graph = static(tmp_path)
    saved, api = [], FakeAPI(graph)
    observer = PacketObserver(graph, sink=saved.append)
    api.frame(0, {'v': packet('a0', 30)}, {'a0': .05})
    observer.capture(api, 0, 'prefix')
    row = packet('a1', 31)
    row['parameters']['accel_mps2'] = 3.
    api.frame(1, {'v': row}, {'a1': .05})
    frame = observer.capture(api, 1, 'measurement')
    assert all(frame['checks'].values())
    assert frame['packets']['v']['body_trail'] == ['a1']
    assert frame['packets']['v']['cell_index'] == 2
    assert frame['reconstructed_lanes']['a0']['O'] == 0
    assert any(event['kind'] == 'parameter_change' for event in frame['events'])
    assert saved == [frame]


def test_ambiguous_skipped_connection_is_unresolved_instead_of_guessed(tmp_path):
    graph = static(tmp_path, ambiguous=True)
    observer, api = PacketObserver(graph), FakeAPI(graph)
    api.frame(0, {'v': packet('a0', 99)}, {'a0': .05})
    observer.capture(api, 0, 'prefix')
    api.frame(1, {'v': packet('b0', .5)}, {'a0': .025, ':i_0': 2.25, 'b0': .005})
    frame = observer.capture(api, 1, 'measurement')
    assert frame['unresolved'][0]['reason'] == 'past_connection_not_unique'
    assert frame['packets']['v']['body_trail'] == ['b0']
    assert not frame['checks']['body_paths_complete']
    assert not frame['checks']['O_close']
    assert frame['checks']['N_exact'] and frame['checks']['H_exact']


def test_initial_interior_missing_history_differs_from_observed_insertion_and_stop_state(tmp_path):
    graph = static(tmp_path)
    api = FakeAPI(graph)
    api.frame(0, {'v': packet('b0', 1)}, {'b0': .01})
    frame = PacketObserver(graph).capture(api, 0, 'prefix')
    assert frame['unresolved'][0]['reason'] == 'initial_body_trail_unobserved'
    assert not frame['checks']['body_paths_complete']
    # A newly inserted vehicle has no earlier on-network tail, even if the
    # insertion edge also has a topological predecessor.
    api.frame(0, {'v': packet('b0', 1)}, {'b0': .01}, departed=['v'])
    assert all(PacketObserver(graph).capture(api, 0, 'measurement')['checks'].values())
    # Scheduled stopping is not the same flag as traffic halting: native mean
    # speed excludes a scheduled-stop vehicle when an edge has a lane changer.
    api.frame(0, {'v': packet('a0', 30, speed=0, stopped=True)}, {'a0': .05})
    stopped = PacketObserver(graph).capture(api, 0, 'prefix')
    assert all(stopped['checks'].values())
    assert stopped['reconstructed_lanes']['a0']['H'] == 1
    assert stopped['reconstructed_lanes']['a0']['mean_speed_mps'] == 10
    observer = PacketObserver(graph)
    observer.capture(api, 0, 'prefix')
    with pytest.raises(ValueError, match='consecutive'):
        observer.capture(api, 2, 'prefix')


def test_complete_transition_paths_terminal_labels_and_last_complete_frame(tmp_path, monkeypatch):
    graph = static(tmp_path)
    observer, api = PacketObserver(graph), FakeAPI(graph)
    assert observer.last_frame is None
    api.frame(0, {'v': packet('a0', 99)}, {'a0': .05})
    observer.capture(api, 0, 'prefix')
    api.frame(1, {'v': packet('b0', 8)}, {'b0': .05})
    crossed = observer.capture(api, 1, 'prefix')
    event = next(row for row in crossed['events'] if row['kind'] == 'lane_transition')
    assert event['transition_type'] == 'connection'
    assert event['observed_lane_path'] == ['a0', ':i_0', 'b0']
    assert crossed['packets']['v']['body_trail'] == ['b0']
    api.frame(2, {}, {})
    monkeypatch.setattr(api.simulation, 'getArrivedIDList', lambda: ['v'])
    terminal = observer.capture(api, 2, 'prefix')
    assert terminal['events'] == [dict(kind='terminal_exit', vehicle_id='v',
                                       last_lane_id='b0', last_position_m=8.)]
    assert not terminal['unresolved']
    monkeypatch.setattr(api.simulation, 'getArrivedIDList', lambda: [])
    api.frame(3, {'u': packet('a0', 30)}, {'a0': .05}, departed=['u'])
    observer.capture(api, 3, 'prefix')
    api.frame(4, {'u': packet('a1', 31)}, {'a1': .05})
    lateral = observer.capture(api, 4, 'prefix')
    event = next(row for row in lateral['events'] if row['kind'] == 'lane_transition')
    assert event['transition_type'] == 'lane_change'
    assert event['observed_lane_path'] == ['a0', 'a1']
    api.frame(5, {}, {})
    missing = observer.capture(api, 5, 'prefix')
    assert missing['events'][0]['kind'] == 'unaccounted_disappearance'
    assert missing['unresolved'][0]['reason'] == 'unaccounted_disappearance'
    assert observer.last_frame is missing

    def fail_getter():
        raise RuntimeError('native observation failed')
    monkeypatch.setattr(api.vehicle, 'getIDList', fail_getter)
    with pytest.raises(RuntimeError, match='native observation failed'):
        observer.capture(api, 6, 'prefix')
    assert observer.last_frame is missing


def test_recorded_jinan_crossing_then_lanechange_remaps_internal_tail():
    # First paid pilot, vehicle963: t643 internal21_1 -> t644 receiving lane0.
    # Native moves first, then changes lanes and remaps its still-internal rear.
    internal0, internal1 = ':intersection_4_1_21_0', ':intersection_4_1_21_1'
    road0, road1 = 'road_4_1_1_0', 'road_4_1_1_1'
    geometry = {lane: dict(edge_id=edge, length_m=length, speed_limit_mps=speed,
                    has_lane_changer=True, internal=internal, index=index)
        for lane, edge, length, speed, internal, index in [
            (internal0, ':intersection_4_1_21', 27.37, 11.11, True, 0),
            (internal1, ':intersection_4_1_21', 27.37, 11.11, True, 1),
            (road0, 'road_4_1_1', 800., 11.11, False, 0),
            (road1, 'road_4_1_1', 800., 11.11, False, 1)]}
    graph = dict(lanes=geometry, successors={internal0: [road0], internal1: [road1], road0: [], road1: []},
                 predecessors={internal0: [], internal1: [], road0: [internal0], road1: [internal1]})
    observer, api = PacketObserver(graph), FakeAPI(graph)
    api.frame(643, {'v963': packet(internal1, 20.877624551513485, speed=8.633979977698276)},
              {internal1: 5/27.37})
    observer.capture(api, 643, 'prefix')
    position = 3.4989075618633017
    recorded_tail_occupancy = 0.0548444442139824
    api.frame(644, {'v963': packet(road0, position, speed=9.99128301034982)},
              {internal0: recorded_tail_occupancy, road0: position/800})
    frame = observer.capture(api, 644, 'measurement')
    assert all(frame['checks'].values())
    assert not frame['unresolved']
    assert frame['packets']['v963']['body_trail'] == [internal0, road0]
    event = next(row for row in frame['events'] if row['kind'] == 'lane_transition')
    assert event['transition_type'] == 'connection_then_lane_change'
    assert event['observed_lane_path'] == [internal1, road1, road0]
    assert frame['reconstructed_lanes'][internal0]['O'] == pytest.approx(recorded_tail_occupancy, abs=1e-16)
    assert frame['reconstructed_lanes'][internal0]['N'] == 0
