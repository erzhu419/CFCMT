"""Causal lane inputs and separately extracted ten-second state-change labels."""
from collections import defaultdict
from copy import deepcopy
from math import isfinite
from pathlib import Path
from statistics import mean
import sys
from xml.etree import ElementTree as ET

import numpy as np

SIGNALS = ('G', 'g', 's', 'O', 'Y', 'y', 'r')
PARAM_GETTERS = dict(length_m='getLength', min_gap_m='getMinGap', accel_mps2='getAccel',
    decel_mps2='getDecel', tau_s='getTau', imperfection='getImperfection',
    max_speed_mps='getMaxSpeed', speed_factor='getSpeedFactor', allowed_speed_mps='getAllowedSpeed')
TARGET_COLUMNS = ('delta_N_storage', 'delta_H_storage', 'delta_M_storage', 'delta_occupancy')


def read_lane_static(cfg, conflicts_by_tls=None):
    """Read exact controlled connections; the optional conflict map is injected.

    Without an injected map, the existing static_context reader uses SUMO XML
    and sumolib only. Neither path starts or loads a simulator.
    """
    cfg = Path(cfg)
    netfile = (cfg.parent/ET.parse(cfg).find('input/net-file').attrib['value']).resolve()
    network = ET.parse(netfile)
    lanes, lengths, limits = {}, {}, {}
    for edge in network.findall('edge'):
        for lane in edge.findall('lane'):
            key = lane.attrib['id']
            lanes[edge.attrib['id'], lane.attrib['index']] = key
            lengths[key], limits[key] = float(lane.attrib['length']), float(lane.attrib['speed'])
    connections = defaultdict(list)
    for node in network.findall('connection'):
        attributes = node.attrib
        if 'tl' in attributes:
            connections[attributes['tl']].append(dict(
                from_lane=lanes[attributes['from'], attributes['fromLane']],
                to_lane=lanes[attributes['to'], attributes['toLane']],
                link_index=int(attributes['linkIndex']), via=attributes.get('via'),
                direction=attributes.get('dir')))
    if conflicts_by_tls is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tsc_joint_response_interface_20261106'))
        from static_context import read_static_context
        conflicts_by_tls = read_static_context(cfg)['conflicts_by_tls']
    return dict(netfile=str(netfile), lane_lengths=lengths, lane_speed_limits=limits,
                connections_by_tls=dict(connections), conflicts_by_tls=deepcopy(conflicts_by_tls))


def record_vehicle_params(api, *, lane_ids=None, seen_ids=None):
    """Read current vehicles' actual type, speed and parameters before the action.

    Supply either observed lane IDs or already observed vehicle IDs. No route
    type declaration or DEFAULT_VEHTYPE assumption fills a missing observation.
    """
    if (lane_ids is None) == (seen_ids is None):
        raise ValueError('Supply either lane_ids or seen_ids for current vehicle observations')
    selected = set(lane_ids) if lane_ids is not None else None
    vehicles = api.vehicle.getIDList() if seen_ids is None else seen_ids
    records = {}
    for vid in vehicles:
        lane = api.vehicle.getLaneID(vid)
        if selected is not None and lane not in selected:
            continue
        records[vid] = dict(lane_id=lane, type_id=api.vehicle.getTypeID(vid),
                           speed_mps=float(api.vehicle.getSpeed(vid)),
                           **{name: float(getattr(api.vehicle, getter)(vid))
                              for name, getter in PARAM_GETTERS.items()})
    return records


def _state(values, lane, static):
    n, h = values['vehicle_count'], values['halted_count']
    limit = static['lane_speed_limits'][lane]
    storage = static['lane_lengths'][lane]/7.5
    mean_speed = float(values['mean_speed_mps']) if n else None
    moment = n*mean_speed/limit if n else 0.
    return dict(N=n, H=h, M=moment, occupancy=float(values['occupancy_raw']),
                storage=storage, length_m=static['lane_lengths'][lane],
                speed_limit_mps=limit, mean_speed_mps=mean_speed)


def _parameters(lane, vehicles, expected_n, vehicle_params0):
    complete, missing = [], []
    for vehicle in vehicles:
        vid = vehicle['vehicle_id']
        record = vehicle_params0.get(vid) if vehicle_params0 is not None else None
        if (record is None or record.get('lane_id') != lane or 'type_id' not in record or
                any(name not in record or not isfinite(record[name]) for name in PARAM_GETTERS)):
            missing.append(vid)
        else:
            complete.append(record)
    observed = len(complete)
    reason = ('empty_lane_no_parameter_samples' if not expected_n else
              'vehicle_parameters_not_recorded' if vehicle_params0 is None else
              'complete' if observed == expected_n else 'incomplete_current_vehicle_parameters')
    return dict(status='complete' if expected_n and observed == expected_n else 'missing',
                reason=reason, observed_vehicle_count=observed, expected_vehicle_count=expected_n,
                missing_vehicle_ids=missing,
                type_ids=sorted({record['type_id'] for record in complete}),
                mean={name: mean(record[name] for record in complete)
                      for name in PARAM_GETTERS} if observed else None)


def build_inputs(endpoint0, frame0, catalog, static, nominal_states, focal_tls, vehicle_params0=None):
    """Build a lane/connection interface using only the two t0 observations.

    All IDs and link indices remain join keys. A caller may enrich frame0 with
    speed_mps from the same-time physical_start snapshot only after exact
    vehicle-ID/lane/position matching (lane_position equals position_m). Missing
    vehicle speeds remain missing; a lane mean never substitutes for them.
    """
    if endpoint0['offset_sec'] != 0 or frame0['offset_sec'] != 0 or endpoint0['time_sec'] != frame0['time_sec']:
        raise ValueError('Lane inputs require matching preaction observations at offset zero')
    movements = {key: value for key, value in catalog.items() if value['tls_id'] == focal_tls}
    incoming = sorted({lane for movement in movements.values() for lane in movement['incoming_lanes']})
    connections = static['connections_by_tls'][focal_tls]
    links = sorted({row['link_index'] for row in connections})
    states = list(nominal_states[focal_tls])
    if (len(states) != 10 or any(set(state)-set(SIGNALS) for state in states) or
            any(link >= len(state) for state in states for link in links)):
        raise ValueError('The nominal joint plan must preserve every focal link for ten seconds')
    lane_links = {lane: sorted({row['link_index'] for row in connections if row['from_lane'] == lane})
                  for lane in incoming}
    receiving = sorted({row['to_lane'] for row in connections if row['from_lane'] in incoming})
    receiving_nodes = {lane: _state(endpoint0['lanes'][lane], lane, static) for lane in receiving}
    internal = sorted({lane for movement in movements.values() for lane in movement['internal_lanes']})
    internal_nodes = {lane: _state(endpoint0['lanes'][lane], lane, static) for lane in internal}
    nodes = []
    for lane in incoming:
        values = _state(endpoint0['lanes'][lane], lane, static)
        limit, length = values['speed_limit_mps'], values['length_m']
        vehicles = [dict(vehicle_id=vid, distance_m=length-float(row['position_m']),
                         speed_mps=float(row['speed_mps']) if row.get('speed_mps') is not None else None)
                    for vid, row in frame0['vehicles'].items() if row['lane_id'] == lane]
        vehicles.sort(key=lambda row: (row['distance_m'], row['vehicle_id']))
        heads = [dict(present=index < len(vehicles),
            distance_ratio=vehicles[index]['distance_m']/(10*limit) if index < len(vehicles) else None,
            speed_ratio=vehicles[index]['speed_mps']/limit
                if index < len(vehicles) and vehicles[index]['speed_mps'] is not None else None)
            for index in range(2)]
        own = lane_links[lane]
        fractions = [[sum(state[link] == signal for link in own)/len(own) if own else None
                      for signal in SIGNALS] for state in states]
        nodes.append(dict(lane_id=lane, **values, vehicles=vehicles, heads=heads,
            to_lanes=sorted({row['to_lane'] for row in connections if row['from_lane'] == lane}),
            near_count=sum(vehicle['distance_m'] <= 10*limit for vehicle in vehicles),
            spatial_complete=len(vehicles) == values['N'] and all(v['speed_mps'] is not None for v in vehicles),
            parameters=_parameters(lane, vehicles, values['N'], vehicle_params0),
            nominal_signal_fractions=fractions))
    conflicts = static['conflicts_by_tls'].get(focal_tls, {})
    shared = sorted({(a['link_index'], b['link_index'], a['to_lane'])
                     for a in connections for b in connections
                     if a['from_lane'] != b['from_lane'] and a['to_lane'] == b['to_lane']})
    relations = dict(foes=deepcopy(conflicts.get('foes', [])),
                     yields_to=deepcopy(conflicts.get('response', [])),
                     shared_to_lane=[dict(from_link=a, to_link=b, receiving_lane=lane) for a, b, lane in shared])
    relation_complete = bool(conflicts.get('full_context_complete')) and all(lane_links.values()) and all(
        a in links and b in links for name in ('foes', 'yields_to') for a, b in relations[name])
    expected = sum(node['N'] for node in nodes)
    observed = sum(node['parameters']['observed_vehicle_count'] for node in nodes)
    parameters_complete = vehicle_params0 is not None and expected == observed
    parameter_status = dict(status='complete_for_current_vehicles' if parameters_complete else 'missing',
        reason='all_current_vehicles_observed' if parameters_complete else
               'vehicle_params0_not_recorded' if vehicle_params0 is None else 'incomplete_current_vehicle_parameters',
        expected_vehicle_count=expected, observed_vehicle_count=observed,
        empty_lanes_without_parameter_samples=sum(node['N'] == 0 for node in nodes))
    return dict(lane_nodes=nodes, receiving_nodes=receiving_nodes, internal_nodes=internal_nodes,
        internal_vehicle_count=sum(node['N'] for node in internal_nodes.values()),
        lane_to_links=lane_links, connections=deepcopy(connections),
        movements=deepcopy(movements), relations=relations, relation_context_complete=bool(relation_complete),
        full_nominal_states=states, signal_order=list(SIGNALS),
        nominal_link_sequences={str(link): ''.join(state[link] for state in states) for link in links},
        executor0=deepcopy(endpoint0['executors'][focal_tls]), param_observation_status=parameter_status,
        can_full_fit=bool(relation_complete and parameters_complete and all(node['spatial_complete'] for node in nodes)),
        identifier_semantics='lane/vehicle/movement IDs and TLS link indices are local join keys, never numeric features')


def extract_targets(endpoint0, endpoint10, lane_ids, static):
    """Extract four response labels; no future endpoint enters build_inputs()."""
    if (endpoint0['offset_sec'] != 0 or endpoint10['offset_sec'] != 10 or
            endpoint10['time_sec']-endpoint0['time_sec'] != 10):
        raise ValueError('Lane targets require the original complete zero-to-ten-second endpoints')
    result = []
    for lane in lane_ids:
        start, end = (_state(endpoint['lanes'][lane], lane, static) for endpoint in (endpoint0, endpoint10))
        storage = start['storage']
        result.append([(end[key]-start[key])/storage for key in ('N', 'H', 'M')]+
                      [end['occupancy']-start['occupancy']])
    return dict(lane_ids=list(lane_ids), columns=list(TARGET_COLUMNS), y=np.asarray(result, dtype=float).reshape(-1, 4))
