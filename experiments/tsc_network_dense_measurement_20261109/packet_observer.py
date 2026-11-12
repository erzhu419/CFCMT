"""Causal one-second packets and SUMO1.22 native lane reconstruction.

No simulator is started here and no route/next-link getter is used. A sink is
called only for measurement frames; prefix captures retain current body trails
in RAM. The caller owns gzip JSONL persistence and executor measurement.

body_trail is SUMO's current partial-occupancy accounting path after lateral
remapping; it is not the history of lanes actually visited by the front. The
separate transition observed_lane_path is inferred from past/current front
observations, unique static connections and native move-before-lanechange order.
"""
from collections import defaultdict
from math import isclose
from pathlib import Path
from xml.etree import ElementTree as ET


PARAM_GETTERS = dict(length_m='getLength', min_gap_m='getMinGap',
    accel_mps2='getAccel', decel_mps2='getDecel', tau_s='getTau',
    imperfection='getImperfection', max_speed_mps='getMaxSpeed',
    speed_factor='getSpeedFactor')
ATOL = 1e-8


def read_static(cfg):
    """Read the actual net XML, including internal continuation connections."""
    cfg = Path(cfg)
    net = cfg.parent/ET.parse(cfg).find('input/net-file').get('value')
    tree = ET.parse(net)
    lanes, lane_key, successors, predecessors = {}, {}, defaultdict(set), defaultdict(set)
    for edge in tree.findall('edge'):
        members = edge.findall('lane')
        for lane in members:
            key = lane.get('id')
            lane_key[edge.get('id'), lane.get('index')] = key
            lanes[key] = dict(edge_id=edge.get('id'), length_m=float(lane.get('length')),
                speed_limit_mps=float(lane.get('speed')), internal=edge.get('function') == 'internal',
                has_lane_changer=len(members) > 1, index=int(lane.get('index')),
                width_m=float(lane.get('width')) if lane.get('width') is not None else None,
                shape=lane.get('shape'))
    connections, controlled, lane_to_tls = [], set(), defaultdict(set)
    for connection in tree.findall('connection'):
        row = connection.attrib
        origin = lane_key[row['from'], row['fromLane']]
        destination = lane_key[row['to'], row['toLane']]
        via = row.get('via')
        following = via or destination
        successors[origin].add(following)
        predecessors[following].add(origin)
        item = dict(from_lane=origin, to_lane=destination, via=via, direction=row.get('dir'),
                    tls_id=row.get('tl'), link_index=int(row['linkIndex']) if 'linkIndex' in row else None)
        connections.append(item)
        if item['tls_id'] is not None:
            controlled.add(origin)
            lane_to_tls[origin].add(item['tls_id'])
            if via:
                lane_to_tls[via].add(item['tls_id'])
    for row in connections:
        if row['tls_id'] is None or not row['via']:
            continue
        pending, seen = [row['via']], set()
        while pending:
            lane = pending.pop()
            if lane in seen or not lanes[lane]['internal']:
                continue
            seen.add(lane)
            lane_to_tls[lane].add(row['tls_id'])
            pending.extend(successors[lane])
    return dict(netfile=str(net.resolve()), lanes=lanes,
        successors={lane: sorted(successors[lane]) for lane in lanes},
        predecessors={lane: sorted(predecessors[lane]) for lane in lanes},
        connections=connections, internal_lane_ids=sorted(lane for lane, row in lanes.items() if row['internal']),
        controlled_lane_ids=sorted(controlled), lane_to_tls={lane: sorted(values) for lane, values in lane_to_tls.items()})


class PacketObserver:
    def __init__(self, static, sink=None):
        self.static, self.sink = static, sink
        self.previous, self.previous_time = {}, None
        self.last_frame = None

    def _paths(self, origin, destination):
        # In these grids a one-second observation cannot traverse a complete
        # ordinary road. Only internal lanes can be unobserved intermediates.
        paths = []

        def visit(path):
            for lane in self.static['successors'][path[-1]]:
                if lane in path:
                    continue
                if lane == destination:
                    paths.append(path+[lane])
                elif self.static['lanes'][lane]['internal']:
                    visit(path+[lane])
        visit([origin])
        return paths

    def _lane_change_trail(self, prior_path, lane, length, position, vid, unresolved):
        path, remaining, current = [lane], length-position, lane
        for old in reversed(prior_path[:-1]):
            if remaining <= 0:
                break
            candidates = [candidate for candidate in self.static['predecessors'][current]
                          if self.static['lanes'][candidate]['edge_id'] == self.static['lanes'][old]['edge_id']]
            if len(candidates) != 1:
                unresolved.append(dict(vehicle_id=vid, reason='lane_change_tail_not_unique', from_lane=old, to_lane=current))
                return path, False
            current = candidates[0]
            path.insert(0, current)
            remaining -= self.static['lanes'][current]['length_m']
        return path, True

    def _trail(self, packet, departed, unresolved):
        vid, lane = packet['packet_id'], packet['lane_id']
        length, position = packet['parameters']['length_m'], packet['position_m']
        previous = self.previous.get(vid)
        complete = True
        observed_path, transition_type = None, None
        if previous is None:
            path = [lane]
            if position < length and vid not in departed and self.static['predecessors'][lane]:
                complete = False
                unresolved.append(dict(vehicle_id=vid, reason='initial_body_trail_unobserved', to_lane=lane))
        elif previous['lane_id'] == lane:
            path, complete = list(previous['body_trail']), previous['body_path_complete']
        elif self.static['lanes'][previous['lane_id']]['edge_id'] == self.static['lanes'][lane]['edge_id']:
            # Instantaneous same-edge lane change replaces the current lane,
            # rather than treating the previous parallel lane as longitudinal.
            observed_path, transition_type = [previous['lane_id'], lane], 'lane_change'
            path, complete = self._lane_change_trail(previous['body_trail'], lane, length, position, vid, unresolved)
            complete = complete and previous['body_path_complete']
        else:
            paths = self._paths(previous['lane_id'], lane)
            traversed = paths[0] if len(paths) == 1 else None
            observed_path = traversed
            transition_type = 'connection' if traversed is not None else 'unresolved_connection'
            # Native1.22 executes longitudinal movement before instantaneous
            # lane changing. A front can therefore cross the junction and then
            # change to the adjacent receiving lane within this one-second step.
            moved_then_changed = []
            if not paths:
                target = self.static['lanes'][lane]
                for candidate, geometry in self.static['lanes'].items():
                    if (geometry['edge_id'] == target['edge_id'] and
                            abs(geometry['index']-target['index']) == 1):
                        moved_then_changed.extend(self._paths(previous['lane_id'], candidate))
            if len(moved_then_changed) == 1:
                traversed = moved_then_changed[0]
                observed_path, transition_type = traversed+[lane], 'connection_then_lane_change'
                path, complete = self._lane_change_trail(list(previous['body_trail'])+traversed[1:],
                                                         lane, length, position, vid, unresolved)
                complete = complete and previous['body_path_complete']
            elif traversed is None:
                path, complete = [lane], False
                unresolved.append(dict(vehicle_id=vid, reason='past_connection_not_unique',
                                       from_lane=previous['lane_id'], to_lane=lane))
            else:
                path = list(previous['body_trail'])+traversed[1:]
                complete = previous['body_path_complete']
        # Once the rear is inside the current lane, unavailable older history
        # cannot affect occupancy and is discarded. Stops preserve the tail.
        if position >= length:
            path, complete = [lane], True
        else:
            keep, remaining = [lane], length-position
            for old in reversed(path[:-1]):
                if remaining <= 0:
                    break
                keep.insert(0, old)
                remaining -= self.static['lanes'][old]['length_m']
            path = keep
        packet.update(body_trail=path, body_path_complete=complete,
                      cell_index=min(int(position//15), int((self.static['lanes'][lane]['length_m']-1e-12)//15)))
        return observed_path, transition_type

    def _reconstruct(self, packets):
        by_lane = defaultdict(list)
        partial = defaultdict(float)
        for packet in packets.values():
            by_lane[packet['lane_id']].append(packet)
            remaining = packet['parameters']['length_m']-packet['position_m']
            for lane in reversed(packet['body_trail'][:-1]):
                if remaining <= 0:
                    break
                # Native getFractionalVehicleLength adds L-backPosition, not
                # clipped overlap. Short internal lanes can therefore exceed1.
                partial[lane] += remaining
                remaining -= self.static['lanes'][lane]['length_m']
        result = {}
        for lane, geometry in self.static['lanes'].items():
            vehicles = sorted(by_lane[lane], key=lambda row: (row['position_m'], row['packet_id']))
            occupied = sum(row['parameters']['length_m'] for row in vehicles)+partial[lane]
            if vehicles:
                upstream = vehicles[0]
                occupied -= max(upstream['parameters']['length_m']-upstream['position_m'], 0.)
            speed_rows = [row for row in vehicles if not row['is_stopped'] or not geometry['has_lane_changer']]
            result[lane] = dict(ids=sorted(row['packet_id'] for row in vehicles), N=len(vehicles),
                H=sum(row['speed_mps'] < .1 for row in vehicles), O=occupied/geometry['length_m'],
                mean_speed_mps=sum(row['speed_mps'] for row in speed_rows)/len(speed_rows)
                               if speed_rows else geometry['speed_limit_mps'])
        return result

    def capture(self, api, time_s, role):
        """Capture one native observation; does not request or advance SUMO."""
        time_s = float(time_s)
        if self.previous_time is not None and time_s != self.previous_time+1:
            raise ValueError('Packet observations require consecutive one-second captures')
        simulation, vehicle = api.simulation, api.vehicle
        active = sorted(vehicle.getIDList())
        departed = sorted(simulation.getDepartedIDList())
        arrived = sorted(simulation.getArrivedIDList())
        packets, events, unresolved = {}, [], []
        for vid in active:
            packet = dict(packet_id=vid, lane_id=str(vehicle.getLaneID(vid)),
                position_m=float(vehicle.getLanePosition(vid)), speed_mps=float(vehicle.getSpeed(vid)),
                acceleration_mps2=float(vehicle.getAcceleration(vid)),
                lateral_position_m=float(vehicle.getLateralLanePosition(vid)),
                lateral_speed_mps=float(vehicle.getLateralSpeed(vid)),
                allowed_speed_mps=float(vehicle.getAllowedSpeed(vid)), is_stopped=bool(vehicle.isStopped(vid)),
                type_id=str(vehicle.getTypeID(vid)),
                parameters={name: float(getattr(vehicle, getter)(vid)) for name, getter in PARAM_GETTERS.items()})
            observed_path, transition_type = self._trail(packet, set(departed), unresolved)
            old = self.previous.get(vid)
            if old is None:
                events.append(dict(kind='first_observed', vehicle_id=vid, lane_id=packet['lane_id']))
            else:
                if packet['lane_id'] != old['lane_id']:
                    events.append(dict(kind='lane_transition', vehicle_id=vid, from_lane=old['lane_id'],
                        to_lane=packet['lane_id'], transition_type=transition_type, observed_lane_path=observed_path))
                if packet['parameters'] != old['parameters'] or packet['type_id'] != old['type_id']:
                    events.append(dict(kind='parameter_change', vehicle_id=vid))
                if (packet['lane_id'], packet['cell_index']) != (old['lane_id'], old['cell_index']):
                    events.append(dict(kind='cell_transition', vehicle_id=vid,
                        from_cell=[old['lane_id'], old['cell_index']], to_cell=[packet['lane_id'], packet['cell_index']]))
            packets[vid] = packet
        for vid in sorted(set(self.previous)-set(active)):
            old = self.previous[vid]
            kind = 'terminal_exit' if vid in arrived else 'unaccounted_disappearance'
            events.append(dict(kind=kind, vehicle_id=vid, last_lane_id=old['lane_id'],
                               last_position_m=old['position_m']))
            if kind == 'unaccounted_disappearance':
                unresolved.append(dict(vehicle_id=vid, reason=kind, from_lane=old['lane_id']))
        native = {lane: dict(ids=sorted(api.lane.getLastStepVehicleIDs(lane)),
                    N=int(api.lane.getLastStepVehicleNumber(lane)), H=int(api.lane.getLastStepHaltingNumber(lane)),
                    O=float(api.lane.getLastStepOccupancy(lane)), mean_speed_mps=float(api.lane.getLastStepMeanSpeed(lane)))
                  for lane in self.static['lanes']}
        reconstructed = self._reconstruct(packets)
        checks = {f'{key}_exact': all(native[lane][key] == reconstructed[lane][key] for lane in native)
                  for key in ('ids', 'N', 'H')}
        checks.update({f'{key}_close': all(isclose(native[lane][key], reconstructed[lane][key],
                      rel_tol=0., abs_tol=ATOL) for lane in native) for key in ('O', 'mean_speed_mps')})
        checks.update(body_paths_complete=all(row['body_path_complete'] for row in packets.values()),
                      native_time_exact=float(simulation.getTime()) == time_s)
        frame = dict(time_s=time_s, role=role, packets=packets, native_lanes=native,
            reconstructed_lanes=reconstructed, native_active_ids=active,
            native_pending_ids=sorted(simulation.getPendingVehicles()), departed_ids=departed,
            arrived_ids=arrived,
            starting_teleport_ids=sorted(simulation.getStartingTeleportIDList()),
            ending_teleport_ids=sorted(simulation.getEndingTeleportIDList()),
            collision_ids=sorted(simulation.getCollidingVehiclesIDList()), checks=checks,
            unresolved=unresolved, events=events)
        self.previous, self.previous_time = packets, time_s
        self.last_frame = frame
        if role == 'measurement' and self.sink is not None:
            self.sink(frame)
        return frame
