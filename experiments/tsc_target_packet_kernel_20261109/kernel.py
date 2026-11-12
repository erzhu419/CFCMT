"""One-second packet propagation with own-state PP and the original executor.

No native simulator, observed future trajectory or learned cost head is used.
This is an approximate traffic model; synthetic closure is not calibration.
"""
from collections import defaultdict
from copy import deepcopy
from math import isfinite
from functools import cmp_to_key
from types import SimpleNamespace

from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_protocol_20261109.packet_contract import (
    bounded_speed_proposal, categorical_choice, persistent_intent, verify_identity_step,
)
from .local_model import FrozenLocalModel, masked_lane_probabilities
from .topology import RoadTopology
from .scenario_noise import ScenarioNoise


class PacketKernel:
    def __init__(self, static, initial_state, executor_seeds, target_laws,
                 coefficients=None, random_seed=0, executor_cls=None):
        self.topology = RoadTopology(static)
        self.static = deepcopy(static)
        self.state = deepcopy(initial_state)
        self.laws = deepcopy(target_laws)
        self.random_seed = random_seed
        self.model = FrozenLocalModel(coefficients)
        self.sensor_operator = PacketObserver(self.static)
        self.state.setdefault('intents', {})
        self.state.setdefault('pending', [])
        self.start_time = self.state['time_s']
        self.cost_lanes = tuple(self.static['controlled_lane_ids'])
        self.integral = 0
        self.steps = 0
        self.serial = 0
        self.seen = set(self.state['packets']) | {p['packet_id'] for p in self.state['pending']}
        self.noise = ScenarioNoise(random_seed,self.seen,self.start_time)
        self.diagnostics = defaultdict(int)
        self.signals = {}
        self.executors = {}
        executor_cls = executor_cls or SafePhaseExecutor
        api = SimpleNamespace(
            trafficlight=SimpleNamespace(setRedYellowGreenState=self._set_signal),
            lane=SimpleNamespace(getLastStepVehicleNumber=lambda lane: self.state['lane_sensors'][lane]['N']))
        for tls, seed in executor_seeds.items():
            snapshot = dict(seed['snapshot'], **self.state['executors'][tls], tls_id=tls)
            timings = [PhaseTiming(**row) for row in seed['timings']]
            executor = executor_cls(sumo_api=api, tls_id=tls, timings=timings,
                                    initial_state=snapshot['current_state'],
                                    clearance_lane_ids=seed['clearance_lane_ids'],
                                    clearance_poll_sec=seed['clearance_poll_sec'])
            executor.restore(snapshot)
            self.executors[tls] = executor
        if set(self.state['executors']) != set(self.executors):
            raise ValueError('Every observed executor needs its static configuration')
        for packet in self.state['packets'].values():
            self._initialize_internal(packet)
        self._prepare_pending()
        self._validate_state()

    def _set_signal(self, tls, value):
        self.signals[tls] = value

    def _initialize_internal(self, packet):
        if not self.topology.lanes[packet['lane_id']]['internal']:
            return
        candidates = list(self.topology.internal_connections[packet['lane_id']])
        # Past body lanes may identify the entry. A future route never does.
        trail = packet['body_trail']
        compatible = [c for c in candidates if all(lane in c['path'] for lane in trail)]
        if compatible:
            candidates = compatible
        if not candidates:
            raise ValueError('An internal packet needs a legal committed movement')
        # Shared internal paths retain a latent path sample instead of reading a route.
        connection = candidates[int(self.noise.intent_uniform(packet['packet_id'])*len(candidates))] if len(candidates) > 1 else candidates[0]
        packet['_movement_key'] = connection['key']

    def _intent(self, packet):
        lane = packet['lane_id']
        edge = self.topology.lanes[lane]['edge_id']
        if self.topology.lanes[lane]['internal']:
            return self.topology.by_key[packet['_movement_key']]['to_edge']
        probabilities = self.laws['intent_probs'].get(edge)
        if probabilities is None:
            if any(self.topology.connections_from(k) for k in self.topology.road_lanes(edge)):
                raise ValueError(f'Missing target-history intent law for {edge}')
            probabilities = {'END': 1.}
        prior = self.state['intents'].get(packet['packet_id'])
        uniform = self.noise.intent_uniform(packet['packet_id']) if prior is None or prior['road_id'] != edge else 0.
        return persistent_intent(self.state['intents'], packet['packet_id'], edge, probabilities, uniform)

    def _connection(self, packet):
        lane = packet['lane_id']
        if self.topology.lanes[lane]['internal']:
            return self.topology.by_key[packet['_movement_key']], True
        intent = self._intent(packet)
        if intent == 'END':
            return None, True
        options = self.topology.compatible(lane, intent)
        if options:
            return options[0], True
        target = []
        current_index = self.topology.lanes[lane]['index']
        for index, sibling in enumerate(self.topology.road_lanes(self.topology.lanes[lane]['edge_id'])):
            matches = self.topology.compatible(sibling, intent)
            if matches:
                target.append((abs(self.topology.lanes[sibling]['index']-current_index), index, matches[0]))
        if not target:
            raise ValueError('An intent must have a reachable static connection or END')
        return min(target, key=lambda row: row[:2])[2], False

    def _signal(self, connection, compatible=True):
        if not compatible:
            return 'r'
        if connection is None or connection['tls_id'] is None:
            return 'uncontrolled'
        return self.signals[connection['tls_id']][connection['link_index']]

    def _road_goal_distance(self, lane, intent):
        if intent == 'END':
            return 0
        current = self.topology.lanes[lane]
        choices = [abs(self.topology.lanes[k]['index']-current['index'])
                   for k in self.topology.road_lanes(current['edge_id'])
                   if self.topology.compatible(k, intent)]
        return min(choices) if choices else float('inf')

    def _lateral_legal(self, packet, target):
        if target is None or self.topology.lanes[target]['internal']:
            return False
        intent = self._intent(packet)
        return self._road_goal_distance(target, intent) <= self._road_goal_distance(packet['lane_id'], intent)

    def _allowed_speed(self, packet):
        profile = packet['parameters']
        return min(profile['max_speed_mps'],
                   self.topology.lanes[packet['lane_id']]['speed_limit_mps']*profile['speed_factor'])

    def _path(self, packet, connection, compatible):
        lane = packet['lane_id']
        if connection is None or not compatible:
            return (lane,)
        path = connection['path']
        return path[path.index(lane):]

    def _leader(self, packet, packets, path):
        progress, offset = {}, -packet['position_m']
        for lane in path:
            progress[lane] = offset
            offset += self.topology.lanes[lane]['length_m']
        candidates = []
        for other in packets.values():
            if other['packet_id'] == packet['packet_id'] or other['lane_id'] not in progress:
                continue
            front = progress[other['lane_id']]+other['position_m']
            if front > 1e-12:
                gap = max(front-other['parameters']['length_m']-packet['parameters']['min_gap_m'], 0.)
                candidates.append((gap, other))
        return min(candidates, key=lambda row: row[0]) if candidates else (None, None)

    def _side_gaps(self, packet, lane, packets):
        if lane is None:
            return 0., 0., 0.
        length = self.topology.lanes[lane]['length_m']
        x = packet['position_m']
        ahead, behind = [], []
        for other in packets.values():
            if other['packet_id'] == packet['packet_id'] or other['lane_id'] != lane:
                continue
            if other['position_m'] >= x:
                ahead.append(other['position_m']-other['parameters']['length_m']-x)
            else:
                behind.append((x-packet['parameters']['length_m']-other['position_m'], other['speed_mps']))
        front = min(ahead) if ahead else length-x
        rear, rear_speed = min(behind, key=lambda row: row[0]) if behind else (x, 0.)
        return front, rear, rear_speed

    def _receiving_space(self, connection, packets):
        if connection is None:
            return 0.
        lane = connection['to_lane']
        rear_positions = [p['position_m']-p['parameters']['length_m']
                          for p in packets.values() if p['lane_id'] == lane]
        return max(0., min(rear_positions)) if rear_positions else self.topology.lanes[lane]['length_m']

    def _foe_context(self, connection, packets):
        if connection is None:
            return 0., 0., 0.
        times, internal_count = [], 0
        for other in packets.values():
            if other['packet_id'] not in self._connections:
                continue
            candidate, legal = self._connections[other['packet_id']]
            if candidate is None or not legal or not self.topology.foes(connection, candidate):
                continue
            if self.topology.lanes[other['lane_id']]['internal']:
                internal_count += 1
                times.append(0.)
            elif self._gives_way(connection,self._signal(connection),candidate,self._signal(candidate)):
                distance = self.topology.lanes[other['lane_id']]['length_m']-other['position_m']
                times.append(distance/max(other['speed_mps']+other['parameters']['accel_mps2'], .1))
        return min(times) if times else 0., float(bool(times)), internal_count

    def _gives_way(self, first, first_signal, second, second_signal):
        """Use directed right of way before the signal-tier/FIFO tie rule."""
        if not self.topology.foes(first,second):
            return False
        forward=self.topology.yields_to(first,second)
        reverse=self.topology.yields_to(second,first)
        if forward!=reverse:
            return forward
        return first_signal=='g' and second_signal=='G'

    def _speed(self, packet, decision, packets, connection, compatible, travel=None):
        current = dict(packet, allowed_speed_mps=self._allowed_speed(packet))
        path = self._path(packet, connection, compatible)
        gap, leader = self._leader(packet, packets, path)
        kwargs = dict(net_gap_m=gap, admissible_travel_m=travel)
        if leader:
            kwargs.update(leader_speed_mps=leader['speed_mps'],
                          leader_decel_mps2=leader['parameters']['decel_mps2'])
        signal = self._signal(connection, compatible)
        ordinary = not self.topology.lanes[packet['lane_id']]['internal']
        distance = self.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
        blocked = not compatible or signal == 'r'
        if signal in ('y', 'Y') and ordinary:
            blocked = packet['speed_mps']**2/(2*packet['parameters']['decel_mps2']) <= distance
        if ordinary and connection is not None and blocked:
            stop_speed = bounded_speed_proposal(current, decision['motion_residual_mps'],
                                                net_gap_m=max(0., distance),
                                                admissible_travel_m=distance)
        else:
            stop_speed = float('inf')
        return min(stop_speed, bounded_speed_proposal(current, decision['motion_residual_mps'], **kwargs))

    def _context(self, packet, packets, connection, compatible):
        geometry = self.topology.lanes[packet['lane_id']]
        params = packet['parameters']
        path = self._path(packet, connection, compatible)
        gap, leader = self._leader(packet, packets, path)
        signal = self._signal(connection, compatible)
        tta, foe_present, foe_count = self._foe_context(connection, packets)
        context = dict(bias=1., speed_mps=packet['speed_mps'],
                       acceleration_mps2=packet['acceleration_mps2'],
                       allowed_speed_mps=self._allowed_speed(packet),
                       **{key: params[key] for key in ('accel_mps2','decel_mps2','tau_s','min_gap_m','length_m','imperfection','speed_factor')},
                       stopline_distance_m=geometry['length_m']-packet['position_m'],
                       net_leader_gap_m=gap if leader else geometry['length_m']-packet['position_m'],
                       leader_speed_mps=leader['speed_mps'] if leader else 0.,
                       leader_present=float(leader is not None), internal_lane=float(geometry['internal']),
                       foe_min_TTA_s=tta, foe_present=foe_present, foe_internal_count=foe_count,
                       receiving_free_space_m=self._receiving_space(connection, packets),
                       lateral_position_m=packet['lateral_position_m'], lateral_speed_mps=packet['lateral_speed_mps'])
        for char in ('G','g','Y','y','r','s','O','uncontrolled'):
            context['signal_'+char] = float(signal == char)
        # Head features remain pre-step observations even when the lateral
        # action will occur on the receiving road later in this second.
        lateral_packet = packet
        base = self._speed(packet, {'motion_residual_mps': 0.}, packets, connection, compatible)
        if compatible and connection is not None:
            distance = sum(self.topology.lanes[k]['length_m'] for k in path[:-1])-packet['position_m']
            if base > distance:
                lateral_packet = dict(packet, lane_id=connection['to_lane'], position_m=base-distance)
        for name, delta in (('left', 1), ('right', -1)):
            target = self.topology.adjacent(lateral_packet['lane_id'], delta)
            front, rear, rear_speed = self._side_gaps(lateral_packet, target, packets)
            context.update({name+'_front_gap_m':front, name+'_rear_gap_m':rear,
                            name+'_rear_speed_mps':rear_speed})
            # Do not sample a receiving-road intent just to form a feature.
            same_road = self.topology.lanes[lateral_packet['lane_id']]['edge_id'] == geometry['edge_id']
            context[name+'_legal'] = float(target is not None and (not same_road or self._lateral_legal(packet, target)))
        return context

    def _pp_requests(self):
        requests = {}
        for tls, executor in sorted(self.executors.items()):
            first_by_link = self.topology.pp_lane_by_link[tls]
            incoming_order = self.static.get('incoming_lanes_by_tls', {}).get(tls)
            if incoming_order is None:
                incoming_order = list(dict.fromkeys(first_by_link[index] for index in sorted(first_by_link)))
            def score(phase):
                permitted = {lane for index, lane in first_by_link.items() if phase[int(index)] in 'Gg'}
                return sum(self.state['lane_sensors'][lane]['H']+.35*self.state['lane_sensors'][lane]['N']
                           +.025*self.state['lane_sensors'][lane]['O']
                           for lane in incoming_order if lane in permitted)
            requests[tls] = max(executor.feasible_states_now(), key=score)
        return requests

    def _reservation(self, packet, connection, speed):
        path = connection['path']
        internal = self.topology.lanes[packet['lane_id']]['internal']
        internal_length = sum(self.topology.lanes[lane]['length_m'] for lane in path[1:-1])
        if internal:
            start = 0.
            progress = self.topology.path_progress(connection, packet['lane_id'], packet['position_m'])-self.topology.lanes[path[0]]['length_m']
            finish = (internal_length-progress+packet['parameters']['length_m'])/max(speed,.1)
        else:
            start = (self.topology.lanes[packet['lane_id']]['length_m']-packet['position_m'])/max(speed,.1)
            finish = start+(internal_length+packet['parameters']['length_m'])/max(speed,.1)
        return (start, max(start,finish))

    def _conflict(self, connection, interval, reservations):
        for other, reserved in reservations:
            if self.topology.foes(connection, other) and max(interval[0],reserved[0]) < min(interval[1],reserved[1]):
                return True
        return False

    def _trim_body(self, packet, trail):
        lane = packet['lane_id']
        remaining = packet['parameters']['length_m']-packet['position_m']
        keep = [lane]
        for prior in reversed(trail[:-1]):
            if remaining <= 0:
                break
            keep.insert(0,prior)
            remaining -= self.topology.lanes[prior]['length_m']
        if remaining > 1e-9 and len(keep)>1:
            raise ValueError('Predicted body trail does not cover the rear')
        packet.update(body_trail=keep, body_path_complete=True,
                      cell_index=min(int(packet['position_m']//15),int((self.topology.lanes[lane]['length_m']-1e-12)//15)))

    def _walk(self, packet, distance, connection, compatible):
        origin = packet['lane_id']
        trace = [origin]
        remaining = distance
        path = self._path(packet, connection, compatible)
        index = 0
        while remaining > 1e-12:
            lane = packet['lane_id']
            rest = self.topology.lanes[lane]['length_m']-packet['position_m']
            if remaining < rest-1e-12:
                packet['position_m'] += remaining
                remaining = 0.
                break
            remaining -= max(rest,0.)
            packet['position_m'] += max(rest,0.)
            if not self.topology.lanes[lane]['internal'] and self._intent(packet) == 'END':
                return trace, True, distance-remaining
            if index+1 >= len(path):
                # Real benchmark roads exceed a one-second travel distance.
                # At a short receiving road, stop at its end until its own
                # next-step request/connection can be evaluated.
                break
            index += 1
            packet['lane_id'], packet['position_m'] = path[index], 0.
            trace.append(path[index])
            if self.topology.lanes[path[index]]['internal']:
                packet['_movement_key'] = connection['key']
            else:
                packet.pop('_movement_key',None)
            if remaining <= 1e-12:
                break
        return trace, False, distance-remaining

    def _move(self, old, decisions, speeds):
        working = deepcopy(old)
        reservations = {}
        approaches = {}
        for key, packet in old.items():
            connection, compatible = self._connections[key]
            if connection is None or not compatible:
                continue
            interval = self._reservation(packet,connection,speeds[key])
            if self.topology.lanes[packet['lane_id']]['internal']:
                reservations[key]=(connection,interval)
            elif self._signal(connection) in ('G','g') and speeds[key]>0:
                approaches[key]=(connection,interval)
        def order(key):
            packet = old[key]
            connection,compatible = self._connections[key]
            internal = self.topology.lanes[packet['lane_id']]['internal']
            signal = self._signal(connection,compatible)
            rank = 0 if internal else 1 if signal=='G' else 2
            distance = self.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
            return rank, distance/max(speeds[key],.1), self._packet_order[key]
        def compare(first,second):
            a,b=old[first],old[second]
            ai=self.topology.lanes[a['lane_id']]['internal']
            bi=self.topology.lanes[b['lane_id']]['internal']
            if ai!=bi:
                return -1 if ai else 1
            if a['lane_id']==b['lane_id'] and a['position_m']!=b['position_m']:
                return -1 if a['position_m']>b['position_m'] else 1
            ac,al=self._connections[first];bc,bl=self._connections[second]
            if ac is not None and bc is not None and al and bl:
                asig,bsig=self._signal(ac),self._signal(bc)
                if self._gives_way(ac,asig,bc,bsig):
                    return 1
                if self._gives_way(bc,bsig,ac,asig):
                    return -1
            return -1 if order(first)<order(second) else 1 if order(first)>order(second) else 0
        ordered=sorted(old,key=cmp_to_key(compare))
        priority_index={key:index for index,key in enumerate(ordered)}
        exited, events = [], []
        for key in ordered:
            packet = working[key]
            original = old[key]
            connection,compatible = self._connections[key]
            internal = self.topology.lanes[packet['lane_id']]['internal']
            signal = self._signal(connection,compatible)
            distance_to_entry = self.topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
            travel = speeds[key]
            max_travel = None
            if connection is not None and compatible and not internal and travel>0 and travel>=distance_to_entry:
                interval = self._reservation(original,connection,travel)
                blockers = list(reservations.values())
                blockers.extend((c,r) for other,(c,r) in approaches.items() if other!=key
                    and priority_index[other]<priority_index[key]
                    and self._gives_way(connection,signal,c,self._signal(c)))
                conflict = self._conflict(connection,interval,blockers)
                receiving = self._receiving_space(connection,working)
                blocked = receiving < packet['parameters']['length_m']+packet['parameters']['min_gap_m']
                if conflict or blocked or (signal=='g' and self._decision_uniforms[key]['yield']>=decisions[key]['yield_probability']):
                    max_travel = max(distance_to_entry-1e-9,0.)
                    self.diagnostics['junction_blocked_steps'] += 1
            path = self._path(original,connection,compatible)
            # Front-to-back resolved packets cannot be overlapped by a later
            # merge or follower even though every proposal used the old state.
            progress,offset = {}, -original['position_m']
            for lane in path:
                progress[lane]=offset
                offset += self.topology.lanes[lane]['length_m']
            for otherkey,other in working.items():
                if otherkey==key or other['lane_id'] not in progress:
                    continue
                front = progress[other['lane_id']]+other['position_m']
                if front>1e-12:
                    bound=max(0.,front-other['parameters']['length_m']-packet['parameters']['min_gap_m'])
                    max_travel=bound if max_travel is None else min(max_travel,bound)
            if max_travel is not None:
                travel=min(travel,max_travel)
            if travel < original['speed_mps']-packet['parameters']['decel_mps2']:
                self.diagnostics['stronger_than_nominal_braking_steps'] += 1
            trace,ended,actual = self._walk(packet,travel,connection,compatible)
            if connection is not None and compatible:
                # Receiving/merge resolution can slow a committed internal
                # vehicle. Its old optimistic clearing time must not survive.
                updated_interval=self._reservation(original,connection,actual)
                if internal or len(trace)>1:
                    reservations[key]=(connection,updated_interval)
                if key in approaches:
                    approaches[key]=(connection,updated_interval)
            if ended:
                exited.append(key)
                events.append(dict(kind='terminal_exit',vehicle_id=key,from_lane=packet['lane_id'],
                                   nonsink=bool(self.topology.connections_from(packet['lane_id']))))
                del working[key]
                self.state['intents'].pop(key,None)
                continue
            packet.update(speed_mps=actual, acceleration_mps2=actual-original['speed_mps'],
                          allowed_speed_mps=self._allowed_speed(packet))
            self._trim_body(packet,list(original['body_trail'])+trace[1:])
            if len(trace)>1:
                events.append(dict(kind='lane_transition',vehicle_id=key,observed_lane_path=trace,
                                   transition_type='connection'))
        return working,exited,events

    def _lane_changes(self, packets, decisions, events):
        for key in sorted(packets,key=lambda k:self._packet_order[k]):
            packet = packets[key]
            if self.topology.lanes[packet['lane_id']]['internal']:
                continue
            targets={'stay':packet['lane_id']}
            for choice,delta in (('left',1),('right',-1)):
                target=self.topology.adjacent(packet['lane_id'],delta)
                if not self._lateral_legal(packet,target):
                    continue
                if packet['position_m']>self.topology.lanes[target]['length_m']:
                    continue
                front,rear,_=self._side_gaps(packet,target,packets)
                rear_vehicles=[p for p in packets.values() if p['lane_id']==target and p['position_m']<packet['position_m']]
                front_vehicles=[p for p in packets.values() if p['lane_id']==target and p['position_m']>=packet['position_m']]
                nearest_rear=max(rear_vehicles,key=lambda p:p['position_m']) if rear_vehicles else None
                needed_rear=nearest_rear['parameters']['min_gap_m'] if nearest_rear else 0.
                if (front_vehicles and front<packet['parameters']['min_gap_m']) or (rear_vehicles and rear<needed_rear):
                    continue
                targets[choice]=target
            probabilities=masked_lane_probabilities(decisions[key]['lane_logits'],
                {choice:choice in targets for choice in ('left','stay','right')})
            chosen=categorical_choice(probabilities,self._decision_uniforms[key]['lane'])
            if chosen=='stay':
                continue
            target=targets[chosen]
            old_lane = packet['lane_id']
            unresolved=[]
            trail,complete = self.sensor_operator._lane_change_trail(
                packet['body_trail'],target,packet['parameters']['length_m'],packet['position_m'],key,unresolved)
            if not complete or unresolved:
                raise ValueError(f'Predicted lane-change body trail is unresolved: {unresolved}')
            packet['lane_id']=target
            packet['lateral_position_m']=packet['lateral_speed_mps']=0.
            packet['allowed_speed_mps']=self._allowed_speed(packet)
            self._trim_body(packet,trail)
            prior = next((event for event in reversed(events) if event.get('vehicle_id')==key
                          and event['kind']=='lane_transition'),None)
            if prior is not None:
                prior['observed_lane_path'].append(target)
                prior['transition_type']='connection_then_lane_change'
            else:
                events.append(dict(kind='lane_transition',vehicle_id=key,
                    observed_lane_path=[old_lane,target],transition_type='lane_change'))

    def _profile(self):
        profiles=self.laws['profiles']
        if not profiles:
            raise ValueError('Boundary packets require a target-history profile law')
        return deepcopy(profiles[self.noise.profile_index(len(profiles))])

    def _prepare_pending(self):
        for packet in self.state['pending']:
            if packet.get('entry_lane_id') is None:
                packet['entry_lane_id']=categorical_choice(self.laws['entry_probs'],self.noise.entry_uniform())
            if 'profile' not in packet:
                packet['profile']=self._profile()

    def _boundary(self,packets):
        rate=self.laws['due_rate_per_sec']
        if rate<0:
            raise ValueError('Boundary intensity must be nonnegative')
        count=self.noise.poisson_due(rate)
        due=[]
        for _ in range(count):
            key=f'model:{self.random_seed}:{self.serial}'
            self.serial+=1
            if key in self.seen:
                raise ValueError('Generated boundary identity already exists')
            self.seen.add(key)
            self.noise.register(key,self.state['time_s'])
            due.append(key)
            self.state['pending'].append(dict(packet_id=key,entry_lane_id=None))
        self._prepare_pending()
        waiting,inserted=[],[]
        for queued in self.state['pending']:
            lane=queued['entry_lane_id']
            profile=queued['profile']
            params=profile['parameters']
            geometry=self.topology.lanes[lane]
            occupied=[p['position_m']-p['parameters']['length_m'] for p in packets.values() if p['lane_id']==lane]
            available=min(occupied) if occupied else geometry['length_m']
            if available<params['length_m']+params['min_gap_m']:
                waiting.append(queued)
                continue
            key=queued['packet_id']
            allowed=min(params['max_speed_mps'],geometry['speed_limit_mps']*params['speed_factor'])
            speed=min(profile.get('entry_speed_mps',profile.get('speed_mps',0.)),allowed)
            packet=dict(packet_id=key,lane_id=lane,position_m=params['length_m'],speed_mps=speed,
                        acceleration_mps2=0.,lateral_position_m=0.,lateral_speed_mps=0.,
                        allowed_speed_mps=allowed,is_stopped=False,type_id=profile.get('type_id','model'),
                        parameters=deepcopy(params),body_trail=[lane],body_path_complete=True)
            self._trim_body(packet,[lane])
            packets[key]=packet
            inserted.append(key)
        self.state['pending']=waiting
        return due,inserted

    def _validate_state(self):
        packets=self.state['packets']
        for key,packet in packets.items():
            geometry=self.topology.lanes[packet['lane_id']]
            if key!=packet['packet_id'] or not packet['body_path_complete']:
                raise ValueError('Packet identity/body state is not closed')
            if not all(isfinite(packet[field]) for field in ('position_m','speed_mps','acceleration_mps2')):
                raise ValueError('Nonfinite physical state')
            if not 0<=packet['position_m']<=geometry['length_m']+1e-8 or packet['speed_mps']<0:
                raise ValueError('Packet is outside its lane or moves backwards')
        sensor_ids=[key for row in self.state['lane_sensors'].values() for key in row['ids']]
        if sorted(sensor_ids)!=sorted(packets):
            raise ValueError('Front-assigned sensor membership is not a packet partition')

    def step(self,initial_requests=None):
        old=deepcopy(self.state['packets'])
        pending_before=[p['packet_id'] for p in self.state['pending']]
        now=self.state['time_s']
        if initial_requests is not None and self.steps!=0:
            raise ValueError('Only the initial focal action can override PP')
        requests=self._pp_requests() if now%10==0 else {}
        if initial_requests is not None:
            if now%10:
                raise ValueError('The initial intervention must be at a decision epoch')
            if not set(initial_requests)<=set(self.executors):
                raise ValueError('Unknown initial intervention TLS')
            requests.update(initial_requests)
        returns={tls:self.executors[tls].request(value) for tls,value in sorted(requests.items())}
        applied_signals=dict(self.signals)
        self._packet_order={key:index for index,key in enumerate(old)}
        self._decision_uniforms={key:self.noise.step_uniforms(key,now) for key in old}
        self._connections={key:self._connection(packet) for key,packet in old.items()}
        contexts={key:self._context(packet,old,*self._connections[key]) for key,packet in old.items()}
        decisions={key:self.model.predict(context) for key,context in contexts.items()}
        speeds={key:self._speed(packet,decisions[key],old,*self._connections[key]) for key,packet in old.items()}
        for key,packet in old.items():
            if packet['speed_mps']<.1 and speeds[key]>0 and self._decision_uniforms[key]['start']>=decisions[key]['start_probability']:
                speeds[key]=0.
            if decisions[key]['motion_residual_mps']>0:
                self.diagnostics['positive_motion_residual_steps']+=1
        packets,exited,events=self._move(old,decisions,speeds)
        self._lane_changes(packets,decisions,events)
        due,inserted=self._boundary(packets)
        self.state['packets']=packets
        self.state['lane_sensors']=self.sensor_operator._reconstruct(packets)
        for executor in self.executors.values():
            executor.advance(1.)
        self.state['executors']={tls:executor.snapshot() for tls,executor in self.executors.items()}
        self.state['time_s']=now+1
        ledger=verify_identity_step(list(old),pending_before,due,inserted,exited,list(packets),
                                   [p['packet_id'] for p in self.state['pending']])
        self._validate_state()
        for event in events:
            if event['kind']=='lane_transition':
                for a,b in zip(event['observed_lane_path'],event['observed_lane_path'][1:]):
                    same_road=self.topology.lanes[a]['edge_id']==self.topology.lanes[b]['edge_id']
                    if b not in self.static['successors'][a] and not(same_road and self.topology.adjacent(a,1)==b
                            or same_road and self.topology.adjacent(a,-1)==b):
                        raise ValueError('Predicted front path violates static connectivity')
        halted=sum(self.state['lane_sensors'][lane]['H'] for lane in self.cost_lanes)
        self.integral+=halted
        self.steps+=1
        return dict(time_s=now+1,requests=requests,request_returns=returns,
                    applied_signals=applied_signals,due_ids=due,inserted_ids=inserted,exited_ids=exited,
                    events=events,halted_total=halted,I_accum=self.integral,ledger_passed=ledger,
                    active_count=len(packets),pending_count=len(self.state['pending']),
                    terminal_count=len(exited),nonsink_terminal_count=sum(e.get('nonsink',False) for e in events),
                    executors=deepcopy(self.state['executors']))

    def rollout(self,horizon=450):
        start_integral=self.integral
        halted=[sum(self.state['lane_sensors'][lane]['H'] for lane in self.cost_lanes)]
        active=[len(self.state['packets'])]
        pending=[len(self.state['pending'])]
        transitions=[]
        for _ in range(horizon):
            row=self.step()
            halted.append(row['halted_total']);active.append(row['active_count']);pending.append(row['pending_count'])
            transitions.append(row)
        integral=self.integral-start_integral
        return dict(horizon_sec=horizon,completed_steps=horizon,steps=transitions,
                    I_accum=integral,ledger_passed=all(row['ledger_passed'] for row in transitions),
                    halted_total=halted,active_count=active,
                    pending_count=pending,terminal_count=sum(row['terminal_count'] for row in transitions),
                    nonsink_terminal_count=sum(row['nonsink_terminal_count'] for row in transitions),
                    I450=integral,C450=integral/(horizon*len(self.cost_lanes)) if self.cost_lanes else None,
                    diagnostics=dict(self.diagnostics),native_starts=0,real_data_fits=0,
                    parameter_origin=getattr(self.model,'parameter_origin','synthetic_test_heads'))
