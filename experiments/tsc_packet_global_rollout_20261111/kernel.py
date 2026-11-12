"""Sample the joint packet model over the complete active network each second."""

from collections import defaultdict
from copy import deepcopy

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.cache import _vectors
from experiments.tsc_packet_discrete_braking_20261110.kernel import PacketKernel as DiscreteKernel
from experiments.tsc_packet_discrete_braking_20261110.motion import resolve_forward
from experiments.tsc_packet_joint_law_definition_20261110.law import CHOICES
from experiments.tsc_packet_lateral_conditioning_20261110.order import _targets
from experiments.tsc_target_packet_kernel_20261109.local_model import _sigmoid, masked_lane_probabilities
from experiments.tsc_target_packet_kernel_20261109.scenario_noise import ScenarioNoise
from experiments.tsc_target_packet_protocol_20261109.packet_contract import (
    categorical_choice, verify_identity_step,
)


class PacketKernel(DiscreteKernel):
    """Use fitted brake/permission logits and masked sequential LC logits.

    All active actors are stochastic. Current ordinary-road intent persists
    across steps and lane changes. A new receiving road gets its own intent
    before lateral execution. The saved local likelihood conditions on a
    fixed background; applying its heads to all actors is the rollout model.
    """

    def __init__(self, static, initial_state, executor_seeds, target_laws, coefficients,
                 random_seed=0, executor_cls=None, conflict_geometry=None):
        self.coefficients = _vectors(coefficients).copy()
        super().__init__(static, initial_state, executor_seeds, target_laws,
                         random_seed=random_seed, executor_cls=executor_cls,
                         conflict_geometry=conflict_geometry)
        self.model.parameter_origin = 'frozen_joint_packet_coefficients'
        # Keep inherited intent and exogenous demand draws. Give brake its own
        # packet/time stream so it cannot perturb start, yield or LC draws.
        self._brake_noise = ScenarioNoise(f'{random_seed}:brake', self.seen, self.start_time)

    def _step_uniforms(self, packet_id, time_s):
        draws = self.noise.step_uniforms(packet_id, time_s)
        draws['brake'] = self._brake_noise.step_uniforms(packet_id, time_s)['start']
        return draws

    def _boundary(self, packets):
        due, inserted = super()._boundary(packets)
        for packet_id in due:
            self._brake_noise.register(packet_id, self.state['time_s'])
        return due, inserted

    def _features(self, packets):
        """Canonical prestate features on the lanes and movements they query."""
        by_lane, by_tls = defaultdict(list), defaultdict(list)
        for key, packet in packets.items():
            by_lane[packet['lane_id']].append(key)
            connection, compatible = self._connections[key]
            if connection is not None and compatible and connection['tls_id'] is not None:
                by_tls[connection['tls_id']].append(key)

        features = []
        for key, packet in packets.items():
            connection, compatible = self._connections[key]
            lanes = set(self._path(packet, connection, compatible))
            lateral_lanes = [packet['lane_id']]
            if connection is not None:
                lanes.add(connection['to_lane'])  # receiving-space feature
                if compatible:
                    lateral_lanes.append(connection['to_lane'])
            for lane in lateral_lanes:
                lanes.update(self.topology.adjacent(lane, delta) for delta in (-1, 1))
            candidates = {actor for lane in lanes for actor in by_lane[lane]}
            if connection is not None:
                candidates.update(by_tls[connection['tls_id']])
            # The canonical full-frame context breaks equal-gap ties in packet
            # insertion order, including when candidates span several lanes.
            view = {actor: packets[actor] for actor in sorted(candidates, key=self._packet_order.__getitem__)}
            context = self._context(packet, view, connection, compatible)
            features.append(self.model.features(context))
        return np.asarray(features, dtype=float).reshape(len(packets), self.coefficients.shape[1])

    def _joint_decisions(self, packets):
        scores = self._features(packets) @ self.coefficients.T
        if not np.isfinite(scores).all():
            raise ArithmeticError('Nonfinite joint packet head score')
        decisions = {}
        for (key, packet), (brake, start, yielding, left, right) in zip(packets.items(), scores, strict=True):
            connection, compatible = self._connections[key]
            geometry = self.topology.lanes[packet['lane_id']]
            permissive = (not geometry['internal'] and connection is not None and compatible
                          and self._signal(connection, compatible) == 'g')
            apply_brake = self._decision_uniforms[key]['brake'] < _sigmoid(brake)
            decisions[key] = dict(
                motion_residual_mps=-packet['parameters']['decel_mps2'] if apply_brake else 0.,
                start_probability=_sigmoid(start) if packet['speed_mps'] < .1 else 1.,
                yield_probability=_sigmoid(yielding) if permissive else 1.,
                lane_logits=dict(left=float(left), stay=0., right=float(right)),
            )
        return decisions

    def _lane_changes(self, packets, decisions, events):
        """Sample in the original packet order using accepted t+1 neighbours."""
        by_lane = defaultdict(set)
        for key, packet in packets.items():
            by_lane[packet['lane_id']].add(key)
        transitions = {event['vehicle_id']: event for event in events if event['kind'] == 'lane_transition'}
        actions = {}
        for key in sorted(packets, key=self._packet_order.__getitem__):
            packet = packets[key]
            lane = packet['lane_id']
            if self.topology.lanes[lane]['internal']:
                actions[key] = 'stay'
                continue
            # A new ordinary road has its own intent before testing LC masks.
            # If the current intent already belongs to this road, reuse it.
            self._intent(packet)
            adjacent = {self.topology.adjacent(lane, delta) for delta in (-1, 1)} - {None}
            if not adjacent:
                actions[key] = 'stay'
                continue
            candidates = {actor for neighbor in adjacent for actor in by_lane[neighbor]}
            view = {actor: packets[actor] for actor in sorted(candidates, key=self._packet_order.__getitem__)}
            targets = _targets(self, packet, view)
            probabilities = masked_lane_probabilities(decisions[key]['lane_logits'],
                {choice: choice in targets for choice in CHOICES})
            chosen = categorical_choice(probabilities, self._decision_uniforms[key]['lane'])
            actions[key] = chosen
            if chosen == 'stay':
                continue
            target = targets[chosen]
            unresolved = []
            trail, complete = self.sensor_operator._lane_change_trail(
                packet['body_trail'], target, packet['parameters']['length_m'],
                packet['position_m'], key, unresolved)
            if not complete or unresolved:
                raise ValueError(f'Predicted lane-change body trail is unresolved: {unresolved}')
            packet['lane_id'] = target
            packet['lateral_position_m'] = packet['lateral_speed_mps'] = 0.
            packet['allowed_speed_mps'] = self._allowed_speed(packet)
            self._trim_body(packet, trail)
            by_lane[lane].remove(key)
            by_lane[target].add(key)
            if key in transitions:
                transitions[key]['observed_lane_path'] = [*transitions[key]['observed_lane_path'], target]
                transitions[key]['transition_type'] = 'connection_then_lane_change'
            else:
                events.append(dict(kind='lane_transition', vehicle_id=key,
                    observed_lane_path=[lane, target], transition_type='lane_change'))
        return actions

    def step(self, initial_requests=None):
        old = self.state['packets']
        pending_before = [packet['packet_id'] for packet in self.state['pending']]
        now = self.state['time_s']
        requests = self._pp_requests() if now % 10 == 0 else {}
        if initial_requests is not None:
            if self.steps or now % 10:
                raise ValueError('An action override requires the initial decision epoch')
            if not initial_requests.keys() <= self.executors.keys():
                raise ValueError('Unknown initial intervention TLS')
            requests.update(initial_requests)
        returns = {tls: self.executors[tls].request(value) for tls, value in sorted(requests.items())}
        applied = dict(self.signals)

        self._packet_order = {key: index for index, key in enumerate(old)}
        self._decision_uniforms = {key: self._step_uniforms(key, now) for key in old}
        self._connections = {key: self._connection(packet) for key, packet in old.items()}
        decisions = self._joint_decisions(old)
        packets, exited, events = resolve_forward(self, old, decisions)
        lane_actions = self._lane_changes(packets, decisions, events)
        due, inserted = self._boundary(packets)

        self.state['packets'] = packets
        self.state['lane_sensors'] = self.sensor_operator._reconstruct(packets)
        for executor in self.executors.values():
            executor.advance(1.)
        self.state['executors'] = {tls: executor.snapshot() for tls, executor in self.executors.items()}
        self.state['time_s'] = now+1
        ledger = verify_identity_step(list(old), pending_before, due, inserted, exited,
                                      list(packets), [packet['packet_id'] for packet in self.state['pending']])
        self._validate_state()
        for event in events:
            if event['kind'] != 'lane_transition':
                continue
            for origin, target in zip(event['observed_lane_path'], event['observed_lane_path'][1:]):
                if (target not in self.static['successors'][origin]
                        and target not in (self.topology.adjacent(origin, -1), self.topology.adjacent(origin, 1))):
                    raise ValueError('Predicted front path violates static connectivity')
        halted = sum(self.state['lane_sensors'][lane]['H'] for lane in self.cost_lanes)
        self.integral += halted
        self.steps += 1
        return dict(time_s=now+1, requests=requests, request_returns=returns, applied_signals=applied,
            due_ids=due, inserted_ids=inserted, exited_ids=exited, events=events,
            lane_change_actions=lane_actions, halted_total=halted, I_accum=self.integral,
            ledger_passed=ledger, active_count=len(packets), pending_count=len(self.state['pending']),
            terminal_count=len(exited), nonsink_terminal_count=sum(event.get('nonsink', False) for event in events),
            executors=deepcopy(self.state['executors']))
