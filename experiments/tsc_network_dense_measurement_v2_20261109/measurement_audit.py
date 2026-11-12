"""Passive dense-pilot sensor, population and original-executor replay audit."""
from collections import Counter
from math import isfinite
from types import SimpleNamespace


class AuditMismatch(Exception):
    def __init__(self, check, detail):
        self.check, self.detail = check, detail


def require(condition, check, detail):
    if not condition:
        raise AuditMismatch(check, detail)


class MeasurementAuditor:
    """Consume frame0 then poststep frames; never write to the live SUMO API.

    PP replay uses cached native observables to preserve exact floating-point
    ties. Independently, packet-reconstructed N/H/IDs must match exactly and
    occupancy/speed must match to 1e-8. Clone clearance uses reconstructed N.
    """

    def __init__(self, v3, infos, graph, context, executors, costlanes, *, focal_tls, horizon=450):
        self.v3, self.infos, self.graph, self.context = v3, infos, graph, context
        self.costlanes, self.focal_tls, self.horizon = tuple(costlanes), focal_tls, int(horizon)
        if not self.costlanes or focal_tls not in executors or set(executors) != set(infos):
            raise ValueError('Expected controlled lanes, one known focal TLS and complete executor/info rosters')
        self.frame = None
        self.start = None
        self.frames_seen = 0
        self.request_epochs = []
        self.halted = []
        self.checks = {}
        self.first_failure = None
        self.maximum_occupancy_error = self.maximum_speed_error = 0.
        self.collisions = Counter()
        self.collision_steps = 0
        self.departed, self.arrived, self.due = set(), set(), set()
        self.initial_active = self.initial_pending = None
        self.active = self.pending = set()
        self.initial_pp_requests = None
        self.initial_focal_request = None
        clone_api = SimpleNamespace(
            trafficlight=SimpleNamespace(setRedYellowGreenState=lambda *_: None),
            lane=SimpleNamespace(getLastStepVehicleNumber=lambda lane: self.frame['reconstructed_lanes'][lane]['N']))
        self.clones = {}
        for tls, executor in sorted(executors.items()):
            clone = type(executor)(sumo_api=clone_api, tls_id=tls, timings=list(executor.timings.values()),
                initial_state=executor.current_state, clearance_lane_ids=executor.clearance_lane_ids,
                clearance_poll_sec=executor.clearance_poll_sec)
            clone.restore(executor.snapshot(), write_state=False)
            self.clones[tls] = clone
        self.policy_api = SimpleNamespace(
            lane=SimpleNamespace(
                getLastStepVehicleNumber=lambda lane: self.frame['native_lanes'][lane]['N'],
                getLastStepHaltingNumber=lambda lane: self.frame['native_lanes'][lane]['H'],
                getLastStepOccupancy=lambda lane: self.frame['native_lanes'][lane]['O'],
                getLastStepMeanSpeed=lambda lane: self.frame['native_lanes'][lane]['mean_speed_mps'],
                getLastStepVehicleIDs=lambda lane: tuple(self.frame['native_lanes'][lane]['ids'])),
            vehicle=SimpleNamespace(getIDCount=lambda: len(self.frame['native_active_ids'])),
            simulation=SimpleNamespace(getTime=lambda: self.frame['time_s']))

    def _records_match(self, records, stage):
        require(set(records) == set(self.clones), 'executor_replay', dict(stage=stage, reason='TLS roster'))
        for tls, clone in self.clones.items():
            actual = records[tls]
            expected = clone.snapshot()
            require(actual['snapshot'] == expected, 'executor_replay',
                    dict(stage=stage, tls=tls, expected=expected, actual=actual['snapshot']))
            require(list(actual['feasible_states']) == list(clone.feasible_states_now()), 'feasible_roster',
                    dict(stage=stage, tls=tls))
            minimum = float(clone.timings[clone.current_green_state].min_green_sec)
            require(actual['minimum_green_sec'] == minimum, 'executor_replay',
                    dict(stage=stage, tls=tls, reason='minimum-green clock'))

    def _sensor_frame(self, frame):
        native, rebuilt = frame['native_lanes'], frame['reconstructed_lanes']
        require(set(native) == set(rebuilt), 'lane_sensor_reconstruction', 'native/reconstructed lane rosters')
        if self.frame is not None:
            require(set(native) == set(self.frame['native_lanes']), 'lane_sensor_reconstruction', 'dense lane roster changed')
        require(set(self.costlanes) <= set(native), 'lane_sensor_reconstruction', 'missing controlled lane')
        for clone in self.clones.values():
            require(set(clone.clearance_lane_ids) <= set(native), 'lane_sensor_reconstruction', 'missing clearance lane')
        ids = []
        for lane in native:
            a, b = native[lane], rebuilt[lane]
            require(Counter(a['ids']) == Counter(b['ids']) and a['N'] == b['N'] == len(a['ids']) and
                    len(set(a['ids'])) == len(a['ids']), 'lane_sensor_reconstruction', dict(lane=lane, field='IDs/N'))
            require(a['H'] == b['H'] and int(a['H']) == a['H'] and 0 <= a['H'] <= a['N'],
                    'lane_sensor_reconstruction', dict(lane=lane, field='H'))
            for key in ('O', 'mean_speed_mps'):
                require(isfinite(a[key]) and isfinite(b[key]), 'lane_sensor_reconstruction', dict(lane=lane, field=key))
            occupancy_error = abs(a['O']-b['O'])
            self.maximum_occupancy_error = max(self.maximum_occupancy_error, occupancy_error)
            require(occupancy_error <= 1e-8, 'lane_sensor_reconstruction', dict(lane=lane, field='O', error=occupancy_error))
            if a['N']:
                speed_error = abs(a['mean_speed_mps']-b['mean_speed_mps'])
                self.maximum_speed_error = max(self.maximum_speed_error, speed_error)
                require(speed_error <= 1e-8, 'lane_sensor_reconstruction', dict(lane=lane, field='speed', error=speed_error))
            ids.extend(a['ids'])
        active, pending = frame['native_active_ids'], frame['native_pending_ids']
        require(len(set(active)) == len(active) and len(set(pending)) == len(pending) and
                not set(active) & set(pending), 'population_identity', 'duplicate or active/pending overlap')
        require(Counter(ids) == Counter(active) and set(frame['packets']) == set(active),
                'population_identity', 'packet/front-lane/active identity partition')
        require(not frame['unresolved'], 'essential_state_resolved', frame['unresolved'][:3])
        require(all(frame['checks'].values()), 'observer_frame_checks', frame['checks'])
        require(not frame['starting_teleport_ids'] and not frame['ending_teleport_ids'], 'no_teleport',
                dict(starting=frame['starting_teleport_ids'], ending=frame['ending_teleport_ids']))
        if self.frame is not None:
            collisions = frame['collision_ids']
            self.collisions.update(collisions)
            self.collision_steps += bool(collisions)
        self.checks.update(lane_sensor_reconstruction=True, population_identity=True,
            essential_state_resolved=True, observer_frame_checks=True, no_teleport=True)

    def _population(self, frame, transition):
        departed, arrived, due = (frame['departed_ids'], frame['arrived_ids'], transition['due_ids'])
        for name, values, seen in (('departed', departed, self.departed), ('arrived', arrived, self.arrived), ('due', due, self.due)):
            require(len(values) == len(set(values)) and not set(values) & seen, 'population_identity',
                    dict(event=name, reason='repeated identity'))
        require(set(departed) <= self.pending | set(due), 'pending_due_ledger', 'departure not pending or newly due')
        require(set(arrived) <= self.active | set(departed), 'population_identity', 'arrival not active or same-step departed')
        expected = Counter(self.active); expected.update(departed); expected.subtract(arrived)
        require(+expected == expected and expected == Counter(frame['native_active_ids']),
                'population_identity', 'active_t != active_prev + departed - arrived')
        waiting = Counter(self.pending); waiting.update(due); waiting.subtract(departed)
        require(+waiting == waiting and waiting == Counter(frame['native_pending_ids']),
                'pending_due_ledger', 'pending_t != pending_prev + newly_due - departed')
        self.departed.update(departed); self.arrived.update(arrived); self.due.update(due)
        self.active, self.pending = set(frame['native_active_ids']), set(frame['native_pending_ids'])
        self.checks['pending_due_ledger'] = True

    def observe(self, frame, transition=None):
        """Return False on the first failed measurement; retain its compact cause."""
        if self.first_failure is not None:
            return False
        self.frames_seen += 1
        try:
            self._sensor_frame(frame)
            if self.frame is None:
                require(transition is None, 'native_clock_exact', 'frame0 must precede the first request')
                self._records_match(frame['executors'], 'initial_frame')
                self.start = float(frame['time_s'])
                self.frame = frame
                self.initial_active = sorted(frame['native_active_ids'])
                self.initial_pending = sorted(frame['native_pending_ids'])
                self.active, self.pending = set(self.initial_active), set(self.initial_pending)
                return True
            offset = len(self.halted)
            require(offset < self.horizon and frame['time_s'] == self.start+offset+1 and
                    transition['offset_sec'] == offset and transition['time_sec'] == self.start+offset,
                    'native_clock_exact', dict(expected_poststep=self.start+offset+1, actual=frame['time_s']))
            self._records_match(transition['before'], 'before_request')
            requests = transition['requests']
            expected = {}
            if offset % 10 == 0:
                states = self.v3._read_graph_states_v3(self.policy_api, self.infos, self.context, self.graph)
                expected = {tls: self.v3._rule_candidate(states[tls], clone, 'phase_pressure').state
                            for tls, clone in sorted(self.clones.items())}
                require(set(requests) == set(expected), 'original_pp_requests', 'all-TLS request roster')
                if offset == 0:
                    self.initial_pp_requests = dict(expected)
                    requested = requests[self.focal_tls]
                    require(requested in self.clones[self.focal_tls].feasible_states_now(),
                            'original_pp_requests', 'initial focal request not feasible')
                    expected[self.focal_tls] = requested
                    self.initial_focal_request = requested
                require(requests == expected, 'original_pp_requests', dict(expected=expected, actual=requests))
                self.request_epochs.append(offset)
            else:
                require(requests == {}, 'request_once_per_epoch', dict(offset=offset, requests=requests))
            returns = {tls: bool(self.clones[tls].request(requests[tls])) for tls in sorted(requests)}
            require(transition['request_returns'] == returns, 'request_acceptance',
                    dict(expected=returns, actual=transition['request_returns']))
            self._records_match(transition['after'], 'after_request')
            applied = {tls: clone.current_state for tls, clone in self.clones.items()}
            require(transition['applied_states'] == applied, 'signals_before_step',
                    dict(expected=applied, actual=transition['applied_states']))
            self.frame = frame
            for clone in self.clones.values():
                clone.advance(1.)  # Observed poststep internal N can extend all-red.
            self._records_match(transition['after_executors'], 'after_step')
            self._population(frame, transition)
            native_h = sum(frame['native_lanes'][lane]['H'] for lane in self.costlanes)
            reconstructed_h = sum(frame['reconstructed_lanes'][lane]['H'] for lane in self.costlanes)
            raw = transition['halted_int']
            require(isfinite(raw) and raw == int(raw) and raw == native_h == reconstructed_h,
                    'poststep_cost_closure', dict(raw=raw, native=native_h, reconstructed=reconstructed_h))
            self.halted.append(int(raw))
            self.checks.update(native_clock_exact=True, original_pp_requests=True, request_once_per_epoch=True,
                request_acceptance=True, executor_replay=True, feasible_roster=True, signals_before_step=True,
                population_identity=True, poststep_cost_closure=True)
            return True
        except AuditMismatch as exc:
            self.first_failure = dict(check=exc.check, time_s=frame.get('time_s'), detail=exc.detail)
            self.checks[exc.check] = False
        except Exception as exc:
            self.first_failure = dict(check='essential_record_complete', time_s=frame.get('time_s'),
                                      detail=f'{type(exc).__name__}: {exc}')
            self.checks['essential_record_complete'] = False
        return False

    def summary(self, final=None):
        """Aggregate every observed frame; final is the collector's last frame."""
        complete = len(self.halted) == self.horizon
        checks = dict(self.checks, full_horizon=complete,
            dense_frames_complete=self.frames_seen == self.horizon+1,
            request_epochs_complete=self.request_epochs == list(range(0, self.horizon, 10)))
        integral = sum(self.halted) if complete else None
        cost = integral/(self.horizon*len(self.costlanes)) if complete else None
        if final is not None:
            checks['final_frame_clock_exact'] = complete and final['time_s'] == self.start+self.horizon and (
                final['offset_sec'] == self.horizon)
        passed = self.first_failure is None and all(checks.values())
        endpoints = (0, 10, 30, 60, self.horizon)
        integrals = {str(end): sum(self.halted[:end]) if len(self.halted) >= end else None for end in endpoints}
        segments = {f'{a}_{b}': integrals[str(b)]-integrals[str(a)]
                    if integrals[str(a)] is not None and integrals[str(b)] is not None else None
                    for a, b in zip(endpoints, endpoints[1:])}
        return dict(status='PASS_DENSE_MEASUREMENT' if passed else 'HOLD_DENSE_MEASUREMENT', checks=checks,
            first_failure=self.first_failure, start_time_sec=self.start, checked_poststeps=len(self.halted),
            frames_seen=self.frames_seen, controlled_lane_count=len(self.costlanes),
            I450=integral, C450=cost if passed else None, integrals=integrals, segments=segments,
            maximum_occupancy_error=self.maximum_occupancy_error, maximum_nonempty_mean_speed_error=self.maximum_speed_error,
            request_epochs=len(self.request_epochs), initial_pp_requests=self.initial_pp_requests,
            initial_focal_request=self.initial_focal_request,
            population=dict(initial_active_count=len(self.initial_active or ()), initial_pending_count=len(self.initial_pending or ()),
                final_active_count=len(self.active), final_pending_count=len(self.pending),
                newly_due_count=len(self.due), departed_count=len(self.departed), arrived_count=len(self.arrived)),
            collision_metadata=dict(vehicle_occurrences=sum(self.collisions.values()),
                steps_with_collisions=self.collision_steps, unique_vehicle_ids=len(self.collisions)),
            new_fits=0, source_data_used=False,
            role='Retrospective measured-state interface admission, not free-rollout prediction or control efficacy')
