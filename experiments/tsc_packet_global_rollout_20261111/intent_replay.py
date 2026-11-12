"""Short offline replay of first road intent under recorded PP signals."""

from collections import Counter

from experiments.tsc_target_packet_protocol_20261109.packet_contract import categorical_choice

from .calibration import kernel_factory
from .kernel import PacketKernel


ARMS = ('baseline', 'lane_conditioned', 'route_oracle')
CHECKPOINTS = (10, 30)
STOP_SPEED = .1
APPROACH_DISTANCE = 25.


def select_cohort(static, initial):
    """Hold the anchor roster fixed across arms, even as routes diverge."""
    lanes = static['lanes']
    return tuple(sorted(
        key for key, packet in initial['packets'].items()
        if not lanes[packet['lane_id']]['internal']
        and static['successors'][packet['lane_id']]
        and lanes[packet['lane_id']]['length_m']-packet['position_m'] <= APPROACH_DISTANCE
    ))


def motion_transition_ids(before_packets, after_packets, cohort_ids):
    stopped, started = set(), set()
    for key in cohort_ids:
        before, after = before_packets.get(key), after_packets.get(key)
        if before is None or after is None:
            continue
        was_stopped = before['speed_mps'] < STOP_SPEED
        is_stopped = after['speed_mps'] < STOP_SPEED
        if is_stopped and not was_stopped:
            stopped.add(key)
        if was_stopped and not is_stopped:
            started.add(key)
    return stopped, started


def _green(connection, applied):
    tls = connection['tls_id']
    return tls is not None and applied[tls][connection['link_index']] in ('G', 'g')


def crossing_green(topology, observed_path, applied, *, next_edge=None, connection=None):
    """Read the crossing link on a partial or complete observed physical path."""
    if not observed_path or len(observed_path) < 2:
        return None
    path = tuple(observed_path)
    candidates = []
    for option in topology.connections_from(path[0]):
        shared = min(len(path), len(option['path']))
        if path[:shared] == option['path'][:shared]:
            candidates.append(option)
    if connection is not None:
        candidates = [option for option in candidates if option['key'] == connection['key']]
    elif next_edge is not None:
        indicated = [option for option in candidates if option['to_edge'] == next_edge]
        if indicated:
            candidates = indicated
    values = {_green(option, applied) for option in candidates}
    return next(iter(values)) if len(values) == 1 else None


class RecordedIntentKernel(PacketKernel):
    """Change the initial intent draw rule and supply observed signal strings."""

    def __init__(self, static, initial, executor_seeds, model, scenario, arm,
                 execution, oracle_choices, conflict_geometry, *, audit_caps=False):
        super().__init__(static, initial, executor_seeds, model['target_laws'],
                         coefficients=model['coefficients'], random_seed=scenario,
                         conflict_geometry=conflict_geometry)
        if arm not in ARMS:
            raise ValueError(f'Unknown replay arm {arm}')
        self.arm, self.execution, self.oracle_choices = arm, execution, oracle_choices
        self.cohort_ids = select_cohort(static, initial)
        self.anchor_lanes = {key: initial['packets'][key]['lane_id'] for key in self.cohort_ids}
        self.anchor_edges = {key: self.topology.lanes[lane]['edge_id']
                             for key, lane in self.anchor_lanes.items()}
        self.anchor_choices = {}
        self.audit_caps = audit_caps

    def _pp_requests(self):
        return {}

    def _intent(self, packet):
        key, lane = packet['packet_id'], packet['lane_id']
        geometry = self.topology.lanes[lane]
        edge = geometry['edge_id']
        if (geometry['internal'] or key not in self.anchor_edges or key in self.anchor_choices
                or self.anchor_edges[key] != edge):
            return super()._intent(packet)

        if self.state['intents'].get(key, {}).get('road_id') == edge:
            raise ValueError('Replay requires an observed anchor with no sampled ordinary intents')
        prior = self.laws['intent_probs'][edge]
        lane_support = {'END'} | {c['to_edge'] for c in self.topology.connections_from(lane)}
        restricted = {choice: weight for choice, weight in prior.items() if choice in lane_support}
        total = sum(restricted.values())
        actual = self.oracle_choices.get(key)
        oracle_legal = actual == 'END' or (actual is not None and any(
            self.topology.compatible(sibling, actual) for sibling in self.topology.road_lanes(edge)))

        # One draw for every arm, even with an oracle. All later road draws stay
        # in the same ordinal stream for this packet and are never overridden.
        uniform = self.noise.intent_uniform(key)
        baseline = categorical_choice(prior, uniform)
        selected = baseline
        if self.arm == 'lane_conditioned' and total > 0:
            selected = categorical_choice({choice: weight/total for choice, weight in restricted.items()}, uniform)
        if self.arm == 'route_oracle' and oracle_legal:
            selected = actual
        self.state['intents'][key] = dict(road_id=edge, choice=selected)
        self.anchor_choices[key] = dict(baseline=baseline, selected=selected,
            baseline_compatible=baseline in lane_support, selected_compatible=selected in lane_support,
            has_lane_support=total > 0, oracle_present=actual is not None, oracle_legal=oracle_legal,
            oracle_lane_compatible=actual in lane_support, agrees_with_oracle=oracle_legal and selected == actual)
        return selected

    def _joint_decisions(self, packets):
        decisions = super()._joint_decisions(packets)
        if self.audit_caps:
            self.last_decisions = decisions
        return decisions

    def step(self):
        applied = self.execution[self.state['time_s']]['applied_states']
        if applied.keys() != self.executors.keys():
            raise ValueError('Recorded signal states must cover the frozen executor roster')
        self.signals.update(applied)
        row = super().step()
        if row['applied_signals'] != applied or row['requests'] or not row['ledger_passed']:
            raise ValueError('Recorded-signal replay lost signal or packet identity closure')
        return row

    def _original_road(self, key, packet):
        lane = self.topology.lanes[packet['lane_id']]
        return not lane['internal'] and lane['edge_id'] == self.anchor_edges[key]

    def _factual_green(self, key, packet, applied):
        choice = self.oracle_choices.get(key)
        options = self.topology.compatible(packet['lane_id'], choice)
        return bool(options) and all(_green(option, applied) for option in options)

    def tally_caps(self, before, after, applied, actual_releases):
        """Count overlapping constraints for stopped cohort actors at releases."""
        counts = Counter()
        for key in self.cohort_ids:
            old, new = before.get(key), after.get(key)
            if old is None or new is None or not self._original_road(key, old):
                continue
            if actual_releases is None:
                released = self._factual_green(key, old, applied)
            else:
                released = key in actual_releases
            if not released or old['speed_mps'] >= STOP_SPEED:
                continue
            counts['stopped_on_factual_release'] += 1
            if new['speed_mps'] >= STOP_SPEED:
                continue
            counts['still_stopped'] += 1
            connection, compatible = self._connections[key]
            if not compatible:
                counts['route_incompatible'] += 1
            elif connection is None:
                counts['end_intent'] += 1
            elif not _green(connection, applied):
                counts['selected_signal_not_green'] += 1
            decision = self.last_decisions[key]
            if self._decision_uniforms[key]['start'] >= decision['start_probability']:
                counts['start_permission_denied'] += 1
                continue
            motion = self._last_receiving_resolution['results'][key]
            proposal = motion['physical_proposal_mps']
            receiving, spatial = motion['same_step_path_bound_m'], motion['spatial_limit_m']
            if proposal < STOP_SPEED:
                counts['start_permitted_but_proposal_below_stop_speed'] += 1
                if decision['motion_residual_mps'] < 0:
                    counts['brake_draw_on_low_proposal'] += 1
            else:
                if receiving < STOP_SPEED and receiving <= spatial:
                    counts['receiving_dominant_bound'] += 1
                if spatial < STOP_SPEED and spatial <= receiving:
                    counts['spatial_dominant_bound'] += 1
                if receiving >= STOP_SPEED and spatial >= STOP_SPEED:
                    counts['stall_after_proposal_and_caps'] += 1
        return counts


def simulate_arm(base, static, initial, executor_seeds, model, scenario, arm, execution,
                 oracle_choices, *, audit_caps=False, factory=None, actual_releases=None):
    """Read out one shared scenario, retaining only checkpoint counts."""
    if factory is None:
        factory = kernel_factory(base, static)
    kernel = RecordedIntentKernel(static, initial, executor_seeds, model, scenario, arm,
        execution, oracle_choices, factory.keywords['conflict_geometry'], audit_caps=audit_caps)
    stopped, started, green_starts = set(), set(), set()
    crossed, green_crossed, green_unresolved, exited = set(), set(), set(), set()
    due_counts, caps, checkpoints = [], Counter(), {}
    inserted = 0
    for step in range(1, CHECKPOINTS[-1]+1):
        before, time_s = kernel.state['packets'], kernel.state['time_s']
        row = kernel.step()
        after, applied = kernel.state['packets'], row['applied_signals']
        stopped_now, started_now = motion_transition_ids(before, after, kernel.cohort_ids)
        stopped.update(stopped_now)
        started.update(started_now)
        for key in started_now:
            connection, compatible = kernel._connections[key]
            if (key not in crossed and kernel._original_road(key, before[key]) and compatible and connection is not None
                    and _green(connection, applied)):
                green_starts.add(key)
        for event in row['events']:
            key = event['vehicle_id']
            if key not in kernel.anchor_edges:
                continue
            if event['kind'] == 'terminal_exit':
                exited.add(key)
            elif key not in crossed and kernel._original_road(key, before[key]):
                path = event['observed_lane_path']
                if not kernel._original_road(key, after[key]):
                    crossed.add(key)
                    connection, _ = kernel._connections[key]
                    green = crossing_green(kernel.topology, path, applied, connection=connection)
                    if green is None:
                        green_unresolved.add(key)
                    elif green:
                        green_crossed.add(key)
        if audit_caps:
            released = actual_releases.get(time_s, ()) if actual_releases is not None else None
            caps.update(kernel.tally_caps(before, after, applied, released))
        due_counts.append(len(row['due_ids']))
        inserted += len(row['inserted_ids'])
        if step in CHECKPOINTS:
            cohort = [after[key] for key in kernel.cohort_ids if key in after]
            checkpoints[str(step)] = dict(halted_total=row['halted_total'], active_total=row['active_count'],
                cohort_active=len(cohort), cohort_halted=sum(p['speed_mps'] < STOP_SPEED for p in cohort),
                cohort_moving_to_halt=len(stopped), cohort_halt_to_moving=len(started),
                cohort_green_starts=len(green_starts), cohort_crossed_stopline=len(crossed),
                cohort_green_entries=len(green_crossed), cohort_crossing_green_unresolved=len(green_unresolved),
                cohort_terminal_exits=len(exited), inserted=inserted, caps=dict(caps) if audit_caps else None)
    choices = kernel.anchor_choices
    if choices.keys() != kernel.anchor_edges.keys():
        raise ValueError('Every anchor cohort packet must consume its first ordinary intent draw')
    counts = dict(cohort_size=len(choices), changed_from_baseline=sum(r['baseline'] != r['selected'] for r in choices.values()),
        baseline_incompatible=sum(not r['baseline_compatible'] for r in choices.values()),
        selected_incompatible=sum(not r['selected_compatible'] for r in choices.values()),
        no_conditioned_support=sum(not r['has_lane_support'] for r in choices.values()),
        oracle_present=sum(r['oracle_present'] for r in choices.values()), oracle_legal=sum(r['oracle_legal'] for r in choices.values()),
        oracle_incompatible=sum(r['oracle_legal'] and not r['oracle_lane_compatible'] for r in choices.values()),
        agrees_with_oracle=sum(r['agrees_with_oracle'] for r in choices.values()),
        baseline_end=sum(r['baseline'] == 'END' for r in choices.values()),
        selected_end=sum(r['selected'] == 'END' for r in choices.values()))
    return dict(arm=arm, scenario=scenario, checked_steps=kernel.steps, checkpoints=checkpoints,
                intent_counts=counts, due_counts=due_counts)
