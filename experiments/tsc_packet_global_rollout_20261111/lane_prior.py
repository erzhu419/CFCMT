"""Training-only lane law for the first intent of vehicles near the stop line."""

from collections import Counter
from pathlib import Path

from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_protocol_20261109.packet_contract import persistent_intent

from .calibration import existing, read
from .intent_replay import select_cohort
from .intent_replay_panel import oracle_at_anchor, read_routes
from .kernel import PacketKernel


INTERVAL_SEC = 10
PRIOR_STRENGTH = 8.0


def estimate_lane_prior(base, static):
    """Count training packet snapshots against compatible route supervision."""
    training_base = base/'target_packet_training'
    protocol = read(training_base/'protocol.json')
    seeds = tuple(protocol['cohorts']['training_seeds'])
    horizon = protocol['proposed_budget']['seconds_per_single_trajectory']
    receipt = read(training_base/'training/result.json')
    cases = {row['seed']:row['case'] for row in receipt['seeds']
             if row['status'] == 'COMPLETE' and row['case']['role'] == 'training'}
    if receipt['status'] != 'COMPLETE' or set(cases) != set(seeds):
        raise ValueError('The four frozen training archives must be complete')
    configs = {Path(case['sumocfg']) for case in cases.values()}
    if len(configs) != 1:
        raise ValueError('The original training archives must share their native route file')
    routes = read_routes(next(iter(configs)))
    topology = RoadTopology(static)
    road_priors = read(training_base/'target_laws_server_only.json')['intent_probs']
    compatible_priors = {}
    for lane, geometry in topology.lanes.items():
        if geometry['internal'] or not static['successors'][lane]:
            continue
        support = {'END'} | {connection['to_edge'] for connection in topology.connections_from(lane)}
        restricted = {choice:p for choice,p in road_priors[geometry['edge_id']].items() if choice in support}
        total = sum(restricted.values())
        if total <= 0.:
            raise ValueError(f'No compatible road-prior mass for training lane {lane}')
        compatible_priors[lane] = {choice:p/total for choice,p in restricted.items()}

    snapshots = set(range(0, horizon+1, INTERVAL_SEC))
    counts = {lane:Counter() for lane in compatible_priors}
    training_counts = {}
    for seed in seeds:
        seen = set()
        summary = dict(sampled_frames=0, cohort_packet_snapshots=0, compatible_labels=0,
                       incompatible_labels=0, unresolved_labels=0)
        frames = training_base/'training'/f'seed{seed}'/'frames_server_only.jsonl.gz'
        for frame in existing.json_lines(frames):
            time_s = frame['time_s']
            if time_s not in snapshots:
                continue
            if time_s in seen:
                raise ValueError(f'Repeated training frame at seed {seed}, time {time_s}')
            seen.add(time_s)
            summary['sampled_frames'] += 1
            choices, _ = oracle_at_anchor(topology, frame, routes)
            for key in select_cohort(static, frame):
                lane = frame['packets'][key]['lane_id']
                choice = choices.get(key)
                summary['cohort_packet_snapshots'] += 1
                if choice is None:
                    summary['unresolved_labels'] += 1
                elif choice not in compatible_priors[lane]:
                    summary['incompatible_labels'] += 1
                else:
                    counts[lane][choice] += 1
                    summary['compatible_labels'] += 1
        if seen != snapshots:
            raise ValueError(f'Incomplete training snapshot grid for seed {seed}')
        training_counts[seed] = summary

    probabilities = {}
    for lane, prior in compatible_priors.items():
        observed = counts[lane]
        total = observed.total() + PRIOR_STRENGTH
        probabilities[lane] = {choice:(observed[choice] + PRIOR_STRENGTH*p)/total
                               for choice,p in prior.items()}
    return dict(status='ESTIMATED_TRAINING_ONLY_LANE_PRIOR', training_seeds=list(seeds),
                training_counts=training_counts, lane_counts={lane:dict(row) for lane,row in counts.items()},
                lane_probabilities=probabilities, prior_strength=PRIOR_STRENGTH,
                interval_sec=INTERVAL_SEC, nuisance_law_estimations=1, numerical_optimizations=0)


class LanePriorKernel(PacketKernel):
    """Replace only the first intent law for eligible vehicles at the anchor."""

    def __init__(self, static, initial_state, executor_seeds, target_laws, coefficients,
                 random_seed=0, executor_cls=None, conflict_geometry=None, *, lane_probabilities):
        super().__init__(static, initial_state, executor_seeds, target_laws, coefficients,
                         random_seed=random_seed, executor_cls=executor_cls,
                         conflict_geometry=conflict_geometry)
        self.lane_probabilities = lane_probabilities
        self._anchor_lanes = {key:initial_state['packets'][key]['lane_id']
                              for key in select_cohort(static, initial_state)}
        if not set(self._anchor_lanes.values()) <= lane_probabilities.keys():
            raise ValueError('The frozen lane prior must cover every eligible anchor lane')

    def _intent(self, packet):
        key = packet['packet_id']
        lane = self._anchor_lanes.pop(key, None)
        if lane is None:
            return super()._intent(packet)
        if lane != packet['lane_id'] or key in self.state['intents']:
            raise ValueError('A lane-prior packet must first draw on its original observed lane')
        edge = self.topology.lanes[lane]['edge_id']
        return persistent_intent(self.state['intents'], key, edge, self.lane_probabilities[lane],
                                 self.noise.intent_uniform(key))
