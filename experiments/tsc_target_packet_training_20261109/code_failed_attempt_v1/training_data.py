"""Streaming training-only history laws, fixed reservoirs and latent features."""
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import json
from pathlib import Path
from random import Random

import numpy as np

from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from .conditional_features import candidate_features, prepare_feature_kernel


HEADS = ('motion', 'start', 'yield_entry', 'lane_change')
RESERVOIR_SEED = 76116999


def iter_transitions(directory):
    """Read one previous/current frame and its interval action at a time."""
    directory = Path(directory)
    with gzip.open(directory/'frames_server_only.jsonl.gz', 'rt') as frames, gzip.open(directory/'execution_server_only.jsonl.gz', 'rt') as execution:
        previous = json.loads(next(frames))
        for line in execution:
            current, action = json.loads(next(frames)), json.loads(line)
            if current['time_s'] != previous['time_s']+1 or action['time_sec'] != previous['time_s']:
                raise ValueError('Training frames/actions must be consecutive and pre-step aligned')
            yield previous, current, action
            previous = current
        if next(frames, None) is not None:
            raise ValueError('Every training poststep requires its action record')


def _front_path(previous, current, packet_id):
    for event in current['events']:
        if event.get('vehicle_id') == packet_id and event['kind'] == 'lane_transition':
            return event['observed_lane_path']
    if packet_id in current['packets']:
        return [previous['packets'][packet_id]['lane_id']]
    return []


def _lane_label(path, topology):
    label = 1  # left0 / stay1 / right2
    for origin, destination in zip(path, path[1:]):
        a, b = topology.lanes[origin], topology.lanes[destination]
        if a['edge_id'] == b['edge_id'] and not a['internal']:
            label = 0 if b['index'] > a['index'] else 2
    return label


def _entry_label(path, topology):
    return int(any(topology.lanes[lane]['internal'] for lane in path[1:]) or
               any(topology.lanes[a]['edge_id'] != topology.lanes[b]['edge_id']
                   for a, b in zip(path, path[1:])))


def _exit_outcome(old, current, packet_id, topology):
    lane = old['lane_id']
    if topology.lanes[lane]['internal']:
        return None
    edge = topology.lanes[lane]['edge_id']
    if packet_id in current['arrived_ids']:
        return 'END'
    if packet_id not in current['packets']:
        raise ValueError('A training packet disappeared without an observed terminal event')
    new_lane = current['packets'][packet_id]['lane_id']
    new = topology.lanes[new_lane]
    if not new['internal']:
        return new['edge_id'] if new['edge_id'] != edge else None
    possible = {c['to_edge'] for c in topology.internal_connections[new_lane] if c['from_edge'] == edge}
    if len(possible) != 1:
        raise ValueError('An observed internal entry does not identify a historical road-exit outcome')
    return next(iter(possible))


def _legal_outcomes(topology):
    result = {}
    for lane, geometry in topology.lanes.items():
        if geometry['internal']:
            continue
        result.setdefault(geometry['edge_id'], {})
    for c in topology.connections:
        outcomes = result[c['from_edge']]
        if c['to_edge'] in outcomes and outcomes[c['to_edge']] != c['direction']:
            raise ValueError('A road-level outcome has ambiguous direction semantics')
        outcomes[c['to_edge']] = c['direction']
    for outcomes in result.values():
        outcomes['END'] = 'END'
    return result


def _cheap_candidates(previous, current, action, topology):
    """Current-state supersets; future observations provide labels only."""
    for key, packet in previous['packets'].items():
        lane = packet['lane_id']
        geometry = topology.lanes[lane]
        path = _front_path(previous, current, key)
        survived = key in current['packets']
        if survived:
            speed = current['packets'][key]['speed_mps']
            yield 'motion', key, float(speed)
            if packet['speed_mps'] < .1:
                yield 'start', key, int(speed >= .1)
        reach = packet['speed_mps']+packet['parameters']['accel_mps2']
        distance = geometry['length_m']-packet['position_m']
        connections = topology.connections_from(lane)
        if not geometry['internal'] and reach >= distance:
            applied = action['applied_states']
            if any(c['tls_id'] is not None and applied[c['tls_id']][c['link_index']] == 'g' for c in connections):
                yield 'yield_entry', key, _entry_label(path, topology)
        lateral = (not geometry['internal'] and len(topology.road_lanes(geometry['edge_id'])) > 1)
        if not lateral:
            possible = topology.internal_connections[lane] if geometry['internal'] else connections
            for c in possible:
                remaining = topology.path_distance(c, lane, packet['position_m'], c['to_lane'], 0.)
                if reach >= remaining and len(topology.road_lanes(c['to_edge'])) > 1:
                    lateral = True
                    break
        if survived and lateral:
            yield 'lane_change', key, _lane_label(path, topology)


def estimate_laws_and_sample(seed_dirs, protocol, static):
    expected = protocol['cohorts']['training_seeds']
    if set(seed_dirs) != set(expected):
        raise ValueError('Only all frozen training seeds may enter history laws or reservoirs')
    topology, legal = RoadTopology(static), None
    legal = _legal_outcomes(topology)
    duration = protocol['proposed_budget']['seconds_per_single_trajectory']
    training = protocol['model']['training']
    cap = training.get('reservoir_cap_per_head_seed', 10000)
    reservoir_seed = training.get('reservoir_seed', RESERVOIR_SEED)
    rng = Random(reservoir_seed)
    counts = {edge: Counter() for edge in legal}
    directions, entries, profiles = Counter(), Counter(), []
    seconds, due_total, censored = 0, 0, 0
    reservoirs, sampling = {}, {}
    for seed in sorted(seed_dirs):
        print(json.dumps(dict(stage='history_scan',seed=seed)),flush=True)
        buckets = {head: [] for head in HEADS}
        seen = Counter()
        transitions = 0
        last = None
        for previous, current, action in iter_transitions(seed_dirs[seed]):
            transitions += 1
            if transitions == 1 and previous['time_s'] != 0:
                raise ValueError('A target training single trajectory starts at native time zero')
            due = len(current['native_pending_ids'])-len(previous['native_pending_ids'])+len(current['departed_ids'])
            if due < 0:
                raise ValueError('Observed global pending/departure accounting implies negative due demand')
            due_total += due
            for key in current['departed_ids']:
                packet = current['packets'][key]
                if topology.lanes[packet['lane_id']]['internal']:
                    raise ValueError('The fixed network must expose entrants on their ordinary entry road')
                entries[packet['lane_id']] += 1
                profiles.append(dict(parameters=deepcopy(packet['parameters']), type_id=packet['type_id'],
                                     entry_speed_mps=packet['speed_mps']))
            for key, packet in previous['packets'].items():
                outcome = _exit_outcome(packet, current, key, topology)
                if outcome is not None:
                    edge = topology.lanes[packet['lane_id']]['edge_id']
                    if outcome not in legal[edge]:
                        raise ValueError('Observed historical road exit is outside static legal support')
                    counts[edge][outcome] += 1
                    directions[legal[edge][outcome]] += 1
            for head, key, label in _cheap_candidates(previous, current, action, topology):
                seen[head] += 1
                reference = dict(seed=seed, time_sec=previous['time_s'], packet_id=key, y=label)
                if len(buckets[head]) < cap:
                    buckets[head].append(reference)
                else:
                    index = rng.randrange(seen[head])
                    if index < cap:
                        buckets[head][index] = reference
            last = current
        if transitions != duration or last['time_s'] != duration:
            raise ValueError('Every training seed must retain its full frozen single trajectory')
        seconds += transitions
        censored += sum(not topology.lanes[p['lane_id']]['internal'] for p in last['packets'].values())
        reservoirs[seed] = {head: sorted(rows, key=lambda r: (r['time_sec'], r['packet_id'])) for head, rows in buckets.items()}
        sampling[str(seed)] = {head: dict(superset_seen=seen[head], sampled=len(buckets[head]), retained=0,
            allzero_excluded=0, allzero_excluded_positive_labels=0) for head in HEADS}
        print(json.dumps(dict(stage='history_complete',seed=seed,seconds=transitions,
                              sampled={head:len(buckets[head]) for head in HEADS})),flush=True)
    probabilities = {}
    for edge, outcomes in legal.items():
        weights = {key: directions[direction]+.5 for key, direction in outcomes.items()}
        total = sum(weights.values())
        prior = {key: value/total for key, value in weights.items()}
        n = sum(counts[edge].values())
        probabilities[edge] = {key: (counts[edge][key]+8.*prior[key])/(n+8.) for key in outcomes}
    if not profiles or not sum(entries.values()):
        raise ValueError('Target history must observe boundary insertions and their profiles')
    laws = dict(intent_probs=probabilities, due_rate_per_sec=due_total/seconds,
        entry_probs={key: value/sum(entries.values()) for key, value in sorted(entries.items())}, profiles=profiles)
    history = dict(history_law_estimations=4, training_seeds=sorted(seed_dirs), trajectory_seconds=seconds,
        inferred_due_count=due_total, observed_insertions=sum(entries.values()),
        observed_road_exit_counts={edge: dict(values) for edge, values in counts.items()},
        global_direction_counts=dict(directions), right_censored_ordinary_visits=censored,
        calibrated_seed_data_used=False, source_data_used=False)
    return laws, reservoirs, sampling, history


def _candidates(kernel, key):
    packet = kernel.state['packets'][key]
    geometry = kernel.topology.lanes[packet['lane_id']]
    if geometry['internal']:
        return {kernel._connections[key][0]['to_edge']: 1.}
    return kernel.laws['intent_probs'][geometry['edge_id']]


def _pack(rows, head):
    width = max((len(row['prior']) for row in rows), default=1)
    n = len(rows)
    result = dict(X=np.zeros((n, width, len(FEATURE_NAMES))), prior=np.zeros((n, width)),
                  risk=np.zeros((n, width), dtype=bool), y=np.asarray([row['y'] for row in rows], dtype=float))
    if head == 'motion':
        result.update({key: np.zeros((n, width)) for key in ('upper', 'a', 'b')})
    if head == 'lane_change':
        result['mask'] = np.zeros((n, width, 3), dtype=bool)
        result['mask'][:, :, 1] = True
    for index, row in enumerate(rows):
        k = len(row['prior'])
        for key in ('X', 'prior', 'risk'):
            result[key][index, :k] = row[key]
        for key in ('upper', 'a', 'b') if head == 'motion' else ('mask',) if head == 'lane_change' else ():
            result[key][index, :k] = row[key]
    return result


def build_training_data(seed_dirs, protocol, static, executor_seeds, *, artifact_dir=None):
    """History/reservoir pass, then selected-reference conditional-feature pass."""
    laws, reservoirs, sampling, history = estimate_laws_and_sample(seed_dirs, protocol, static)
    if artifact_dir is not None:
        artifact_dir = Path(artifact_dir)
        for filename, value in (('target_laws_server_only.json', laws),
                                ('training_history_result.json', history),
                                ('training_reservoir_server_only.json', reservoirs)):
            (artifact_dir/filename).write_text(json.dumps(value, allow_nan=False, separators=(',', ':'))+'\n')
    rows, references = {head: [] for head in HEADS}, {head: [] for head in HEADS}
    for seed in sorted(seed_dirs):
        print(json.dumps(dict(stage='feature_scan',seed=seed)),flush=True)
        selected = defaultdict(lambda: defaultdict(list))
        for head, samples in reservoirs[seed].items():
            for reference in samples:
                selected[reference['time_sec']][reference['packet_id']].append((head, reference))
        for frame, unused_next, action in iter_transitions(seed_dirs[seed]):
            if frame['time_s']%300==0:
                print(json.dumps(dict(stage='feature_progress',seed=seed,time_sec=frame['time_s'])),flush=True)
            if frame['time_s'] not in selected:
                continue
            feature_seed = RESERVOIR_SEED+int(seed)*100000+int(frame['time_s'])
            kernel = prepare_feature_kernel(static, frame, executor_seeds, laws, action['requests'], feature_seed)
            if kernel.signals != action['applied_states']:
                raise ValueError('Current requests/executor do not reproduce the recorded current interval signal')
            for key, head_refs in selected[frame['time_s']].items():
                priors = _candidates(kernel, key)
                candidates = [candidate_features(kernel, key, intent) for intent in priors]
                features = [kernel.model.features(candidate['context']) for candidate in candidates]
                for head, reference in head_refs:
                    risk = [bool(candidate[head+'_risk']) for candidate in candidates] if head in ('motion', 'start') else (
                        [candidate['yield_risk'] for candidate in candidates] if head == 'yield_entry' else
                        [candidate['lane_mask']['left'] or candidate['lane_mask']['right'] for candidate in candidates])
                    stat = sampling[str(seed)][head]
                    if not any(risk):
                        stat['allzero_excluded'] += 1
                        positive = reference['y'] != 1 if head == 'lane_change' else reference['y'] > 0
                        stat['allzero_excluded_positive_labels'] += int(positive)
                        continue
                    row = dict(X=features, prior=list(priors.values()), risk=risk, y=reference['y'])
                    if head == 'motion':
                        row.update({name: [candidate[name] for candidate in candidates] for name in ('upper', 'a', 'b')})
                    if head == 'lane_change':
                        row['mask'] = [[candidate['lane_mask'][choice] for choice in ('left', 'stay', 'right')]
                                       for candidate in candidates]
                    rows[head].append(row)
                    references[head].append(dict(reference, candidate_intents=list(priors)))
                    stat['retained'] += 1
    return dict(laws=laws, heads={head: _pack(values, head) for head, values in rows.items()},
        sampling=sampling, history=history, record_references=references,
        feature_names=FEATURE_NAMES, feature_seed_rule='76116999 + training_seed*100000 + integer_second',
        role='Training only; sampled current-risk structural exclusions never use label values')
