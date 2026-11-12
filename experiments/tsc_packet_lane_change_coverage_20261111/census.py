"""Census and match archived lane-change labels using only recorded states."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
from statistics import median
import time

from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


DIRECTIONS = {0: 1, 2: -1}
SPEED_TOLERANCE = 2.
DISTANCE_TOLERANCE = 15.


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def key(row):
    return int(row['seed']), int(row['time_sec']), row['packet_id']


def vehicle(row):
    return int(row['seed']), row['packet_id']


def domain_rows(path, expected):
    domain = {}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            ref = key(row)
            if ref in expected:
                if ref in domain or not row['eligibility']['lane_change']:
                    raise ValueError(f'Invalid saved LC domain for {ref}')
                domain[ref] = row['required_actor_count']
    if domain.keys() != expected:
        raise ValueError('Saved geometry census misses archived LC references')
    return domain


def current_stratum(packet, direction, participants, topology, lanes, applied):
    lane = packet['lane_id']
    target = topology.adjacent(lane, DIRECTIONS[direction])
    pos = packet['position_m']
    if target is None or pos > topology.lanes[target]['length_m']:
        return None
    others = lanes[target]
    leader = min((p for p in others if p['position_m'] >= pos),
                 key=lambda p: p['position_m'], default=None)
    follower = max((p for p in others if p['position_m'] < pos),
                   key=lambda p: p['position_m'], default=None)
    own = packet['parameters']
    front_clear = (leader is None or
        leader['position_m'] - leader['parameters']['length_m'] - pos >= own['min_gap_m'])
    rear_clear = (follower is None or
        pos - own['length_m'] - follower['position_m'] >= follower['parameters']['min_gap_m'])
    colors = tuple(applied[c['tls_id']][c['link_index']] for c in topology.connections_from(lane)
                   if c['tls_id'] is not None)
    speed = packet['speed_mps']
    distance = topology.lanes[lane]['length_m'] - pos
    return (lane, direction, speed < .1, participants <= 4, colors, front_clear, rear_clear), speed, distance


def observe_positives(topology, expected, post, direct):
    paths = {}
    for event in post['events']:
        if event['kind'] == 'lane_transition':
            paths.setdefault(event['vehicle_id'], event['observed_lane_path'])
    for row, lane in expected:
        pid = row['packet_id']
        path = paths.get(pid, [])
        hops = [(a, b) for a, b in zip(path, path[1:]) if not topology.lanes[a]['internal']
                and topology.lanes[a]['edge_id'] == topology.lanes[b]['edge_id']]
        if pid not in post['packets'] or not hops:
            raise ValueError(f'Missing archived positive LC event at {key(row)}')
        a, b = (topology.lanes[lane_id] for lane_id in hops[-1])
        if row['y'] != (0 if b['index'] > a['index'] else 2):
            raise ValueError(f'Archived positive LC direction differs at {key(row)}')
        if path == [lane, topology.adjacent(lane, DIRECTIONS[row['y']])]:
            direct.add(key(row))


def match(positives, controls, direct):
    # Process scarce strata first. Keep each sampled vehicle in at most one pair.
    choices = []
    for row, stratum, speed, distance in positives:
        if key(row) not in direct:
            continue
        candidates = []
        for other, v, d in controls[stratum]:
            dv, dd = abs(speed-v), abs(distance-d)
            if dv <= SPEED_TOLERANCE and dd <= DISTANCE_TOLERANCE:
                candidates.append((dv/SPEED_TOLERANCE + dd/DISTANCE_TOLERANCE, dv, dd, key(other), other))
        choices.append((len(candidates), key(row), row, stratum, sorted(candidates)))
    pairs, used = [], set()
    for _, _, row, stratum, candidates in sorted(choices):
        if vehicle(row) in used:
            continue
        for _, dv, dd, _, control in candidates:
            if vehicle(control) not in used:
                pairs.append(dict(positive=row, stay=control, current_stratum=stratum,
                                  speed_difference_mps=dv, lane_end_distance_difference_m=dd))
                used.update((vehicle(row), vehicle(control)))
                break
    return pairs, Counter(row['y'] for count, _, row, _, _ in choices if count)


def seed_scan(seed, rows, base, static, current_train, current_dev, inspected, old_fit):
    topology = RoadTopology(static)
    domain = domain_rows(base/'target_packet_conditional_census/output'/f'seed{seed}_cases_server_only.jsonl.gz',
                         {key(r) for r in rows})
    positive_vehicles = {vehicle(r) for r in rows if r['y'] != 1}
    exclude = current_train | current_dev | inspected
    times = defaultdict(list)
    for row in rows:
        times[key(row)[1]].append(row)
    counts = {y: Counter() for y in (0, 1, 2)}
    unique = {y: set() for y in (0, 1, 2)}
    positives, controls, pending, direct = [], defaultdict(list), [], set()
    decoded, verified = 0, 0
    archive = base/'target_packet_training/training'/f'seed{seed}'
    with gzip.open(archive/'frames_server_only.jsonl.gz', 'rt') as frames, gzip.open(
            archive/'execution_server_only.jsonl.gz', 'rt') as actions:
        for t, line in enumerate(frames):
            action_line = next(actions, None)
            rows_here = times.get(t, ())
            if not rows_here and not pending:
                continue
            frame = json.loads(line)
            decoded += 1
            if frame['time_s'] != t:
                raise ValueError(f'Current frame clock differs in seed {seed} at {t}')
            observe_positives(topology, pending, frame, direct)
            verified += len(pending)
            pending = []
            if not rows_here:
                continue
            action = json.loads(action_line)
            if action['time_sec'] != t:
                raise ValueError(f'Current action clock differs in seed {seed} at {t}')
            packets = frame['packets']
            lanes = defaultdict(list)
            for packet in packets.values():
                lanes[packet['lane_id']].append(packet)
            for row in rows_here:
                pid, label = row['packet_id'], row['y']
                actor, ref = vehicle(row), key(row)
                packet, j = packets[pid], domain[ref]
                c = counts[label]
                c['archive'] += 1
                unique[label].add(actor)
                c['old_clock_fit'] += ref in old_fit
                c['current_train_vehicle'] += actor in current_train
                c['current_development_vehicle'] += actor in current_dev
                c['old_diagnostic_vehicle'] += actor in inspected
                c['outside_previous_vehicles'] += actor not in exclude
                c['within_particle_domain'] += 1 <= j <= 10
                if label != 1:
                    pending.append((row, packet['lane_id']))
                if actor in exclude or not 1 <= j <= 10:
                    continue
                if label == 1 and actor in positive_vehicles:
                    continue
                for direction in DIRECTIONS if label == 1 else (label,):
                    current = current_stratum(packet, direction, j, topology, lanes, action['applied_states'])
                    if current is None:
                        continue
                    stratum, speed, distance = current
                    if label == 1:
                        controls[stratum].append((row, speed, distance))
                    else:
                        positives.append((row, stratum, speed, distance))
    if sum(c['archive'] for c in counts.values()) != len(rows) or verified != sum(
            counts[y]['archive'] for y in DIRECTIONS):
        raise ValueError(f'Incomplete archive scan in seed {seed}')
    pairs, matchable = match(positives, controls, direct)
    matched = Counter(pair['positive']['y'] for pair in pairs)
    for y in DIRECTIONS:
        counts[y]['verified_direct_one_lane_change'] = sum(key(r) in direct and r['y'] == y for r in rows)
        counts[y]['candidate_direct_after_vehicle_exclusion'] = sum(
            r['y'] == y and key(r) in direct for r, *_ in positives)
        counts[y]['has_matching_stay'] = matchable[y]
        counts[y]['disjoint_pairs'] = matched[y]
        counts[y]['pair_positive_300s_blocks'] = dict(Counter(key(pair['positive'])[1]//300
            for pair in pairs if pair['positive']['y'] == y))
    for y in counts:
        counts[y]['distinct_archive_vehicles'] = len(unique[y])
    report = dict(seed=seed, labels=counts, decoded_frames=decoded, verified_positives=verified,
                  pairs=len(pairs), median_speed_difference_mps=median(p['speed_difference_mps'] for p in pairs) if pairs else None,
                  median_lane_end_difference_m=median(p['lane_end_distance_difference_m'] for p in pairs) if pairs else None)
    print(json.dumps(dict(seed=seed, verified_positives=verified, matched=matched)), flush=True)
    return report, pairs


def run(base, output):
    began = time.monotonic()
    output.mkdir(exist_ok=False)
    training = base/'target_packet_training'
    manifest = read(training/'training_reservoir_server_only.json')
    history = read(training/'training_history_result.json')
    cohort = read(base/'target_packet_particle_first_fit/preflight/cohort.json')['references']
    old_fit_rows = read(base/'target_packet_clock_corrected/training_references_server_only.json')['lane_change']
    old_fit = {key(row): row['y'] for row in old_fit_rows}
    pilot = read(base/'target_packet_real_particle_pilot/evaluation/summary.json')['cases']
    rows_by_seed = {int(seed): data['lane_change'] for seed, data in manifest.items()}
    archived = {key(row): row['y'] for rows in rows_by_seed.values() for row in rows}
    expected_seeds = list(range(76116901, 76116905))
    if (sorted(rows_by_seed) != expected_seeds or history['training_seeds'] != expected_seeds
            or any(len(rows) != 10000 for rows in rows_by_seed.values()) or len(archived) != 40000
            or set(archived.values()) != {0, 1, 2} or len(cohort) != 512 or len(pilot) != 15
            or any(archived.get(ref) != y for ref, y in old_fit.items())):
        raise ValueError('LC archive or frozen fit/diagnostic registration changed')
    hard = {key(r['reference']) for r in pilot if r['case_index'] < 9}
    positive_not_old_fit = {ref for ref, y in archived.items() if y != 1 and ref not in old_fit}
    if hard != positive_not_old_fit:
        raise ValueError('Previously inspected hard-case provenance differs from the archive')
    fit_label_counts = {split: dict(Counter(archived[key(r)] for r in cohort
        if r['split'] == split and r['head'] == 'lane_change')) for split in ('train', 'development')}
    train = {vehicle(r) for r in cohort if r['split'] == 'train'}
    development = {vehicle(r) for r in cohort if r['split'] == 'development'}
    inspected = {vehicle(r['reference']) for r in pilot}
    static = read(training/'static_graph_server_only.json')
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(seed_scan, s, rows_by_seed[s], base, static, train, development,
                                   inspected, old_fit) for s in expected_seeds]
        report, pairs = zip(*(f.result() for f in futures))
    save(output/'matched_pairs_server_only.json', [pair for group in pairs for pair in group])
    result = dict(status='COMPLETE_ARCHIVED_LC_COVERAGE', seeds=report, label_count=40000,
        coefficient_lc_label_counts=fit_label_counts, prior_seed_seconds=history['trajectory_seconds'],
        old_hard_positive_count=len(hard), current_coefficient_holdout_seed=76116904,
        new_native_starts=0, new_full_forward_calls=0, new_canonical_feature_evaluations=0,
        new_nuisance_estimations=0, new_fits=0, optimizer_calls=0,
        matched_pairs_file='matched_pairs_server_only.json',
        elapsed_sec=time.monotonic()-began, completed_at_utc=datetime.now(timezone.utc).isoformat())
    save(output/'result.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'elapsed_sec', 'label_count')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.base, args.output)
