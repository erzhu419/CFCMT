"""Archived current-state coverage, without kernels or feature construction."""

from collections import Counter, defaultdict
import gzip
import json
from math import fsum
from pathlib import Path

from experiments.tsc_packet_lane_change_coverage_20261111.census import (
    DISTANCE_TOLERANCE, SPEED_TOLERANCE, current_stratum,
)
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology


DIRECTIONS = {0: 'left', 2: 'right'}


def _reference(task):
    return task.get('reference', task)


def _identity(row):
    ref = _reference(row)
    return int(ref['seed']), int(ref['time_sec']), ref['packet_id']


def _stratum_key(row):
    values = row['current_stratum']
    return tuple(values[:4]) + (tuple(values[4]),) + tuple(values[5:])


def _rows(tasks, cohort):
    result = []
    for task in tasks:
        for direction in (DIRECTIONS if cohort == 'training' and task['y'] == 1
                          else (task['y'] if cohort == 'training' else task['direction'],)):
            row = dict(cohort=cohort, case_index=task['case_index'],
                reference=dict(zip(('seed', 'time_sec', 'packet_id'), _identity(task))),
                y=task['y'], direction=direction, status='MISSING_CURRENT_FRAME',
                current_stratum=None, speed_mps=None, lane_end_distance_m=None)
            if cohort == 'training':
                row['objective_weight'] = task['objective_weight']
            else:
                row.update(role=task['role'], pair_index=task['pair_index'])
            result.append(row)
    return result


def _record_current(rows, tasks, frame, action, topology):
    missing = [name for name, present in (
        ('frame.packets', 'packets' in frame), ('frame.time_s', 'time_s' in frame),
        ('action.time_sec', 'time_sec' in action), ('action.applied_states', 'applied_states' in action))
        if not present]
    if missing:
        for row in rows:
            row.update(status='MISSING_CURRENT_FIELDS', missing_current_fields=missing)
        return
    lanes = defaultdict(list)
    for packet in frame['packets'].values():
        lanes[packet['lane_id']].append(packet)
    for row in rows:
        task = tasks[row['case_index']]
        ref = row['reference']
        if int(frame['time_s']) != ref['time_sec'] or int(action['time_sec']) != ref['time_sec']:
            row['status'] = 'CURRENT_CLOCK_MISMATCH'
            continue
        if ref['packet_id'] not in frame['packets']:
            row['status'] = 'MISSING_CURRENT_PACKET'
            continue
        if 'required_actor_count' not in task:
            row['status'] = 'MISSING_REGISTERED_ACTOR_COUNT'
            continue
        packet = frame['packets'][ref['packet_id']]
        current = current_stratum(packet, row['direction'], task['required_actor_count'],
                                  topology, lanes, action['applied_states'])
        row['lane_id'] = packet['lane_id']
        row['required_actor_count'] = task['required_actor_count']
        if current is None:
            row['status'] = 'NO_ADJACENT_CURRENT_STRATUM'
            row['speed_mps'] = packet['speed_mps']
            row['lane_end_distance_m'] = topology.lanes[packet['lane_id']]['length_m']-packet['position_m']
            continue
        stratum, speed, distance = current
        row.update(status='RECORDED', current_stratum=[*stratum[:4], list(stratum[4]), *stratum[5:]],
                   speed_mps=speed, lane_end_distance_m=distance)


def _saved_training(base, tasks, rows, topology):
    root = base/'target_packet_positive_covered_registration/registration'
    by_index = {task['case_index']: task for task in tasks}
    grouped = defaultdict(list)
    for row in rows:
        path = by_index[row['case_index']].get('current_cache')
        if path is None:
            row['status'] = 'MISSING_REGISTERED_CURRENT_PATH'
        else:
            grouped[path].append(row)
    decoded, missing = 0, []
    for relative, selected in sorted(grouped.items()):
        path = root/relative
        if not path.is_file():
            missing.append(str(path))
            continue
        with gzip.open(path, 'rt') as stream:
            saved = json.load(stream)
        _record_current(selected, by_index, saved.get('frame', {}), saved.get('action', {}), topology)
        decoded += 1
    return dict(saved_current_frames_decoded=decoded, missing_saved_current_paths=missing)


def _archived_regression(base, tasks, rows, topology):
    by_index = {task['case_index']: task for task in tasks}
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ref = row['reference']
        grouped[ref['seed']][ref['time_sec']].append(row)
    decoded, streamed, missing = 0, 0, []
    for seed, times in sorted(grouped.items()):
        root = base/'target_packet_training/training'/f'seed{seed}'
        frame_path, action_path = (root/f'{name}_server_only.jsonl.gz'
                                   for name in ('frames', 'execution'))
        absent = [str(path) for path in (frame_path, action_path) if not path.is_file()]
        if absent:
            missing.extend(absent)
            continue
        with gzip.open(frame_path, 'rt') as frames, gzip.open(action_path, 'rt') as actions:
            for time, line in enumerate(frames):
                action_line = next(actions, None)
                streamed += 1
                if time in times:
                    if action_line is None:
                        for row in times[time]:
                            row['status'] = 'MISSING_CURRENT_ACTION'
                    else:
                        _record_current(times[time], by_index, json.loads(line), json.loads(action_line), topology)
                        decoded += 1
                if time >= max(times):
                    break
    return dict(archive_current_frames_decoded=decoded, archive_current_action_line_pairs_streamed=streamed,
                missing_archive_paths=missing)


def _coverage(rows, stays):
    groups = defaultdict(list)
    for stay in stays:
        if stay['status'] == 'RECORDED':
            groups[_stratum_key(stay)].append(stay)
    for row in rows:
        candidates = []
        if row['status'] == 'RECORDED':
            candidates = [other for other in groups[_stratum_key(row)]
                          if _identity(other) != _identity(row)]
        matches = [other for other in candidates
                   if abs(row['speed_mps']-other['speed_mps']) <= SPEED_TOLERANCE
                   and abs(row['lane_end_distance_m']-other['lane_end_distance_m']) <= DISTANCE_TOLERANCE]
        reason = ('MATCHED' if matches else row['status'] if row['status'] != 'RECORDED' else
                  'NO_TRAINING_STAY_IN_STRATUM' if not candidates else 'OUTSIDE_FIXED_TOLERANCES')
        row.update(same_stratum_training_stays=len(candidates), coverage_reason=reason,
            matching_train_stay_count=len(matches),
            matching_training_stay_case_indices=sorted(other['case_index'] for other in matches))


def _counts(rows):
    recorded = [row for row in rows if row['status'] == 'RECORDED']
    result = dict(direction_rows=len(rows), distinct_references=len({(r['cohort'], r['case_index']) for r in rows}),
        distinct_current_strata=len({_stratum_key(row) for row in recorded}),
        status_counts=dict(Counter(row['status'] for row in rows)),
        coverage_reason_counts=dict(Counter(row['coverage_reason'] for row in rows)),
        recorded=len(recorded), with_same_stratum_training_stay=sum(r['same_stratum_training_stays'] > 0 for r in rows),
        with_matching_training_stay=sum(r['matching_train_stay_count'] > 0 for r in rows),
        recorded_without_matching_training_stay=sum(r['matching_train_stay_count'] == 0 for r in recorded))
    if rows and 'objective_weight' in rows[0]:
        result['objective_weight'] = fsum(r['objective_weight'] for r in rows)
    return result


def _training_strata(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['status'] == 'RECORDED':
            grouped[_stratum_key(row)].append(row)
    return [dict(current_stratum=values[0]['current_stratum'], counts=_counts(values),
                 by_label={str(y):_counts([r for r in values if r['y'] == y]) for y in (0, 1, 2)},
                 by_seed=dict(Counter(r['reference']['seed'] for r in values)))
            for _, values in sorted(grouped.items())]


def coverage_readouts(base, training_references, regression_cases):
    """Return a small summary and server-only rows keyed by cohort/case/direction.

    Every registered case is retained. Training stay cases have both directions;
    matching reuses the original current-state strata and tolerances. Seed is
    omitted from strata, lane identity is preserved, and self matches are excluded.
    """
    base = Path(base)
    training_references, regression_cases = list(training_references), list(regression_cases)
    with (base/'target_packet_training/static_graph_server_only.json').open() as stream:
        topology = RoadTopology(json.load(stream))
    training, regression = _rows(training_references, 'training'), _rows(regression_cases, 'regression')
    retrieval = dict(training=_saved_training(base, training_references, training, topology),
                     regression=_archived_regression(base, regression_cases, regression, topology))
    stays = [row for row in training if row['y'] == 1]
    _coverage(training+regression, stays)
    summary = dict(training_references=len(training_references), regression_references=len(regression_cases),
        speed_tolerance_mps=SPEED_TOLERANCE, lane_end_distance_tolerance_m=DISTANCE_TOLERANCE,
        matching='Original current_stratum; exact lane/direction; seed omitted; all candidates, self excluded.',
        retrieval=retrieval, training_by_direction={}, regression_by_direction={}, training_strata=_training_strata(training))
    for direction, name in DIRECTIONS.items():
        selected = [row for row in training if row['direction'] == direction]
        summary['training_by_direction'][name] = dict(all=_counts(selected),
            positive=_counts([r for r in selected if r['y'] == direction]),
            stay=_counts([r for r in selected if r['y'] == 1]))
        selected = [row for row in regression if row['direction'] == direction]
        summary['regression_by_direction'][name] = dict(all=_counts(selected),
            by_role={role:_counts([r for r in selected if r['role'] == role]) for role in ('positive', 'stay')},
            covered_positive_300s_blocks=dict(Counter(str(r['reference']['time_sec']//300) for r in selected
                if r['role'] == 'positive' and r['matching_train_stay_count'] > 0)),
            by_300s_block={str(block):_counts([r for r in selected if r['reference']['time_sec']//300 == block])
                           for block in range(6)})
    return summary, training+regression
