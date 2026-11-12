"""Run the frozen short intent diagnostic from existing native archives."""

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import json
from pathlib import Path
from statistics import mean
import xml.etree.ElementTree as ET

from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology

from . import calibration
from .intent_replay import crossing_green, motion_transition_ids, select_cohort, simulate_arm


ARMS = ('baseline', 'lane_conditioned', 'route_oracle')
HORIZON = 30
CHECKPOINTS = (10, HORIZON)
_WORKER = None


def read_routes(cfg):
    """Read the explicit vehicle routes from the archive's original config."""
    route_file = cfg.parent / ET.parse(cfg).find('input/route-files').attrib['value']
    root = ET.parse(route_file).getroot()
    routes = {node.attrib['id']: tuple(node.find('route').attrib['edges'].split())
              for node in root.findall('vehicle')}
    if not routes:
        raise ValueError(f'No explicit vehicle routes in {route_file}')
    return routes


def oracle_at_anchor(topology, initial, routes):
    """Resolve only current-road choices identified by the frozen route."""
    choices = {}
    ordinary_count = 0
    for key, packet in initial['packets'].items():
        lane = topology.lanes[packet['lane_id']]
        if lane['internal']:
            continue
        ordinary_count += 1
        edge, route = lane['edge_id'], routes.get(key, ())
        outgoing = {route[i+1] if i+1 < len(route) else 'END'
                    for i, road in enumerate(route) if road == edge}
        if len(outgoing) == 1:
            choices[key] = next(iter(outgoing))
    return choices, ordinary_count


def select_rows(path, field, times):
    selected = {}
    last = max(times)
    for row in calibration.existing.json_lines(path):
        clock = row[field]
        if clock in times:
            if clock in selected:
                raise ValueError(f'Repeated {field}={clock} in {path}')
            selected[clock] = row
        if clock >= last:
            break
    if set(selected) != times:
        raise ValueError(f'Incomplete frozen window in {path}: missing {sorted(times - selected.keys())}')
    return selected


def _on_green(topology, lane, target, applied):
    if target is None:
        return False
    return any(c['tls_id'] is not None and applied[c['tls_id']][c['link_index']] in 'Gg'
               for c in topology.compatible(lane, target))


def native_checkpoints(topology, lanes, initial, frames, execution, cohort, choices):
    """Read first observed crossings and unique motion transitions for the cohort."""
    anchor = initial['time_s']
    cohort = set(cohort)
    origins = {key: topology.lanes[initial['packets'][key]['lane_id']]['edge_id'] for key in cohort}
    stops, starts, green_starts, crossed, entered, unresolved, exited = (set() for _ in range(7))
    checkpoints, actual_releases = {}, {}
    inserted = 0
    for offset in range(1, HORIZON + 1):
        before, after = frames[anchor+offset-1]['packets'], frames[anchor+offset]['packets']
        applied = execution[anchor+offset-1]['applied_states']
        release_ids = set()
        newly_halted, newly_started = motion_transition_ids(before, after, cohort)
        stops.update(newly_halted)
        starts.update(newly_started)
        for key in newly_started:
            lane = before[key]['lane_id']
            if key not in crossed and topology.lanes[lane]['edge_id'] == origins[key] and (
                    _on_green(topology, lane, choices.get(key), applied)):
                green_starts.add(key)
                release_ids.add(key)
        frame = frames[anchor+offset]
        inserted += len(frame['departed_ids'])
        exited.update(cohort.intersection(frame['arrived_ids']))
        for event in frame['events']:
            key = event['vehicle_id']
            if event['kind'] != 'lane_transition' or key not in cohort or key in crossed:
                continue
            path = event['observed_lane_path']
            crossing_index = next((i for i, lane in enumerate(path[1:]) if topology.lanes[lane]['internal']
                                   or topology.lanes[lane]['edge_id'] != origins[key]), None)
            if crossing_index is None:
                continue
            crossed.add(key)
            green = crossing_green(topology, path[crossing_index:], applied, next_edge=choices.get(key))
            if green is None:
                unresolved.add(key)
            elif green:
                entered.add(key)
                release_ids.add(key)
        actual_releases[anchor+offset-1] = tuple(sorted(release_ids))
        if offset in CHECKPOINTS:
            live = cohort & after.keys()
            checkpoints[str(offset)] = dict(
                halted_total=sum(frame['native_lanes'][lane]['H'] for lane in lanes),
                active_total=len(after), cohort_active=len(live),
                cohort_halted=sum(after[key]['speed_mps'] < .1 for key in live),
                cohort_moving_to_halt=len(stops), cohort_halt_to_moving=len(starts),
                cohort_green_starts=len(green_starts), cohort_crossed_stopline=len(crossed),
                cohort_green_entries=len(entered), cohort_crossing_green_unresolved=len(unresolved),
                cohort_terminal_exits=len(exited), inserted=inserted)
    return checkpoints, actual_releases


def load_archives(archive, static):
    collection = archive/'collection'
    receipt = calibration.read(collection/'result.json')
    expected = set(calibration.existing.CALIBRATION_SEEDS)
    cases = {row['seed']: row['case'] for row in receipt['seeds'] if row['status'] == 'COMPLETE'}
    if receipt['status'] != 'COMPLETE' or set(cases) != expected:
        raise ValueError('Both archived native calibration seeds are required')
    cfgs = {Path(case['sumocfg']) for case in cases.values()}
    if len(cfgs) != 1:
        raise ValueError('The native calibration archive did not use one common config')
    routes = read_routes(next(iter(cfgs)))
    topology = RoadTopology(static)
    anchors = calibration.existing.ANCHORS
    frame_times = {t for anchor in anchors for t in range(anchor, anchor+HORIZON+1)}
    execution_times = {t for anchor in anchors for t in range(anchor, anchor+HORIZON)}
    windows = {}
    for seed in calibration.existing.CALIBRATION_SEEDS:
        directory = collection/f'seed{seed}'
        frames = select_rows(directory/'frames_server_only.jsonl.gz', 'time_s', frame_times)
        execution = select_rows(directory/'execution_server_only.jsonl.gz', 'time_sec', execution_times)
        executors = calibration.read(directory/'executor_seed_server_only.json')
        for anchor in anchors:
            initial = calibration.initialize_observation(frames[anchor], anchor)
            cohort = select_cohort(static, initial)
            choices, ordinary_count = oracle_at_anchor(topology, initial, routes)
            window_execution = {t: execution[t] for t in range(anchor, anchor+HORIZON)}
            checkpoints, releases = native_checkpoints(topology, static['controlled_lane_ids'], initial,
                                                        frames, window_execution, cohort, choices)
            windows[seed, anchor] = dict(
                initial=initial, executor_seeds=executors, execution=window_execution,
                oracle_choices=choices, actual_releases=releases,
                native=dict(cohort_size=len(cohort), oracle_ordinary_covered=len(choices),
                            ordinary_at_anchor=ordinary_count,
                            oracle_cohort_covered=sum(key in choices for key in cohort),
                            checkpoints=checkpoints))
        print(json.dumps(dict(stage='archive_loaded', seed=seed, anchors=list(anchors))), flush=True)
    return windows


def _init_worker(base, static, windows, model, factory):
    global _WORKER
    _WORKER = base, static, windows, model, factory


def _run_case(seed, anchor, scenario, arm):
    base, static, windows, model, factory = _WORKER
    window = windows[seed, anchor]
    choices = window['oracle_choices'] if arm == 'route_oracle' else {}
    result = simulate_arm(base, static, window['initial'], window['executor_seeds'], model, scenario,
                          arm, window['execution'], choices, factory=factory,
                          actual_releases=window['actual_releases'], audit_caps=True)
    steps = len(result['due_counts'])
    if steps != HORIZON:
        raise ValueError(f'Expected {HORIZON} recorded model steps, got {steps}')
    return dict(result, status='COMPLETE', checked_steps=steps, seed=seed,
                anchor_sec=anchor, scenario_seed=scenario, arm=arm)


def _gate(windows):
    by_seed = {}
    for seed in calibration.existing.CALIBRATION_SEEDS:
        rows = [row for row in windows if row['seed'] == seed]
        by_seed[str(seed)] = {}
        for arm in ARMS:
            values = [row['arms'][arm]['errors_at_30'] for row in rows]
            by_seed[str(seed)][arm] = {field: mean(row[field] for row in values)
                                      for field in ('signed_halt_error', 'positive_halt_excess',
                                                    'absolute_halt_error', 'absolute_green_entry_error')}
    candidates = {}
    for arm in ARMS[1:]:
        effects = {}
        for seed, values in by_seed.items():
            baseline, candidate = values['baseline'], values[arm]
            effects[seed] = {key: baseline[key]-candidate[key] for key in baseline}
        passed = all(all(deltas[field] > 0 for field in ('positive_halt_excess', 'absolute_halt_error',
                                                        'absolute_green_entry_error'))
                     for deltas in effects.values())
        candidates[arm] = dict(pass_both_seeds=passed, seed_improvement=effects)
    return dict(scope='Strict paired improvement on each seed at +30 seconds, equally weighting its two starts.',
                per_seed=by_seed, candidates=candidates)


def summarize(inputs, rows):
    windows = []
    for (seed, anchor), context in inputs.items():
        window = dict(seed=seed, anchor_sec=anchor, native=context['native'], arms={})
        windows.append(window)
        for arm in ARMS:
            group = [row for row in rows if row['status'] == 'COMPLETE' and
                     (row['seed'], row['anchor_sec'], row['arm']) == (seed, anchor, arm)]
            if len(group) != len(calibration.existing.SCENARIOS):
                continue
            summary = {str(t): {key: mean(row['checkpoints'][str(t)][key] for row in group)
                                for key in context['native']['checkpoints'][str(t)]}
                       for t in CHECKPOINTS}
            for t in map(str, CHECKPOINTS):
                caps = [row['checkpoints'][t]['caps'] for row in group]
                summary[t]['mean_caps_on_native_green_release'] = {
                    key: mean(draw.get(key, 0) for draw in caps)
                    for key in sorted({key for draw in caps for key in draw})}
            counts = [row['intent_counts'] for row in group]
            actual, predicted = context['native']['checkpoints']['30'], summary['30']
            halt = predicted['halted_total']-actual['halted_total']
            entry = predicted['cohort_green_entries']-actual['cohort_green_entries']
            window['arms'][arm] = dict(checkpoints=summary,
                mean_intent_counts={key: mean(draw[key] for draw in counts) for key in counts[0]},
                errors_at_30=dict(
                signed_halt_error=halt, positive_halt_excess=max(halt, 0), absolute_halt_error=abs(halt),
                signed_green_entry_error=entry, absolute_green_entry_error=abs(entry)))
    return windows


def run(base, archive, output, workers):
    output.mkdir(parents=True, exist_ok=False)
    cases_dir = output/'cases'
    cases_dir.mkdir()
    expected = len(calibration.existing.CALIBRATION_SEEDS)*len(calibration.existing.ANCHORS)*len(
        calibration.existing.SCENARIOS)*len(ARMS)
    result = dict(status='RUNNING', horizon_sec=HORIZON, planned_cases=expected,
                  planned_model_steps=expected*HORIZON, completed_cases=0, checked_model_steps=0,
                  new_native_starts=0, new_fits=0, workers=workers, failure=None,
                  native_seeds=list(calibration.existing.CALIBRATION_SEEDS),
                  anchor_seconds=list(calibration.existing.ANCHORS),
                  scenario_seeds=list(calibration.existing.SCENARIOS), arms=list(ARMS),
                  green_entry='First stopline crossing by an anchor-cohort vehicle with applied G/g at the physical link.',
                  signal_source='Archived native applied signal states for each second.')
    calibration.save(output/'result.json', result)
    rows, inputs = [], {}
    try:
        static, _ = calibration.load_widths(calibration.read(archive/'static_graph_server_only.json'))
        model, factory = calibration.load_model(base), calibration.kernel_factory(base, static)
        inputs = load_archives(archive, static)
        tasks = deque((seed, anchor, scenario, arm) for seed, anchor in inputs
                      for scenario in calibration.existing.SCENARIOS for arm in ARMS)
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                 initargs=(base, static, inputs, model, factory)) as pool:
            pending = {}
            while pending or (tasks and not result['failure']):
                while tasks and len(pending) < workers and not result['failure']:
                    case = tasks.popleft()
                    pending[pool.submit(_run_case, *case)] = case
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    seed, anchor, scenario, arm = pending.pop(future)
                    try:
                        row = future.result()
                    except Exception as exc:
                        row = dict(status='FAILED', seed=seed, anchor_sec=anchor, scenario_seed=scenario,
                                   arm=arm, checked_steps=0, error=f'{type(exc).__name__}: {exc}')
                    calibration.save(cases_dir/f'seed{seed}_t{anchor}_scenario{scenario}_{arm}.json', row)
                    rows.append(row)
                    if row['status'] != 'COMPLETE' and result['failure'] is None:
                        result['failure'] = row
                    print(json.dumps(dict(stage='short_replay_case', seed=seed, anchor_sec=anchor,
                                          scenario_seed=scenario, arm=arm, status=row['status'],
                                          error=row.get('error'))), flush=True)
    except Exception as exc:
        result['failure'] = dict(error=f'{type(exc).__name__}: {exc}')
    try:
        result['windows'] = summarize(inputs, rows)
        result['completed_cases'] = sum(row['status'] == 'COMPLETE' for row in rows)
        result['checked_model_steps'] = sum(row.get('checked_steps', 0) for row in rows)
        if result['completed_cases'] == expected and result['checked_model_steps'] == result['planned_model_steps']:
            due_streams = {}
            for row in rows:
                pair = row['seed'], row['anchor_sec'], row['scenario_seed']
                if due_streams.setdefault(pair, row['due_counts']) != row['due_counts']:
                    raise ValueError(f'Shared demand stream diverged in {pair}')
            result['gate'] = _gate(result['windows'])
            result['shared_due_streams_checked'] = len(due_streams)
            unresolved = any(window['native']['checkpoints']['30']['cohort_crossing_green_unresolved']
                             or any(arm['checkpoints']['30']['cohort_crossing_green_unresolved']
                                    for arm in window['arms'].values()) for window in result['windows'])
            result['gate']['all_crossing_signals_resolved'] = not unresolved
            result['status'] = 'COMPLETE_SHORT_INTENT_DIAGNOSTIC'
        else:
            result['status'] = 'HOLD_SHORT_INTENT_REPLAY'
    except Exception as exc:
        result.update(status='HOLD_SHORT_INTENT_REPLAY', failure=dict(error=f'{type(exc).__name__}: {exc}'))
    calibration.save(output/'result.json', result)
    print(json.dumps(dict(stage='short_replay_panel', status=result['status'],
                          completed_cases=result['completed_cases'],
                          checked_model_steps=result['checked_model_steps'],
                          failure=result['failure'])), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=range(1, 13), default=12)
    args = parser.parse_args()
    result = run(args.base, args.archive, args.output, args.workers)
    return 0 if result['status'] == 'COMPLETE_SHORT_INTENT_DIAGNOSTIC' else 2


if __name__ == '__main__':
    raise SystemExit(main())
