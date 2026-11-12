"""Independent frozen packet-model calibration; no fitting or native starts."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
import gzip
import json
from math import isfinite
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.kernel import PacketKernel
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation
from experiments.tsc_target_packet_protocol_20261109.calibration_gates import evaluate_calibration

CALIBRATION_SEEDS = (76116911, 76116912)
ANCHORS = (600, 1200)
SCENARIOS = tuple(range(76117001, 76117009))
HORIZON = 450


def require(condition, message):
    if not condition:
        raise ValueError(message)


class StructuralAudit:
    """Independently check each predicted transition and original actuator."""
    def __init__(self, static, state, executor_seeds):
        self.static, self.topology = static, RoadTopology(static)
        self.operator = PacketObserver(static)
        self.post = state
        api = type('MemoryAPI', (), {})()
        api.trafficlight = type('Signals', (), dict(setRedYellowGreenState=lambda *_: None))()
        api.lane = type('Lanes', (), {})()
        api.lane.getLastStepVehicleNumber = lambda lane: self.post['lane_sensors'][lane]['N']
        self.executors = {}
        for tls, seed in executor_seeds.items():
            snapshot = dict(seed['snapshot'], **state['executors'][tls], tls_id=tls)
            clone = SafePhaseExecutor(sumo_api=api, tls_id=tls, timings=[PhaseTiming(**row) for row in seed['timings']],
                initial_state=snapshot['current_state'], clearance_lane_ids=seed['clearance_lane_ids'],
                clearance_poll_sec=seed['clearance_poll_sec'])
            clone.restore(snapshot, write_state=False)
            self.executors[tls] = clone
        self.integral = 0
        self.steps = 0
        self.seen = set(state['packets']) | {row['packet_id'] for row in state['pending']}

    def _pp(self, state):
        result = {}
        for tls, executor in self.executors.items():
            mapping = self.topology.pp_lane_by_link[tls]
            incoming = self.static.get('incoming_lanes_by_tls', {}).get(tls)
            if incoming is None:
                incoming = list(dict.fromkeys(mapping[index] for index in sorted(mapping)))
            def score(phase):
                permitted = {mapping[index] for index, signal in enumerate(phase)
                             if signal in 'Gg' and index in mapping}
                return sum(state['lane_sensors'][lane]['H']+.35*state['lane_sensors'][lane]['N']+
                           .025*state['lane_sensors'][lane]['O'] for lane in incoming
                           if lane in permitted)
            result[tls] = max(executor.feasible_states_now(), key=score)
        return result

    def check(self, before, after, row, initial_requests=None):
        require(after['time_s'] == before['time_s']+1 == row['time_s'], 'Predicted one-second clock drift')
        a0, p0 = set(before['packets']), {p['packet_id'] for p in before['pending']}
        a1, p1 = set(after['packets']), {p['packet_id'] for p in after['pending']}
        due, inserted, exited = (row[key] for key in ('due_ids', 'inserted_ids', 'exited_ids'))
        require(all(len(values) == len(set(values)) for values in (due, inserted, exited)), 'Repeated event identity')
        require(not a0 & p0 and not a1 & p1 and not self.seen & set(due), 'Active/pending/due identity overlap or reused identity')
        require(set(inserted) <= p0 | set(due) and set(exited) <= a0 | set(inserted), 'Impossible insertion or exit')
        active = Counter(a0); active.update(inserted); active.subtract(exited)
        pending = Counter(p0); pending.update(due); pending.subtract(inserted)
        require(active == Counter(a1) and pending == Counter(p1), 'Independent packet identity ledger failed')
        require(row['ledger_passed'], 'Kernel reported its own failed identity ledger')
        self.seen.update(due)
        events = {event['vehicle_id']: event for event in row['events'] if event['kind'] == 'lane_transition'}
        for key, packet in after['packets'].items():
            geometry = self.static['lanes'][packet['lane_id']]
            require(key == packet['packet_id'] and packet['body_path_complete'], 'Unclosed predicted packet/body identity')
            require(all(isfinite(packet[name]) for name in ('position_m', 'speed_mps', 'acceleration_mps2')) and
                    0 <= packet['position_m'] <= geometry['length_m']+1e-8 and packet['speed_mps'] >= 0,
                    'Nonfinite or out-of-lane predicted packet')
            body = packet['body_trail']
            require(body and body[-1] == packet['lane_id'], 'Predicted body trail does not end at front lane')
            require(all(b in self.static['successors'][a] for a, b in zip(body, body[1:])), 'Illegal body accounting path')
            if key in before['packets'] and before['packets'][key]['lane_id'] != packet['lane_id']:
                require(key in events, 'Changed lane without a front-path event')
                path = events[key]['observed_lane_path']
                require(path[0] == before['packets'][key]['lane_id'] and path[-1] == packet['lane_id'],
                        'Front-path event endpoints changed')
                require(all(b in self.static['successors'][a] or self.topology.adjacent(a, 1) == b or
                            self.topology.adjacent(a, -1) == b for a, b in zip(path, path[1:])), 'Illegal front transition')
        sensors = self.operator._reconstruct(after['packets'])
        require(set(sensors) == set(after['lane_sensors']), 'Predicted dense lane roster changed')
        for lane, expected in sensors.items():
            actual = after['lane_sensors'][lane]
            require(all(actual[key] == expected[key] for key in ('ids', 'N', 'H')), 'Predicted N/H/IDs not derived from packets')
            require(all(isfinite(actual[key]) and abs(actual[key]-expected[key]) <= 1e-10
                        for key in ('O', 'mean_speed_mps')), 'Predicted occupancy/speed sensor mismatch')
        require(sum(value['N'] for value in sensors.values()) == len(a1), 'Predicted packet mass counted twice')
        requests = self._pp(before) if before['time_s'] % 10 == 0 else {}
        if initial_requests is not None:
            requests.update(initial_requests)
        require(row['requests'] == requests, 'Predicted PP request or cadence mismatch')
        returns = {tls: bool(self.executors[tls].request(request)) for tls, request in sorted(requests.items())}
        require(row['request_returns'] == returns, 'Predicted actuator request acceptance mismatch')
        require(row['applied_signals'] == {tls: ex.current_state for tls, ex in self.executors.items()},
                'Predicted signal was taken after movement instead of before movement')
        self.post = after
        for tls, executor in self.executors.items():
            executor.advance(1.)
            require(after['executors'][tls] == executor.snapshot() == row['executors'][tls],
                    'Original executor replay failed on predicted internal N')
        halted = sum(sensors[lane]['H'] for lane in self.static['controlled_lane_ids'])
        self.integral += halted
        self.steps += 1
        require(row['halted_total'] == halted and row['I_accum'] == self.integral, 'Predicted poststep integral mismatch')
        require(row['active_count'] == len(a1) and row['pending_count'] == len(p1), 'Predicted summary population mismatch')
        require(row['terminal_count'] == len(exited), 'Predicted terminal summary mismatch')
        terminal_events = [event for event in row['events'] if event['kind'] == 'terminal_exit']
        require(Counter(event['vehicle_id'] for event in terminal_events) == Counter(exited),
                'Predicted terminal events do not identify every exit')
        require(row['nonsink_terminal_count'] == sum(bool(self.static['successors'][event['from_lane']])
                                                   for event in terminal_events), 'Predicted nonsink terminal summary mismatch')


def _run_draw(task, kernel_cls=None):
    """This function receives no truth/execution streams or demand schedules."""
    kernel_cls = kernel_cls or PacketKernel
    static, initial, executor_seeds, model, seed, anchor, scenario = task
    result = dict(seed=seed, anchor_sec=anchor, scenario_seed=scenario, status='STARTED', checked_steps=0)
    kernel = before = halted = active = pending = None
    try:
        kernel = kernel_cls(static, deepcopy(initial), executor_seeds, model['target_laws'],
                            coefficients=model['coefficients'], random_seed=scenario)
        audit = StructuralAudit(static, kernel.state, executor_seeds)
        halted = [sum(kernel.state['lane_sensors'][lane]['H'] for lane in static['controlled_lane_ids'])]
        active, pending = [len(kernel.state['packets'])], [len(kernel.state['pending'])]
        terminals = nonsinks = 0
        for _ in range(HORIZON):
            before = deepcopy(kernel.state)
            row = kernel.step()
            audit.check(before, kernel.state, row)
            halted.append(row['halted_total']); active.append(row['active_count']); pending.append(row['pending_count'])
            terminals += len(row['exited_ids'])
            nonsinks += sum(bool(static['successors'][event['from_lane']])
                            for event in row['events'] if event['kind'] == 'terminal_exit')
            result['checked_steps'] += 1
        result.update(status='COMPLETE', structural_passed=True,
            halted_total=halted, active_count=active, pending_count=pending,
            terminal_count=terminals, nonsink_terminal_count=nonsinks,
            I450=audit.integral, C450=audit.integral/(HORIZON*len(static['controlled_lane_ids'])),
            diagnostics=dict(kernel.diagnostics), parameter_origin=getattr(kernel.model, 'parameter_origin', None))
    except Exception as exc:
        result.update(status='FAILED', structural_passed=False, error=f'{type(exc).__name__}: {exc}',
            failure_evidence=dict(partial_halted_total=deepcopy(halted), partial_active_count=deepcopy(active),
                partial_pending_count=deepcopy(pending), last_before_state=deepcopy(before),
                last_after_state=deepcopy(kernel.state) if kernel is not None else None))
    return result


def _free_tasks(tasks, workers, on_draw, kernel_cls):
    records = [dict(seed=task[4], anchor_sec=task[5], scenario_seed=task[6], status='NOT_RUN',
                    structural_passed=False, checked_steps=0) for task in tasks]
    def started(index):
        records[index]['status'] = 'STARTED'
        if on_draw:
            on_draw(deepcopy(records[index]))
    def completed(index, result):
        records[index] = result
        if on_draw:
            on_draw(deepcopy(result))
    if workers == 1:
        for index, task in enumerate(tasks):
            started(index)
            result = _run_draw(task, kernel_cls)
            completed(index, result)
            if result['status'] != 'COMPLETE':
                break
    else:
        require(kernel_cls is None, 'Synthetic class injection uses one worker')
        with ProcessPoolExecutor(max_workers=workers) as pool:
            inflight, next_index, stopped = {}, 0, False
            while next_index < len(tasks) and len(inflight) < workers:
                started(next_index)
                inflight[pool.submit(_run_draw, tasks[next_index])] = next_index
                next_index += 1
            while inflight:
                done, unused = wait(inflight, return_when=FIRST_COMPLETED)
                batch = []
                for future in sorted(done, key=lambda item: inflight[item]):
                    index = inflight.pop(future)
                    try:
                        value = future.result()
                    except Exception as exc:
                        value = dict(records[index], status='FAILED', structural_passed=False,
                                     error=f'{type(exc).__name__}: {exc}')
                    completed(index, value)
                    batch.append(value)
                stopped |= any(row['status'] != 'COMPLETE' for row in batch)
                while not stopped and next_index < len(tasks) and len(inflight) < workers:
                    started(next_index)
                    inflight[pool.submit(_run_draw, tasks[next_index])] = next_index
                    next_index += 1
    for record in records:
        if record['status'] == 'NOT_RUN' and on_draw:
            on_draw(deepcopy(record))
    return records


def _truth_window(static, frames, seed, anchor):
    halted, active, terminals, nonsinks = [], [], 0, 0
    for time_s in range(anchor, anchor+HORIZON+1):
        frame = frames[time_s]
        halted.append(sum(frame['native_lanes'][lane]['H'] for lane in static['controlled_lane_ids']))
        active.append(len(frame['native_active_ids']))
        if time_s > anchor:
            prior = frames[time_s-1]['packets']
            for key in frame['arrived_ids']:
                require(key in prior, 'An actual terminal has no preceding observed lane')
                terminals += 1
                nonsinks += bool(static['successors'][prior[key]['lane_id']])
    return dict(halted_total=halted, active_count=active, terminal_count=terminals, nonsink_terminal_count=nonsinks)


def _one_step(static, before, actual, execution, executor_seeds, model, kernel_cls=None):
    kernel_cls = kernel_cls or PacketKernel
    initial = initialize_observation(before, before['time_s'])
    kernel = kernel_cls(static, initial, executor_seeds, model['target_laws'],
                        coefficients=model['coefficients'], random_seed=SCENARIOS[0])
    audit = StructuralAudit(static, kernel.state, executor_seeds)
    observed_requests = execution['requests']
    previous = deepcopy(kernel.state)
    row = kernel.step(initial_requests=observed_requests)
    audit.check(previous, kernel.state, row, initial_requests=observed_requests)
    require(row['applied_signals'] == execution['applied_states'], 'Conditional one-step initial action/signal mismatch')
    survivors = set(before['packets']) & set(actual['packets'])
    matched = survivors & set(kernel.state['packets'])
    speed_error = sum(abs(kernel.state['packets'][key]['speed_mps']-actual['packets'][key]['speed_mps']) for key in matched)
    errors = {key: sum(abs(kernel.state['lane_sensors'][lane][key]-actual['native_lanes'][lane][key])
                       for lane in static['lanes']) for key in ('N', 'H', 'O')}
    return dict(observed_survivors=len(survivors), speed_pairs=len(matched),
        missing_predicted_survivors=len(survivors-matched), speed_absolute_error_sum=speed_error,
        lane_error_sums=errors, lane_pairs=len(static['lanes']),
        predicted_inserted=len(row['inserted_ids']), actual_inserted=len(actual['departed_ids']),
        predicted_terminals=len(row['exited_ids']), actual_terminals=len(actual['arrived_ids']))


def _one_step_panel(static, inputs, model, kernel_cls):
    rows = []
    for seed in CALIBRATION_SEEDS:
        source = inputs[seed]
        summary = dict(seed=seed, samples=0, observed_survivors=0, speed_pairs=0,
            missing_predicted_survivors=0, speed_absolute_error_sum=0., lane_pairs=0,
            lane_error_sums=dict(N=0., H=0., O=0.), predicted_inserted=0,
            actual_inserted=0, predicted_terminals=0, actual_terminals=0, failures=[])
        for time_s in range(0, 1800, 10):
            try:
                value = _one_step(static, source['frames'][time_s], source['frames'][time_s+1],
                    source['execution'][time_s], source['executor_seeds'], model, kernel_cls)
                summary['samples'] += 1
                for name in ('observed_survivors', 'speed_pairs', 'missing_predicted_survivors',
                             'speed_absolute_error_sum', 'lane_pairs', 'predicted_inserted', 'actual_inserted',
                             'predicted_terminals', 'actual_terminals'):
                    summary[name] += value[name]
                for name in ('N', 'H', 'O'):
                    summary['lane_error_sums'][name] += value['lane_error_sums'][name]
            except Exception as exc:
                summary['failures'].append(dict(time_s=time_s, error=f'{type(exc).__name__}: {exc}'))
                break
        summary['speed_MAE'] = summary['speed_absolute_error_sum']/summary['speed_pairs'] if summary['speed_pairs'] else None
        summary['survivor_prediction_coverage'] = summary['speed_pairs']/summary['observed_survivors'] if summary['observed_survivors'] else None
        summary['lane_MAE'] = {name: total/summary['lane_pairs'] if summary['lane_pairs'] else None
                               for name, total in summary['lane_error_sums'].items()}
        rows.append(summary)
    return dict(status='PASS_CONDITIONAL_DIAGNOSTIC' if all(not row['failures'] for row in rows) else 'HOLD_CONDITIONAL_DIAGNOSTIC',
                scenario_seed=SCENARIOS[0], timestep_interval_sec=10, per_seed=rows,
                scope='Observed current state and executed current action, one step only; separate from free-rollout gates')


def evaluate_panel(static, seed_inputs, model, protocol, *, workers=1, on_draw=None, kernel_cls=None):
    evaluation = protocol['calibration']
    require(tuple(protocol['cohorts']['calibration_seeds']) == CALIBRATION_SEEDS and
            tuple(evaluation['anchors_sec']) == ANCHORS and evaluation['horizon_sec'] == HORIZON and
            tuple(evaluation['scenario_random_seeds']) == SCENARIOS and evaluation['model_scenarios'] == 8,
            'The fixed two-seed, two-anchor, eight-scenario panel is required')
    require(1 <= workers <= 4 and set(seed_inputs) == set(CALIBRATION_SEEDS), 'Use both frozen calibration seeds and at most four workers')
    require(sorted(model['training_seeds']) == sorted(protocol['cohorts']['training_seeds']) and
            not set(model['training_seeds']) & set(CALIBRATION_SEEDS), 'Model training/calibration seed leakage')
    require(len(static['controlled_lane_ids']) == protocol['cost']['lane_count'] == 144, 'The unchanged 144-lane cost roster is required')
    indexed = {}
    for seed in CALIBRATION_SEEDS:
        source = seed_inputs[seed]
        frames, execution = {}, {}
        for frame in source['frames']:
            time_s = frame['time_s']
            require(time_s not in frames and not frame['unresolved'] and all(frame['checks'].values()), 'Incomplete or repeated calibration frame')
            frames[time_s] = frame
        for row in source['execution']:
            require(row['time_sec'] not in execution, 'Repeated calibration execution timestamp')
            execution[row['time_sec']] = row
        require(set(frames) == set(range(1801)) and set(execution) == set(range(1800)), 'Calibration requires every native second0..1800')
        indexed[seed] = dict(frames=frames, execution=execution, executor_seeds=source['executor_seeds'])
    tasks = []
    for seed in CALIBRATION_SEEDS:
        source = indexed[seed]
        for anchor in ANCHORS:
            initial = initialize_observation(source['frames'][anchor], anchor)
            tasks.extend((static, initial, source['executor_seeds'], model, seed, anchor, scenario) for scenario in SCENARIOS)
    draws = _free_tasks(tasks, workers, on_draw, kernel_cls)
    structurally_complete = all(row['status'] == 'COMPLETE' and row['structural_passed'] for row in draws)
    windows, numerical, one_step = [], None, None
    if structurally_complete:
        # This is the first stage that reads held future outcome fields. No
        # truth object or execution stream was passed to a free worker.
        for seed in CALIBRATION_SEEDS:
            for anchor in ANCHORS:
                windows.append(dict(seed=seed, anchor_sec=anchor,
                    actual=_truth_window(static, indexed[seed]['frames'], seed, anchor),
                    predicted_scenarios=[row for row in draws if row['seed'] == seed and row['anchor_sec'] == anchor]))
        numerical = evaluate_calibration(windows, protocol)
        one_step = _one_step_panel(static, indexed, model, kernel_cls)
    conditional_complete = one_step is not None and all(row['samples'] == 180 and not row['failures']
                                                        for row in one_step['per_seed'])
    passed = structurally_complete and numerical['status'] == 'PASS_NUMERICAL_CALIBRATION' and conditional_complete
    compact_draws = [{key: value for key, value in row.items() if key not in
                     ('halted_total', 'active_count', 'pending_count', 'failure_evidence')}
                     for row in draws]
    return dict(status='PASS_TARGET_PACKET_TECHNICAL_CALIBRATION' if passed else 'HOLD_TARGET_PACKET_CALIBRATION',
        checks=dict(all_scenarios_structurally_complete=structurally_complete,
                    numerical_calibration_passed=numerical is not None and numerical['status'] == 'PASS_NUMERICAL_CALIBRATION',
                    conditional_records_complete=conditional_complete),
        planned_draws=32, completed_draws=sum(row['status'] == 'COMPLETE' for row in draws),
        failed_draws=sum(row['status'] == 'FAILED' for row in draws), unstarted_draws=sum(row['status'] == 'NOT_RUN' for row in draws),
        configured_workers=workers, planned_free_model_seconds=32*HORIZON,
        checked_free_model_seconds=sum(row['checked_steps'] for row in draws),
        planned_conditional_steps=360,
        completed_conditional_steps=sum(row['samples'] for row in one_step['per_seed']) if one_step else 0,
        draws=compact_draws, numerical_calibration=numerical, one_step=one_step,
        new_native_starts=0, new_real_data_fits=0, source_data_used=False,
        scope='Frozen technical calibration only; no action selection, PP superiority, source benefit or full-hour control test')


def json_lines(path):
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            yield json.loads(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--seed-dir', action='append', required=True, metavar='SEED=DIRECTORY')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {}
    def persist(draw):
        key = f"seed{draw['seed']}_t{draw['anchor_sec']}_scenario{draw['scenario_seed']}"
        (args.output/f'{key}_server_only.json').write_text(json.dumps(draw, indent=2, allow_nan=False)+'\n')
        manifest[key] = {name: draw.get(name) for name in ('seed', 'anchor_sec', 'scenario_seed', 'status', 'checked_steps', 'error')}
        (args.output/'draw_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    try:
        inputs = {}
        for value in args.seed_dir:
            seed_text, path_text = value.split('=', 1)
            seed, directory = int(seed_text), Path(path_text)
            require(seed not in inputs, 'Repeated seed directory')
            inputs[seed] = dict(frames=json_lines(directory/'frames_server_only.jsonl.gz'),
                execution=json_lines(directory/'execution_server_only.jsonl.gz'),
                executor_seeds=json.loads((directory/'executor_seed_server_only.json').read_text()))
        result = evaluate_panel(json.loads(args.static.read_text()), inputs, json.loads(args.model.read_text()),
            json.loads(args.protocol.read_text()), workers=args.workers, on_draw=persist)
    except Exception as exc:
        result = dict(status='HOLD_TARGET_PACKET_EVALUATION_ERROR', error=f'{type(exc).__name__}: {exc}',
                      new_native_starts=0, new_real_data_fits=0)
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'])
    return 0 if result['status'] == 'PASS_TARGET_PACKET_TECHNICAL_CALIBRATION' else 2


if __name__ == '__main__':
    raise SystemExit(main())
