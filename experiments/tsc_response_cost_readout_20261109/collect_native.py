"""Collect the frozen Hangzhou10s/Jinan450s native action-response cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace

from advance_native import advance_horizon, _endpoint
from lane_interface import build_inputs, extract_targets, read_lane_static, record_vehicle_params, _state

HERE = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, allow_nan=False, separators=(',', ':'))+'\n')


def cases(protocol, input_base):
    budget = protocol['proposed_native_budget']
    result = []
    for city, seeds, horizon, filename in (
        ('hangzhou', budget['source_seeds'], 10, 'hangzhou_aligned.sumocfg'),
        ('jinan', budget['target_seeds'], 450, 'jinan_3x4_real.sumocfg')):
        for j, seed in enumerate(seeds):
            for k, anchor in enumerate(budget['anchors']):
                result.append(dict(case_id=f'{city}_{seed}_{anchor}', city=city, seed=seed,
                    anchor=anchor, checkpoint_sec=anchor, checkpoint_search_sec=60,
                    tls_id=budget['tls_pool'][(2*j+k) % len(budget['tls_pool'])],
                    horizon_sec=horizon, sumocfg=str(input_base/'inputs'/city/filename)))
    if len(result)*8 != budget['total_native_starts']:
        raise ValueError('Case roster differs from the frozen224 native-start budget')
    return result


def nominal_plan(executor, request):
    """Advance an isolated executor copy; SUMO and future lane state are never read."""
    from cf_h2o.traffic_signal.safe_phase_controller import SafePhaseExecutor
    memory = SimpleNamespace(trafficlight=SimpleNamespace(setRedYellowGreenState=lambda tls, state: None))
    clone = SafePhaseExecutor(sumo_api=memory, tls_id=executor.tls_id,
        timings=list(executor.timings.values()), initial_state=executor.current_state,
        clearance_lane_ids=(), clearance_poll_sec=executor.clearance_poll_sec)
    clone.restore(executor.snapshot(), write_state=False)
    accepted, signals = bool(clone.request(request)), []
    for _ in range(10):
        signals.append(clone.current_state)
        clone.advance(1.)
    endpoint = dict(snapshot=clone.snapshot(), feasible_states=list(clone.feasible_states_now()),
        minimum_green_sec=float(clone.timings[clone.current_green_state].min_green_sec))
    return dict(states=signals, request_accepted=accepted, executor10=endpoint)


def collect_branch(api, v3, case, candidate_index, output, lane_static):
    import collect_tsc_physical_response_trial as native
    from cologne_action_branch_checks import physical_snapshot
    from tsc_joint_response_observer import JointResponseObserver
    from tsc_physical_response_data import build_preaction_rows, extract_response_labels

    infos, graph, context, executors, states, prefix = native.native_prefix(api, v3, case)
    candidates, reference = native.candidate_roster(v3, states, executors, case)
    index = reference if candidate_index is None else candidate_index
    t0 = int(api.simulation.getTime())
    start = physical_snapshot(api, executors)
    start['pending_ids'] = sorted(api.simulation.getPendingVehicles())
    lanes = tuple(sorted(v3._controlled_lane_set(infos)))
    actions = {tls: v3._rule_candidate(states[tls], executors[tls], 'phase_pressure').state
               for tls in sorted(infos)}
    actions[case['tls_id']] = candidates[index]
    observer = JointResponseObserver(api, infos, t0, windows=(10,), focal_tls=case['tls_id'])
    preaction = build_preaction_rows(api, infos, executors, actions, observer.movements,
                                    focal_tls=case['tls_id'], horizon_sec=10)
    network = native.network_preaction(v3, states, executors, graph)
    observer.sample(t0)
    endpoint0 = _endpoint(api, executors, sorted(set(lanes) | set(observer.lane_geometry)), t0)
    params = record_vehicle_params(api, seen_ids=sorted(observer.frames[0]['vehicles']))
    for vid, record in params.items():
        row = observer.frames[0]['vehicles'][vid]
        if row['lane_id'] != record['lane_id']:
            raise ValueError('Current vehicle parameter/frame lane join changed at t0')
        row['speed_mps'] = record['speed_mps']
    for tls, executor in executors.items():
        endpoint0['executors'][tls]['timings'] = [asdict(timing) for timing in executor.timings.values()]
    nominal = nominal_plan(executors[case['tls_id']], actions[case['tls_id']])
    if nominal['states'] != preaction['nominal_states'][case['tls_id']]:
        raise ValueError('Isolated nominal sequence differs from the admitted preaction plan')
    inputs = build_inputs(endpoint0, observer.frames[0], observer.movements, lane_static,
                          preaction['nominal_states'], case['tls_id'], params)
    inputs.update(global_nodes={lane: _state(endpoint0['lanes'][lane], lane, lane_static) for lane in lanes},
        nominal_executor10=nominal['executor10'],
        request_differs=actions[case['tls_id']] != executors[case['tls_id']].current_green_state)
    if not inputs['can_full_fit']:
        raise ValueError('Missing actual parameters, spatial state or exact joint graph at t0')
    write_json(output/'predecision_server_only.json', dict(inputs=inputs, vehicle_params0=params, endpoint0=endpoint0))
    ledger = v3._new_safety_ledger()
    future = advance_horizon(api, v3, infos, graph, context, executors, actions, lanes, ledger, observer,
                             case['horizon_sec'])
    observation = dict(preaction=preaction, events=observer.events, frames=observer.frames,
        catalog=observer.movements, lane_geometry=observer.lane_geometry, costs=future['costs'])
    write_json(output/'observations_server_only.json', observation)
    write_json(output/'future_execution_server_only.json', future)
    checks = dict(future['checks'], native_checkpoint_exact=t0 == case.get('resolved_checkpoint_sec', t0),
        inside_frozen_window=case['checkpoint_sec'] <= t0 <= case['checkpoint_sec']+60,
        prefix_no_teleport=prefix['no_teleport_passed'])
    labels, targets, measurement = None, None, None
    if all(checks.values()):
        measurement = observer.summary()
        labels = extract_response_labels(measurement, preaction['movement_ids'], horizon_sec=10)
        targets = extract_targets(endpoint0, future['endpoint_states']['10'],
                                  [row['lane_id'] for row in inputs['lane_nodes']], lane_static)
        targets['y'] = targets['y'].tolist()
        checks.update(response_samples_complete=measurement['sample_count'] == 10,
            movement_accounting_complete=measurement['complete'],
            first_receiving_entries_observed=native.unobserved_first_receiving_count(observer.frames, observer.movements) == 0)
    return dict(start=start, candidate_states=candidates, reference_index=reference, candidate_index=index,
        resolved_checkpoint_sec=t0, preaction=preaction, network_preaction=network, inputs=inputs, targets=targets,
        labels=labels, measurement=measurement, checks=checks, observation=observation,
        cost=future['cost'], integrals=future['integrals'], controlled_lane_count=len(lanes), future=future,
        prefix_safety=prefix, safety=v3._finalize_safety_ledger(ledger))


def collect_case(api, v3, case, output, static, lane_static, clearance_lanes, stop=None):
    from cologne_action_branch_checks import compare_physical
    from recover_tsc_joint_response_trial import revised_audit

    output.mkdir(parents=True, exist_ok=False)
    resolved, indices, first = dict(case), [None], None
    summaries, rows = [], []
    for position in range(8):
        if stop is not None and stop.is_set():
            break
        requested = indices[position]
        directory = output/('first_pp' if requested is None else f'action_{requested}')
        directory.mkdir()
        summary = dict(candidate_index=requested, status='INVALID')
        try:
            branch = collect_branch(api, v3, resolved, requested, directory, lane_static)
            index = branch['candidate_index']
            summary['candidate_index'] = index
            if first is None:
                first = branch
                resolved['resolved_checkpoint_sec'] = branch['resolved_checkpoint_sec']
                indices.extend(i for i in range(8) if i != branch['reference_index'])
                write_json(output/'physical_start_server_only.json', branch['start'])
            checks = dict(branch['checks'])
            if all(checks.values()):
                extra, audit = revised_audit(branch, branch['observation'], static, case['tls_id'],
                                             first['observation']['frames'][0], clearance_lanes)
                extra.pop('prefix_no_collision')
                extra.pop('branch_no_collision')
                checks.update(extra)
                summary['execution_audit'] = audit
                checks.update(same_native_state=compare_physical(first['start'], branch['start'], tolerance=0)['passed'],
                    same_pending_population=first['start']['pending_ids'] == branch['start']['pending_ids'],
                    same_candidate_roster=first['candidate_states'] == branch['candidate_states'],
                    same_pp_reference=first['reference_index'] == branch['reference_index'],
                    same_movement_roster=first['preaction']['movement_ids'] == branch['preaction']['movement_ids'],
                    same_network_preaction=first['network_preaction'] == branch['network_preaction'])
            summary.update(status='COMPLETE' if all(checks.values()) else 'INVALID', checks=checks,
                cost=branch['cost'], integrals=branch['integrals'], prefix_safety=branch['prefix_safety'],
                safety=branch['safety'], advance_error=branch['future'].get('error'),
                elapsed_sec=api.simulation.getTime()-branch['resolved_checkpoint_sec'])
            if summary['status'] == 'COMPLETE':
                rows.append(dict(candidate_index=index, inputs=branch['inputs'], targets=branch['targets'],
                                 cost=branch['cost'], raw_directory=directory.name))
        except Exception as error:
            summary['error'] = f'{type(error).__name__}: {error}'
        finally:
            try:
                api.close()
            except Exception:
                pass
        summaries.append(summary)
        write_json(directory/'result.json', summary)
        print(json.dumps(dict(case_id=case['case_id'], attempted=len(summaries), total=8,
            candidate_index=summary['candidate_index'], status=summary['status'])), flush=True)
        if summary['status'] != 'COMPLETE':
            if stop is not None:
                stop.set()
            break
    complete = len(summaries) == 8 and all(row['status'] == 'COMPLETE' for row in summaries)
    result = dict(case=case, status='COMPLETE' if complete else 'HOLD_INCOMPLETE_MEASUREMENT',
        resolved_checkpoint_sec=resolved.get('resolved_checkpoint_sec'),
        new_branches_attempted=len(summaries), branches=summaries)
    if first is not None:
        result['reference_index'] = first['reference_index']
        write_json(output/'group_data_server_only.json', dict(case=case, city=case['city'], seed=case['seed'],
            anchor=case['anchor'], complete=complete, pp_index=first['reference_index'],
            branches=sorted(rows, key=lambda row: row['candidate_index'])))
    write_json(output/'result.json', result)
    return result


def _collect_one(case, base, stop):
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
    from static_context import read_static_context
    from tsc_clearance_execution_audit import read_clearance_lanes

    if stop.is_set():
        return dict(case=case, status='NOT_STARTED_AFTER_HOLD', new_branches_attempted=0, branches=[])
    api = load_libsumo()
    if libsumo_version(api) != '1.22.0':
        raise ValueError('Use the fixed SUMO1.22 runtime')
    static = read_static_context(Path(case['sumocfg']))
    lane_static = read_lane_static(case['sumocfg'], static['conflicts_by_tls'])
    return collect_case(api, v3, case, base/'measurement'/case['case_id'], static, lane_static,
        read_clearance_lanes(Path(case['sumocfg']), case['tls_id']), stop)


def run(base, input_base):
    protocol = json.loads((HERE/'protocol.json').read_text())
    roster = cases(protocol, input_base)
    if (base/'measurement').exists():
        raise ValueError('This cohort has already been started; never rerun or replace it')
    (base/'measurement').mkdir()
    write_json(base/'case_roster.json', roster)
    results, groups = [], []
    with multiprocessing.Manager() as manager:
        stop = manager.Event()
        with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(_collect_one, case, base, stop): case for case in roster}
            for future in as_completed(futures):
                case = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    stop.set()
                    result = dict(case=case, status='HOLD_WORKER_ERROR', error=f'{type(error).__name__}: {error}',
                                  new_branches_attempted=0, branches=[])
                    path = base/'measurement'/case['case_id']
                    path.mkdir(exist_ok=True)
                    write_json(path/'worker_error.json', result)
                results.append(result)
                complete = len(results) == len(roster) and all(row['status'] == 'COMPLETE' for row in results)
                receipt = dict(status='PASS_COLLECTION' if complete else 'HOLD_INCOMPLETE_MEASUREMENT'
                    if stop.is_set() else 'COLLECTING', groups=sorted(results, key=lambda row: row['case']['case_id']),
                    scheduled_native_branches=224,
                    new_branches_attempted=sum(row['new_branches_attempted'] for row in results), new_fits=0)
                write_json(base/'collection_result.json', receipt)
    for case in roster:
        path = base/'measurement'/case['case_id']/'group_data_server_only.json'
        if path.exists():
            groups.append(json.loads(path.read_text()))
    if receipt['status'] == 'PASS_COLLECTION':
        write_json(base/'training_dataset_server_only.json', dict(groups=groups))
    print(receipt['status'], flush=True)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--input-base', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.base, args.input_base)
    raise SystemExit(0 if result['status'] == 'PASS_COLLECTION' else 2)
