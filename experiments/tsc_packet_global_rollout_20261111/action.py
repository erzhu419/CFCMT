"""One same-state native action pair and the frozen old/new LC predictions."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from math import fsum
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from experiments.tsc_network_dense_measurement_v2_20261109.collect_dense import executor_rows, executor_seed, write_json
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_repaired_action_cost_pilot_20261109.advance_measurement import ENDPOINTS, advance_horizon
from experiments.tsc_target_packet_protocol_20261109.packet_contract import initialize_observation
from experiments.tsc_target_packet_training_20261109.evaluate import StructuralAudit


NATIVE_SEED = 76116921
FOCAL_TLS = 'intersection_2_2'
CHECKPOINTS = range(600, 661, 10)
HORIZON = 450
SCENARIOS = tuple(range(76117001, 76117009))
MODELS = ('old_lc', 'new_lc')
ARMS = ('pp', 'alternative')
MAX_WORKERS = 16


def _read(path):
    return json.loads(path.read_text())


def _pp_requests(api, v3, infos, context, graph, executors):
    states = v3._read_graph_states_v3(api, infos, context, graph)
    return {tls: v3._rule_candidate(states[tls], executors[tls], 'phase_pressure').state
            for tls in sorted(executors)}


def _switch_expected(initial, requests):
    return requests[FOCAL_TLS] != initial['executors'][FOCAL_TLS]['current_green_state']


def _native_prefix(api, v3, cfg, static, fixed_time=None):
    """Rebuild body trails during the same one-second PP prefix in both arms."""
    v3._start_sumo(api, cfg, NATIVE_SEED)
    infos = v3._tls_phase_infos(api)
    graph = v3.build_signal_routing_graph(api, infos)
    context = v3._scenario_context_v3(sumocfg=cfg, infos=infos, control_interval_sec=10)
    executors = v3.build_safe_phase_executors(api, infos)
    lanes = tuple(sorted(v3._controlled_lane_set(infos)))
    if lanes != tuple(sorted(static['controlled_lane_ids'])) or len(lanes) != 144:
        raise ValueError('The live and frozen Jinan cost lanes must match exactly')
    if FOCAL_TLS not in executors or api.simulation.getTime() != 0:
        raise ValueError('The fixed TLS and zero-time native start are required')
    observer = PacketObserver(static)
    frame = observer.capture(api, 0, 'prefix_observation')
    ledger = v3._new_safety_ledger()
    chosen = None
    for second in range(CHECKPOINTS.stop):
        if api.simulation.getTime() != second:
            raise ValueError('Native prefix clock changed')
        if (second == fixed_time if fixed_time is not None else second in CHECKPOINTS):
            candidates = tuple(executors[FOCAL_TLS].feasible_states_now())
            if len(candidates) >= 2:
                chosen = second
                break
            if fixed_time is not None:
                raise ValueError('The replayed checkpoint lost its legal action pair')
        if second >= CHECKPOINTS[-1]:
            break
        requests = _pp_requests(api, v3, infos, context, graph, executors) if second % 10 == 0 else {}
        v3._advance_with_actions(sumo_api=api, executors=executors, actions=requests,
            control_interval_sec=1, collision_mode='audit', safety_ledger=ledger)
        frame = observer.capture(api, second+1, 'prefix_observation')
    if chosen is None:
        raise ValueError('No two-action opportunity at the fixed TLS in seconds600–660')
    safety = v3._finalize_safety_ledger(ledger)
    if not safety['no_teleport_passed']:
        raise ValueError('The native prefix failed the existing teleport accounting check')
    frame.update(offset_sec=0, role='preaction_observation', executors=executor_rows(executors))
    initial = initialize_observation(frame, chosen)
    return infos, graph, context, executors, lanes, chosen, candidates, initial, safety


def collect_native_pair(base, output):
    """Measure two starts and retain the shared causal input separately."""
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
    from cologne_action_branch_checks import compare_physical, physical_snapshot
    from tsc_joint_response_observer import JointResponseObserver

    output.mkdir(parents=True, exist_ok=False)
    cfg = base/'repaired_action_cost_pilot/inputs/jinan/jinan_3x4_real.sumocfg'
    static = _read(base/'target_packet_training/static_graph_server_only.json')
    api = load_libsumo()
    if libsumo_version(api) != '1.22.0':
        raise ValueError('Use the frozen SUMO1.22 runtime')
    result = dict(status='RUNNING_NATIVE_ACTION_PAIR', seed=NATIVE_SEED, focal_tls=FOCAL_TLS,
        checkpoint_window_sec=[CHECKPOINTS[0], CHECKPOINTS[-1]], horizon_sec=HORIZON,
        native_start_budget=2, new_native_starts=0, new_fits=0, branches={})
    first = None
    for arm in ARMS:
        directory = output/arm
        directory.mkdir()
        branch = dict(status='STARTED')
        result['branches'][arm] = branch
        result['new_native_starts'] += 1
        write_json(directory/'result.json', branch)
        write_json(output/'result.json', result)
        try:
            fixed_time = first['checkpoint_sec'] if first else None
            prefix = _native_prefix(api, v3, cfg, static, fixed_time)
            infos, graph, context, executors, lanes, checkpoint, candidates, initial, safety = prefix
            snapshot = physical_snapshot(api, executors)
            snapshot['pending_ids'] = sorted(api.simulation.getPendingVehicles())
            seeds = executor_seed(executors)
            requests = _pp_requests(api, v3, infos, context, graph, executors)
            pp = requests[FOCAL_TLS]
            if pp not in candidates or len(set(candidates)) != len(candidates):
                raise ValueError('The native PP request or feasible roster changed')
            alternative = next(candidate for candidate in candidates if candidate != pp)
            receipt = dict(checkpoint_sec=checkpoint, feasible_states=list(candidates),
                pp_request=pp, alternative_request=alternative, pp_requests=requests,
                initial=initial, executor_seeds=seeds, physical_snapshot=snapshot)
            write_json(directory/'initial_server_only.json', receipt)
            if first is None:
                first = receipt
                result.update(checkpoint_sec=checkpoint, feasible_states=list(candidates),
                    pp_request=pp, alternative_request=alternative)
                pairing = True
            else:
                pairing = (compare_physical(first['physical_snapshot'], snapshot, tolerance=0)['passed']
                    and first['physical_snapshot']['pending_ids'] == snapshot['pending_ids']
                    and all(first[key] == receipt[key] for key in ('checkpoint_sec', 'feasible_states',
                        'pp_requests', 'alternative_request', 'initial', 'executor_seeds')))
            branch.update(same_initial_state=pairing, prefix_safety=safety)
            if not pairing:
                raise ValueError('The two native prefixes did not reproduce the same initial state')
            initial_requests = dict(requests)
            if arm == 'alternative':
                initial_requests[FOCAL_TLS] = alternative
            observer = JointResponseObserver(api, infos, checkpoint, windows=(10,), focal_tls=FOCAL_TLS)
            observer.sample(checkpoint)
            ledger = v3._new_safety_ledger()
            future = advance_horizon(api, v3, infos, graph, context, executors, initial_requests,
                                     lanes, ledger, observer)
            write_json(directory/'execution_server_only.json', future)
            checks = dict(future['checks'])
            checks['initial_request_executed'] = bool(future['execution_trace'] and
                future['execution_trace'][0]['request_returns'].get(FOCAL_TLS) ==
                    _switch_expected(initial, initial_requests))
            branch.update(checks=checks, safety=v3._finalize_safety_ledger(ledger),
                initial_request=initial_requests[FOCAL_TLS], completed_steps=future['completed_steps'],
                cost=future['cost'], integrals=future['integrals'], error=future.get('error'),
                status='COMPLETE' if all(checks.values()) else 'HOLD_NATIVE_ACTION_BRANCH')
        except Exception as error:
            branch.update(status='HOLD_NATIVE_ACTION_BRANCH', error=f'{type(error).__name__}: {error}')
        finally:
            try:
                api.close()
            except Exception as error:
                branch.update(status='HOLD_NATIVE_ACTION_BRANCH', close_error=f'{type(error).__name__}: {error}')
        write_json(directory/'result.json', branch)
        print(json.dumps(dict(stage='native_action_pair', arm=arm, status=branch['status'],
                             steps=branch.get('completed_steps'), error=branch.get('error'))), flush=True)
        if branch['status'] != 'COMPLETE':
            result['status'] = 'HOLD_NATIVE_ACTION_PAIR'
            write_json(output/'result.json', result)
            return result
        write_json(output/'result.json', result)
    result.update(status='COMPLETE_NATIVE_ACTION_PAIR', controlled_lane_count=144,
                  native_seconds=sum(result['checkpoint_sec']+HORIZON for _ in ARMS))
    write_json(output/'result.json', result)
    return result


def _predict_draw(scenario, factory, static, initial, executor_seeds, model, requests):
    """Run one model and action using the shared scenario seed and causal anchor."""
    halted = []
    try:
        kernel = factory(static, deepcopy(initial), executor_seeds, model['target_laws'],
                         coefficients=model['coefficients'], random_seed=scenario)
        audit = StructuralAudit(static, kernel.state, executor_seeds)
        for step in range(HORIZON):
            before = deepcopy(kernel.state)
            initial_requests = requests if step == 0 else None
            row = kernel.step(initial_requests=initial_requests)
            audit.check(before, kernel.state, row, initial_requests=initial_requests)
            if step == 0 and row['request_returns'].get(FOCAL_TLS) != _switch_expected(initial, requests):
                raise ValueError('The model did not execute the measured focal phase request')
            halted.append(row['halted_total'])
    except Exception as error:
        return dict(status='FAILED', checked_steps=len(halted), error=f'{type(error).__name__}: {error}')
    return dict(status='COMPLETE', cost=audit.integral/(HORIZON*len(static['controlled_lane_ids'])),
                integrals={str(end):sum(halted[:end]) for end in ENDPOINTS}, checked_steps=audit.steps)


def _group_scenario(scenario, branches):
    """Rebuild the fixed paired layout after the four independent tasks finish."""
    models, failures = {model:{} for model in MODELS}, []
    for model in MODELS:
        for arm in ARMS:
            branch = branches[model, arm]
            if branch['status'] == 'COMPLETE':
                models[model][arm] = {key:value for key,value in branch.items() if key != 'status'}
            else:
                failures.append(dict(branch, model=model, arm=arm))
    result = dict(scenario_seed=scenario, status='FAILED' if failures else 'COMPLETE', models=models)
    if failures:
        result.update(failures=failures, **failures[0])
    return result


def compare_costs(native, draws):
    """Compare model choices using the same two measured costs for both models."""
    measured = {arm:native['branches'][arm]['cost'] for arm in ARMS}
    native_delta = measured['alternative']-measured['pp']
    lower_cost = min(measured.values())
    models = {}
    for model in MODELS:
        estimates = {arm:fsum(draw['models'][model][arm]['cost'] for draw in draws)/len(draws) for arm in ARMS}
        deltas = [draw['models'][model]['alternative']['cost']-draw['models'][model]['pp']['cost'] for draw in draws]
        delta = estimates['alternative']-estimates['pp']
        selected = 'alternative' if delta < 0 else 'pp'
        models[model] = dict(predicted_cost=estimates, predicted_alternative_minus_pp=delta,
            paired_scenario_deltas=deltas, predicted_cost_difference_absolute_error=abs(delta-native_delta),
            pair_direction_matches_measured=(delta > 0)-(delta < 0) == (native_delta > 0)-(native_delta < 0),
            selected_action=selected, measured_selected_cost=measured[selected],
            measured_selected_cost_minus_pp=measured[selected]-measured['pp'],
            regret_against_measured_pair_minimum=measured[selected]-lower_cost)
    return dict(measured_cost=measured, measured_alternative_minus_pp=native_delta,
        measured_integral_difference={str(end):native['branches']['alternative']['integrals'][str(end)]-
            native['branches']['pp']['integrals'][str(end)] for end in ENDPOINTS if end},
        models=models, lc_replacement_changes_choice=models['old_lc']['selected_action'] != models['new_lc']['selected_action'],
        lc_replacement_measured_cost_difference=models['new_lc']['measured_selected_cost']-
            models['old_lc']['measured_selected_cost'])


def predict_pair(base, native_output, output, workers):
    from experiments.tsc_packet_spatial_conflict_20261110.diagnose import load_widths
    from .calibration import kernel_factory, load_model

    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f'Use between 1 and {MAX_WORKERS} action prediction workers')
    native = _read(native_output/'result.json')
    if native['status'] != 'COMPLETE_NATIVE_ACTION_PAIR':
        raise ValueError('Both native actions must be measured from the same state')
    anchor = _read(native_output/'pp/initial_server_only.json')
    static, _ = load_widths(_read(base/'target_packet_training/static_graph_server_only.json'))
    models = {name:load_model(base, point=name) for name in MODELS}
    old, new = (models[name] for name in MODELS)
    if old['target_laws'] != new['target_laws'] or not np.array_equal(
            np.asarray(old['coefficients'])[:3], np.asarray(new['coefficients'])[:3]):
        raise ValueError('The old and new forecasts must differ only in LC coefficients')
    factory = kernel_factory(base, static)
    requests = {arm:dict(anchor['pp_requests']) for arm in ARMS}
    requests['alternative'][FOCAL_TLS] = anchor['alternative_request']
    output.mkdir(parents=True, exist_ok=False)
    draws = {}
    branches = {seed:{} for seed in SCENARIOS}
    branches_per_scenario = len(MODELS)*len(ARMS)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as executor:
        tasks = {
            executor.submit(_predict_draw, seed, factory, static, anchor['initial'],
                            anchor['executor_seeds'], models[model], requests[arm]):(seed, model, arm)
            for seed in SCENARIOS for model in MODELS for arm in ARMS
        }
        for task in as_completed(tasks):
            seed, model, arm = tasks[task]
            try:
                branch = task.result()
            except Exception as error:
                branch = dict(status='FAILED', error=f'{type(error).__name__}: {error}')
            branches[seed][model, arm] = branch
            print(json.dumps(dict(stage='action_prediction', scenario_seed=seed, model=model, arm=arm,
                                 status=branch['status'], error=branch.get('error'))), flush=True)
            if len(branches[seed]) == branches_per_scenario:
                draws[seed] = _group_scenario(seed, branches[seed])
                write_json(output/f'scenario{seed}_server_only.json', draws[seed])
    complete = all(draws[seed]['status'] == 'COMPLETE' for seed in SCENARIOS)
    result = dict(status='COMPLETE_ACTION_PREDICTION' if complete else 'HOLD_ACTION_PREDICTION',
        scenario_seeds=list(SCENARIOS), planned_model_rollouts=branches_per_scenario*len(SCENARIOS),
        completed_scenarios=sum(draw['status'] == 'COMPLETE' for draw in draws.values()),
        failures=[draw for draw in draws.values() if draw['status'] != 'COMPLETE'],
        new_fits=0, new_native_starts=0,
        comparison=compare_costs(native,[draws[seed] for seed in SCENARIOS]) if complete else None)
    write_json(output/'result.json', result)
    return result


def run(base, output, calibration_result, *, phase='all', workers=MAX_WORKERS):
    calibration = _read(calibration_result)
    if (calibration['status'] not in ('PASS_TARGET_PACKET_TECHNICAL_CALIBRATION', 'HOLD_TARGET_PACKET_CALIBRATION')
            or not calibration['checks']['all_scenarios_structurally_complete']):
        raise ValueError('The frozen full-horizon calibration must first be structurally complete')
    began = time.monotonic()
    native_output, prediction_output = output/'measurement', output/'prediction'
    native = (collect_native_pair(base, native_output) if phase != 'predict'
              else _read(native_output/'result.json'))
    prediction = (predict_pair(base, native_output, prediction_output, workers)
                  if phase != 'measure' and native['status'] == 'COMPLETE_NATIVE_ACTION_PAIR' else None)
    status = ('COMPLETE_PAIRED_ACTION_DIAGNOSTIC' if prediction and prediction['status'] == 'COMPLETE_ACTION_PREDICTION'
              else prediction['status'] if prediction else native['status'])
    result = dict(status=status, seed=NATIVE_SEED, focal_tls=FOCAL_TLS, calibration_status=calibration['status'],
        native=native, prediction=prediction, new_fits=0, new_native_starts=native['new_native_starts'],
        horizon_sec=HORIZON, elapsed_sec=time.monotonic()-began,
        cost_unit='mean halted vehicles per controlled lane over450s',
        scope='One fixed Jinan state, two legal initial requests and native PP continuation thereafter.')
    write_json(output/'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-result', type=Path, required=True)
    parser.add_argument('--phase', choices=('all', 'measure', 'predict'), default='all')
    parser.add_argument('--workers', type=int, choices=range(1, MAX_WORKERS+1), default=MAX_WORKERS)
    args = parser.parse_args()
    receipt = run(args.base, args.output, args.calibration_result, phase=args.phase, workers=args.workers)
    print(receipt['status'], flush=True)
    raise SystemExit(0 if receipt['status'].startswith('COMPLETE') else 2)
