"""Measure complete spaced-intervention windows using fresh paired native prefixes."""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time
import sys

import numpy as np

from cologne_action_branch_checks import physical_snapshot, compare_physical
from run_cologne_accepted_action_counterfactual import load_tool

PROTOCOL = 'tsc-v154y-cologne-spaced-action-counterfactual-v1'
RAW_TARGET_PROTOCOL = 'tsc-v155c-cologne-raw-target-action-counterfactual-v1'
ARMS = ('model_once_then_pp', 'pp_only')
TLS = 'cluster_357187_359543'


def apply_branch_policy(arm, time_sec, decision, target_time):
    use_pp = (arm == 'pp_only' and time_sec >= target_time
              or arm == 'model_once_then_pp' and time_sec >= target_time + 10)
    if not use_pp:
        return decision
    return replace(decision, selected_index=decision.reference_index,
                   learned_differs=False, eligible=False, priority=0.0,
                   predicted_score=decision.reference_score, rejection=None)


def population(api):
    return {'active': len(api.vehicle.getIDList()),
            'pending': len(api.simulation.getPendingVehicles())}


def reference_trace_exact(arm, actual, reference, target_time):
    cutoff = target_time + (10 if arm == 'model_once_then_pp' else 0)
    return actual == [row for row in reference if row['time_sec'] < cutoff]


def validate_reference(protocol, seed, reference):
    trace = reference['metrics']['accepted_intervention_trace']
    guard = reference['metrics']['guard_audit']
    windows = [row for row in protocol['windows'] if row['seed'] == seed]
    complete = [row for row in trace if row['time_sec'] + 450 <= 28800]
    expected_protocol = ('tsc-v155b-cologne-raw-target-closed-loop-v1'
                         if protocol['protocol'] == RAW_TARGET_PROTOCOL
                         else 'tsc-v154x-cologne-spaced-no-stay-closed-loop-v1')
    if (reference['protocol'] != expected_protocol
            or reference['run_valid'] is not True or reference['seed'] != seed
            or reference['frozen_threshold'] != protocol['frozen_threshold']
            or reference['source_root'] != protocol['source_root']
            or reference['repaired_sumocfg'] != protocol['sumocfg']
            or reference['ablation_originator_path'] != protocol['ablation_originator_path']
            or len(trace) != guard['accepted_overrides'] or len(trace) != guard['executed_overrides']
            or any(row['override_kind'] != 'switch' or not row['execution_effective'] for row in trace)
            or len(windows) != 7 or [row['selected_trace'] for row in windows] != complete
            or any(row['time_sec'] != row['selected_trace']['time_sec']
                   or row['window_end_sec'] != row['time_sec'] + 450 for row in windows)):
        raise ValueError('Use all seven complete windows from the frozen accepted trajectory')
    return trace, windows


def branch_status(checks, prefix_safety, safety):
    # Native collision reports are separate diagnostic outcomes, not a cost filter.
    safety_passed = all(row['raw_collision_events'] == 0 and row['no_teleport_passed']
                        for row in (prefix_safety, safety))
    return {'status': 'PASS' if all(checks.values()) else 'INVALID',
            'safety_status': 'PASS' if safety_passed else 'FAIL'}


def run(protocol, seed, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY

    started = time.monotonic()
    if protocol['protocol'] not in (PROTOCOL, RAW_TARGET_PROTOCOL) or os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the frozen paired-window protocol and original simulation core')
    if seed not in protocol['seeds'] or protocol['cooldown_intervals'] != 44 or tuple(protocol['branches']) != ARMS:
        raise ValueError('Keep the three selected seeds and frozen intervention spacing')
    if output.exists():
        raise FileExistsError(output)
    frozen_path = Path(protocol['frozen_threshold_path'])
    frozen = json.loads(frozen_path.read_text())
    raw_target = protocol['protocol'] == RAW_TARGET_PROTOCOL
    expected_threshold = 0.0 if raw_target else 0.20790450432178334
    expected_calibration = ('tsc-v155a-cologne-raw-target-refit-v1' if raw_target
                            else 'tsc-v154t-cologne-no-stay-oof-threshold-v1')
    if (frozen != protocol['frozen_threshold'] or frozen['threshold'] != expected_threshold
            or frozen.get('calibration_protocol') != expected_calibration
            or raw_target and (frozen['mode'] != 'minimum_advantage'
                or frozen['score_units'] != protocol['score_units']
                or protocol['score_units'] != 'raw_native_halted_per_lane_cost_difference')
            or frozen['ablation_originator_path'] != protocol['ablation_originator_path']):
        raise ValueError('Keep the frozen model, calibrated threshold units and S no-stay rule')
    reference = json.loads(Path(protocol['closed_loop_references'][str(seed)]).read_text())
    trace, windows = validate_reference(protocol, seed, reference)
    sumocfg = Path(protocol['sumocfg'])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != json.loads(Path(protocol['geometry_reference']).read_text())['bilateral_static_geometry']:
        raise ValueError('The fixed repaired geometry changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    sys.path.insert(0, str(Path(protocol['threshold_originator_path']).parent))
    ablation = load_tool(protocol['ablation_originator_path'], 'v154y_no_stay')
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    scratch = Path(protocol['scratch_root']) / f'seed_{seed}'
    scratch.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True)
    groups, all_full_records = [], {}

    for window in windows:
        target = window['selected_trace']
        target_time, end_time = int(window['time_sec']), int(window['window_end_sec'])
        runs, captures, full_records = {}, {}, {}

        for arm in ARMS:
            capture, decisions, samples, collision_events = {}, [], [], []
            prefix_ledger, window_ledger = v3._new_safety_ledger(), v3._new_safety_ledger()

            class BranchOriginator(ablation.NoStayFractionRigidOriginator):
                def prepare_interval(self, **kwargs):
                    super().prepare_interval(**kwargs)
                    if float(api.simulation.getTime()) == target_time:
                        states, executors = kwargs['states'], kwargs['executors']
                        infos = {tls: state.info for tls, state in states.items()}
                        capture.update(physical=physical_snapshot(api, executors), population=population(api),
                            lanes=tuple(sorted(v3._controlled_lane_set(infos))), tls_ids=list(states),
                            pending_ids=sorted(api.simulation.getPendingVehicles()))
                        capture['pp_phase'] = v3._rule_candidate(states[TLS], executors[TLS], 'phase_pressure').state

                def select(self, contrast, *, reference_index):
                    decision = super().select(contrast, reference_index=reference_index)
                    now = float(api.simulation.getTime())
                    phases = list(contrast.metadata['candidate_states'])
                    record = {'time_sec': now, 'selected_phase': str(phases[decision.selected_index]),
                        'reference_phase': str(phases[decision.reference_index]),
                        'selected_index': int(decision.selected_index), 'reference_index': int(decision.reference_index),
                        'priority': float(decision.priority),
                        'predicted_advantage': float(decision.reference_score - decision.predicted_score),
                        'originator_decision': decision.to_dict()}
                    actual = apply_branch_policy(arm, now, decision, target_time)
                    record['requested_phase'] = str(phases[actual.selected_index])
                    record['forced_pp'] = actual is not decision
                    decisions.append(record)
                    if now == target_time:
                        capture['decision'] = record
                    return actual

            def observer(*, stage, time_sec, executors):
                if stage != 'after_executor_advance':
                    return
                collisions = tuple(api.simulation.getCollisions())
                collision_events.extend({'time_sec': time_sec, 'repr': repr(event)} for event in collisions)
                ledger = prefix_ledger if time_sec <= target_time else window_ledger
                v3._record_safety_step(ledger=ledger, sumo_api=api, executors=executors,
                    starting_teleports=int(api.simulation.getStartingTeleportNumber()),
                    ending_teleports=int(api.simulation.getEndingTeleportNumber()),
                    collisions=collisions)
                if target_time < time_sec <= end_time:
                    snap = executors[TLS].snapshot()
                    samples.append({'time_sec': time_sec,
                        'cost': float(sum(api.lane.getLastStepHaltingNumber(lane) for lane in capture['lanes'])) / 8,
                        **population(api), 'arrived': int(api.simulation.getArrivedNumber()),
                        'departed': int(api.simulation.getDepartedNumber()),
                        'current_green_state': snap['current_green_state'],
                        'executor_mode': snap['mode'], 'green_elapsed_sec': snap['green_elapsed_sec']})

            originator = BranchOriginator.load(Path(frozen['model_path']), frozen_threshold_path=frozen_path,
                expected_city='cologne', network_context=runtime.read_network_right_of_way_context(runtime.net_file_from_sumocfg(sumocfg)))
            tripinfo = scratch / f't{target_time}_{arm}.xml'
            try:
                metrics = runtime.evaluate_policy_v3(sumo_api=api, sumocfg=sumocfg, scenario='cologne1', policy=RUNTIME_POLICY,
                    models=_runtime_models(originator, prediction_horizon_sec=450), duration_sec=end_time-25200,
                    control_interval_sec=10, warmup_sec=60, seed=seed, tripinfo_output=tripinfo,
                    residual_coordination_mode='direct', residual_cooldown_intervals_override=44, step_observer=observer)
            finally:
                tripinfo.unlink(missing_ok=True)
            captures[arm] = capture
            costs = [row['cost'] for row in samples]
            checks = {'evaluator_complete': metrics.get('ok') is True,
                'native_reference_trace_exact': reference_trace_exact(arm, metrics.get('accepted_intervention_trace'), trace, target_time),
                'full_horizon': [s['time_sec'] for s in samples] == list(range(target_time+1, end_time+1)),
                'target_captured': 'decision' in capture}
            prefix_safety = v3._finalize_safety_ledger(prefix_ledger)
            safety = v3._finalize_safety_ledger(window_ledger)
            branch = {
                'checks': checks, 'prefix_safety': prefix_safety, 'safety': safety,
                'sample_count': len(costs), 'first_interval_costs': costs[:10],
                'mean_halted_per_lane': float(np.mean(costs)) if costs else None,
                'accepted_trace_count': len(metrics.get('accepted_intervention_trace', [])),
                'prefix_decision_count': sum(d['time_sec'] < target_time for d in decisions),
                'runtime_error': metrics.get('error')}
            if capture.get('decision'):
                raw = capture['decision']
                checks.update(selected_action_exact=raw['selected_phase'] == target['selected_phase_state'],
                    requested_action_exact=raw['requested_phase'] == (target['selected_phase_state'] if arm == 'model_once_then_pp' else target['prior_phase_state']),
                    pp_reference_exact=raw['reference_phase'] == target['prior_phase_state'] == capture['pp_phase'],
                    predicted_advantage_exact=raw['predicted_advantage'] == target['priority'],
                    single_tls_eight_lanes=capture['tls_ids'] == [TLS] and len(capture['lanes']) == 8)
                branch['start_population'] = capture['population']
                branch['first_decision'] = raw
            if samples:
                branch.update(end_population={k:samples[-1][k] for k in ('active','pending')},
                    window_arrived=sum(s['arrived'] for s in samples), window_departed=sum(s['departed'] for s in samples),
                    window_active_vehicle_hours=sum(s['active'] for s in samples)/3600,
                    window_pending_vehicle_hours=sum(s['pending'] for s in samples)/3600,
                    window_system_vehicle_hours=sum(s['active']+s['pending'] for s in samples)/3600,
                    end_green_elapsed_sec=samples[-1]['green_elapsed_sec'],
                    per_150_seconds=[{'start_sec':target_time+i,'end_sec':target_time+i+150,
                                     'mean_halted_per_lane':float(np.mean(costs[i:i+150]))} for i in (0,150,300)] if len(costs)==450 else [])
                checks['active_population_accounting'] = (capture['population']['active'] + branch['window_departed']
                    - branch['window_arrived'] == branch['end_population']['active'])
            branch.update(branch_status(checks, prefix_safety, safety))
            full_records[arm] = {'capture':capture, 'decisions':decisions, 'samples':samples,
                                 'accepted_intervention_trace':metrics.get('accepted_intervention_trace', []),
                                 'collision_events': collision_events}
            runs[arm] = branch
            print(json.dumps({'seed': seed, 'target_time_sec': target_time, 'arm':arm,
                              'status':branch['status'], 'safety_status': branch['safety_status'],
                              'cost':branch['mean_halted_per_lane']}), flush=True)

        checks, comparisons = {}, {}
        base, alternate = captures['model_once_then_pp'], captures['pp_only']
        checks['both_target_captured'] = all('decision' in capture for capture in captures.values())
        if checks['both_target_captured']:
            same = compare_physical(base['physical'], alternate['physical'], tolerance=0)
            comparisons['model_once_vs_pp'] = same
            checks.update(physical_start_exact=same['passed'],
                pending_start_exact=base['pending_ids'] == alternate['pending_ids'],
                prefix_decisions_exact=[d for d in full_records['model_once_then_pp']['decisions'] if d['time_sec']<target_time]
                    == [d for d in full_records['pp_only']['decisions'] if d['time_sec']<target_time])
        group = {'seed':seed,'target_time_sec':target_time,'end_time_sec':end_time,
            'status':'INVALID','safety_status':'PASS' if all(row['safety_status']=='PASS' for row in runs.values()) else 'FAIL',
            'selected_trace':target,'checks':checks,'physical_start_comparisons':comparisons,'runs':runs}
        if all(checks.values()) and all(row['status']=='PASS' for row in runs.values()):
            a,b = [runs[arm]['mean_halted_per_lane'] for arm in ARMS]
            group.update(status='PASS',comparison={'model_once_then_pp_cost':a,'pp_only_cost':b,
                'one_action_minus_pp':a-b,'predicted_advantage':target['priority'],'actual_advantage':b-a,
                'one_action_ranking':'harmful' if a>b+1e-12 else 'beneficial' if a<b-1e-12 else 'tie'})
        groups.append(group)
        all_full_records[str(target_time)] = full_records

    result = {'protocol':protocol['protocol'],'status':'PASS' if all(row['status']=='PASS' for row in groups) else 'INVALID',
        'safety_status':'PASS' if all(row['safety_status']=='PASS' for row in groups) else 'FAIL',
        'seed':seed,'source_root':protocol['source_root'],'sumocfg':str(sumocfg),'frozen_threshold':frozen,
        'cooldown_intervals':44,'label_contract':protocol['label_contract'],'groups':groups,
        'new_simulation_count':2*len(windows),'new_model_fits':0,'threshold_retuning':False,
        'full_traces_server_only':str(output/'full_traces_server_only.json')}
    if raw_target:
        result['score_units'] = protocol['score_units']
    result['elapsed_sec']=time.monotonic()-started
    (output/'full_traces_server_only.json').write_text(json.dumps(all_full_records,separators=(',',':'),allow_nan=False)+'\n')
    (output/'result.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('protocol','status','elapsed_sec')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    if run(json.loads(args.protocol.read_text()),args.seed,args.output)['status']!='PASS':raise SystemExit(2)
