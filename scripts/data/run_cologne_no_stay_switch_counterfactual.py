"""Replay the preselected V154U switch against PP from an identical native prefix."""

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

PROTOCOL = 'tsc-v154v-cologne-no-stay-switch-counterfactual-v1'
ARMS = ('model_once_then_pp', 'pp_only')
TLS = 'cluster_357187_359543'
SEED, TARGET_TIME, END_TIME = 63792, 27360, 27810


def apply_branch_policy(arm, time_sec, decision):
    use_pp = (arm == 'pp_only' and time_sec >= TARGET_TIME
              or arm == 'model_once_then_pp' and time_sec >= TARGET_TIME + 10)
    if not use_pp:
        return decision
    return replace(decision, selected_index=decision.reference_index,
                   learned_differs=False, eligible=False, priority=0.0,
                   predicted_score=decision.reference_score, rejection=None)


def population(api):
    return {'active': len(api.vehicle.getIDList()),
            'pending': len(api.simulation.getPendingVehicles())}


def reference_trace_exact(arm, actual, reference):
    cutoff = TARGET_TIME + (10 if arm == 'model_once_then_pp' else 0)
    return actual == [row for row in reference if row['time_sec'] < cutoff]


def validate_reference(reference, target, frozen):
    trace = reference['metrics']['accepted_intervention_trace']
    if (reference['protocol'] != 'tsc-v154u-cologne-recalibrated-no-stay-closed-loop-v1'
            or reference['status'] != 'PASS' or reference['seed'] != SEED
            or reference['frozen_threshold'] != frozen
            or reference['ablation_originator_path'] != frozen['ablation_originator_path']
            or len(trace) != 64
            or reference['originator_diagnostics']['phase_pressure_override_count'] != len(trace)
            or any(row['override_kind'] != 'switch' or not row['execution_effective'] for row in trace)
            or target not in trace or target != min(trace, key=lambda row: (-row['priority'], row['time_sec']))
            or target['time_sec'] != TARGET_TIME or target['priority'] != 0.8757098725229305
            or target['tls_id'] != TLS
            or target['prior_phase_state'] != 'GGGGGrrrrrsrrrrrrrrr'
            or target['selected_phase_state'] != 'rrrrrGGGggrrrrrGGGgg'):
        raise ValueError('The complete V154U trace or its preselected highest-score switch changed')
    return trace


def run(protocol, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY

    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the frozen V154V protocol and original simulation core')
    if (protocol['seed'], protocol['target_time_sec'], protocol['end_time_sec']) != (SEED, TARGET_TIME, END_TIME) or tuple(protocol['branches']) != ARMS:
        raise ValueError('Keep the selected state and two frozen branches')
    if output.exists():
        raise FileExistsError(output)
    frozen_path = Path(protocol['frozen_threshold_path'])
    frozen = json.loads(frozen_path.read_text())
    if (frozen != protocol['frozen_threshold'] or frozen['threshold'] != 0.20790450432178334
            or frozen.get('calibration_protocol') != 'tsc-v154t-cologne-no-stay-oof-threshold-v1'
            or frozen['ablation_originator_path'] != protocol['ablation_originator_path']):
        raise ValueError('Keep the P model, T threshold and S no-stay rule')
    reference = json.loads(Path(protocol['closed_loop_reference']).read_text())
    target = protocol['selected_trace']
    trace = validate_reference(reference, target, frozen)
    sumocfg = Path(protocol['sumocfg'])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != json.loads(Path(protocol['geometry_reference']).read_text())['bilateral_static_geometry']:
        raise ValueError('The fixed repaired geometry changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    sys.path.insert(0, str(Path(protocol['threshold_originator_path']).parent))
    ablation = load_tool(protocol['ablation_originator_path'], 'v154v_no_stay')
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    scratch = Path(protocol['scratch_root'])
    scratch.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True)
    runs, captures, full_records = {}, {}, {}

    for arm in ARMS:
        capture, decisions, samples = {}, [], []
        prefix_ledger, window_ledger = v3._new_safety_ledger(), v3._new_safety_ledger()

        class BranchOriginator(ablation.NoStayFractionRigidOriginator):
            def prepare_interval(self, **kwargs):
                super().prepare_interval(**kwargs)
                if float(api.simulation.getTime()) == TARGET_TIME:
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
                actual = apply_branch_policy(arm, now, decision)
                record['requested_phase'] = str(phases[actual.selected_index])
                record['forced_pp'] = actual is not decision
                decisions.append(record)
                if now == TARGET_TIME:
                    capture['decision'] = record
                return actual

        def observer(*, stage, time_sec, executors):
            if stage != 'after_executor_advance':
                return
            ledger = prefix_ledger if time_sec <= TARGET_TIME else window_ledger
            v3._record_safety_step(ledger=ledger, sumo_api=api, executors=executors,
                starting_teleports=int(api.simulation.getStartingTeleportNumber()),
                ending_teleports=int(api.simulation.getEndingTeleportNumber()),
                collisions=tuple(api.simulation.getCollisions()))
            if TARGET_TIME < time_sec <= END_TIME:
                snap = executors[TLS].snapshot()
                samples.append({'time_sec': time_sec,
                    'cost': float(sum(api.lane.getLastStepHaltingNumber(lane) for lane in capture['lanes'])) / 8,
                    **population(api), 'arrived': int(api.simulation.getArrivedNumber()),
                    'departed': int(api.simulation.getDepartedNumber()),
                    'current_green_state': snap['current_green_state'],
                    'executor_mode': snap['mode'], 'green_elapsed_sec': snap['green_elapsed_sec']})

        originator = BranchOriginator.load(Path(frozen['model_path']), frozen_threshold_path=frozen_path,
            expected_city='cologne', network_context=runtime.read_network_right_of_way_context(runtime.net_file_from_sumocfg(sumocfg)))
        tripinfo = scratch / f'{arm}.xml'
        try:
            metrics = runtime.evaluate_policy_v3(sumo_api=api, sumocfg=sumocfg, scenario='cologne1', policy=RUNTIME_POLICY,
                models=_runtime_models(originator, prediction_horizon_sec=450), duration_sec=END_TIME-25200,
                control_interval_sec=10, warmup_sec=60, seed=SEED, tripinfo_output=tripinfo,
                residual_coordination_mode='direct', residual_cooldown_intervals_override=0, step_observer=observer)
        finally:
            tripinfo.unlink(missing_ok=True)
        captures[arm] = capture
        costs = [row['cost'] for row in samples]
        checks = {'evaluator_complete': metrics.get('ok') is True,
            'native_reference_trace_exact': reference_trace_exact(arm, metrics.get('accepted_intervention_trace'), trace),
            'full_horizon': [s['time_sec'] for s in samples] == list(range(TARGET_TIME+1, END_TIME+1)),
            'target_captured': 'decision' in capture}
        prefix_safety = v3._finalize_safety_ledger(prefix_ledger)
        safety = v3._finalize_safety_ledger(window_ledger)
        safe = all(s['raw_collision_events'] == 0 and s['no_teleport_passed'] for s in (prefix_safety, safety))
        branch = {'status': 'FAIL_UNSAFE' if not safe else 'PASS' if all(checks.values()) else 'INVALID',
            'checks': checks, 'prefix_safety': prefix_safety, 'safety': safety,
            'sample_count': len(costs), 'first_interval_costs': costs[:10],
            'mean_halted_per_lane': float(np.mean(costs)) if costs else None,
            'accepted_trace_count': len(metrics.get('accepted_intervention_trace', [])),
            'prefix_decision_count': sum(d['time_sec'] < TARGET_TIME for d in decisions),
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
                per_150_seconds=[{'start_sec':TARGET_TIME+i,'end_sec':TARGET_TIME+i+150,
                                 'mean_halted_per_lane':float(np.mean(costs[i:i+150]))} for i in (0,150,300)] if len(costs)==450 else [])
            checks['active_population_accounting'] = (capture['population']['active'] + branch['window_departed']
                - branch['window_arrived'] == branch['end_population']['active'])
        if branch['status'] == 'PASS' and not all(checks.values()):
            branch['status'] = 'INVALID'
        full_records[arm] = {'capture':capture, 'decisions':decisions, 'samples':samples,
                             'accepted_intervention_trace':metrics.get('accepted_intervention_trace', [])}
        runs[arm] = branch
        print(json.dumps({'arm':arm, 'status':branch['status'], 'cost':branch['mean_halted_per_lane']}), flush=True)

    checks, comparisons = {}, {}
    base, alternate = captures['model_once_then_pp'], captures['pp_only']
    checks['both_target_captured'] = all('decision' in capture for capture in captures.values())
    if checks['both_target_captured']:
        same = compare_physical(base['physical'], alternate['physical'], tolerance=0)
        comparisons['model_once_vs_pp'] = same
        checks.update(physical_start_exact=same['passed'],
            pending_start_exact=base['pending_ids'] == alternate['pending_ids'],
            prefix_decisions_exact=[d for d in full_records['model_once_then_pp']['decisions'] if d['time_sec']<TARGET_TIME]
                == [d for d in full_records['pp_only']['decisions'] if d['time_sec']<TARGET_TIME])
    result = {'protocol':PROTOCOL,'status':'INVALID','seed':SEED,'target_time_sec':TARGET_TIME,'end_time_sec':END_TIME,
        'source_root':protocol['source_root'],'sumocfg':str(sumocfg),'frozen_threshold':frozen,
        'selected_trace':target,'checks':checks,'physical_start_comparisons':comparisons,
        'runs':runs,'label_contract':protocol['label_contract'],
        'new_simulation_count':2,'new_model_fits':0,'threshold_retuning':False,
        'full_traces_server_only':str(output/'full_traces_server_only.json')}
    if all(checks.values()) and all(r['status']=='PASS' for r in runs.values()):
        a,b = [runs[arm]['mean_halted_per_lane'] for arm in ARMS]
        result.update(status='PASS',comparison={'model_once_then_pp_cost':a,'pp_only_cost':b,
            'one_action_minus_pp':a-b,
            'one_action_relative_to_pp_percent':100*(a/b-1) if b else None,
            'predicted_advantage':target['priority'],'actual_advantage':b-a,
            'one_action_ranking':'harmful' if a>b+1e-12 else 'beneficial' if a<b-1e-12 else 'tie'})
    elif any(r['status']=='FAIL_UNSAFE' for r in runs.values()):
        result['status']='FAIL_UNSAFE'
    result['elapsed_sec']=time.monotonic()-started
    (output/'full_traces_server_only.json').write_text(json.dumps(full_records,separators=(',',':'),allow_nan=False)+'\n')
    (output/'result.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('protocol','status','elapsed_sec')}))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if run(json.loads(args.protocol.read_text()),args.output)['status']!='PASS':raise SystemExit(2)
