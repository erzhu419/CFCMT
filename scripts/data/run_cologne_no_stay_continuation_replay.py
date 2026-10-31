"""Add one native repeated-model branch to the completed V154V comparison."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from cologne_action_branch_checks import physical_snapshot, compare_physical
from run_cologne_accepted_action_counterfactual import load_tool, compare_vectors
from run_cologne_no_stay_switch_counterfactual import (
    SEED, TARGET_TIME, END_TIME, TLS, population, validate_reference,
)

PROTOCOL = 'tsc-v154w-cologne-no-stay-continuation-replay-v1'
ARMS = ('model_once_then_pp', 'pp_only', 'model_repeated')


def repeated_trace_exact(actual, reference):
    return actual == [row for row in reference if row['time_sec'] < END_TIME]


def compare_reused_prefix(current, reused):
    if 'decision' not in current['capture']:
        return {'target_captured': False}, {}
    a, b = current['capture'], reused['capture']
    physical = compare_physical(a['physical'], b['physical'], tolerance=0)
    first = compare_vectors([row['cost'] for row in current['samples'][:10]],
                            [row['cost'] for row in reused['samples'][:10]], tolerance=0)
    checks = {
        'target_captured': True,
        'physical_start_exact': physical['passed'],
        'pending_start_exact': a['pending_ids'] == b['pending_ids'],
        'prefix_decisions_exact': [row for row in current['decisions'] if row['time_sec'] < TARGET_TIME]
            == [row for row in reused['decisions'] if row['time_sec'] < TARGET_TIME],
        'target_decision_exact': a['decision'] == b['decision'],
        'first_ten_costs_exact': first['passed'] and first['left_count'] == first['right_count'] == 10,
    }
    return checks, {'physical_start': physical, 'first_interval': first}


def comparison_metrics(runs):
    a, b, c = [runs[arm]['mean_halted_per_lane'] for arm in ARMS]
    return {'model_once_then_pp_cost': a, 'pp_only_cost': b, 'model_repeated_cost': c,
        'one_action_minus_pp': a-b, 'repeated_minus_one_action': c-a, 'repeated_minus_pp': c-b,
        'one_action_relative_to_pp_percent': 100*(a/b-1) if b else None,
        'repeated_relative_to_one_action_percent': 100*(c/a-1) if a else None,
        'repeated_relative_to_pp_percent': 100*(c/b-1) if b else None,
        'one_action_ranking': 'harmful' if a>b+1e-12 else 'beneficial' if a<b-1e-12 else 'tie'}


def run(protocol, output):
    from cf_h2o.eval import traffic_signal_cologne_collision_replay as runtime
    from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import _runtime_models, RUNTIME_POLICY

    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or os.environ['CFCMT_SOURCE_ROOT'] != protocol['source_root']:
        raise ValueError('Use the frozen V154W protocol and original simulation core')
    if ((protocol['seed'], protocol['target_time_sec'], protocol['end_time_sec']) != (SEED, TARGET_TIME, END_TIME)
            or tuple(protocol['branches']) != ARMS or protocol['new_simulation_count'] != 1
            or protocol['reused_branch_count'] != 2):
        raise ValueError('Add only the selected repeated-model branch to V154V')
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
    reused = json.loads(Path(protocol['reuse_result']).read_text())
    if (reused['protocol'] != 'tsc-v154v-cologne-no-stay-switch-counterfactual-v1'
            or reused['status'] != 'PASS' or not all(reused['checks'].values())
            or reused['seed'] != SEED or reused['target_time_sec'] != TARGET_TIME or reused['end_time_sec'] != END_TIME
            or reused['frozen_threshold'] != frozen or reused['selected_trace'] != target
            or reused['source_root'] != protocol['source_root'] or reused['sumocfg'] != protocol['sumocfg']
            or reused['label_contract'] != protocol['label_contract']
            or reused['full_traces_server_only'] != protocol['reuse_traces']
            or tuple(reused['runs']) != ARMS[:2]
            or any(row['status'] != 'PASS' or not all(row['checks'].values()) for row in reused['runs'].values())):
        raise ValueError('Reuse the two completed V154V branches with their exact model and label contract')
    reused_traces = json.loads(Path(protocol['reuse_traces']).read_text())
    sumocfg = Path(protocol['sumocfg'])
    geometry = runtime.static_geometry(runtime.net_file_from_sumocfg(sumocfg))
    if geometry != json.loads(Path(protocol['geometry_reference']).read_text())['bilateral_static_geometry']:
        raise ValueError('The fixed repaired geometry changed')
    os.environ['CFCMT_EXTERNAL_CONVERSION_ROOT'] = protocol['conversion_root']
    sys.path.insert(0, str(Path(protocol['threshold_originator_path']).parent))
    ablation = load_tool(protocol['ablation_originator_path'], 'v154w_no_stay')
    api = runtime.load_libsumo()
    if '1.22.0' not in str(runtime.libsumo_version()):
        raise ValueError('Use SUMO 1.22.0')
    scratch = Path(protocol['scratch_root'])
    scratch.mkdir(parents=True, exist_ok=False)
    output.mkdir(parents=True)
    capture, decisions, samples = {}, [], []
    prefix_ledger, window_ledger = v3._new_safety_ledger(), v3._new_safety_ledger()

    class RepeatedOriginator(ablation.NoStayFractionRigidOriginator):
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
                'originator_decision': decision.to_dict(),
                'requested_phase': str(phases[decision.selected_index]), 'forced_pp': False}
            decisions.append(record)
            if now == TARGET_TIME:
                capture['decision'] = record
            return decision

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

    originator = RepeatedOriginator.load(Path(frozen['model_path']), frozen_threshold_path=frozen_path,
        expected_city='cologne', network_context=runtime.read_network_right_of_way_context(runtime.net_file_from_sumocfg(sumocfg)))
    tripinfo = scratch / 'model_repeated.xml'
    try:
        metrics = runtime.evaluate_policy_v3(sumo_api=api, sumocfg=sumocfg, scenario='cologne1', policy=RUNTIME_POLICY,
            models=_runtime_models(originator, prediction_horizon_sec=450), duration_sec=END_TIME-25200,
            control_interval_sec=10, warmup_sec=60, seed=SEED, tripinfo_output=tripinfo,
            residual_coordination_mode='direct', residual_cooldown_intervals_override=0, step_observer=observer)
    finally:
        tripinfo.unlink(missing_ok=True)
    costs = [row['cost'] for row in samples]
    accepted = metrics.get('accepted_intervention_trace', [])
    window_accepted = [row for row in accepted if TARGET_TIME <= row['time_sec'] < END_TIME]
    current = {'capture': capture, 'decisions': decisions, 'samples': samples, 'accepted_intervention_trace': accepted}
    checks, matching = compare_reused_prefix(current, reused_traces['model_once_then_pp'])
    checks.update(evaluator_complete=metrics.get('ok') is True,
        native_reference_trace_exact=repeated_trace_exact(accepted, trace),
        full_horizon=[row['time_sec'] for row in samples] == list(range(TARGET_TIME+1, END_TIME+1)))
    prefix_safety = v3._finalize_safety_ledger(prefix_ledger)
    safety = v3._finalize_safety_ledger(window_ledger)
    safe = all(row['raw_collision_events'] == 0 and row['no_teleport_passed'] for row in (prefix_safety, safety))
    branch = {'status': 'INVALID', 'checks': checks, 'prefix_safety': prefix_safety, 'safety': safety,
        'sample_count': len(costs), 'first_interval_costs': costs[:10],
        'mean_halted_per_lane': float(np.mean(costs)) if costs else None,
        'accepted_trace_count': len(accepted), 'prefix_decision_count': sum(row['time_sec'] < TARGET_TIME for row in decisions),
        'window_accepted_switch_count': sum(row['override_kind'] == 'switch' for row in window_accepted),
        'window_accepted_stay_count': sum(row['override_kind'] == 'stay' for row in window_accepted),
        'maximum_recorded_green_elapsed_sec': max((row['green_elapsed_sec'] for row in samples), default=None),
        'runtime_error': metrics.get('error')}
    if capture.get('decision'):
        branch.update(start_population=capture['population'], first_decision=capture['decision'])
        checks['single_tls_eight_lanes'] = capture['tls_ids'] == [TLS] and len(capture['lanes']) == 8
    if samples:
        branch.update(end_population={key:samples[-1][key] for key in ('active','pending')},
            window_arrived=sum(row['arrived'] for row in samples), window_departed=sum(row['departed'] for row in samples),
            window_active_vehicle_hours=sum(row['active'] for row in samples)/3600,
            window_pending_vehicle_hours=sum(row['pending'] for row in samples)/3600,
            window_system_vehicle_hours=sum(row['active']+row['pending'] for row in samples)/3600,
            end_green_elapsed_sec=samples[-1]['green_elapsed_sec'],
            per_150_seconds=[{'start_sec':TARGET_TIME+i,'end_sec':TARGET_TIME+i+150,
                             'mean_halted_per_lane':float(np.mean(costs[i:i+150]))} for i in (0,150,300)] if len(costs)==450 else [])
        checks['active_population_accounting'] = (capture['population']['active'] + branch['window_departed']
            - branch['window_arrived'] == branch['end_population']['active'])
    branch['status'] = 'FAIL_UNSAFE' if not safe else 'PASS' if all(checks.values()) else 'INVALID'
    runs = {**reused['runs'], 'model_repeated': branch}
    result = {'protocol': PROTOCOL, 'status': branch['status'], 'seed': SEED, 'target_time_sec': TARGET_TIME, 'end_time_sec': END_TIME,
        'source_root': protocol['source_root'], 'sumocfg': str(sumocfg), 'frozen_threshold': frozen,
        'selected_trace': target, 'checks': checks, 'reuse_matching': matching, 'runs': runs,
        'label_contract': protocol['label_contract'], 'reuse_result': protocol['reuse_result'], 'reuse_traces': protocol['reuse_traces'],
        'new_simulation_count': 1, 'reused_branch_count': 2, 'new_model_fits': 0, 'threshold_retuning': False,
        'full_traces_server_only': str(output/'full_traces_server_only.json'), 'elapsed_sec': time.monotonic()-started}
    if result['status'] == 'PASS':
        result['comparison'] = comparison_metrics(runs)
    (output/'full_traces_server_only.json').write_text(json.dumps({'model_repeated': current}, separators=(',', ':'), allow_nan=False)+'\n')
    (output/'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('protocol', 'status', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if run(json.loads(args.protocol.read_text()), args.output)['status'] != 'PASS':
        raise SystemExit(2)
