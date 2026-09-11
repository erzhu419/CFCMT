"""Summarize eighteen fresh-seed runs of the fixed B195, A and PP policies."""

import argparse
import json
from pathlib import Path

if __package__:
    from .summarize_cologne_expanded_data_closed_loop import (
        NEW, ORIGINAL_A, PP, ARMS, ORIGINATOR_PROTOCOL, MODEL_PROTOCOL, SPACING, LANES, SAMPLES,
        SERVICE_KEYS, METRIC_KEYS, OCCUPANCY_PROTOCOL, COMPARISON, THRESHOLD_PROTOCOL, service_summary)
    from .summarize_cologne_spaced_no_stay import execution_summary
else:
    from summarize_cologne_expanded_data_closed_loop import (
        NEW, ORIGINAL_A, PP, ARMS, ORIGINATOR_PROTOCOL, MODEL_PROTOCOL, SPACING, LANES, SAMPLES,
        SERVICE_KEYS, METRIC_KEYS, OCCUPANCY_PROTOCOL, COMPARISON, THRESHOLD_PROTOCOL, service_summary)
    from summarize_cologne_spaced_no_stay import execution_summary

PROTOCOL = 'tsc-v155n-cologne-expanded-data-fresh-seed-validation-v1'
SEEDS = (152282, 141242, 163792, 252282, 241242, 263792)
SCORE_UNITS = 'raw_native_halted_per_lane_cost_difference'
ORIGINAL_ORIGINATOR_PROTOCOL = 'tsc-v154s-no-stay-fraction-rigid-originator-v1'


def task_signature(arm, seed):
    return f'CFCMT/v155n/fresh-seed-v1/{arm}/{seed}'


def effective_task_records(launch):
    if 'effective_tasks' in launch:
        records = launch['effective_tasks']
        if (len(records) != 18 or len({row['logical_signature'] for row in records}) != 18
                or len({row['task_id'] for row in records}) != 18):
            raise ValueError('The effective launch record must map eighteen unique logical cells to tasks')
        return {row['logical_signature']: row for row in records}
    submitted = json.loads(launch['scheduler_stdout'])['submitted']
    return {row['signature']: {'logical_signature': row['signature'],
            'scheduler_signature': row['signature'], 'task_id': row['id'], 'attempt': 1}
            for row in submitted}


def fresh_safety_summary(rows):
    valid = all(row['run_valid'] for row in rows)
    keys = ('collision_incidents', 'collision_events', 'starting_teleports', 'ending_teleports')
    zero = sum(row['run_valid'] and all(row['metrics'][key] == 0 for key in keys) for row in rows)
    return {'all_seed_runs_valid': valid, 'zero_incident_seed_count': zero,
            'zero_incident_status': 'UNAVAILABLE' if not valid else 'PASS' if zero == len(rows) else 'FAIL',
            **{key + '_total': sum(row['metrics'][key] for row in rows) if valid else None for key in keys}}


def _execution(rows, frozen, originator_protocol):
    # Only current-run accounting is requested; no previously run comparison arm is reused.
    empty_baseline = {'execution_diagnosis': {'per_seed': [], 'totals': {}, 'maximum_green_elapsed_sec_by_seed': []}}
    result = execution_summary(rows, frozen, empty_baseline, originator_protocol=originator_protocol)
    result.pop('continuous_baseline')
    return result


def _frozen_models(protocol):
    models = {NEW: protocol['frozen_threshold'], ORIGINAL_A: protocol['original_a_frozen_threshold']}
    for arm, frozen in models.items():
        calibration = ('tsc-v155k-cologne-expanded-data-oof-v1' if arm == NEW else 'tsc-v155a-cologne-raw-target-refit-v1')
        if (frozen['protocol'] != THRESHOLD_PROTOCOL or frozen['calibration_protocol'] != calibration
                or frozen['threshold'] != 0 or frozen['mode'] != 'minimum_advantage'
                or frozen['comparison'] != COMPARISON or frozen['score_units'] != SCORE_UNITS
                or frozen['source_root'] != protocol['source_root']):
            raise ValueError('Both models must retain their own frozen raw-score threshold zero')
    if (models[NEW]['fit_result'] != protocol['full_fit_result']
            or models[NEW]['calibration_result'] != protocol['calibration_result']
            or models[ORIGINAL_A]['ablation_originator_path'] != protocol['ablation_originator_path']):
        raise ValueError('The L/K and original A fit bindings must remain frozen')
    return models


def aggregate(protocol, cells):
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    evaluation = protocol['evaluation']
    timing = {'scenario': 'cologne1', 'duration_sec': 3600, 'warmup_sec': 60, 'control_interval_sec': 10,
              'prediction_horizon_sec': 450, 'residual_coordination_mode': 'direct', 'residual_cooldown_intervals': 44}
    if (protocol['protocol'] != PROTOCOL or evaluation['seeds'] != list(SEEDS)
            or any(evaluation[key] != value for key, value in timing.items())
            or protocol['arms'] != list(ARMS) or protocol['spacing_contract'] != SPACING
            or len(cells) != 18 or {(row['seed'], row['arm']) for row in cells} != expected
            or len({row['task_id'] for row in cells}) != 18):
        raise ValueError('Require all eighteen newly run cells on the six frozen fresh seeds')
    frozen_models = _frozen_models(protocol)
    rows = sorted(cells, key=lambda row: (SEEDS.index(row['seed']), ARMS.index(row['arm'])))
    compact = []
    for row in rows:
        arm, metrics = row['arm'], row['metrics']
        runtime_bindings = {'evaluator': protocol['source_root'] + '/cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py',
                            'executor': protocol['source_root'] + '/cf_h2o/traffic_signal/safe_phase_controller.py'}
        yellow = {'passed': True, 'initial_state': 'Ggrr', 'target_state': 'rrGG', 'clearance_state': 'Yyrr'}
        if (row['protocol'] != PROTOCOL or row['scenario'] != 'cologne1' or row.get('reused', False)
                or row['source_root'] != protocol['source_root'] or row['repaired_sumocfg'] != protocol['sumocfg']
                or row['runtime_bindings'] != runtime_bindings or '1.22.0' not in str(row['sumo_version'])
                or row['yellow_priority_check'] != yellow or row['occupancy_equation_protocol'] != OCCUPANCY_PROTOCOL
                or row['deployed_policy'] != ('phase_pressure' if arm == PP else 'causal_rigid_advantage_contrast_source_utility')
                or row['task_signature'] != task_signature(arm, row['seed'])
                or (row['scheduler_task_signature'] != row['task_signature']
                    and row['scheduler_task_signature'] != row['task_signature'] + '/reporting-fix1')
                or not isinstance(row['task_attempt'], int) or row['task_attempt'] < 1
                or not row['validity_checks'] or row['run_valid'] != all(row['validity_checks'].values())
                or row['status'] != ('COMPLETE' if row['run_valid'] else 'INVALID')):
            raise ValueError('Every cell must bind its own fresh task, frozen source/network and runtime validity')
        if arm == PP:
            if (any(row[key] is not None for key in ('model_path', 'frozen_threshold', 'originator_diagnostics'))
                    or metrics['accepted_intervention_trace'] or metrics['guard_audit'] != {}
                    or metrics['residual_deployment'] != {}):
                raise ValueError('Fresh PP runs must have no fitted model or learned originator')
        else:
            frozen, diag = frozen_models[arm], row['originator_diagnostics']
            if (row['frozen_threshold'] != frozen or row['model_path'] != frozen['model_path']
                    or row['score_units'] != SCORE_UNITS or row['spacing_contract'] != SPACING
                    or any(row[key] != protocol[key] for key in ('threshold_originator_path', 'ablation_originator_path'))
                    or diag['target_budget'] != (195 if arm == NEW else 100)):
                raise ValueError('Model cells must preserve their own payload, threshold and S/spacing execution rule')
            if arm == NEW and (any(row[key] != protocol[key] for key in
                    ('expanded_originator_path', 'full_fit_result', 'calibration_result'))
                    or diag['model_protocol'] != MODEL_PROTOCOL or diag['training_group_count'] != 195):
                raise ValueError('The expanded-model runs must bind the complete L B195 payload and K threshold')
        if row['run_valid'] and (metrics['controlled_lane_count'] != LANES or metrics['fixed_horizon_sample_seconds'] != SAMPLES
                or abs(metrics['mean_queue'] - metrics['halted_vehicle_seconds'] / SAMPLES) > 1e-12
                or abs(metrics['mean_queue_per_lane'] - metrics['mean_queue'] / LANES) > 1e-12):
            raise ValueError('All three arms must use the same fixed-horizon halted-queue denominator')
        compact.append({**{key: row[key] for key in ('seed', 'arm', 'task_id', 'task_signature', 'source_result', 'status',
            'run_valid', 'validity_checks', 'zero_incident_status', 'elapsed_sec', 'source_root', 'model_path', 'deployed_policy',
            'yellow_priority_check', 'runtime_bindings', 'sumo_version', 'occupancy_equation_protocol',
            'scheduler_task_signature', 'task_attempt')},
            'reused': False, 'source_protocol': PROTOCOL,
            'metrics': {key: metrics.get(key) for key in (*METRIC_KEYS, 'mean_queue_per_lane')},
            'collision_lane_pairs': row['collision_inventory']['incident_counts_by_first_participant_lane_pair']})
    execution = {arm: _execution([row for row in rows if row['arm'] == arm], frozen_models[arm],
                                ORIGINATOR_PROTOCOL if arm == NEW else ORIGINAL_ORIGINATOR_PROTOCOL)
                 for arm in (NEW, ORIGINAL_A)}
    complete = all(row['run_valid'] for row in compact)
    service = service_summary(compact, SEEDS) if complete else None
    if service:
        for comparison in ('expanded_vs_original_a', 'expanded_vs_phase_pressure'):
            for metric in (*SERVICE_KEYS, 'mean_queue_per_lane'):
                paired = service[comparison][metric]['paired_differences']
                for pair in paired:
                    difference = pair['expanded_minus_reference']
                    pair['sign'] = 'lower' if difference < 0 else 'higher' if difference > 0 else 'equal'
                service[comparison][metric]['paired_sign_counts'] = {sign: sum(pair['sign'] == sign for pair in paired)
                                                                     for sign in ('lower', 'higher', 'equal')}
    decision = None if service is None else {
        'b195_data_effect_retained': service['expanded_vs_original_a']['mean_queue']['difference_mean'] < 0,
        'b195_beats_phase_pressure': service['expanded_vs_phase_pressure']['mean_queue']['difference_mean'] < 0,
        'criterion': 'Strict sign of the frozen six-seed mean_queue differences, without retuning.'}
    return {'protocol': PROTOCOL, 'matrix_status': 'COMPLETE' if complete else 'INVALID', 'seeds': list(SEEDS), 'arms': list(ARMS),
            'source_root': protocol['source_root'], 'repaired_sumocfg': protocol['sumocfg'], 'spacing_contract': SPACING,
            'frozen_models': frozen_models, 'cells': compact, 'all_seed_service': service, 'decision': decision,
            'primary_efficiency': {'metric': 'mean_queue', 'weighting': 'equal_seed', 'samples_per_run': SAMPLES,
                                  'primary_contrast': 'expanded_vs_phase_pressure', 'data_size_contrast': 'expanded_vs_original_a',
                                  'controlled_lane_count': LANES, 'supporting_metrics': 'Descriptive waiting, system vehicle hours and arrivals.'},
            'primary_analysis': protocol['primary_analysis'], 'limitations': protocol['limits'],
            'execution_by_model_arm': execution,
            'safety_by_arm': {arm: fresh_safety_summary([row for row in compact if row['arm'] == arm]) for arm in ARMS},
            'new_task_ids': [row['task_id'] for row in rows], 'new_simulation_count': 18, 'reused_cell_count': 0,
            'new_model_fit_count': 0, 'threshold_retuned': False,
            'new_simulation_elapsed_seconds_sum': sum(row['elapsed_sec'] for row in rows),
            'interpretation': 'All three arms are newly run on the six preselected seeds. Complete colliding runs remain in efficiency means; '
                              'an invalid cell disables the complete-matrix means and decision. The B195 model uses the expanded training budget.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'results-root', 'launch-record', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads(args.launch_record.read_text())
    tasks = effective_task_records(launch)
    signatures = {task_signature(arm, seed) for seed in SEEDS for arm in ARMS}
    if set(tasks) != signatures:
        raise ValueError('The launch record must contain exactly the eighteen frozen N task signatures')
    cells, retrieved = [], 0
    for seed in SEEDS:
        for arm in ARMS:
            path = args.results_root / 'evaluation' / f'seed_{seed}' / arm / 'result.json'
            row = json.loads(path.read_text())
            if (row['seed'], row['arm']) != (seed, arm):
                raise ValueError('The fresh-seed result must match its arm and seed path')
            signature = task_signature(arm, seed)
            task = tasks[signature]
            row.update(source_result=str(path), task_signature=signature, task_id=task['task_id'],
                       scheduler_task_signature=task['scheduler_signature'], task_attempt=task['attempt'])
            cells.append(row)
            retrieved += path.stat().st_size
    result = aggregate(json.loads(args.protocol.read_text()), cells)
    result.update(retrieved_result_bytes=retrieved,
                  operational_attempts=launch.get('operational_attempts'),
                  refs={'protocol': str(args.protocol), 'launch_record': str(args.launch_record)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'matrix_status': result['matrix_status'], 'decision': result['decision'], 'retrieved_result_bytes': retrieved}))


if __name__ == '__main__':
    main()
