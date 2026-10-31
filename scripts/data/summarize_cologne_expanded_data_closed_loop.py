"""Compare the B195 closed-loop controller with the frozen A and PP runs."""

import argparse
import json
from pathlib import Path
from statistics import mean

if __package__:
    from .summarize_cologne_raw_target_closed_loop import (
        PROTOCOL as BASELINE_PROTOCOL, NEW as ORIGINAL_A, PP, GATED, PP_PROTOCOL, THRESHOLD_PROTOCOL,
        OCCUPANCY_PROTOCOL, COMPARISON, SEEDS, SERVICE_KEYS, METRIC_KEYS, SPACING, LANES, SAMPLES, safety_summary)
    from .summarize_cologne_spaced_no_stay import execution_summary
else:
    from summarize_cologne_raw_target_closed_loop import (
        PROTOCOL as BASELINE_PROTOCOL, NEW as ORIGINAL_A, PP, GATED, PP_PROTOCOL, THRESHOLD_PROTOCOL,
        OCCUPANCY_PROTOCOL, COMPARISON, SEEDS, SERVICE_KEYS, METRIC_KEYS, SPACING, LANES, SAMPLES, safety_summary)
    from summarize_cologne_spaced_no_stay import execution_summary

PROTOCOL = 'tsc-v155m-cologne-expanded-data-closed-loop-v1'
MODEL_PROTOCOL = 'tsc-v155l-cologne-expanded-raw-target-model-v1'
ORIGINATOR_PROTOCOL = 'tsc-v155m-expanded-data-no-stay-originator-v1'
NEW = 'expanded_data_spaced_no_stay_rigid'
ARMS = (NEW, ORIGINAL_A, PP)


def service_summary(cells, seeds):
    indexed = {(row['seed'], row['arm']): row for row in cells}
    keys = (*SERVICE_KEYS, 'mean_queue_per_lane')
    means = {arm: {key: mean(indexed[seed, arm]['metrics'][key] for seed in seeds) for key in keys} for arm in ARMS}
    return {'seeds': list(seeds), 'weighting': 'equal_seed', 'primary_metric': 'mean_queue', 'means_by_arm': means,
            **{name: {key: {'difference_mean': means[NEW][key] - means[reference][key],
                           'relative_difference_of_means_percent': 100 * (means[NEW][key] / means[reference][key] - 1)
                           if means[reference][key] != 0 else None,
                           'paired_differences': [{'seed': seed, 'expanded_minus_reference':
                               indexed[seed, NEW]['metrics'][key] - indexed[seed, reference]['metrics'][key]} for seed in seeds]}
                      for key in keys} for name, reference in (('expanded_vs_original_a', ORIGINAL_A), ('expanded_vs_phase_pressure', PP))}}


def aggregate(protocol, cells, baseline):
    frozen, controls = protocol['frozen_threshold'], protocol['controls']
    if (protocol['protocol'] != PROTOCOL or protocol['spacing_contract'] != SPACING
            or len(cells) != 3 or {row['seed'] for row in cells} != set(SEEDS)
            or frozen['protocol'] != THRESHOLD_PROTOCOL
            or frozen['calibration_protocol'] != 'tsc-v155k-cologne-expanded-data-oof-v1'
            or frozen['threshold'] != 0 or frozen['mode'] != 'minimum_advantage' or frozen['comparison'] != COMPARISON
            or frozen['score_units'] != 'raw_native_halted_per_lane_cost_difference'
            or frozen['source_root'] != protocol['source_root']
            or frozen['fit_result'] != protocol['full_fit_result'] or frozen['calibration_result'] != protocol['calibration_result']):
        raise ValueError('Use the completed L B195 model and K-fixed threshold on the three frozen evaluation seeds')
    if (baseline['protocol'] != BASELINE_PROTOCOL or baseline['seeds'] != list(SEEDS)
            or baseline['source_root'] != protocol['source_root'] or baseline['repaired_sumocfg'] != protocol['sumocfg']
            or baseline['model_path'] != controls['old_model_path'] or baseline['frozen_threshold'] != controls['old_frozen_threshold']
            or baseline['spacing_contract'] != SPACING or baseline['new_task_ids'] != controls['original_a_tasks']
            or baseline['reused_task_ids'][PP] != controls['phase_pressure_tasks']
            or any(baseline[key] != protocol[key] for key in ('threshold_originator_path', 'ablation_originator_path'))):
        raise ValueError('Reuse the original B A-model runs and same-network PP cells with unchanged executor and timing')
    reused = [row for row in baseline['cells'] if row['arm'] in (ORIGINAL_A, PP)]
    if len(reused) != 6 or {(row['seed'], row['arm']) for row in reused} != {(seed, arm) for seed in SEEDS for arm in (ORIGINAL_A, PP)}:
        raise ValueError('Require exactly six original A and PP comparison cells')
    for row in reused:
        is_model = row['arm'] == ORIGINAL_A
        if (row['source_protocol'] != (BASELINE_PROTOCOL if is_model else PP_PROTOCOL)
                or row['task_id'] != controls['original_a_tasks' if is_model else 'phase_pressure_tasks'][SEEDS.index(row['seed'])]
                or (is_model and (row['source_root'] != protocol['source_root'] or row['model_path'] != controls['old_model_path']))):
            raise ValueError('A comparison cell changed its source, model or original task')
    rows = sorted(cells, key=lambda row: SEEDS.index(row['seed']))
    compact = []
    for row in rows:
        if (row['protocol'] != PROTOCOL or row['arm'] != GATED or row['scenario'] != 'cologne1'
                or row['source_root'] != protocol['source_root'] or row['repaired_sumocfg'] != protocol['sumocfg']
                or row['frozen_threshold'] != frozen or row['model_path'] != frozen['model_path']
                or row['score_units'] != frozen['score_units'] or row['occupancy_equation_protocol'] != OCCUPANCY_PROTOCOL
                or any(row[key] != protocol[key] for key in ('threshold_originator_path', 'ablation_originator_path',
                                                            'expanded_originator_path', 'full_fit_result', 'calibration_result'))):
            raise ValueError('New cells must bind the B195 model, original source/network and actual expanded-model loader')
        diag, metrics = row['originator_diagnostics'], row['metrics']
        if diag['target_budget'] != 195 or diag['training_group_count'] != 195 or diag['model_protocol'] != MODEL_PROTOCOL:
            raise ValueError('Runtime diagnostics must identify the complete B195 payload')
        if row['run_valid'] and (metrics['controlled_lane_count'] != LANES or metrics['fixed_horizon_sample_seconds'] != SAMPLES
                or abs(metrics['mean_queue'] - metrics['halted_vehicle_seconds'] / SAMPLES) > 1e-12
                or abs(metrics['mean_queue_per_lane'] - metrics['mean_queue'] / LANES) > 1e-12):
            raise ValueError('Preserve the original fixed-horizon queue and lane denominators')
        compact.append({**{key: row[key] for key in ('seed', 'run_valid', 'status', 'zero_incident_status', 'task_id', 'source_result',
            'elapsed_sec', 'source_root', 'model_path', 'occupancy_equation_protocol')}, 'arm': NEW, 'source_arm': GATED,
            'source_protocol': PROTOCOL, 'reused': False,
            'metrics': {key: metrics.get(key) for key in (*METRIC_KEYS, 'mean_queue_per_lane')},
            'collision_lane_pairs': row['collision_inventory']['incident_counts_by_first_participant_lane_pair']})
    compact.extend({**row, 'reused': True} for row in reused)
    indexed = {(row['seed'], row['arm']): row for row in compact}
    compact = [indexed[seed, arm] for seed in SEEDS for arm in ARMS]
    prior = baseline['execution_diagnosis']
    prior_view = {'execution_diagnosis': {'per_seed': prior['per_seed'], 'totals': prior['totals'],
        'maximum_green_elapsed_sec_by_seed': [{'seed': row['seed'], 'by_tls': row['maximum_green_elapsed_sec_by_tls']} for row in prior['spacing_by_seed']]}}
    execution = execution_summary(rows, frozen, prior_view, originator_protocol=ORIGINATOR_PROTOCOL)
    execution['original_a_baseline'] = execution.pop('continuous_baseline')
    complete = all(row['run_valid'] for row in compact)
    service = service_summary(compact, SEEDS) if complete else None
    decision = None if service is None else {
        'expanded_improves_original_a': service['expanded_vs_original_a']['mean_queue']['difference_mean'] < 0,
        'expanded_beats_phase_pressure': service['expanded_vs_phase_pressure']['mean_queue']['difference_mean'] < 0,
        'criterion': 'Sign of the frozen equal-seed mean_queue difference; descriptive service metrics do not replace it.',
    }
    return {'protocol': PROTOCOL, 'matrix_status': 'COMPLETE' if complete else 'INVALID', 'seeds': list(SEEDS),
            'source_root': protocol['source_root'], 'repaired_sumocfg': protocol['sumocfg'], 'model_path': frozen['model_path'],
            'frozen_threshold': frozen, 'spacing_contract': SPACING, 'target_budget': 195,
            **{key: protocol[key] for key in ('expanded_originator_path', 'threshold_originator_path', 'ablation_originator_path',
                                            'full_fit_result', 'calibration_result')},
            'primary_efficiency': {'metric': 'mean_queue', 'controlled_lane_count': LANES, 'samples_per_run': SAMPLES,
                                  'training_target_equivalent': 'mean_queue_per_lane = mean_queue / 8'},
            'cells': compact, 'safety_by_arm': {arm: safety_summary([row for row in compact if row['arm'] == arm]) for arm in ARMS},
            'all_seed_service': service,
            'additional_seed_service': service_summary(compact, (52282, 63792)) if complete else None,
            'execution_diagnosis': execution, 'new_task_ids': [row['task_id'] for row in rows],
            'reused_task_ids': {arm: [indexed[seed, arm]['task_id'] for seed in SEEDS] for arm in (ORIGINAL_A, PP)},
            'new_simulation_count': 3, 'reused_cell_count': 6, 'new_model_fit_count': 0, 'threshold_retuned': False,
            'new_simulation_elapsed_seconds_sum': sum(row['elapsed_sec'] for row in rows),
            'interpretation': 'B195 replaces the original A B100 model under the same top1 selection, S veto and450-second spacing. '
                              'Complete colliding runs remain in efficiency means; invalid runs disable means. '
                              'Extra labels increase the training budget; these inspected evaluation seeds provide development evidence.',
            'decision': decision, 'limitations': protocol['limits'],
            'next_step': {'protocol': 'fresh-seed-fixed-policy-validation-v1',
                          'seeds': [152282, 141242, 163792, 252282, 241242, 263792],
                          'selection_rule': 'Add100000 and200000 to each original evaluation seed, preserving original seed order.',
                          'arms': [NEW, ORIGINAL_A, PP], 'runs': 18, 'duration_sec': 3600,
                          'retune_after_outcomes': False}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'results-root', 'launch-record', 'baseline-summary', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    launch = json.loads(args.launch_record.read_text())
    tasks = {row['signature']: row['id'] for row in json.loads(launch['scheduler_stdout'])['submitted']}
    cells, retrieved = [], 0
    for seed in SEEDS:
        path = args.results_root / 'evaluation' / f'seed_{seed}' / GATED / 'result.json'
        row = json.loads(path.read_text())
        if (row['seed'], row['arm']) != (seed, GATED):
            raise ValueError('Evaluation result must match its seed and arm path')
        row.update(source_result=str(path), task_id=tasks[f'CFCMT/v155m/expanded-data-spaced-v1/evaluate/{seed}'])
        cells.append(row)
        retrieved += path.stat().st_size
    result = aggregate(json.loads(args.protocol.read_text()), cells, json.loads(args.baseline_summary.read_text()))
    result.update(retrieved_result_bytes=retrieved, refs={'baseline_summary': str(args.baseline_summary),
                  'protocol': str(args.protocol), 'launch_record': str(args.launch_record)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'matrix_status': result['matrix_status'], 'retrieved_result_bytes': retrieved}))


if __name__ == '__main__':
    main()
