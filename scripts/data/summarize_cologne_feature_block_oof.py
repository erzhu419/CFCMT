"""Compare the fixed42-feature OOF experiment against A and PhasePressure."""

import argparse
import json
from pathlib import Path

from cologne_feature_block_comparison import compare_actions

PROTOCOL = 'tsc-v155g-cologne-feature-block-oof-v1'
BOUND = ('source_root', 'sumocfg', 'reference_model_path', 'training_group_ids', 'folds',
         'original_feature_names', 'added_feature_names', 'full_feature_names', 'threshold')


def aggregate(protocol, result):
    if (protocol['protocol'] != PROTOCOL or result['protocol'] != PROTOCOL or result['status'] != 'COMPLETE'
            or any(result[key] != protocol[key] for key in BOUND)
            or result['threshold'] != 0 or len(result['full_feature_names']) != 42
            or result['new_model_fits'] != 3 or result['oof_fold_fits'] != 3
            or any(result[key] != 0 for key in ('full_model_fits', 'inactive_correction_fits', 'new_labels', 'new_simulations'))
            or result['threshold_retuned'] is not False or not all(result['checks'].values())
            or result['metrics']['original_a'] != protocol['expected_original_metrics']):
        raise ValueError('Require the original fixed inputs and exactly three OOF fits')
    diagnostics = result['fit_diagnostics']
    if [row['heldout_seed'] for row in diagnostics] != [row['heldout_seed'] for row in protocol['folds']]:
        raise ValueError('All three original heldout folds must be present in order')
    for row, spec in zip(diagnostics, protocol['folds']):
        diag = row['diagnostics']
        if (row['weights_exactly_match_a'] is not True or diag['feature_names'] != protocol['full_feature_names']
                or diag['feature_count'] != 42 or diag['training_rows'] != 7 * spec['fit_group_count']
                or diag['rows'] != 8 * spec['fit_group_count'] or diag['raw_target_clipped_count'] != 0
                or diag['sample_weight_source_protocol'] != 'group_action_range_v1'):
            raise ValueError('Each fold must change only the input columns while preserving A weights')
    if {row['group_id'] for row in result['matched_actions']} != set(protocol['training_group_ids']):
        raise ValueError('Keep the original100 groups in the comparison')
    comparison = compare_actions(result['matched_actions'])
    for local, remote in (('model', 'model'), ('previous_model', 'original_a'), ('phase_pressure', 'phase_pressure')):
        actual, expected = comparison[local], result['metrics'][remote]
        for part, reference in [(actual, expected), *[(actual['per_seed'][seed], expected['per_seed'][seed])
                              for seed in expected['per_seed']]]:
            if (any(part[key] != reference[key] for key in ('groups', 'accepted_overrides', 'beneficial', 'harmful', 'equal_cost'))
                    or any(abs(part[key] - reference[key]) > 1e-12 for key in (
                        'mean_cost', 'mean_phase_pressure_cost', 'mean_cost_minus_phase_pressure'))):
                raise ValueError('Matched actions must reproduce model, A and PP metrics independently')
    return {'protocol': PROTOCOL, 'status': 'COMPLETE', **{key: result[key] for key in BOUND},
        'score_units': result['score_units'], 'comparison': comparison, 'fit_diagnostics': diagnostics,
        'proposal_accounting': result['proposal_accounting'], 'mask': result['mask'],
        'checks': {**result['checks'], 'independent_action_metrics_match': True},
        'accounting': {key: result[key] for key in ('new_model_fits', 'oof_fold_fits', 'full_model_fits',
            'inactive_correction_fits', 'new_labels', 'new_simulations', 'threshold_retuned', 'elapsed_sec')},
        'limitations': result['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'result', 'launch-record', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(json.loads(args.protocol.read_text()), json.loads(args.result.read_text()))
    launch = json.loads(args.launch_record.read_text())
    tasks = json.loads(launch['scheduler_stdout'])['submitted']
    task = next(row['id'] for row in tasks if row['signature'] == 'CFCMT/v155g/feature-block-oof/v1')
    result['accounting'].update(task_id=task, retrieved_result_bytes=args.result.stat().st_size)
    result['refs'] = {key: str(getattr(args, key).resolve()) for key in ('protocol', 'result', 'launch_record')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'comparison': result['comparison']['comparisons'],
                      'action_changes': result['comparison']['action_changes'], 'accounting': result['accounting']}))


if __name__ == '__main__':
    main()
