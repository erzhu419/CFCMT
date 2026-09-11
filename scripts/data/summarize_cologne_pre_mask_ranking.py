"""Summarize the fixed A cached-score comparison of S mask ordering."""

import argparse
import json
from pathlib import Path

from cologne_pre_mask_summary import summarize_pre_mask
from run_cologne_pre_mask_ranking import BOUND, PROTOCOL


def aggregate(protocol, result):
    if (protocol['protocol'] != PROTOCOL or result['protocol'] != PROTOCOL
            or result['status'] != 'COMPLETE'
            or any(result[key] != protocol[key] for key in BOUND)
            or result['threshold'] != 0 or len(result['original_feature_names']) != 29
            or result['original_metrics'] != protocol['expected_original_metrics']
            or any(result[key] != 0 for key in ('new_model_fits', 'new_labels', 'new_simulations'))
            or result['threshold_retuned'] is not False
            or not all(result['checks'].values()) or not all(result['audit']['checks'].values())):
        raise ValueError('Require the frozen A scores and a completed mask-order comparison')
    rows = result['audit']['rows']
    if (result['audit']['group_count'] != 100
            or [row['group_id'] for row in rows] != protocol['training_group_ids']):
        raise ValueError('Retain every original group in the fixed order')
    summary = summarize_pre_mask(rows)
    for arm, key in [('model', 'metrics'), ('previous_model', 'original_metrics')]:
        for actual, expected in [(summary[arm], result[key]),
                *[(summary[arm]['per_seed'][seed], values)
                  for seed, values in result[key]['per_seed'].items()]]:
            if (any(actual[k] != expected[k] for k in
                    ('groups', 'accepted_overrides', 'beneficial', 'harmful', 'equal_cost'))
                    or any(abs(actual[k] - expected[k]) > 1e-12 for k in
                           ('mean_cost', 'mean_phase_pressure_cost', 'mean_cost_minus_phase_pressure'))):
                raise ValueError('Independent cost accounting must match both policies and all seeds')
    improvement = summary['comparisons']['overall']['model_minus_previous_model_cost'] < -1e-12
    return {'protocol': PROTOCOL, 'status': 'COMPLETE', **{key: result[key] for key in BOUND},
        'summary': summary, 'decision': {
            'offline_cost_improves_a': improvement,
            'advance_to_closed_loop': improvement,
            'criterion': protocol['decision']},
        'allowed_actions': protocol['allowed_actions'], 'selection': protocol['selection'],
        'checks': {**result['checks'], **result['audit']['checks'], **summary['checks'],
                   'both_policy_metrics_reproduced': True},
        'accounting': {key: result[key] for key in ('new_model_fits', 'new_labels', 'new_simulations',
                                                  'threshold_retuned', 'elapsed_sec')},
        'limitations': result['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'result', 'launch-record', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(json.loads(args.protocol.read_text()), json.loads(args.result.read_text()))
    launch = json.loads(args.launch_record.read_text())
    tasks = json.loads(launch['scheduler_stdout'])['submitted']
    task = next(row['id'] for row in tasks if row['signature'] == 'CFCMT/v155i/pre-mask-ranking/v1')
    result['accounting'].update(task_id=task, retrieved_result_bytes=args.result.stat().st_size)
    result['refs'] = {key: str(getattr(args, key).resolve()) for key in ('protocol', 'result', 'launch_record')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'comparisons': result['summary']['comparisons'],
        'action_changes': result['summary']['action_changes'], 'decision': result['decision'],
        'accounting': result['accounting']}))


if __name__ == '__main__':
    main()
