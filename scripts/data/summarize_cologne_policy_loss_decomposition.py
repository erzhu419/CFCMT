"""Bind and summarize the original A policy's S-allowed label regret."""

import argparse
import json
from pathlib import Path

from cologne_policy_loss_summary import summarize_loss

PROTOCOL = 'tsc-v155h-cologne-policy-loss-decomposition-v1'
BOUND = ('source_root', 'sumocfg', 'reference_model_path', 'training_group_ids',
         'folds', 'original_feature_names', 'threshold')


def aggregate(protocol, result):
    if (protocol['protocol'] != PROTOCOL or result['protocol'] != PROTOCOL or result['status'] != 'COMPLETE'
            or any(result[key] != protocol[key] for key in BOUND)
            or result['threshold'] != 0 or len(result['original_feature_names']) != 29
            or result['metrics'] != protocol['expected_original_metrics']
            or any(result[key] != 0 for key in ('new_model_fits', 'new_labels', 'new_simulations'))
            or result['threshold_retuned'] is not False
            or not all(result['checks'].values()) or not all(result['audit']['checks'].values())):
        raise ValueError('Require the original A policy and a completed data-only decomposition')
    rows = result['audit']['rows']
    if (result['audit']['group_count'] != 100
            or [row['group_id'] for row in rows] != protocol['training_group_ids']):
        raise ValueError('Retain every original group in the fixed order')
    summary = summarize_loss(rows)
    for actual, expected in [(summary, result['metrics']),
            *[(summary['per_seed'][seed], result['metrics']['per_seed'][seed]) for seed in result['metrics']['per_seed']]]:
        if (any(actual['policy_outcomes'][key] != expected[key] for key in (
                'accepted_overrides', 'beneficial', 'harmful', 'equal_cost'))
                or abs(actual['means']['policy_cost'] - expected['mean_cost']) > 1e-12
                or abs(actual['means']['reference_cost'] - expected['mean_phase_pressure_cost']) > 1e-12):
            raise ValueError('Decomposition must reproduce original A costs and accepted outcomes')
    return {'protocol': PROTOCOL, 'status': 'COMPLETE', **{key: result[key] for key in BOUND},
        'summary': summary, 'original_metrics': result['metrics'],
        'category_definitions': protocol['categories'], 'allowed_actions': protocol['allowed_actions'],
        'checks': {**result['checks'], **result['audit']['checks'], **summary['checks'],
                   'original_policy_metrics_reproduced': True},
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
    task = next(row['id'] for row in tasks if row['signature'] == 'CFCMT/v155h/policy-loss-decomposition/v1')
    result['accounting'].update(task_id=task, retrieved_result_bytes=args.result.stat().st_size)
    result['refs'] = {key: str(getattr(args, key).resolve()) for key in ('protocol', 'result', 'launch_record')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    summary = result['summary']
    print(json.dumps({'status': result['status'], 'means': summary['means'],
        'categories': {key: {'groups': value['groups'], 'regret_share': value['regret_share']}
                       for key, value in summary['categories'].items()},
        'accounting': result['accounting']}))


if __name__ == '__main__':
    main()
