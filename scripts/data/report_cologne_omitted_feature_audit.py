"""Bind the V155F server result and summarize its unchanged144 pairs."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from summarize_cologne_omitted_feature_audit import summarize_audit

PROTOCOL = 'tsc-v155f-cologne-omitted-feature-audit-v1'
BOUND = ('source_root', 'sumocfg', 'model_path', 'training_group_ids', 'folds',
         'frozen_threshold', 'selected_policy', 'proposal_accounting',
         'original_model_feature_names', 'inspected_feature_names')


def report(protocol, result):
    if (protocol['protocol'] != PROTOCOL or result['protocol'] != PROTOCOL
            or result['status'] != 'COMPLETE'
            or any(result[key] != protocol[key] for key in BOUND)
            or any(result[key] != 0 for key in ('new_model_fits', 'new_labels', 'new_simulations'))
            or result['threshold_retuned'] is not False or result['neighbors_reselected'] is not False
            or not all(result['checks'].values()) or not all(result['audit']['checks'].values())
            or result['audit']['feature_names'] != protocol['inspected_feature_names']):
        raise ValueError('Require the frozen A/E inputs and an unchanged144-pair diagnosis')
    queries = deepcopy(result['audit']['queries'])
    for query in queries:
        query.pop('feature_values')
        for neighbor in query['neighbors']:
            neighbor.pop('feature_values')
    if queries != protocol['expected_queries']:
        raise ValueError('Query and neighbor identities, order, gains and original distances must match E')
    expected_folds = {fold['heldout_seed']: fold for fold in protocol['folds']}
    for fold in result['audit']['folds']:
        expected = expected_folds[fold['heldout_seed']]
        if (fold['fit_group_count'] != expected['fit_group_count']
                or fold['fit_row_count'] != 7 * expected['fit_group_count']
                or fold['training_reference_rows'] != 0
                or fold['training_seeds'] != sorted(set(expected_folds) - {fold['heldout_seed']})):
            raise ValueError('Feature scaling must use the corresponding original fit fold')
    summary = summarize_audit(result['audit'])
    return {'protocol': PROTOCOL, 'status': 'COMPLETE',
        **{key: result[key] for key in BOUND}, **summary,
        'fold_diagnostics': result['audit']['folds'],
        'feature_semantics': {**protocol['feature_semantics'],
            'queue_max': 'Maximum incoming-lane queue proxy: halted vehicles +0.35*vehicle count '
                         '+0.025*TraCI occupancy in percent, among candidate green/red lanes.',
            'protocol_description_correction': 'The frozen protocol called queue_max a halted count. '
                'The frozen extractor confirms it is the weighted queue proxy; all analyzed columns and values are unchanged.'},
        'checks': {**result['checks'], **result['audit']['checks'], **summary['checks'],
                   'local_e_roster_exact': True},
        'accounting': {key: result[key] for key in ('new_model_fits', 'new_labels', 'new_simulations',
                      'threshold_retuned', 'neighbors_reselected', 'elapsed_sec')},
        'limitations': protocol['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'result', 'launch-record', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = report(json.loads(args.protocol.read_text()), json.loads(args.result.read_text()))
    launch = json.loads(args.launch_record.read_text())
    tasks = json.loads(launch['scheduler_stdout'])['submitted']
    task_id = next(task['id'] for task in tasks if task['signature'] == 'CFCMT/v155f/omitted-feature-audit/v1')
    result['accounting'].update(task_id=task_id, retrieved_result_bytes=args.result.stat().st_size)
    result['refs'] = {key: str(getattr(args, key).resolve()) for key in ('protocol', 'result', 'launch_record')}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'accounting': result['accounting']}))


if __name__ == '__main__':
    main()
