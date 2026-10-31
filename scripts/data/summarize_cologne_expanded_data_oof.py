"""Bind the six native-label shards and the fixed three-fold B195 comparison."""

import argparse
import json
from pathlib import Path

PROTOCOL = 'tsc-v155k-cologne-expanded-data-oof-v1'


def summarize(protocol, fit, local_root):
    if (fit['protocol'] != PROTOCOL or fit['status'] != 'COMPLETE'
            or fit['new_model_fits'] != 3 or fit['oof_fold_fits'] != 3
            or fit['full_model_fits'] != 0 or fit['threshold_retuned'] is not False
            or not all(fit['checks'].values())
            or any(fit[key] != protocol[key] for key in ('source_root', 'sumocfg', 'reference_model_path',
                'original_group_ids', 'additional_group_ids', 'original_feature_names', 'threshold', 'folds'))):
        raise ValueError('Require the completed frozen K comparison with only three OOF fits')
    cells, files = [], []
    for spec in protocol['collection_shards']:
        path = local_root / Path(spec['result']).relative_to(protocol['results_root'])
        cell = json.loads(path.read_text())
        if (cell['protocol'] != PROTOCOL or cell['status'] != 'COMPLETE'
                or cell['requested_groups'] != spec['group_ids'] or cell['retained_groups'] != spec['group_ids']
                or cell['rows'] != 8 * len(spec['group_ids']) or cell['invalid_groups']
                or cell['state_restores'] != 0 or cell['collision_admission_gate'] is not False):
            raise ValueError('Require every prescribed native label, retaining collision diagnostics')
        cells.append(cell)
        files.append(path)
    total = {key: sum(cell[key] for cell in cells) for key in ('rows', 'new_branches_attempted',
             'completed_branches', 'prefix_replay_sec', 'window_simulation_sec',
             'prefix_collision_events', 'window_collision_events', 'elapsed_sec')}
    if (sum(len(cell['retained_groups']) for cell in cells) != 95 or total['rows'] != 760
            or total['completed_branches'] != 760 or total['new_branches_attempted'] != 760
            or total['window_simulation_sec'] != 760 * 450):
        raise ValueError('All95 new groups need eight complete450-second outcomes')
    comparison = fit['action_comparison']
    return {'protocol': PROTOCOL, 'status': 'COMPLETE',
        **{key: fit[key] for key in ('source_root', 'sumocfg', 'reference_model_path', 'original_group_ids',
                                    'additional_group_ids', 'original_feature_names', 'threshold', 'folds')},
        'metrics': fit['metrics'], 'comparison': comparison, 'fit_diagnostics': fit['fit_diagnostics'],
        'proposal_accounting': fit['proposal_accounting'], 'mask': fit['mask'],
        'decision': {'improves_original_a': comparison['comparisons']['overall']['model_minus_previous_model_cost'] < -1e-12,
                     'criterion': protocol['decision']},
        'checks': {**fit['checks'], 'all760_new_native_labels_complete': True},
        'accounting': {'original_label_groups': 100, 'new_label_groups': 95, 'total_label_groups': 195,
            'reused_labels': 800, 'new_labels': 760, 'total_labels': 1560, 'new_native_branches': 760,
            'prefix_replay_simulation_sec': total['prefix_replay_sec'], 'label_window_simulation_sec': total['window_simulation_sec'],
            'collection_elapsed_sec_sum': total['elapsed_sec'], 'fit_elapsed_sec': fit['elapsed_sec'],
            'new_model_fits': 3, 'new_full_fits': 0, 'new_closed_loop_runs': 0, 'threshold_retuned': False,
            'retrieved_json_bytes': sum(path.stat().st_size for path in files) + (local_root/'fit_v1/result.json').stat().st_size},
        'collision_diagnostics': {key: total[key] for key in ('prefix_collision_events', 'window_collision_events')},
        'limitations': protocol['limitations']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    ppath, rpath = args.root/'protocol_v1.json', args.root/'fit_v1/result.json'
    result = summarize(json.loads(ppath.read_text()), json.loads(rpath.read_text()), args.root)
    result['refs'] = {'protocol': str(ppath.resolve()), 'fit_result': str(rpath.resolve())}
    result['accounting']['task_ids'] = [task['id'] for name in ('launch_v1.json', 'fit_launch_v1.json')
        for task in json.loads(json.loads((args.root/name).read_text())['scheduler_stdout'])['submitted']]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': result['status'], 'comparison': result['comparison']['comparisons'],
                      'accounting': result['accounting']}))


if __name__ == '__main__':
    main()
