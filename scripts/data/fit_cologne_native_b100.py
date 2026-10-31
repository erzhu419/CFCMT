"""Refit the complete native B100 and calibrate the unchanged seed-OOF rule."""

import argparse
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

from cologne_native_bank import assemble_native_bank, assert_complete_b100
from calibrate_cologne_rigid_threshold import (
    PROTOCOL as THRESHOLD_PROTOCOL, score_records, select_threshold, threshold_metrics)


PROTOCOL = 'tsc-v154p-cologne-native-b100-v1'
SEEDS = (2027, 3037, 4047)


def write_json(path, value):
    with path.open('x') as file:
        json.dump(value, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')


def collection_gate(protocol, results_root):
    """Read every seed result before admitting any bank to model fitting."""
    expected = protocol['training_group_ids']
    cells, failures, retained = [], [], []
    for seed in SEEDS:
        path = results_root / f'collection/seed_{seed}/result.json'
        wanted = [g for g in expected if g.startswith(f'cologne1:seed{seed}:')]
        try:
            cell = json.loads(path.read_text())
        except (OSError, ValueError) as error:
            failures.append({'seed': seed, 'result': str(path), 'error': str(error),
                             'missing_groups': wanted})
            continue
        cells.append(cell)
        retained.extend(cell.get('retained_groups', []))
        invalid = (cell.get('protocol') != PROTOCOL or cell.get('status') != 'PASS'
            or cell.get('seed') != seed or cell.get('source_root') != protocol['source_root']
            or cell.get('sumocfg') != protocol['sumocfg']
            or cell.get('requested_groups') != wanted or cell.get('retained_groups') != wanted
            or cell.get('rows') != 8 * len(wanted) or cell.get('unsafe_groups')
            or cell.get('invalid_groups') or cell.get('unsafe_native_branches') != 0)
        if invalid:
            failures.append({'seed': seed, 'result': str(path), 'status': cell.get('status'),
                'unsafe_groups': cell.get('unsafe_groups', []),
                'invalid_groups': cell.get('invalid_groups', []),
                'missing_groups': sorted(set(wanted) - set(cell.get('retained_groups', []))),
                'failed_actions': [g for g in cell.get('groups', []) if g.get('status') != 'PASS']})
    if len(expected) != 100 or len(set(expected)) != 100 or retained != expected:
        failures.append({'error': 'Collected roster is not the unchanged complete B100',
            'missing_groups': sorted(set(expected) - set(retained)),
            'unexpected_groups': sorted(set(retained) - set(expected))})
    return cells, failures


def label_comparison(protocol, native, load_dataset):
    """Compare matched original labels; no other groups enter this diagnostic."""
    from cf_h2o.traffic_signal.action_contrast import rule_reference_indices
    references = rule_reference_indices(native, policy='phase_pressure')
    groups = np.asarray(native.metadata['action_group_ids'])
    states = np.asarray(native.metadata['candidate_states'])
    banks, differences, flips, best_changed = {}, [], 0, 0
    for spec in protocol['groups']:
        path, group = spec['bank_path'], spec['group_id']
        if path not in banks:
            banks[path] = load_dataset(Path(path))
        old = banks[path]
        before_rows = np.flatnonzero(np.asarray(old.metadata['action_group_ids']) == group)
        after_rows = np.flatnonzero(groups == group)
        if not np.array_equal(np.asarray(old.metadata['candidate_states'])[before_rows], states[after_rows]):
            raise ValueError(f'{group}: original label candidate order changed')
        before = np.asarray(old.targets['interval_cost'])[before_rows]
        after = np.asarray(native.targets['interval_cost'])[after_rows]
        reference = int(np.flatnonzero(after_rows == references[group])[0])
        old_gain, new_gain = before[reference] - before, after[reference] - after
        flips += int(np.sum(((old_gain > 1e-12) & (new_gain < -1e-12))
                           | ((old_gain < -1e-12) & (new_gain > 1e-12))))
        best_changed += int(not np.array_equal(before <= before.min() + 1e-12,
                                                after <= after.min() + 1e-12))
        differences.extend((after - before).tolist())
    values = np.asarray(differences)
    return {'groups': 100, 'rows': len(values), 'non_reference_rows': 700,
        'strict_advantage_sign_flips': flips, 'optimal_action_set_changed_groups': best_changed,
        'mean_native_minus_original': float(np.mean(values)),
        'mean_absolute_difference': float(np.mean(np.abs(values))),
        'max_absolute_difference': float(np.max(np.abs(values)))}


def run(protocol, results_root, output):
    started = time.monotonic()
    if protocol['protocol'] != PROTOCOL or tuple(protocol['training_seeds']) != SEEDS:
        raise ValueError('Use the frozen V154P native B100 protocol')
    if os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']:
        raise ValueError('Use the frozen V154J model implementation')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    cells, failures = collection_gate(protocol, results_root)
    base = {'protocol': PROTOCOL, 'source_root': protocol['source_root'],
        'sumocfg': protocol['sumocfg'], 'results_root': str(results_root),
        'old_model_and_threshold_unchanged': True, 'new_model_fits': 0}
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset, save_mechanism_dataset
    if not failures:
        try:
            old_fit = json.loads(Path(protocol['fit_result']).read_text())
            if (old_fit['status'] != 'PASS' or old_fit['selected_groups'] != protocol['training_group_ids']
                    or old_fit['frozen_candidate'] != 'constant_alpha_0'
                    or json.loads(Path(protocol['frozen_threshold_path']).read_text()) != protocol['frozen_threshold']):
                raise ValueError('Original B100/model/threshold binding changed')
            bank = assemble_native_bank([load_mechanism_dataset(Path(c['bank'])) for c in cells],
                                        protocol['training_group_ids'])
            assert_complete_b100(bank, protocol['training_group_ids'])
            comparison = label_comparison(protocol, bank, load_mechanism_dataset)
        except (OSError, ValueError, KeyError) as error:
            failures.append({'error': str(error)})
    if failures:
        result = {**base, 'status': 'FAIL_B100', 'failures': failures,
                  'elapsed_sec': time.monotonic() - started}
        write_json(output / 'result.json', result)
        return result
    from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import _target_model
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
    from cf_h2o.traffic_signal.fraction_target_rigid import MODEL_PROTOCOL
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    from cf_h2o.traffic_signal.right_of_way_context import (augment_dataset_with_right_of_way_context,
        net_file_from_sumocfg, read_network_right_of_way_context)
    groups = protocol['training_group_ids']
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    training = augment_dataset_with_right_of_way_context(bank, {'cologne1': context})
    full_model = _target_model(training, group_ids=groups, candidate='constant_alpha_0', target_domain='cologne')
    if list(full_model.anchor_model.feature_names) != old_fit['anchor_feature_names']:
        raise ValueError('Native refit anchor features differ from the original B100 model')
    records, folds = [], []
    for seed in SEEDS:
        held = {g for g in groups if g.startswith(f'cologne1:seed{seed}:')}
        fitting = set(groups) - held
        if len(held) != protocol['fitting']['heldout_group_counts'][str(seed)]:
            raise ValueError('The frozen seed fold changed')
        model = _target_model(training, group_ids=sorted(fitting), candidate='constant_alpha_0', target_domain='cologne')
        heldout = _group_subset_v3(training, selected_groups=held)
        records.extend(score_records(model, heldout))
        folds.append({'heldout_seed': seed, 'fit_group_count': len(fitting), 'heldout_group_count': len(held)})
    if len(records) != 100 or {r['group_id'] for r in records} != set(groups):
        raise ValueError('OOF records must cover the original B100 exactly once')
    calibration = select_threshold(records)
    selected = calibration['selected']
    model_path = output / 'model.pkl'
    payload = {'protocol': MODEL_PROTOCOL, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'city': 'cologne', 'target_budget': 100, 'selected_groups': groups, 'fit_protocol': PROTOCOL,
        'training_scenarios': ['cologne1'], 'frozen_candidate': 'constant_alpha_0', 'rigid_model': full_model}
    with model_path.open('xb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    save_mechanism_dataset(output / 'bank.npz', bank)
    frozen = {'protocol': THRESHOLD_PROTOCOL, 'calibration_protocol': PROTOCOL,
        'source_root': protocol['source_root'], 'fit_result': str(output / 'result.json'),
        'model_path': str(model_path), 'threshold': selected['threshold'], 'mode': selected['mode'],
        'comparison': 'predicted_advantage > threshold + 1e-12',
        'selection_data': 'Exactly 100 training-seed OOF predictions using native B100 labels',
        'calibration_mean_cost': selected['mean_cost']}
    write_json(output / 'frozen_threshold.json', frozen)
    write_json(output / 'oof_server_only.json', {'folds': folds, 'records': records, 'calibration': calibration})
    result = {**base, 'status': 'PASS', 'new_model_fits': 4, 'full_model_fits': 1, 'oof_fold_fits': 3,
        'training_group_count': 100, 'training_row_count': 800, 'selected_groups': groups,
        'summed_fold_training_groups': sum(f['fit_group_count'] for f in folds), 'folds': folds,
        'model_path': str(model_path), 'bank': str(output / 'bank.npz'), 'frozen_threshold': frozen,
        'anchor_feature_names': list(full_model.anchor_model.feature_names),
        'calibration': {k: v for k, v in calibration.items() if k != 'curve'},
        'old_threshold_on_new_oof': threshold_metrics(records, protocol['frozen_threshold']['threshold']),
        'native_vs_original_labels': comparison, 'new_closed_loop_evaluations': 0,
        'limitations': ['Selected OOF cost is calibration performance, not independent validation.',
            'Transfer of the fold-calibrated threshold to the full B100 model awaits closed-loop evaluation.'],
        'elapsed_sec': time.monotonic() - started}
    write_json(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'results-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.protocol.read_text()), args.results_root, args.output)
    print(json.dumps({k: result[k] for k in ('protocol', 'status', 'new_model_fits', 'elapsed_sec')}))
    if result['status'] != 'PASS':
        raise SystemExit(2)
