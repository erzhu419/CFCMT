"""Calibrate the fixed no-stay policy from cached native B100 OOF scores."""

import argparse
import json
import os
from pathlib import Path
import time

from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset

if __package__:
    from .cologne_native_bank import assert_complete_b100
    from .cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from .calibrate_cologne_rigid_threshold import (
        PROTOCOL as THRESHOLD_PROTOCOL, select_threshold, threshold_metrics)
else:
    from cologne_native_bank import assert_complete_b100
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import (
        PROTOCOL as THRESHOLD_PROTOCOL, select_threshold, threshold_metrics)


PROTOCOL = 'tsc-v154t-cologne-no-stay-oof-threshold-v1'


def write_json(path, value):
    with path.open('x') as file:
        json.dump(value, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL
            or protocol['training_seeds'] != [2027, 3037, 4047]
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']):
        raise ValueError('Use the frozen V154T protocol and source implementation')
    if output.exists():
        raise FileExistsError(output)
    fit = json.loads(Path(protocol['fit_result']).read_text())
    original = json.loads(Path(protocol['original_frozen_threshold_path']).read_text())
    groups = protocol['training_group_ids']
    if (fit['status'] != 'PASS' or fit['protocol'] != 'tsc-v154p-cologne-native-b100-v1'
            or fit['source_root'] != protocol['source_root']
            or fit['sumocfg'] != protocol['sumocfg']
            or fit['selected_groups'] != groups or fit['training_group_count'] != 100
            or fit['training_row_count'] != 800 or fit['model_path'] != protocol['model_path']
            or fit['bank'] != protocol['native_bank']
            or fit['folds'] != protocol['folds']
            or original != protocol['original_frozen_threshold']
            or original != fit['frozen_threshold'] or original['threshold'] != 0.0):
        raise ValueError('Original native B100 model, labels, folds or threshold binding changed')
    cached = json.loads(Path(protocol['oof_source']).read_text())
    if cached['folds'] != protocol['folds']:
        raise ValueError('Cached predictions do not use the original seed folds')
    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, groups)
    effective, mask = apply_no_stay_to_oof(cached['records'], bank, groups)
    actual_folds = [{'heldout_seed': seed,
        'fit_group_count': 100 - sum(row['seed'] == seed for row in effective),
        'heldout_group_count': sum(row['seed'] == seed for row in effective)}
        for seed in protocol['training_seeds']]
    if actual_folds != protocol['folds']:
        raise ValueError('OOF group seed counts differ from the fixed folds')
    calibration = select_threshold(effective)
    selected = calibration['selected']
    accounting = {name: proposal_accounting(effective, threshold) for name, threshold in
        [('selected', selected['threshold']), ('no_stay_threshold_zero', 0.0), ('phase_pressure', None)]}
    if accounting['selected']['accepted_overrides'] != selected['accepted_overrides']:
        raise ValueError('Deployment accounting differs from the calibrated policy')
    frozen = {'protocol': THRESHOLD_PROTOCOL, 'calibration_protocol': PROTOCOL,
        'source_root': protocol['source_root'], 'fit_result': protocol['fit_result'],
        'calibration_result': str(output / 'result.json'), 'model_path': protocol['model_path'],
        'ablation_originator_path': protocol['ablation_originator_path'],
        'threshold': selected['threshold'], 'mode': selected['mode'],
        'comparison': 'predicted_advantage > threshold + 1e-12',
        'selection_data': 'Original 100 native B100 seed-OOF records with fixed post-selection stay veto',
        'calibration_mean_cost': selected['mean_cost']}
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE',
        'source_root': protocol['source_root'], 'sumocfg': protocol['sumocfg'],
        'model_path': protocol['model_path'], 'fit_result': protocol['fit_result'],
        'oof_source': protocol['oof_source'], 'native_bank': protocol['native_bank'],
        'frozen_threshold': frozen, 'original_threshold': original['threshold'],
        'model_and_original_threshold_unchanged': True,
        'new_model_fits': 0, 'new_labels': 0, 'new_simulations': 0,
        'training_group_count': 100, 'training_row_count': 800, 'folds': actual_folds,
        'mask': mask, 'accounting': accounting,
        'calibration': {'selected': selected, 'phase_pressure': calibration['phase_pressure'],
            'no_stay_threshold_zero': calibration['ungated_rigid'],
            'original_rigid_threshold_zero': threshold_metrics(cached['records'], 0.0),
            'candidate_count': calibration['candidate_count']},
        'selected_improvement_percent_vs_pp': 100 * (1 - selected['mean_cost']
            / calibration['phase_pressure']['mean_cost']),
        'selected_improvement_percent_vs_no_stay_zero': 100 * (1 - selected['mean_cost']
            / calibration['ungated_rigid']['mean_cost']),
        'verdict': ('PP_ONLY_SELECTED' if selected['threshold'] is None else
                    'UNCHANGED_ZERO_SELECTED' if selected['threshold'] == 0 else
                    'POSITIVE_SWITCH_THRESHOLD_SELECTED'),
        'selection_excludes': protocol['forbidden_selection_inputs'],
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started}
    output.mkdir(parents=True)
    write_json(output / 'frozen_threshold.json', frozen)
    write_json(output / 'calibration_server_only.json', {'folds': actual_folds,
        'effective_records': effective, 'curve': calibration['curve']})
    write_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.protocol.read_text()), args.output)
    print(json.dumps({key: result[key] for key in
        ('status', 'verdict', 'frozen_threshold', 'mask', 'elapsed_sec')}))


if __name__ == '__main__':
    main()
