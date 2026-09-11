"""Audit the fixed T/S accepted OOF actions without fitting or simulation."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
from cologne_oof_normalization_audit import audit_records


PROTOCOL = 'tsc-v154z-cologne-oof-normalization-audit-v1'


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']):
        raise ValueError('Use the frozen V154Z protocol and original model implementation')
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, protocol['calibration_tooling_root'])
    from cologne_native_bank import assert_complete_b100
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import threshold_metrics

    original = json.loads(Path(protocol['original_calibration_protocol']).read_text())
    calibration = json.loads(Path(protocol['original_calibration_result']).read_text())
    frozen = json.loads(Path(protocol['frozen_threshold_path']).read_text())
    fit = json.loads(Path(protocol['fit_result']).read_text())
    cached = json.loads(Path(protocol['oof_source']).read_text())
    masked_cache = json.loads(Path(protocol['original_calibration_records']).read_text())
    keys = ('source_root', 'sumocfg', 'model_path', 'fit_result', 'oof_source',
            'native_bank', 'training_group_ids', 'training_seeds', 'folds',
            'ablation_originator_path')
    if (any(protocol[k] != original[k] for k in keys)
            or original['protocol'] != 'tsc-v154t-cologne-no-stay-oof-threshold-v1'
            or calibration['status'] != 'COMPLETE'
            or calibration['protocol'] != original['protocol']
            or frozen != protocol['frozen_threshold'] or frozen != calibration['frozen_threshold']
            or frozen['threshold'] != 0.20790450432178334
            or calibration['calibration']['selected'] != protocol['expected_selected_calibration']
            or calibration['accounting']['selected'] != protocol['expected_proposal_accounting']
            or fit['status'] != 'PASS' or fit['protocol'] != 'tsc-v154p-cologne-native-b100-v1'
            or fit['selected_groups'] != protocol['training_group_ids']
            or fit['training_group_count'] != 100 or fit['training_row_count'] != 800
            or fit['model_path'] != protocol['model_path'] or fit['bank'] != protocol['native_bank']
            or fit['source_root'] != protocol['source_root'] or fit['sumocfg'] != protocol['sumocfg']
            or any(x['folds'] != protocol['folds'] for x in (fit, cached, masked_cache))):
        raise ValueError('Original B100 labels, seed folds, scores or T/S selection changed')
    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, protocol['training_group_ids'])
    effective, mask = apply_no_stay_to_oof(cached['records'], bank, protocol['training_group_ids'])
    threshold = frozen['threshold']
    selected = threshold_metrics(effective, threshold)
    accounting = proposal_accounting(effective, threshold)
    if (effective != masked_cache['effective_records'] or mask != calibration['mask']
            or selected != protocol['expected_selected_calibration']
            or accounting != protocol['expected_proposal_accounting']
            or selected['accepted_overrides'] != protocol['expected_accepted_count']
            or any(selected[k] != v for k, v in protocol['expected_outcomes'].items())):
        raise ValueError('The complete original 44-action policy must reproduce before diagnosis')
    diagnostic = audit_records(effective, threshold)
    if (diagnostic['accepted_count'] != 44
            or diagnostic['outcome_counts'] != {'beneficial': 23, 'harmful': 15, 'tie': 6}
            or any(abs(diagnostic['raw_policy'][k] - selected[k]) > 1e-12 for k in
                   ('mean_cost', 'mean_phase_pressure_cost', 'mean_cost_minus_phase_pressure'))):
        raise ValueError('Audit must retain the original 44 actions and their raw policy cost')
    result = {
        'protocol': PROTOCOL, 'status': 'COMPLETE', 'source_root': protocol['source_root'],
        'sumocfg': protocol['sumocfg'], 'model_path': protocol['model_path'],
        'frozen_threshold': frozen, 'training_group_count': 100, 'training_row_count': 800,
        'folds': protocol['folds'], 'normalization': protocol['normalization'],
        'checks': {'original_t_effective_records_exact': True,
                   'native_bank_labels_and_candidate_order_exact': True,
                   'original_t_selection_cost_and_counts_exact': True,
                   'diagnostic_raw_policy_cost_matches_t': True},
        'selected_policy': selected, 'proposal_accounting': accounting, 'audit': diagnostic,
        'new_model_fits': 0, 'new_labels': 0, 'new_simulations': 0,
        'threshold_retuned': False, 'spacing_retuned': False,
        'refs': {k: protocol[k] for k in ('original_calibration_result',
                'original_calibration_records', 'oof_source', 'native_bank', 'fit_result')},
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started,
    }
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('protocol', 'status', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
