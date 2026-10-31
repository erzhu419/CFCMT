"""Decompose the fixed A OOF policy loss using the S-allowed native labels."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from cologne_policy_loss_decomposition import decompose_policy_loss

PROTOCOL = 'tsc-v155h-cologne-policy-loss-decomposition-v1'


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL or protocol['threshold'] != 0
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']):
        raise ValueError('Use the original A policy at threshold0')
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, protocol['calibration_tooling_root'])
    from cologne_native_bank import assert_complete_b100
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import threshold_metrics
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset

    previous = json.loads(Path(protocol['reference_fit_result']).read_text())
    cached = json.loads(Path(protocol['reference_oof_source']).read_text())
    if (previous['protocol'] != 'tsc-v155a-cologne-raw-target-refit-v1'
            or previous['status'] != 'COMPLETE' or not all(previous['checks'].values())
            or previous['selected_groups'] != protocol['training_group_ids']
            or previous['model_path'] != protocol['reference_model_path']
            or previous['anchor_feature_names'] != protocol['original_feature_names']
            or previous['refs']['native_bank'] != protocol['native_bank']
            or any(previous[key] != protocol[key] for key in ('source_root', 'sumocfg', 'folds'))
            or previous['frozen_threshold'] != protocol['reference_frozen_threshold']
            or previous['frozen_threshold']['threshold'] != 0 or cached['folds'] != protocol['folds']):
        raise ValueError('The original A model, labels, features or folds changed')
    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, protocol['training_group_ids'])
    effective, mask = apply_no_stay_to_oof(cached['records'], bank, protocol['training_group_ids'])
    metrics = threshold_metrics(effective, 0.)
    accounting = proposal_accounting(effective, 0.)
    if (effective != cached['effective_records'] or mask != previous['mask']
            or metrics != previous['calibration']['selected'] or metrics != protocol['expected_original_metrics']
            or accounting != previous['proposal_accounting']):
        raise ValueError('All100 cached A decisions must reproduce before decomposition')
    audit = decompose_policy_loss(effective, bank, protocol['training_group_ids'], threshold=0.)
    if (audit['group_count'] != 100 or len(audit['rows']) != 100
            or [row['group_id'] for row in audit['rows']] != protocol['training_group_ids']
            or not all(audit['checks'].values())):
        raise ValueError('Retain all100 original groups with their S-allowed actions')
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE',
        **{key: protocol[key] for key in ('source_root', 'sumocfg', 'reference_model_path', 'training_group_ids',
                                        'folds', 'original_feature_names', 'threshold')},
        'metrics': metrics, 'mask': mask, 'proposal_accounting': accounting, 'audit': audit,
        'checks': {'original_a_decisions_exact': True, 'original_b100_bank_complete': True},
        'new_model_fits': 0, 'new_labels': 0, 'new_simulations': 0, 'threshold_retuned': False,
        'refs': {key: protocol[key] for key in ('reference_fit_result', 'reference_oof_source', 'native_bank')},
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started}
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'groups': audit['group_count'], 'elapsed_sec': result['elapsed_sec']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
