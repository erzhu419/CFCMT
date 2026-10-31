"""Compare S action filtering before ranking with the frozen A stay veto."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

PROTOCOL = 'tsc-v155i-cologne-pre-mask-ranking-v1'
BOUND = ('source_root', 'sumocfg', 'reference_model_path', 'training_group_ids',
         'folds', 'original_feature_names', 'threshold')


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL or protocol['threshold'] != 0
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']):
        raise ValueError('Use frozen A scores with threshold0')
    if output.exists():
        raise FileExistsError(output)
    tooling = str(Path(__file__).resolve().parent)
    sys.path.insert(0, protocol['calibration_tooling_root'])
    sys.path.insert(0, tooling)
    from cologne_native_bank import assert_complete_b100
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import threshold_metrics
    from cologne_pre_mask_selection import pre_mask_records
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
            or cached['folds'] != protocol['folds']):
        raise ValueError('The original A inputs or OOF folds changed')
    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, protocol['training_group_ids'])
    effective, mask = apply_no_stay_to_oof(cached['records'], bank, protocol['training_group_ids'])
    baseline = threshold_metrics(effective, 0.)
    if (effective != cached['effective_records'] or mask != previous['mask']
            or baseline != previous['calibration']['selected']
            or baseline != protocol['expected_original_metrics']
            or proposal_accounting(effective, 0.) != previous['proposal_accounting']):
        raise ValueError('Original A decisions must reproduce before the comparison')
    audit = pre_mask_records(cached['records'], bank, protocol['training_group_ids'], threshold=0.)
    metrics = threshold_metrics(audit.pop('records'), 0.)
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE',
        **{key: protocol[key] for key in BOUND},
        'original_metrics': baseline, 'metrics': metrics, 'original_mask': mask, 'audit': audit,
        'checks': {'original_a_decisions_exact': True, 'original_b100_bank_complete': True},
        'new_model_fits': 0, 'new_labels': 0, 'new_simulations': 0, 'threshold_retuned': False,
        'refs': {key: protocol[key] for key in ('reference_fit_result', 'reference_oof_source', 'native_bank')},
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started}
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'status': 'COMPLETE', 'groups': len(audit['rows']),
                      'mean_cost': metrics['mean_cost'], 'elapsed_sec': result['elapsed_sec']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
