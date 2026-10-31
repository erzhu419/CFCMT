"""Inspect training-fold neighborhoods of the fixed A accepted OOF actions."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

from cologne_oof_neighborhood import audit_neighborhood

PROTOCOL = 'tsc-v155e-cologne-oof-neighborhood-v1'


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']
            or protocol['neighbors_per_query'] != 3):
        raise ValueError('Use the frozen three-neighbor A OOF diagnosis')
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, protocol['calibration_tooling_root'])
    from cologne_native_bank import assert_complete_b100
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import threshold_metrics
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    from cf_h2o.traffic_signal.action_ranker import CAUSAL_RIGID_RANKING_PARENTS
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg,
        read_network_right_of_way_context)

    fit = json.loads(Path(protocol['fit_result']).read_text())
    cached = json.loads(Path(protocol['oof_source']).read_text())
    frozen = json.loads(Path(protocol['frozen_threshold_path']).read_text())
    if (fit['protocol'] != 'tsc-v155a-cologne-raw-target-refit-v1' or fit['status'] != 'COMPLETE'
            or not all(fit['checks'].values()) or fit['selected_groups'] != protocol['training_group_ids']
            or fit['refs']['native_bank'] != protocol['native_bank']
            or any(fit[k] != protocol[k] for k in ('source_root', 'sumocfg', 'model_path', 'folds'))
            or fit['anchor_feature_names'] != protocol['feature_names']
            or cached['folds'] != protocol['folds']
            or frozen != protocol['frozen_threshold'] or frozen != fit['frozen_threshold']
            or frozen['threshold'] != 0 or frozen['mode'] != 'minimum_advantage'
            or frozen['score_units'] != 'raw_native_halted_per_lane_cost_difference'):
        raise ValueError('The original A model, B100, feature order or seed folds changed')
    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, protocol['training_group_ids'])
    effective, mask = apply_no_stay_to_oof(cached['records'], bank, protocol['training_group_ids'])
    selected = threshold_metrics(effective, frozen['threshold'])
    accounting = proposal_accounting(effective, frozen['threshold'])
    if (effective != cached['effective_records'] or mask != fit['mask']
            or selected != protocol['expected_selected_policy'] or selected != fit['calibration']['selected']
            or accounting != protocol['expected_proposal_accounting'] or accounting != fit['proposal_accounting']):
        raise ValueError('All100 A decisions, labels and selected policy costs must reproduce')
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    absolute = augment_dataset_with_right_of_way_context(bank, {'cologne1': context})
    contrast = build_action_contrast_dataset(absolute, reference_policy='phase_pressure',
                                            contrast_features=CONTRAST_FEATURES_V3)
    if [name for name in CAUSAL_RIGID_RANKING_PARENTS if name in contrast.feature_names] != protocol['feature_names']:
        raise ValueError('Use the actual A model feature selection and order')
    audit = audit_neighborhood(contrast, effective, protocol['feature_names'], protocol['folds'],
                               threshold=frozen['threshold'])
    expected = {row['group_id']: row for row in protocol['expected_queries']}
    if (audit['query_count'] != 48 or len(audit['queries']) != 48
            or {row['group_id'] for row in audit['queries']} != set(expected)
            or not all(audit['checks'].values())):
        raise ValueError('Retain exactly the48 accepted A queries and their training-fold neighbors')
    for row in audit['queries']:
        original = expected[row['group_id']]
        if (any(row[k] != original[k] for k in ('seed', 'selected_index', 'reference_index'))
                or any(abs(row[k] - original[k]) > 1e-12 for k in ('predicted_gain', 'actual_gain'))):
            raise ValueError('An accepted A query action, prediction or raw label changed')
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE',
        **{k: protocol[k] for k in ('source_root', 'sumocfg', 'model_path', 'training_group_ids', 'feature_names', 'folds')},
        'frozen_threshold': frozen, 'selected_policy': selected, 'proposal_accounting': accounting,
        'checks': {k: True for k in ('original_a_effective_records_exact',
            'native_bank_labels_and_candidate_order_exact', 'original_a_selection_cost_and_counts_exact',
            'model_feature_order_exact', 'accepted_query_roster_exact')},
        'audit': audit, 'new_model_fits': 0, 'new_labels': 0, 'new_simulations': 0,
        'threshold_retuned': False,
        'refs': {k: protocol[k] for k in ('fit_result', 'oof_source', 'native_bank')},
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started}
    output.mkdir(parents=True)
    (output / 'result.json').write_text(json.dumps(result, separators=(',', ':'), allow_nan=False) + '\n')
    print(json.dumps({'protocol': PROTOCOL, 'status': result['status'], 'queries': audit['query_count'],
                      'elapsed_sec': result['elapsed_sec']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
