"""Test the frozen thirteen added features using only three original seed folds."""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from cologne_raw_target_rigid import fit_raw_target_anchor, SCORE_UNITS

PROTOCOL = 'tsc-v155g-cologne-feature-block-oof-v1'


def write_json(path, data):
    with path.open('x') as file:
        json.dump(data, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')


def deployed_index(row):
    return (row['selected_index'] if row['different_action'] and row['predicted_advantage'] > 1e-12
            else row['reference_index'])


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL or protocol['threshold'] != 0
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']
            or len(protocol['full_feature_names']) != 42
            or protocol['full_feature_names'] != protocol['original_feature_names'] + protocol['added_feature_names']):
        raise ValueError('Use the frozen42-column block and threshold0')
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, protocol['calibration_tooling_root'])
    from cologne_native_bank import assert_complete_b100
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import score_records, threshold_metrics
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
    from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg,
        read_network_right_of_way_context)

    previous = json.loads(Path(protocol['reference_fit_result']).read_text())
    cached = json.loads(Path(protocol['reference_oof_source']).read_text())
    previous_details = json.loads(Path(protocol['reference_fit_details']).read_text())
    groups = protocol['training_group_ids']
    if (previous['protocol'] != 'tsc-v155a-cologne-raw-target-refit-v1' or previous['status'] != 'COMPLETE'
            or not all(previous['checks'].values()) or previous['selected_groups'] != groups
            or previous['refs']['native_bank'] != protocol['native_bank']
            or previous['source_root'] != protocol['source_root'] or previous['sumocfg'] != protocol['sumocfg']
            or previous['model_path'] != protocol['reference_model_path']
            or previous['anchor_feature_names'] != protocol['original_feature_names']
            or previous['frozen_threshold'] != protocol['reference_frozen_threshold']
            or previous['frozen_threshold']['threshold'] != 0 or previous['score_units'] != SCORE_UNITS
            or previous['folds'] != protocol['folds'] or cached['folds'] != protocol['folds']):
        raise ValueError('The original A model, native bank or seed folds changed')
    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, groups)
    oldeffective, oldmask = apply_no_stay_to_oof(cached['records'], bank, groups)
    oldmetrics = threshold_metrics(oldeffective, 0.)
    if (oldeffective != cached['effective_records'] or oldmask != previous['mask']
            or oldmetrics != previous['calibration']['selected']
            or oldmetrics != protocol['expected_original_metrics']):
        raise ValueError('The original100 A decisions and costs must reproduce before fitting')
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    training = augment_dataset_with_right_of_way_context(bank, {'cologne1': context})
    output.mkdir(parents=True)
    records, diagnostics, details = [], [], {}
    for spec in protocol['folds']:
        seed = spec['heldout_seed']
        held = {group for group in groups if group.startswith(f'cologne1:seed{seed}:')}
        fitting = set(groups) - held
        if len(held) != spec['heldout_group_count'] or len(fitting) != spec['fit_group_count']:
            raise ValueError('Use the original training-seed partition')
        absolute = _group_subset_v3(training, selected_groups=fitting)
        contrast = build_action_contrast_dataset(absolute, reference_policy='phase_pressure',
                                                contrast_features=CONTRAST_FEATURES_V3)
        anchor, diag = fit_raw_target_anchor(contrast, feature_names=protocol['full_feature_names'])
        tag = f'oof_{seed}'
        original = previous_details[tag]
        config = protocol['fit_contract']
        if (diag['feature_names'] != protocol['full_feature_names'] or diag['feature_count'] != 42
                or diag['training_rows'] != 7 * len(fitting) or diag['rows'] != 8 * len(fitting)
                or diag['actual_group_count'] != len(fitting)
                or diag['sample_weight_source_protocol'] != 'group_action_range_v1'
                or diag['raw_target_clipped_count'] != 0 or diag['score_units'] != SCORE_UNITS
                or original['groups'] != sorted(fitting)
                or diag['server_only_fit_row_indices'] != original['server_only_fit_row_indices']
                or not np.array_equal(diag['server_only_sample_weights'], original['server_only_sample_weights'])
                or any(getattr(anchor.config, key) != config[key] for key in (
                    'learning_rate', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'l2_regularization',
                    'random_state', 'target_clip', 'candidate_only', 'balance_candidate_signs'))
                or anchor.model is None or anchor.model.early_stopping != config['early_stopping']):
            raise ValueError('Only the feature columns may change; preserve exact A fold weights and HGB settings')
        # The inactive slot aliases this fold's fitted anchor, adding no model fit or labels.
        model = FrozenAnchoredBlendModel(anchor_model=anchor, correction_model=anchor,
            candidate='constant_alpha_0', anchor_objective_mode='control_only', correction_objective_mode='control_only')
        scores = model.predict(contrast)['control_cost']['mean']
        reference_rows = np.unique(np.asarray(contrast.metadata['reference_rows'], dtype=int))
        if np.any(scores[reference_rows] != 0.):
            raise ValueError('PP-reference action advantages must remain zero')
        fold_records = score_records(model, _group_subset_v3(training, selected_groups=held))
        if {record['group_id'] for record in fold_records} != held:
            raise ValueError('OOF scores must cover only the heldout seed')
        records.extend(fold_records)
        details[tag] = {'groups': sorted(fitting),
            **{key: diag.pop(key) for key in list(diag) if key.startswith('server_only_')}}
        diagnostics.append({'heldout_seed': seed, 'weights_exactly_match_a': True, 'diagnostics': diag})
        print(json.dumps({'heldout_seed': seed, 'status': 'FITTED', 'heldout_groups': len(held)}), flush=True)
    effective, mask = apply_no_stay_to_oof(records, bank, groups)
    metrics = threshold_metrics(effective, 0.)
    ppmetrics = threshold_metrics(effective, None)
    old_by_group = {record['group_id']: record for record in oldeffective}
    matched = []
    for new in effective:
        old = old_by_group[new['group_id']]
        if any(old[key] != new[key] for key in ('candidate_states', 'reference_index', 'label_costs')):
            raise ValueError('A and G comparisons must use the same candidate order, PP action and native labels')
        matched.append({'group_id': new['group_id'], 'seed': new['seed'],
            'reference_index': new['reference_index'], 'label_costs': new['label_costs'],
            'old_effective_index': old['selected_index'], 'new_effective_index': new['selected_index'],
            'old_deployed_index': deployed_index(old), 'new_deployed_index': deployed_index(new),
            'old_raw_predicted_advantage': old['predicted_advantage'],
            'new_raw_predicted_advantage': new['predicted_advantage']})
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE',
        **{key: protocol[key] for key in ('source_root', 'sumocfg', 'reference_model_path', 'training_group_ids',
                                        'folds', 'original_feature_names', 'added_feature_names', 'full_feature_names', 'threshold')},
        'score_units': SCORE_UNITS, 'fit_diagnostics': diagnostics,
        'metrics': {'model': metrics, 'original_a': oldmetrics, 'phase_pressure': ppmetrics},
        'matched_actions': matched, 'proposal_accounting': proposal_accounting(effective, 0.), 'mask': mask,
        'checks': {key: True for key in ('original_a_baseline_exact', 'training_roster_exact', 'feature_order_exact',
            'a_fold_weights_exact', 'hgb_settings_unchanged', 'raw_targets_unchanged', 'reference_predictions_zero')},
        'new_model_fits': 3, 'oof_fold_fits': 3, 'full_model_fits': 0, 'inactive_correction_fits': 0,
        'new_labels': 0, 'new_simulations': 0, 'threshold_retuned': False,
        'refs': {key: protocol[key] for key in ('reference_fit_result', 'reference_oof_source', 'reference_fit_details', 'native_bank')},
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started}
    write_json(output / 'fit_details_server_only.json', details)
    write_json(output / 'oof_server_only.json', {'records': records, 'effective_records': effective, 'folds': protocol['folds']})
    write_json(output / 'result.json', result)
    print(json.dumps({'status': 'COMPLETE', 'metrics': metrics, 'elapsed_sec': result['elapsed_sec']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
