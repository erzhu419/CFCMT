"""Refit native B100 on raw action deltas with the frozen A or D weighting."""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from cologne_raw_target_rigid import fit_raw_target_anchor, SCORE_UNITS


PROTOCOL = 'tsc-v155a-cologne-raw-target-refit-v1'
BALANCED_PROTOCOL = 'tsc-v155d-cologne-group-balanced-raw-refit-v1'
SEEDS = (2027, 3037, 4047)


def write_json(path, value):
    with path.open('x') as file:
        json.dump(value, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')


def deployed_index(row, threshold):
    accepted = (threshold is not None and row['different_action']
                and row['predicted_advantage'] > threshold + 1e-12)
    return row['selected_index'] if accepted else row['reference_index']


def run(protocol, output):
    started = time.monotonic()
    balanced = protocol['protocol'] == BALANCED_PROTOCOL
    weight_protocol = 'domain_group_balanced_v1' if balanced else 'group_action_range_v1'
    if (protocol['protocol'] not in (PROTOCOL, BALANCED_PROTOCOL) or tuple(protocol['training_seeds']) != SEEDS
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']):
        raise ValueError('Use the fixed raw-target ablation and original implementation')
    if output.exists():
        raise FileExistsError(output)
    sys.path.insert(0, protocol['calibration_tooling_root'])
    from cologne_native_bank import assert_complete_b100
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import score_records, select_threshold, threshold_metrics
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
    from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
    from cf_h2o.traffic_signal.fraction_target_rigid import MODEL_PROTOCOL
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg, read_network_right_of_way_context)

    oldfit = json.loads(Path(protocol['fit_result']).read_text())
    oldcal = (oldfit if protocol['original_calibration_result'] == protocol['fit_result'] else
              json.loads(Path(protocol['original_calibration_result']).read_text()))
    oldcached = json.loads(Path(protocol['oof_source']).read_text())
    oldmasked = (oldcached if protocol['original_calibration_records'] == protocol['oof_source'] else
                 json.loads(Path(protocol['original_calibration_records']).read_text()))
    oldfrozen = json.loads(Path(protocol['old_frozen_threshold_path']).read_text())
    groups = protocol['training_group_ids']
    old_bank = oldfit['refs']['native_bank'] if balanced else oldfit['bank']
    if (oldfit['status'] != ('COMPLETE' if balanced else 'PASS')
            or oldfit['protocol'] != (PROTOCOL if balanced else 'tsc-v154p-cologne-native-b100-v1')
            or oldfit['selected_groups'] != groups or old_bank != protocol['native_bank']
            or oldfit['source_root'] != protocol['source_root'] or oldfit['sumocfg'] != protocol['sumocfg']
            or oldfit['model_path'] != protocol['old_model_path']
            or oldfit['anchor_feature_names'] != protocol['expected_anchor_feature_names']
            or oldcal['status'] != 'COMPLETE'
            or oldcal['protocol'] != (PROTOCOL if balanced else 'tsc-v154t-cologne-no-stay-oof-threshold-v1')
            or oldfrozen != protocol['old_frozen_threshold'] or oldfrozen != oldcal['frozen_threshold']
            or oldcal['calibration']['selected'] != protocol['expected_original_selected']
            or any(x['folds'] != protocol['folds'] for x in (oldfit, oldcal, oldcached, oldmasked))):
        raise ValueError('The frozen B100, model family, folds or baseline calibration changed')
    if balanced and (oldfit['score_units'] != SCORE_UNITS
                     or oldfit['fit_diagnostics']['full']['sample_weight_source_protocol'] != 'group_action_range_v1'
                     or oldfrozen['threshold'] != 0 or not all(oldfit['checks'].values())):
        raise ValueError('Use the completed A raw-target model with its inherited weights as the D control')
    bank = load_mechanism_dataset(Path(protocol['native_bank']))
    assert_complete_b100(bank, groups)
    oldeffective, _ = apply_no_stay_to_oof(oldcached['records'], bank, groups)
    if (oldeffective != oldmasked['effective_records']
            or threshold_metrics(oldeffective, oldfrozen['threshold']) != protocol['expected_original_selected']):
        raise ValueError('Baseline actions and costs must reproduce before fitting')
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    training = augment_dataset_with_right_of_way_context(bank, {'cologne1': context})
    full_contrast = build_action_contrast_dataset(training, reference_policy='phase_pressure',
                                                  contrast_features=CONTRAST_FEATURES_V3)
    output.mkdir(parents=True)
    detailed_fits, fold_diagnostics, records = {}, [], []

    def fit_subset(absolute, tag):
        contrast = build_action_contrast_dataset(absolute, reference_policy='phase_pressure',
                                                contrast_features=CONTRAST_FEATURES_V3)
        anchor, diag = fit_raw_target_anchor(contrast, sample_weight_protocol=weight_protocol)
        config = protocol['fit_contract']
        if (diag['feature_names'] != protocol['expected_anchor_feature_names']
                or diag['training_rows'] != 7 * diag['actual_group_count']
                or diag['rows'] != 8 * diag['actual_group_count']
                or diag['sample_weight_source_protocol'] != weight_protocol
                or diag['raw_target_clipped_count'] != 0
                or diag['raw_target_max_abs'] > config['maximum_absolute_raw_delta'] + 1e-12
                or diag['score_units'] != config['score_units']
                or any(getattr(anchor.config, k) != config[k] for k in (
                    'learning_rate', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf',
                    'l2_regularization', 'random_state', 'target_clip', 'candidate_only',
                    'balance_candidate_signs'))):
            raise ValueError('Keep the fixed raw targets, feature order, weight protocol and HGB settings')
        if balanced and (not np.allclose(diag['server_only_sample_weights'], 1., rtol=0., atol=1e-12)
                         or diag['sign_balance']['enabled']):
            raise ValueError('Every D fit must use equal weights without sign rebalancing')
        detailed_fits[tag] = {'groups': sorted(set(absolute.metadata['action_group_ids'])),
                             **{k: diag.pop(k) for k in list(diag) if k.startswith('server_only_')}}
        # Both slots use this fit subset; the inactive correction imports no labels from another fold.
        model = FrozenAnchoredBlendModel(anchor_model=anchor, correction_model=anchor,
            candidate='constant_alpha_0', anchor_objective_mode='control_only',
            correction_objective_mode='control_only')
        return model, diag

    for spec in protocol['folds']:
        seed = spec['heldout_seed']
        held = {g for g in groups if g.startswith(f'cologne1:seed{seed}:')}
        fitting = set(groups) - held
        if len(fitting) != spec['fit_group_count'] or len(held) != spec['heldout_group_count']:
            raise ValueError('The original seed folds changed')
        model, diag = fit_subset(_group_subset_v3(training, selected_groups=fitting), f'oof_{seed}')
        fold_records = score_records(model, _group_subset_v3(training, selected_groups=held))
        if {r['group_id'] for r in fold_records} != held:
            raise ValueError('OOF scores must cover only the corresponding heldout groups')
        records.extend(fold_records)
        fold_diagnostics.append({'heldout_seed': seed, 'diagnostics': diag})
        print(json.dumps({'heldout_seed': seed, 'groups': len(fold_records), 'status': 'FITTED'}), flush=True)
    effective, mask = apply_no_stay_to_oof(records, bank, groups)
    calibration = select_threshold(effective)
    selected = calibration['selected']
    accounting = proposal_accounting(effective, selected['threshold'])
    if accounting['accepted_overrides'] != selected['accepted_overrides']:
        raise ValueError('Selected threshold and runtime veto accounting disagree')
    full_model, full_diag = fit_subset(training, 'full_b100')
    model_path = output / 'model.pkl'
    payload = {'protocol': MODEL_PROTOCOL, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'city': 'cologne', 'target_budget': 100, 'selected_groups': groups,
        'training_scenarios': ['cologne1'], 'frozen_candidate': 'constant_alpha_0',
        'fit_protocol': protocol['protocol'], 'score_units': SCORE_UNITS,
        'sample_weight_source_protocol': weight_protocol, 'rigid_model': full_model}
    with model_path.open('xb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(model_path.read_bytes())['rigid_model']
    before = full_model.predict(full_contrast)['control_cost']['mean']
    after = restored.predict(full_contrast)['control_cost']['mean']
    reference_rows = np.unique(np.asarray(full_contrast.metadata['reference_rows'], dtype=int))
    if (not np.array_equal(before, after) or np.any(after[reference_rows] != 0.)
            or restored.anchor_model is not restored.correction_model):
        raise ValueError('Saved raw model must preserve all scores, zero PP references and slot aliasing')
    frozen = {'protocol': 'tsc-v154l-cologne-rigid-oof-threshold-v1', 'calibration_protocol': protocol['protocol'],
        'source_root': protocol['source_root'], 'fit_result': str(output / 'result.json'),
        'calibration_result': str(output / 'result.json'), 'model_path': str(model_path),
        'ablation_originator_path': protocol['ablation_originator_path'],
        'threshold': selected['threshold'], 'mode': selected['mode'],
        'comparison': 'predicted_advantage > threshold + 1e-12', 'score_units': SCORE_UNITS,
        'selection_data': 'Original100 native B100 OOF records; frozen raw-target weight protocol and S stay veto',
        'calibration_mean_cost': selected['mean_cost']}
    old_by_group = {r['group_id']: r for r in oldeffective}
    matched = []
    for new in effective:
        old = old_by_group[new['group_id']]
        if old['reference_index'] != new['reference_index'] or old['label_costs'] != new['label_costs']:
            raise ValueError('New and old action comparisons must share native labels and PP references')
        matched.append({'group_id': new['group_id'], 'seed': new['seed'],
            'reference_index': new['reference_index'], 'label_costs': new['label_costs'],
            'old_effective_index': old['selected_index'], 'new_effective_index': new['selected_index'],
            'old_deployed_index': deployed_index(old, oldfrozen['threshold']),
            'new_deployed_index': deployed_index(new, selected['threshold']),
            ('old_raw_predicted_advantage' if balanced else 'old_normalized_predicted_advantage'): old['predicted_advantage'],
            'new_raw_predicted_advantage': new['predicted_advantage']})
    result = {'protocol': protocol['protocol'], 'status': 'COMPLETE', 'source_root': protocol['source_root'],
        'sumocfg': protocol['sumocfg'], 'old_model_path': protocol['old_model_path'],
        'model_path': str(model_path), 'frozen_threshold': frozen, 'score_units': SCORE_UNITS,
        'selected_groups': groups, 'folds': protocol['folds'],
        'anchor_feature_names': full_diag['feature_names'],
        'fit_diagnostics': {'full': full_diag, 'oof': fold_diagnostics},
        'checks': {k: True for k in ('training_roster_exact', 'feature_order_exact',
            'original_a_baseline_exact' if balanced else 'original_t_baseline_exact', 'raw_target_not_clipped',
            'base_group_weights_uniform' if balanced else 'sample_weight_source_unchanged',
            'serialized_scores_exact', 'reference_predictions_zero')},
        'calibration': {'selected': selected, 'phase_pressure': calibration['phase_pressure'],
            'no_stay_threshold_zero': calibration['ungated_rigid'], 'candidate_count': calibration['candidate_count']},
        'old_calibration': oldcal['calibration'], 'matched_actions': matched,
        'proposal_accounting': accounting, 'mask': mask,
        'new_model_fits': 4, 'full_model_fits': 1, 'oof_fold_fits': 3, 'inactive_correction_fits': 0,
        'new_labels': 0, 'new_simulations': 0,
        'refs': {k: protocol[k] for k in ('fit_result', 'original_calibration_result', 'native_bank')},
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started}
    write_json(output / 'frozen_threshold.json', frozen)
    write_json(output / 'fit_details_server_only.json', detailed_fits)
    write_json(output / 'oof_server_only.json', {'records': records, 'effective_records': effective,
                                              'calibration': calibration, 'folds': protocol['folds']})
    write_json(output / 'result.json', result)
    print(json.dumps({'status': result['status'], 'selected': selected, 'elapsed_sec': result['elapsed_sec']}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
