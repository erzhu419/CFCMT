"""Fit one full B195 model after K, retaining its fixed zero threshold."""

import argparse
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

if __package__:
    from .cologne_native_bank import assert_complete_b100, assemble_expanded_native_bank, assert_complete_b195
    from .cologne_raw_target_rigid import fit_raw_target_anchor, SCORE_UNITS
    from .fit_cologne_expanded_data_oof import PROTOCOL as K_PROTOCOL, collection_gate, write_json
else:
    from cologne_native_bank import assert_complete_b100, assemble_expanded_native_bank, assert_complete_b195
    from cologne_raw_target_rigid import fit_raw_target_anchor, SCORE_UNITS
    from fit_cologne_expanded_data_oof import PROTOCOL as K_PROTOCOL, collection_gate, write_json


PROTOCOL = 'tsc-v155l-cologne-expanded-data-full-fit-v1'
MODEL_PROTOCOL = 'tsc-v155l-cologne-expanded-raw-target-model-v1'
BOUND_INPUTS = ('source_root', 'sumocfg', 'original_group_ids', 'additional_group_ids',
                'original_feature_names', 'fit_contract', 'reference_native_bank', 'collection_shards')


def validate_k_binding(protocol, previous_protocol, previous):
    if (protocol['protocol'] != PROTOCOL or protocol['threshold'] != 0
            or previous_protocol['protocol'] != K_PROTOCOL or previous_protocol['threshold'] != 0
            or any(protocol[key] != previous_protocol[key] for key in BOUND_INPUTS)
            or previous['protocol'] != K_PROTOCOL or previous['status'] != 'COMPLETE'
            or not previous['checks'] or not all(previous['checks'].values())
            or previous['oof_fold_fits'] != 3 or previous['full_model_fits'] != 0
            or previous['score_units'] != SCORE_UNITS
            or any(previous[key] != protocol[key] for key in (
                'source_root', 'sumocfg', 'original_group_ids', 'additional_group_ids', 'original_feature_names'))
            or previous['refs']['reference_native_bank'] != protocol['reference_native_bank']
            or previous['refs']['collection_shards'] != protocol['collection_shards']
            or previous['metrics']['model']['threshold'] != 0
            or previous['metrics']['model']['mean_cost'] >= previous['metrics']['original_a']['mean_cost']-1e-12):
        raise ValueError('Full fitting requires the completed qualifying K result and its exact frozen data/configuration')


def fit_full_model(contrast, protocol):
    from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
    groups = protocol['original_group_ids'] + protocol['additional_group_ids']
    actual = list(dict.fromkeys(contrast.metadata['action_group_ids']))
    if len(groups) != 195 or len(set(groups)) != 195 or actual != groups or contrast.size != 1560:
        raise ValueError('Fit exactly the original100 plus new95 groups in their frozen row order')
    anchor, diagnostics = fit_raw_target_anchor(contrast, sample_weight_protocol='group_action_range_v1',
                                               feature_names=protocol['original_feature_names'])
    config = protocol['fit_contract']
    if (diagnostics['actual_group_count'] != 195 or diagnostics['rows'] != 1560
            or diagnostics['training_rows'] != 1365 or diagnostics['feature_count'] != 29
            or diagnostics['feature_names'] != protocol['original_feature_names']
            or diagnostics['score_units'] != SCORE_UNITS or diagnostics['raw_target_clipped_count'] != 0
            or diagnostics['sample_weight_source_protocol'] != 'group_action_range_v1'
            or any(getattr(anchor.config, key) != config[key] for key in (
                'learning_rate', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'l2_regularization',
                'random_state', 'target_clip', 'candidate_only', 'balance_candidate_signs'))
            or anchor.model is None or anchor.model.early_stopping != config['early_stopping']):
        raise ValueError('Preserve K/A raw-target units,29 columns,HGB settings and original weighting formula')
    return FrozenAnchoredBlendModel(anchor_model=anchor, correction_model=anchor,
        candidate='constant_alpha_0', anchor_objective_mode='control_only',
        correction_objective_mode='control_only'), diagnostics


def serialize_model(model, contrast, protocol, model_path):
    from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL
    groups = protocol['original_group_ids'] + protocol['additional_group_ids']
    payload = {'protocol': MODEL_PROTOCOL, 'occupancy_equation_protocol': OCCUPANCY_EQUATION_PROTOCOL,
        'city': 'cologne', 'target_budget': 195, 'selected_groups': groups,
        'training_scenarios': ['cologne1'], 'frozen_candidate': 'constant_alpha_0',
        'fit_protocol': PROTOCOL, 'score_units': SCORE_UNITS,
        'sample_weight_source_protocol': 'group_action_range_v1', 'rigid_model': model}
    with model_path.open('xb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    restored = pickle.loads(model_path.read_bytes())['rigid_model']
    before = model.predict(contrast)['control_cost']['mean']
    after = restored.predict(contrast)['control_cost']['mean']
    references = np.asarray(contrast.metadata['is_reference'], dtype=bool)
    if (not np.array_equal(before, after) or np.sum(references) != 195 or np.any(after[references] != 0.)
            or model.anchor_model is not model.correction_model
            or restored.anchor_model is not restored.correction_model):
        raise ValueError('Serialized B195 scores, zero PP references or anchor/correction alias changed')


def run(protocol, output):
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg, read_network_right_of_way_context)
    started = time.monotonic()
    if os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']:
        raise ValueError('Use the unchanged frozen J model source')
    if output.exists():
        raise FileExistsError(output)
    previous_protocol = json.loads(Path(protocol['k_protocol_path']).read_text())
    previous = json.loads(Path(protocol['k_result']).read_text())
    validate_k_binding(protocol, previous_protocol, previous)
    cells, failures = collection_gate(previous_protocol)
    if failures:
        raise ValueError(f'All six original K collection shards must remain complete: {failures}')
    original = load_mechanism_dataset(Path(protocol['reference_native_bank']))
    assert_complete_b100(original, protocol['original_group_ids'])
    groups = protocol['original_group_ids'] + protocol['additional_group_ids']
    bank = assemble_expanded_native_bank(original, [load_mechanism_dataset(Path(cell['bank'])) for cell in cells], groups)
    assert_complete_b195(bank, groups)
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    training = augment_dataset_with_right_of_way_context(bank, {'cologne1': context})
    contrast = build_action_contrast_dataset(training, reference_policy='phase_pressure', contrast_features=CONTRAST_FEATURES_V3)
    output.mkdir(parents=True)
    model, diagnostics = fit_full_model(contrast, protocol)
    model_path = output / 'model.pkl'
    serialize_model(model, contrast, protocol, model_path)
    details = {key: diagnostics.pop(key) for key in list(diagnostics) if key.startswith('server_only_')}
    frozen = {'protocol': 'tsc-v154l-cologne-rigid-oof-threshold-v1', 'calibration_protocol': K_PROTOCOL,
        'source_root': protocol['source_root'], 'fit_result': str(output / 'result.json'),
        'calibration_result': protocol['k_result'], 'model_path': str(model_path),
        'threshold': 0.0, 'mode': 'minimum_advantage',
        'comparison': 'predicted_advantage > threshold + 1e-12', 'score_units': SCORE_UNITS,
        'selection_data': 'Retain K fixed threshold0; the full B195 fit performs no threshold calibration.'}
    result = {'protocol': PROTOCOL, 'status': 'COMPLETE', 'source_root': protocol['source_root'],
        'sumocfg': protocol['sumocfg'], 'model_path': str(model_path), 'model_protocol': MODEL_PROTOCOL,
        'selected_groups': groups, 'original_group_ids': protocol['original_group_ids'],
        'additional_group_ids': protocol['additional_group_ids'], 'anchor_feature_names': diagnostics['feature_names'],
        'score_units': SCORE_UNITS, 'frozen_threshold': frozen, 'fit_diagnostics': {'full': diagnostics},
        'training_group_count': 195, 'training_row_count': 1560,
        'new_model_fits': 1, 'full_model_fits': 1, 'oof_fold_fits': 0, 'inactive_correction_fits': 0,
        'new_labels': 0, 'new_simulations': 0, 'threshold_retuned': False,
        'checks': {key: True for key in ('completed_k_admitted', 'native_b195_roster_exact', 'feature_order_exact',
            'a_weight_and_hgb_settings_unchanged', 'raw_targets_not_clipped', 'single_full_fit',
            'serialized_scores_exact', 'reference_predictions_zero', 'anchor_correction_alias_preserved',
            'threshold_zero_unchanged')},
        'refs': {key: protocol[key] for key in ('k_protocol_path', 'k_result', 'reference_native_bank', 'collection_shards')},
        'elapsed_sec': time.monotonic()-started}
    write_json(output / 'fit_details_server_only.json', {'full_b195': {'groups': groups, **details}})
    write_json(output / 'frozen_threshold.json', frozen)
    write_json(output / 'result.json', result)
    print(json.dumps({key: result[key] for key in ('protocol', 'status', 'full_model_fits', 'elapsed_sec')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.protocol.read_text()), args.output)
