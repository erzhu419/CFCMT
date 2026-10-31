"""Fit three B195 OOF anchors and compare only the original100 heldout groups."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

if __package__:
    from .cologne_feature_block_comparison import compare_actions
    from .cologne_raw_target_rigid import fit_raw_target_anchor, SCORE_UNITS
else:
    from cologne_feature_block_comparison import compare_actions
    from cologne_raw_target_rigid import fit_raw_target_anchor, SCORE_UNITS

PROTOCOL = 'tsc-v155k-cologne-expanded-data-oof-v1'
SEEDS = (2027, 3037, 4047)


def write_json(path, data):
    with path.open('x') as file:
        json.dump(data, file, separators=(',', ':'), allow_nan=False)
        file.write('\n')


def group_seed(group):
    return int(group.split(':')[1][4:])


def expanded_fold_plan(original, additional, specs):
    """Exclude the heldout seed from both banks; score only its original groups."""
    if (len(original) != 100 or len(additional) != 95 or len(set(original + additional)) != 195
            or Counter(map(group_seed, original)) != {2027: 34, 3037: 33, 4047: 33}
            or Counter(map(group_seed, additional)) != {2027: 32, 3037: 31, 4047: 32}
            or len(specs) != 3 or [spec['heldout_seed'] for spec in specs] != list(SEEDS)):
        raise ValueError('Require the frozen disjoint100+95 groups and original three training seeds')
    plans = []
    for spec in specs:
        seed = spec['heldout_seed']
        original_fit = [group for group in original if group_seed(group) != seed]
        additional_fit = [group for group in additional if group_seed(group) != seed]
        held = [group for group in original if group_seed(group) == seed]
        excluded = [group for group in additional if group_seed(group) == seed]
        actual = {'original_fit_groups': len(original_fit), 'expanded_fit_groups': len(original_fit + additional_fit),
                  'additional_fit_groups': len(additional_fit), 'evaluation_groups': len(held),
                  'new_heldout_groups_excluded_from_fit': len(excluded)}
        if any(spec[key] != value for key, value in actual.items()):
            raise ValueError('Expanded folds must train129/131/130 groups and evaluate only34/33/33 original groups')
        plans.append({**spec, 'fit_group_ids': original_fit + additional_fit,
                      'evaluation_group_ids': held, 'excluded_additional_group_ids': excluded})
    return plans


def fold_datasets(training, plan):
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
    fitting = _group_subset_v3(training, selected_groups=set(plan['fit_group_ids']))
    evaluation = _group_subset_v3(training, selected_groups=set(plan['evaluation_group_ids']))
    for bank, key in ((fitting, 'fit_group_ids'), (evaluation, 'evaluation_group_ids')):
        groups = np.asarray(bank.metadata['action_group_ids'])
        if (Counter(groups) != Counter({group: 8 for group in plan[key]})
                or (key == 'fit_group_ids' and any(group_seed(group) == plan['heldout_seed'] for group in set(groups)))):
            raise ValueError('Every fit/evaluation group must have eight rows and no heldout-seed fit labels')
    return fitting, evaluation


def compare_common_records(old_records, new_records, expected_groups):
    maps = [{row['group_id']: row for row in records} for records in (old_records, new_records)]
    if (len(expected_groups) != 100 or len(set(expected_groups)) != 100
            or any(len(records) != 100 or len(mapping) != 100 or set(mapping) != set(expected_groups)
                   for records, mapping in zip((old_records, new_records), maps))):
        raise ValueError('Primary comparisons require the same original100 records, excluding the95 additional groups')
    matched = []
    for group in expected_groups:
        old, new = (mapping[group] for mapping in maps)
        if any(old[key] != new[key] for key in ('seed', 'candidate_states', 'reference_index', 'label_costs')):
            raise ValueError('Preserve original native labels, candidate order and PP reference for the common100')
        row = {key: new[key] for key in ('group_id', 'seed', 'reference_index', 'label_costs')}
        for prefix, record in (('old', old), ('new', new)):
            selected, reference = record['selected_index'], record['reference_index']
            deployed = selected if selected != reference and record['predicted_advantage'] > 1e-12 else reference
            row.update({prefix + '_effective_index': selected, prefix + '_deployed_index': deployed,
                        prefix + '_raw_predicted_advantage': record['predicted_advantage']})
        matched.append(row)
    return matched, compare_actions(matched)


def collection_gate(protocol):
    """Read all six shards; collisions do not invalidate otherwise complete labels."""
    cells, failures = [], []
    specs = protocol['collection_shards']
    if (len(specs) != 6 or {(spec['seed'], spec['shard']) for spec in specs} != {(seed, shard) for seed in SEEDS for shard in (0, 1)}):
        raise ValueError('Use the six frozen collection shards')
    for spec in specs:
        expected = [group for group in protocol['additional_group_ids'] if group_seed(group) == spec['seed']][spec['shard']::2]
        if spec['group_ids'] != expected:
            raise ValueError('The shard roster must partition exactly the preselected95 groups')
        try:
            cell = json.loads(Path(spec['result']).read_text())
            if (cell['protocol'] != PROTOCOL or cell['status'] != 'COMPLETE'
                    or any(cell[key] != protocol[key] for key in ('source_root', 'sumocfg'))
                    or any(cell[key] != spec[key] for key in ('seed', 'shard', 'bank'))
                    or cell['requested_groups'] != expected or cell['retained_groups'] != expected
                    or cell['rows'] != 8 * len(expected) or cell['invalid_groups']):
                raise ValueError('Shard is incomplete, invalid or differs from its frozen source/roster')
            cells.append(cell)
        except (OSError, KeyError, ValueError) as error:
            failures.append({'seed': spec['seed'], 'shard': spec['shard'], 'result': spec['result'], 'error': str(error)})
    return cells, failures


def run(protocol, output):
    started = time.monotonic()
    if (protocol['protocol'] != PROTOCOL or protocol['threshold'] != 0
            or tuple(protocol['training_seeds']) != SEEDS or len(protocol['original_feature_names']) != 29
            or os.environ.get('CFCMT_SOURCE_ROOT') != protocol['source_root']):
        raise ValueError('Use the frozen B195 ablation, original29 features and zero threshold')
    plans = expanded_fold_plan(protocol['original_group_ids'], protocol['additional_group_ids'], protocol['folds'])
    if output.exists():
        raise FileExistsError(output)
    # Load the staged K bank API before adding the older calibration-tool directory.
    if __package__:
        from .cologne_native_bank import assert_complete_b100, assemble_expanded_native_bank, assert_complete_b195
    else:
        from cologne_native_bank import assert_complete_b100, assemble_expanded_native_bank, assert_complete_b195
    sys.path.insert(0, protocol['calibration_tooling_root'])
    from cologne_no_stay_calibration import apply_no_stay_to_oof, proposal_accounting
    from calibrate_cologne_rigid_threshold import score_records, threshold_metrics
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
    from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg, read_network_right_of_way_context)

    output.mkdir(parents=True)
    base = {'protocol': PROTOCOL, **{key: protocol[key] for key in (
        'source_root', 'sumocfg', 'reference_model_path', 'original_group_ids', 'additional_group_ids', 'folds',
        'original_feature_names', 'threshold')}, 'score_units': SCORE_UNITS,
        'new_model_fits': 0, 'oof_fold_fits': 0, 'full_model_fits': 0, 'inactive_correction_fits': 0,
        'new_labels': 0, 'new_simulations': 0, 'threshold_retuned': False}
    cells, failures = collection_gate(protocol)
    groups = protocol['original_group_ids']
    if not failures:
        try:
            previous = json.loads(Path(protocol['reference_fit_result']).read_text())
            cached = json.loads(Path(protocol['reference_oof_source']).read_text())
            if (previous['protocol'] != 'tsc-v155a-cologne-raw-target-refit-v1' or previous['status'] != 'COMPLETE'
                    or not all(previous['checks'].values()) or previous['selected_groups'] != groups
                    or previous['refs']['native_bank'] != protocol['reference_native_bank']
                    or any(previous[key] != protocol[key] for key in ('source_root', 'sumocfg'))
                    or previous['model_path'] != protocol['reference_model_path']
                    or previous['anchor_feature_names'] != protocol['original_feature_names']
                    or previous['frozen_threshold'] != protocol['reference_frozen_threshold']
                    or previous['frozen_threshold']['threshold'] != 0 or previous['score_units'] != SCORE_UNITS
                    or previous['folds'] != protocol['original_folds'] or cached['folds'] != protocol['original_folds']):
                raise ValueError('The original A model, native labels, features or OOF folds changed')
            original = load_mechanism_dataset(Path(protocol['reference_native_bank']))
            assert_complete_b100(original, groups)
            oldeffective, oldmask = apply_no_stay_to_oof(cached['records'], original, groups)
            oldmetrics = threshold_metrics(oldeffective, 0.)
            if (oldeffective != cached['effective_records'] or oldmask != previous['mask']
                    or oldmetrics != previous['calibration']['selected'] or oldmetrics != protocol['expected_original_metrics']):
                raise ValueError('Cached A scores must reproduce original top1 then S-veto actions and costs')
            bank = assemble_expanded_native_bank(original, [load_mechanism_dataset(Path(cell['bank'])) for cell in cells],
                                                 groups + protocol['additional_group_ids'])
            assert_complete_b195(bank, groups + protocol['additional_group_ids'])
        except (OSError, KeyError, ValueError) as error:
            failures.append({'error': str(error)})
    if failures:
        result = {**base, 'status': 'INVALID_DATA', 'failures': failures, 'elapsed_sec': time.monotonic() - started}
        write_json(output / 'result.json', result)
        return result

    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    training = augment_dataset_with_right_of_way_context(bank, {'cologne1': context})
    records, diagnostics, details = [], [], {}
    for plan in plans:
        seed = plan['heldout_seed']
        fitting, evaluation = fold_datasets(training, plan)
        contrast = build_action_contrast_dataset(fitting, reference_policy='phase_pressure', contrast_features=CONTRAST_FEATURES_V3)
        anchor, diag = fit_raw_target_anchor(contrast, sample_weight_protocol='group_action_range_v1',
                                              feature_names=protocol['original_feature_names'])
        count, config = plan['expanded_fit_groups'], protocol['fit_contract']
        if (diag['feature_names'] != protocol['original_feature_names'] or diag['feature_count'] != 29
                or diag['rows'] != 8 * count or diag['training_rows'] != 7 * count or diag['actual_group_count'] != count
                or diag['sample_weight_source_protocol'] != 'group_action_range_v1'
                or len(diag['server_only_sample_weights']) != 7 * count
                or diag['raw_target_clipped_count'] != 0 or diag['score_units'] != SCORE_UNITS
                or any(getattr(anchor.config, key) != config[key] for key in (
                    'learning_rate', 'max_iter', 'max_leaf_nodes', 'min_samples_leaf', 'l2_regularization',
                    'random_state', 'target_clip', 'candidate_only', 'balance_candidate_signs'))
                or anchor.model is None or anchor.model.early_stopping != config['early_stopping']):
            raise ValueError('Preserve A HGB/raw-target settings and recompute its weighting formula only within this expanded fit fold')
        model = FrozenAnchoredBlendModel(anchor_model=anchor, correction_model=anchor,
            candidate='constant_alpha_0', anchor_objective_mode='control_only', correction_objective_mode='control_only')
        fold_records = score_records(model, evaluation)
        if (len(fold_records) != plan['evaluation_groups']
                or {record['group_id'] for record in fold_records} != set(plan['evaluation_group_ids'])
                or any(record['predicted_scores'][record['reference_index']] != 0. for record in fold_records)):
            raise ValueError('Score only the original heldout groups, with zero PP reference predictions')
        records.extend(fold_records)
        details[f'oof_{seed}'] = {**plan, **{key: diag.pop(key) for key in list(diag) if key.startswith('server_only_')}}
        diagnostics.append({'heldout_seed': seed, 'fit_seed_group_counts': dict(Counter(map(group_seed, plan['fit_group_ids']))),
                            'evaluation_groups': plan['evaluation_groups'], 'diagnostics': diag})
        print(json.dumps({'heldout_seed': seed, 'fit_groups': count, 'evaluation_groups': len(fold_records), 'status': 'FITTED'}), flush=True)
    effective, mask = apply_no_stay_to_oof(records, original, groups)
    matched, comparison = compare_common_records(oldeffective, effective, groups)
    metrics, ppmetrics = threshold_metrics(effective, 0.), threshold_metrics(effective, None)
    accounting = proposal_accounting(effective, 0.)
    if accounting['accepted_overrides'] != metrics['accepted_overrides']:
        raise ValueError('The unchanged top1 then S-veto accounting must match deployed overrides')
    result = {**base, 'status': 'COMPLETE', 'new_model_fits': 3, 'oof_fold_fits': 3,
        'expanded_training_groups': 195, 'expanded_training_rows': 1560, 'evaluation_groups': 100,
        'fit_diagnostics': {'oof': diagnostics}, 'metrics': {'model': metrics, 'original_a': oldmetrics, 'phase_pressure': ppmetrics},
        'matched_actions': matched, 'action_comparison': comparison, 'proposal_accounting': accounting, 'mask': mask,
        'checks': {key: True for key in ('original_a_baseline_exact', 'complete_native195_roster', 'heldout_seed_excluded_from_all195',
            'original100_evaluation_only', 'feature_order_exact', 'sample_weights_recomputed_fit_fold_only',
            'hgb_settings_unchanged', 'raw_targets_not_clipped', 'original_top1_then_s_veto_preserved', 'reference_predictions_zero')},
        'refs': {key: protocol[key] for key in ('reference_fit_result', 'reference_oof_source', 'reference_native_bank', 'collection_shards')},
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
    result = run(json.loads(args.protocol.read_text()), args.output)
    if result['status'] != 'COMPLETE':
        raise SystemExit(2)
