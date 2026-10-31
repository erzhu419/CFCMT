"""Calibrate a PP fallback threshold using only seed-held-out B100 predictions."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__:
    from .audit_cologne_local_rigid_ranking import group_record
else:
    from audit_cologne_local_rigid_ranking import group_record


PROTOCOL = 'tsc-v154l-cologne-rigid-oof-threshold-v1'
TOLERANCE = 1e-12


def threshold_metrics(records, threshold):
    if not records:
        raise ValueError('Threshold calibration requires nonempty records')
    by_seed = {}
    for seed in sorted({row['seed'] for row in records}):
        rows = [row for row in records if row['seed'] == seed]
        accepted = [row for row in rows if threshold is not None
                    and row['different_action']
                    and row['predicted_advantage'] > threshold + TOLERANCE]
        gains = [row['actual_advantage'] for row in accepted]
        pp = float(np.mean([row['phase_pressure_cost'] for row in rows]))
        improvement = float(sum(gains) / len(rows))
        by_seed[str(seed)] = {
            'groups': len(rows), 'accepted_overrides': len(accepted),
            'beneficial': sum(gain > TOLERANCE for gain in gains),
            'harmful': sum(gain < -TOLERANCE for gain in gains),
            'equal_cost': sum(abs(gain) <= TOLERANCE for gain in gains),
            'mean_phase_pressure_cost': pp, 'mean_cost': pp - improvement,
            'mean_cost_minus_phase_pressure': -improvement,
        }
    return {
        'threshold': threshold, 'mode': 'phase_pressure_only' if threshold is None else 'minimum_advantage',
        'groups': len(records), 'weighting': 'equal_seed', 'per_seed': by_seed,
        'accepted_overrides': sum(row['accepted_overrides'] for row in by_seed.values()),
        **{key: sum(row[key] for row in by_seed.values()) for key in ('beneficial', 'harmful', 'equal_cost')},
        **{key: float(np.mean([row[key] for row in by_seed.values()])) for key in
           ('mean_cost', 'mean_phase_pressure_cost', 'mean_cost_minus_phase_pressure')},
    }


def select_threshold(records):
    cutpoints = sorted({float(row['predicted_advantage']) for row in records
                       if row['different_action'] and row['predicted_advantage'] > TOLERANCE})
    candidates = [threshold_metrics(records, value) for value in [None, 0.0, *cutpoints]]
    minimum = min(row['mean_cost'] for row in candidates)
    tied = [row for row in candidates if abs(row['mean_cost'] - minimum) <= TOLERANCE]
    selected = max(tied, key=lambda row: (row['threshold'] is None,
                                        row['threshold'] if row['threshold'] is not None else 0.0))
    return {'selected': selected, 'phase_pressure': candidates[0], 'ungated_rigid': candidates[1],
            'candidate_count': len(candidates), 'curve': candidates}


def score_records(model, absolute):
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
        StateConditionedSourceUtilityOriginator, _rigid_score)
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    contrast = build_action_contrast_dataset(absolute, reference_policy='phase_pressure',
                                           contrast_features=CONTRAST_FEATURES_V3)
    scores = _rigid_score(model, contrast)
    groups = np.asarray(absolute.metadata['action_group_ids'], dtype=str)
    states = np.asarray(absolute.metadata['candidate_states'], dtype=str)
    references = np.asarray(contrast.metadata['reference_rows'], dtype=int)
    target = np.asarray(absolute.targets['prefix_mean_cost_450s'], dtype=float)
    result = []
    for group in sorted(set(groups)):
        rows = np.flatnonzero(groups == group)
        reference = int(np.flatnonzero(rows == references[rows[0]])[0])
        selected = StateConditionedSourceUtilityOriginator._argmin_with_reference_tie_break(
            scores[rows], reference=reference, candidate_states=states[rows])
        result.append(group_record(group, states[rows], target[rows], scores[rows], reference, selected))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    p = json.loads(args.protocol.read_text())
    if p['protocol'] != PROTOCOL:
        raise ValueError('Use the frozen V154L calibration protocol')
    if args.output.exists():
        raise FileExistsError(args.output)
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import merge_counterfactual_datasets_v3
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _group_subset_v3
    from cf_h2o.eval.traffic_signal_target_budget_source_value_curve import _target_model
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg,
        read_network_right_of_way_context)

    fit = json.loads(Path(p['fit_result']).read_text())
    groups = set(fit['selected_groups'])
    if (fit['status'] != 'PASS' or fit['protocol'] != p['refit_protocol'] or len(groups) != 100
            or sorted(groups) != p['training_group_ids'] or fit['frozen_candidate'] != 'constant_alpha_0'):
        raise ValueError('The original B100 training roster or model family changed')
    bank_paths = [str(Path(cell['result']).with_name('bank.npz')) for cell in fit['cells']]
    pool = merge_counterfactual_datasets_v3([load_mechanism_dataset(Path(path)) for path in bank_paths])
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(p['sumocfg'])))
    augmented = augment_dataset_with_right_of_way_context(pool, {'cologne1': context})
    training = _group_subset_v3(augmented, selected_groups=groups,
                              metadata_updates={'target_data_role': 'v154l_oof_calibration_only'})
    actual_groups, counts = np.unique(training.metadata['action_group_ids'], return_counts=True)
    if set(actual_groups) != groups or np.any(counts != 8):
        raise ValueError('B100 must contain exactly the saved 100 groups and all eight actions per group')
    if not np.allclose(training.targets['interval_cost'], training.targets['prefix_mean_cost_450s'],
                       rtol=0, atol=TOLERANCE):
        raise ValueError('Training targets differ from the frozen 450-second queue labels')
    records, folds = [], []
    for seed in p['training_seeds']:
        fold_started = time.monotonic()
        held = {g for g in groups if g.split(':')[1] == f'seed{seed}'}
        fitting = groups - held
        if len(held) != p['heldout_group_counts'][str(seed)]:
            raise ValueError('Seed-held-out group count changed')
        model = _target_model(training, group_ids=sorted(fitting),
                              candidate='constant_alpha_0', target_domain='cologne')
        if list(model.anchor_model.feature_names) != fit['anchor_feature_names']:
            raise ValueError('Fold score features differ from the original B100 model')
        heldout = _group_subset_v3(training, selected_groups=held,
                                  metadata_updates={'target_data_role': 'v154l_seed_held_out'})
        fold_records = score_records(model, heldout)
        if {row['group_id'] for row in fold_records} != held:
            raise ValueError('Fold predictions do not cover exactly the held-out groups')
        records.extend(fold_records)
        folds.append({'heldout_seed': seed, 'training_groups': sorted(fitting),
                      'heldout_groups': sorted(held), 'fit_group_count': len(fitting),
                      'heldout_group_count': len(held), 'elapsed_sec': time.monotonic() - fold_started})
    if len(records) != 100 or {row['group_id'] for row in records} != groups:
        raise ValueError('OOF predictions must cover B100 exactly once')
    calibration = select_threshold(records)
    selected = calibration['selected']
    selected_threshold = selected['threshold']
    # Freeze the choice before reading any outcomes from the inspected 95 development groups.
    args.output.mkdir(parents=True)
    frozen = {'protocol': PROTOCOL, 'source_root': p['source_root'], 'fit_result': p['fit_result'],
              'model_path': fit['model_path'], 'threshold': selected_threshold, 'mode': selected['mode'],
              'comparison': 'predicted_advantage > threshold + 1e-12',
              'selection_data': 'Only 100 seed-held-out predictions from the original B100 training roster',
              'calibration_mean_cost': selected['mean_cost']}
    with (args.output / 'frozen_threshold.json').open('x') as file:
        json.dump(frozen, file, indent=2, allow_nan=False)
        file.write('\n')
    diagnostic = json.loads(Path(p['unused_diagnostic_result']).read_text())
    unused = diagnostic['unused_groups']
    if (diagnostic['status'] != 'COMPLETE' or diagnostic['model_path'] != fit['model_path']
            or len(unused) != 95 or len({r['group_id'] for r in unused}) != 95
            or groups & {r['group_id'] for r in unused}):
        raise ValueError('Development diagnostics do not match the unchanged original full B100 model')
    result = {
        'protocol': PROTOCOL, 'status': 'COMPLETE', 'source_root': p['source_root'],
        'fit_result': p['fit_result'], 'bank_paths': bank_paths, 'frozen_threshold': frozen,
        'folds': folds, 'oof_records': records, 'calibration': calibration,
        'development_95': {'phase_pressure': threshold_metrics(unused, None),
                           'ungated_rigid': threshold_metrics(unused, 0.0),
                           'frozen_threshold': threshold_metrics(unused, selected_threshold)},
        'positive_threshold_reduces_oof_cost': (selected_threshold is not None and selected_threshold > 0
            and selected['mean_cost'] < min(calibration['phase_pressure']['mean_cost'],
                                           calibration['ungated_rigid']['mean_cost']) - TOLERANCE),
        'new_simulation_count': 0, 'new_fold_model_fit_count': 3, 'unique_training_groups': 100,
        'summed_fold_training_groups': sum(fold['fit_group_count'] for fold in folds),
        'label_units': p['label_units'], 'score_units': p['score_units'],
        'limitations': p['limitations'], 'elapsed_sec': time.monotonic() - started,
    }
    with (args.output / 'result.json').open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'status': result['status'], 'selected': selected,
                      'development_95': result['development_95'], 'elapsed_sec': result['elapsed_sec']}))


if __name__ == '__main__':
    main()
