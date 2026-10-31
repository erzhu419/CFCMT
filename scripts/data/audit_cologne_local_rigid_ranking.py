"""Diagnose the frozen Cologne1 rigid model on its 95 unused label groups."""

import argparse
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np


PROTOCOL = 'tsc-v154k-cologne-local-rigid-ranking-v1'
TOLERANCE = 1e-12


def group_record(group_id, states, costs, scores, reference, selected):
    costs, scores = np.asarray(costs, dtype=float), np.asarray(scores, dtype=float)
    if (costs.shape != scores.shape or costs.shape != (len(states),)
            or not np.all(np.isfinite(costs)) or not np.all(np.isfinite(scores))):
        raise ValueError('Aligned finite candidate scores and absolute label costs are required')
    oracle = float(np.min(costs))
    return {
        'group_id': str(group_id), 'seed': int(str(group_id).split(':')[1][4:]),
        'interval_index': int(str(group_id).split(':')[2]),
        'candidate_states': list(map(str, states)), 'label_costs': costs.tolist(),
        'predicted_scores': scores.tolist(), 'reference_index': int(reference),
        'selected_index': int(selected), 'different_action': selected != reference,
        'phase_pressure_cost': float(costs[reference]), 'rigid_cost': float(costs[selected]),
        'oracle_cost': oracle,
        'rigid_minus_phase_pressure': float(costs[selected] - costs[reference]),
        'rigid_regret': float(costs[selected]) - oracle,
        'phase_pressure_regret': float(costs[reference]) - oracle,
        'predicted_advantage': float(scores[reference] - scores[selected]),
        'actual_advantage': float(costs[reference] - costs[selected]),
    }


def summarize(records):
    if not records:
        raise ValueError('Cannot summarize an empty diagnostic split')
    mean_keys = ('phase_pressure_cost', 'rigid_cost', 'oracle_cost',
                 'rigid_minus_phase_pressure', 'rigid_regret', 'phase_pressure_regret',
                 'predicted_advantage', 'actual_advantage')
    overrides = [row for row in records if row['different_action']]
    positive = [row for row in records if row['predicted_advantage'] > TOLERANCE]
    def outcomes(rows):
        return {
            'beneficial': sum(row['actual_advantage'] > TOLERANCE for row in rows),
            'harmful': sum(row['actual_advantage'] < -TOLERANCE for row in rows),
            'equal_cost': sum(abs(row['actual_advantage']) <= TOLERANCE for row in rows),
        }
    return {
        'groups': len(records), 'candidate_rows': sum(len(row['candidate_states']) for row in records),
        'weighting': 'equal_group',
        'mean': {key: float(np.mean([row[key] for row in records])) for key in mean_keys},
        'rigid_optimal_groups': sum(row['rigid_regret'] <= TOLERANCE for row in records),
        'phase_pressure_optimal_groups': sum(row['phase_pressure_regret'] <= TOLERANCE for row in records),
        'all_groups_outcomes': outcomes(records),
        'different_action_groups': len(overrides), 'different_action_outcomes': outcomes(overrides),
        'predicted_positive_advantage_groups': len(positive),
        'predicted_positive_advantage_outcomes': outcomes(positive),
        'positive_prediction_mean_actual_advantage':
            float(np.mean([row['actual_advantage'] for row in positive])) if positive else None,
        'rigid_regret_p90': float(np.quantile([row['rigid_regret'] for row in records], 0.9)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol = json.loads(args.protocol.read_text())
    if protocol['protocol'] != PROTOCOL:
        raise ValueError('Use the frozen V154K diagnostic protocol')
    if args.out.exists():
        raise FileExistsError(args.out)
    sys.path.insert(0, str(Path(protocol['source_root']) / 'scripts/data'))
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
        CONTRAST_FEATURES_V3, merge_counterfactual_datasets_v3)
    from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
        StateConditionedSourceUtilityOriginator, _rigid_score)
    from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
    from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset
    from cf_h2o.traffic_signal.fraction_target_rigid import FractionTargetRigidOriginator
    from cf_h2o.traffic_signal.right_of_way_context import (
        augment_dataset_with_right_of_way_context, net_file_from_sumocfg,
        read_network_right_of_way_context)

    fit = json.loads(Path(protocol['fit_result']).read_text())
    payload = pickle.loads(Path(fit['model_path']).read_bytes())
    trained = set(fit['selected_groups'])
    if (fit['status'] != 'PASS' or fit['protocol'] != protocol['refit_protocol']
            or trained != set(payload['selected_groups']) or len(trained) != 100
            or payload['frozen_candidate'] != 'constant_alpha_0'):
        raise ValueError('Use the unchanged successful V154J local B100 fit')
    context = read_network_right_of_way_context(net_file_from_sumocfg(Path(protocol['sumocfg'])))
    FractionTargetRigidOriginator(runtime_payload=payload, network_context=context)
    bank_paths = [str(Path(cell['result']).with_name('bank.npz')) for cell in fit['cells']]
    pool = merge_counterfactual_datasets_v3([load_mechanism_dataset(Path(path)) for path in bank_paths])
    group_ids = np.asarray(pool.metadata['action_group_ids'], dtype=str)
    unused = set(group_ids) - trained
    if len(set(group_ids)) != 195 or len(unused) != 95 or not trained <= set(group_ids):
        raise ValueError('The original 195-group pool / 100-fit / 95-unused partition changed')
    if sorted(unused) != protocol['unused_group_ids']:
        raise ValueError('The preselected unused group roster changed')
    target = np.asarray(pool.targets[protocol['target_name']], dtype=float)
    if not np.allclose(target, pool.targets['interval_cost'], rtol=0, atol=TOLERANCE):
        raise ValueError('The fitted interval_cost differs from the 450-second label')
    augmented = augment_dataset_with_right_of_way_context(pool, {'cologne1': context})
    contrast = build_action_contrast_dataset(augmented, reference_policy='phase_pressure',
                                           contrast_features=CONTRAST_FEATURES_V3)
    scores = _rigid_score(payload['rigid_model'], contrast)
    states = np.asarray(pool.metadata['candidate_states'], dtype=str)
    references = np.asarray(contrast.metadata['reference_rows'], dtype=int)
    records = []
    for group in sorted(set(group_ids), key=lambda g: (int(g.split(':')[1][4:]), int(g.split(':')[2]))):
        rows = np.flatnonzero(group_ids == group)
        reference = int(np.flatnonzero(rows == references[rows[0]])[0])
        selected = StateConditionedSourceUtilityOriginator._argmin_with_reference_tie_break(
            scores[rows], reference=reference, candidate_states=states[rows])
        record = group_record(group, states[rows], target[rows], scores[rows], reference, selected)
        record['split'] = 'training' if group in trained else 'unused'
        records.append(record)
    split_records = {name: [r for r in records if r['split'] == name] for name in ('unused', 'training')}
    result = {
        'protocol': PROTOCOL, 'status': 'COMPLETE', 'source_root': protocol['source_root'],
        'fit_result': protocol['fit_result'], 'model_path': fit['model_path'], 'bank_paths': bank_paths,
        'target_name': protocol['target_name'], 'label_units': protocol['label_units'],
        'score_units': protocol['score_units'],
        'scoring': 'Original _rigid_score and reference-first / state-lexical tie-break; existing model, no refit',
        'tolerance': TOLERANCE, 'frozen_candidate': payload['frozen_candidate'],
        'anchor_feature_names': list(payload['rigid_model'].anchor_model.feature_names),
        'splits': {name: summarize(rows) for name, rows in split_records.items()},
        'per_seed': {name: {str(seed): summarize([r for r in rows if r['seed'] == seed])
                           for seed in protocol['training_seeds']} for name, rows in split_records.items()},
        'unused_groups': split_records['unused'],
        'new_simulation_count': 0, 'new_model_fit_count': 0,
        'limitations': protocol['limitations'], 'elapsed_sec': time.monotonic() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write('\n')
    print(json.dumps({'status': result['status'], 'splits': result['splits'], 'elapsed_sec': result['elapsed_sec']}))


if __name__ == '__main__':
    main()
