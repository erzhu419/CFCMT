"""Zero-fit sensitivity of frozen readers to measured ten-second responses."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np


def run(study_base):
    study_base = Path(study_base)
    sys.path.insert(0, str(study_base))
    import numeric_projection as projection
    from diagnose_frozen import _load
    from prior_ridge import predict, reader_predict, select_action

    result = json.loads((study_base/'nested_fit/result.json').read_text())
    dataset = json.loads((study_base/'training_dataset_server_only.json').read_text())
    groups = sorted((g for g in dataset['groups'] if g['city'] == 'jinan'),
                    key=lambda g: (g['seed'], g['anchor']))
    models = {}
    for entry in result['fit_ledger']:
        if entry['role'] in ('outer_target_dynamics', 'target_cost_reader'):
            models[entry['outer_fold'], entry['arm'], entry['role']] = _load(
                study_base/'nested_fit'/entry['model_file'], entry['role'] == 'outer_target_dynamics')
    if len(models) != 30 or len(groups) != 20 or len({g['seed'] for g in groups}) != 10:
        raise ValueError('Expected the existing fifteen readers/dynamics and twenty held-out groups')
    rows, max_reconstruction, max_identity = [], 0., 0.
    for group in groups:
        identity = f"jinan/seed{group['seed']}/anchor{group['anchor']}"
        records = {r['arm']: r for r in result['records'] if r['group_id'] == identity}
        branches = sorted(group['branches'], key=lambda b: b['candidate_index'])
        if len(records) != 3 or [b['candidate_index'] for b in branches] != list(range(8)):
            raise ValueError('All three frozen arms and eight actual branches must be present')
        pp = group['pp_index']
        actual = np.asarray([b['cost'] for b in branches], dtype=float)
        delta = actual-actual[pp]
        nonpp = np.asarray([a for a in range(8) if a != pp])
        true_absolute, actual_responses = [], []
        for branch in branches:
            inputs, target = branch['inputs'], branch['targets']
            if target['lane_ids'] != [n['lane_id'] for n in inputs['lane_nodes']]:
                raise ValueError('Measured ten-second response lane alignment changed')
            actual_responses.append(np.asarray(target['y'], dtype=float))
            truth = projection.project_predicted_state(inputs, actual_responses[-1])
            true_absolute.append(projection.project_reader(inputs, truth))
        true_features = np.asarray(true_absolute)-true_absolute[pp]
        actual_responses = np.asarray(actual_responses)
        actual_contrasts = actual_responses-actual_responses[pp]
        for arm in ('T', 'S', 'P'):
            record = records[arm]
            outer = record['outer_fold']
            dynamics = models[outer, arm, 'outer_target_dynamics']
            reader = models[outer, arm, 'target_cost_reader']
            predicted_absolute, predicted_responses = [], []
            for branch in branches:
                inputs = branch['inputs']
                unused_lanes, X = projection.project_dynamics(inputs)
                predicted_responses.append(predict(dynamics, X))
                state = projection.project_predicted_state(inputs, predicted_responses[-1])
                predicted_absolute.append(projection.project_reader(inputs, state))
            predicted_features = np.asarray(predicted_absolute)-predicted_absolute[pp]
            predicted_responses = np.asarray(predicted_responses)
            predicted_contrasts = predicted_responses-predicted_responses[pp]
            if not np.array_equal(predicted_features[:, :195], true_features[:, :195]):
                raise ValueError('The response substitution changed nominal or t0 context columns')
            scores = reader_predict(reader, predicted_features)
            error = float(np.max(np.abs(scores-record['predicted_delta_costs'])))
            max_reconstruction = max(max_reconstruction, error)
            if error > 1e-12:
                raise ValueError('Frozen held-out scores were not reproduced')
            oracle = reader_predict(reader, true_features)
            if scores[pp] != 0 or oracle[pp] != 0:
                raise ValueError('Response substitution changed the structural PP zero')
            selected, oracle_selected = select_action(scores, pp), select_action(oracle, pp)
            if selected != record['selected_action']:
                raise ValueError('The original frozen choice was not reproduced')
            total_error = scores-delta
            response_change = scores-oracle
            remaining = oracle-delta
            identity_error = float(np.max(np.abs(total_error-response_change-remaining)))
            max_identity = max(max_identity, identity_error)
            if identity_error > 1e-12:
                raise ValueError('Prediction/substitution/remaining residual identity failed')
            fixed = dict(total_score_error=float(total_error[selected]),
                prediction_to_measured_score_change=float(response_change[selected]),
                measured_input_remaining_error=float(remaining[selected]))
            rows.append(dict(group_id=identity, seed=group['seed'], anchor=group['anchor'], arm=arm,
                outer_fold=outer, pp_index=pp, original_action=selected, measured_input_action=oracle_selected,
                original_cost=float(actual[selected]), measured_input_cost=float(actual[oracle_selected]),
                PP_cost=float(actual[pp]), original_delta=float(delta[selected]),
                measured_input_delta=float(delta[oracle_selected]),
                measured_input_scores=oracle.tolist(), actual_delta_costs=delta.tolist(),
                original_all_nonPP_score_mae=float(np.abs(total_error[nonpp]).mean()),
                measured_input_all_nonPP_score_mae=float(np.abs(remaining[nonpp]).mean()),
                original_all_nonPP_sign_accuracy=float((np.sign(np.round(scores[nonpp], 12)) ==
                    np.sign(np.round(delta[nonpp], 12))).mean()),
                measured_input_all_nonPP_sign_accuracy=float((np.sign(np.round(oracle[nonpp], 12)) ==
                    np.sign(np.round(delta[nonpp], 12))).mean()),
                response_contrast_mae_by_output=np.abs(predicted_contrasts[nonpp]-actual_contrasts[nonpp]).mean(axis=(0, 1)).tolist(),
                actual_response_contrast_meanabs_by_output=np.abs(actual_contrasts[nonpp]).mean(axis=(0, 1)).tolist(),
                exact_zero_actual_response_nonPP_pairs=int(np.sum(np.all(actual_contrasts[nonpp] == 0, axis=(1, 2)))),
                original_choice_error_decomposition=fixed))
    seeds = sorted({r['seed'] for r in rows})
    summaries = {}
    for arm in ('T', 'S', 'P'):
        arm_rows = [r for r in rows if r['arm'] == arm]
        per_seed = []
        for seed in seeds:
            selected = [r for r in arm_rows if r['seed'] == seed]
            if len(selected) != 2:
                raise ValueError('Every target seed must retain both states')
            per_seed.append(dict(seed=seed, original_cost=mean(r['original_cost'] for r in selected),
                measured_input_cost=mean(r['measured_input_cost'] for r in selected),
                PP_cost=mean(r['PP_cost'] for r in selected),
                measured_minus_original=mean(r['measured_input_cost']-r['original_cost'] for r in selected),
                measured_minus_PP=mean(r['measured_input_delta'] for r in selected)))
        summaries[arm] = dict(
            original_mean_cost=mean(r['original_cost'] for r in per_seed),
            measured_input_mean_cost=mean(r['measured_input_cost'] for r in per_seed),
            PP_mean_cost=mean(r['PP_cost'] for r in per_seed),
            measured_minus_original=mean(r['measured_minus_original'] for r in per_seed),
            measured_minus_PP=mean(r['measured_minus_PP'] for r in per_seed),
            changed_actions=sum(r['original_action'] != r['measured_input_action'] for r in arm_rows),
            improved_states=sum(r['measured_input_cost'] < r['original_cost'] for r in arm_rows),
            worsened_states=sum(r['measured_input_cost'] > r['original_cost'] for r in arm_rows),
            measured_harmful_overrides=sum(r['measured_input_delta'] > 0 for r in arm_rows),
            measured_beneficial_overrides=sum(r['measured_input_delta'] < 0 for r in arm_rows),
            all_nonPP_score_errors={key: mean(r[key] for r in arm_rows) for key in (
                'original_all_nonPP_score_mae', 'measured_input_all_nonPP_score_mae',
                'original_all_nonPP_sign_accuracy', 'measured_input_all_nonPP_sign_accuracy')},
            fixed_original_choice_error_decomposition={key: mean(r['original_choice_error_decomposition'][key]
                for r in arm_rows) for key in arm_rows[0]['original_choice_error_decomposition']},
            response_contrast_mae_by_output=np.mean([r['response_contrast_mae_by_output'] for r in arm_rows], axis=0).tolist(),
            actual_response_contrast_meanabs_by_output=np.mean([r['actual_response_contrast_meanabs_by_output'] for r in arm_rows], axis=0).tolist(),
            exact_zero_actual_response_nonPP_pairs=sum(r['exact_zero_actual_response_nonPP_pairs'] for r in arm_rows),
            per_seed=per_seed)
    return dict(status='DESCRIPTIVE_FROZEN_READER_MEASURED_RESPONSE_SENSITIVITY', new_fits=0,
        new_simulations=0, frozen_readers=15, frozen_outer_dynamics_reconstructed=15,
        target_states=20, target_seeds=10, nonPP_actions_per_arm=140,
        max_saved_score_reconstruction_error=max_reconstruction, max_error_decomposition_error=max_identity,
        substitution='Only Phi(predicted lane state10) becomes Phi(measured lane state10), including the PP branch. '
                     'All nominal/t0 columns, fitted coefficients, scales and tie rules stay fixed.',
        averaging='Equal two states within seed, then equal ten seeds. All-action errors first average seven nonPP actions.',
        cost_units='450-second mean halted vehicles per controlled lane', arms=summaries, records=rows,
        response_output_order=['delta_N_storage', 'delta_H_storage', 'delta_M_storage', 'delta_occupancy'],
        limitations='Measured future responses are unavailable at decision time. These readers were trained on inner-OOF '
                    'predicted responses, so measured-response substitution changes their input distribution. '
                    'No improvement does not prove that local true states lack information; improvement is not deployable '
                    'or an upper bound. The score-error split is an algebraic identity, not causal shares. '
                    'The original development HOLD remains unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study_base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], new_fits=0, new_simulations=0,
        arms={arm: {key: value for key, value in row.items() if key != 'per_seed'}
              for arm, row in result['arms'].items()}), allow_nan=False))
