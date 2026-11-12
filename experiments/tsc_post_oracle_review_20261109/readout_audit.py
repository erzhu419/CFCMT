"""Zero-fit reconstruction of measured-response reader errors and input support."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

BLOCKS = dict(nominal=slice(0, 15), global_nominal=slice(15, 195), response=slice(195, 211))


def quantiles(values):
    values = np.asarray(values, dtype=float).ravel()
    return dict(zip(('min', 'q25', 'median', 'q75', 'max'),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1])))) if len(values) else None


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None


def state_metrics(group, scores, select):
    selected, pp, costs = select(scores, group['pp']), group['pp'], group['costs']
    delta, rounded = costs-costs[pp], np.round(scores, 12)
    comparable = correct = reverse = ties = 0
    for a in range(8):
        for b in range(a+1, 8):
            actual, predicted = costs[a]-costs[b], rounded[a]-rounded[b]
            if abs(actual) > 1e-12:
                comparable += 1
                correct += actual*predicted > 0
                reverse += actual*predicted < 0
                ties += predicted == 0
    nonpp = [a for a in range(8) if a != pp]
    return dict(group_id=group['id'], seed=group['seed'], anchor=group['anchor'], selected_action=selected,
        pp_index=pp, predicted_delta=float(scores[selected]), actual_delta=float(delta[selected]),
        selected_regret=float(costs[selected]-costs.min()),
        selected_is_observed_best=bool(abs(costs[selected]-costs.min()) <= 1e-12),
        mean_nonpp_delta_mae=float(np.abs(scores[nonpp]-delta[nonpp]).mean()),
        correct_pairs=int(correct), reversed_pairs=int(reverse), predicted_tie_pairs=int(ties),
        true_comparable_pairs=int(comparable), pair_accuracy=correct/comparable if comparable else None)


def summarize(rows):
    return dict(state_evaluations=len(rows), unique_states=len({row['group_id'] for row in rows}),
        mean_predicted_chosen_delta=mean(row['predicted_delta'] for row in rows),
        mean_actual_chosen_delta=mean(row['actual_delta'] for row in rows),
        mean_actual_minus_predicted=mean(row['actual_delta']-row['predicted_delta'] for row in rows),
        mean_nonpp_delta_mae=mean(row['mean_nonpp_delta_mae'] for row in rows),
        mean_selected_regret=mean(row['selected_regret'] for row in rows),
        mean_state_pair_accuracy=mean(row['pair_accuracy'] for row in rows if row['pair_accuracy'] is not None),
        overrides=sum(row['selected_action'] != row['pp_index'] for row in rows),
        harmful=sum(row['actual_delta'] > 1e-12 for row in rows),
        beneficial=sum(row['actual_delta'] < -1e-12 for row in rows),
        observed_best_chosen=sum(row['selected_is_observed_best'] for row in rows),
        pair_counts={key: sum(row[key] for row in rows) for key in
                     ('correct_pairs', 'reversed_pairs', 'predicted_tie_pairs', 'true_comparable_pairs')})


def block_scores(features, model):
    return {name: (features[:, part]/model.scale[part])@model.coef[part]
            for name, part in BLOCKS.items()}


def feature_support(train, held, model, names):
    ztrain, zheld = train/model.scale, held/model.scale
    zero = np.all(train == 0, axis=0)
    new_active = zero & np.any(held != 0, axis=0)
    outside = (held < train.min(axis=0)-1e-12) | (held > train.max(axis=0)+1e-12)
    singular = np.linalg.svd(ztrain/np.sqrt(14), compute_uv=False)
    maximum_train = np.max(np.abs(ztrain), axis=0)
    maximum_held = np.max(np.abs(zheld), axis=0)
    top = np.argsort(maximum_held)[::-1][:12]
    return dict(training_rows=len(train), held_rows=len(held), columns=train.shape[1],
        weighted_design_rank=int(np.sum(singular > singular[0]*max(ztrain.shape)*np.finfo(float).eps)),
        ridge_effective_df=float(np.sum(singular**2/(singular**2+1))),
        training_zero_columns=int(zero.sum()), held_activated_zero_training_columns=int(new_active.sum()),
        held_activated_zero_training_names=[names[index] for index in np.flatnonzero(new_active)],
        maximum_abs_coefficient_on_training_zero_columns=float(np.abs(model.coef[zero]).max()) if zero.any() else 0.,
        held_outside_training_range_coordinates=int(outside.sum()),
        held_outside_training_range_fraction=float(outside.mean()),
        blocks={name: dict(training_row_norm=quantiles(np.linalg.norm(ztrain[:, part], axis=1)),
            held_row_norm=quantiles(np.linalg.norm(zheld[:, part], axis=1)),
            training_max_abs_coordinate=float(np.abs(ztrain[:, part]).max()),
            held_max_abs_coordinate=float(np.abs(zheld[:, part]).max()),
            held_rows_above_training_max_norm=int(np.sum(np.linalg.norm(zheld[:, part], axis=1) >
                                                         np.linalg.norm(ztrain[:, part], axis=1).max()+1e-12)))
                for name, part in BLOCKS.items()},
        largest_held_standardized_coordinates=[dict(feature=names[index], scale=float(model.scale[index]),
            maximum_abs_training=float(maximum_train[index]), maximum_abs_held=float(maximum_held[index]),
            coefficient=float(model.coef[index])) for index in top])


def run(study_base, readout_base, output):
    study_base, readout_base = Path(study_base), Path(readout_base)
    sys.path.insert(0, str(study_base))
    import numeric_projection as projection
    from prior_ridge import DynamicsModel, Scaler, ReaderModel, predict, reader_predict, select_action

    result = json.loads((readout_base/'fit/result.json').read_text())
    old_result = json.loads((study_base/'nested_fit/result.json').read_text())
    dataset = json.loads((study_base/'training_dataset_server_only.json').read_text())
    groups = {}
    for raw in dataset['groups']:
        if raw['city'] != 'jinan':
            continue
        branches = sorted(raw['branches'], key=lambda branch: branch['candidate_index'])
        inputs = [branch['inputs'] for branch in branches]
        absolute = np.stack([projection.project_reader(data, projection.project_predicted_state(
            data, np.asarray(branch['targets']['y']))) for data, branch in zip(inputs, branches)])
        identity = f"jinan/seed{raw['seed']}/anchor{raw['anchor']}"
        groups[identity] = dict(id=identity, seed=raw['seed'], anchor=raw['anchor'], pp=raw['pp_index'],
            inputs=inputs, H=absolute, F=absolute-absolute[raw['pp_index']],
            costs=np.asarray([branch['cost'] for branch in branches]))
    if len(groups) != 20 or len(result['fit_ledger']) != 5:
        raise ValueError('This audit requires the complete frozen twenty-state/five-reader result')
    old_entries = {(entry['outer_fold'], entry['role']): entry for entry in old_result['fit_ledger']
                   if entry['arm'] == 'T' and entry['role'] in ('target_cost_reader', 'outer_target_dynamics')}
    old_records = {row['group_id']: row for row in old_result['records'] if row['arm'] == 'T'}
    new_records = {row['group_id']: row for row in result['records']}
    training_all, held_all, folds, held_details = [], [], [], []
    maximum_gradient = maximum_rms_error = maximum_score_error = maximum_old_score_error = 0.
    for entry in result['fit_ledger']:
        outer = entry['outer_fold']
        with np.load(readout_base/'fit'/entry['model_file']) as arrays:
            model = ReaderModel(arrays['scale'], arrays['coef'])
        with np.load(study_base/'nested_fit'/old_entries[outer, 'target_cost_reader']['model_file']) as arrays:
            old_reader = ReaderModel(arrays['scale'], arrays['coef'])
        with np.load(study_base/'nested_fit'/old_entries[outer, 'outer_target_dynamics']['model_file']) as arrays:
            old_dynamics = DynamicsModel(Scaler(arrays['mean'], arrays['scale']), arrays['coef'],
                                         arrays['intercept'], bool(arrays['single_output'].item()))
        train = [group for group in groups.values() if group['seed'] in entry['training_seeds']]
        held = [group for group in groups.values() if group['seed'] in entry['held_out_seeds']]
        design, labels, held_design = [], [], []
        train_metrics, held_metrics = [], []
        train_blocks = {name: [] for name in BLOCKS}
        held_blocks = {name: [] for name in BLOCKS}
        selected_blocks = {name: [] for name in BLOCKS}
        for group in train:
            nonpp = [a for a in range(8) if a != group['pp']]
            design.append(group['F'][nonpp]); labels.extend((group['costs'][nonpp]-group['costs'][group['pp']]).tolist())
            scores = reader_predict(model, group['F'])
            train_metrics.append(state_metrics(group, scores, select_action))
            for name, values in block_scores(group['F'], model).items():
                train_blocks[name].extend(values[nonpp].tolist())
        design, labels = np.concatenate(design), np.asarray(labels)
        weights = np.full(len(labels), 1/14)
        rms = np.sqrt((weights[:, None]*design**2).sum(axis=0)/weights.sum()); rms[rms == 0] = 1.
        z = design/model.scale
        gradient = z.T@(weights*(z@model.coef-labels))+model.coef
        maximum_gradient = max(maximum_gradient, float(np.abs(gradient).max()))
        maximum_rms_error = max(maximum_rms_error, float(np.abs(rms-model.scale).max()))
        for group in held:
            nonpp = [a for a in range(8) if a != group['pp']]
            held_design.append(group['F'][nonpp])
            scores = reader_predict(model, group['F'])
            saved = new_records[group['id']]
            maximum_score_error = max(maximum_score_error, float(np.abs(scores-saved['predicted_delta_costs']).max()))
            metric = state_metrics(group, scores, select_action)
            if metric['selected_action'] != saved['selected_action']:
                raise ValueError('Rebuilt measured-response selection differs from the frozen result')
            held_metrics.append(metric)
            X = np.stack([projection.project_dynamics(inputs)[1] for inputs in group['inputs']])
            response = predict(old_dynamics, X.reshape(-1, 476)).reshape(8, 12, 4)
            old_absolute = np.stack([projection.project_reader(inputs, projection.project_predicted_state(inputs, delta))
                                     for inputs, delta in zip(group['inputs'], response)])
            old_F = old_absolute-old_absolute[group['pp']]
            if not np.array_equal(old_F[:, :195], group['F'][:, :195]):
                raise ValueError('The nominal and t0 interaction features changed in the measured-response recipe')
            old_scores = reader_predict(old_reader, old_F)
            old_saved = old_records[group['id']]
            maximum_old_score_error = max(maximum_old_score_error,
                float(np.abs(old_scores-old_saved['predicted_delta_costs']).max()))
            components, old_components = block_scores(group['F'], model), block_scores(old_F, old_reader)
            if not np.allclose(sum(components.values()), scores, rtol=0, atol=1e-12):
                raise ValueError('Three reader components do not reproduce the complete cost score')
            selected = metric['selected_action']
            runner_up = next(index for index in np.argsort(np.round(scores, 12), kind='stable') if index != selected)
            observed_best = int(np.argmin(group['costs']))
            outside = (group['F'][selected] < design.min(axis=0)-1e-12) | (group['F'][selected] > design.max(axis=0)+1e-12)
            old_selected = old_saved['selected_action']
            detail = dict(metric, old_T_selected_action=old_selected,
                old_T_actual_delta=float(group['costs'][old_selected]-group['costs'][group['pp']]),
                old_T_predicted_delta=float(old_scores[old_selected]),
                old_T_components_at_old_selected={name: float(values[old_selected]) for name, values in old_components.items()},
                components_at_measured_selected={name: float(values[selected]) for name, values in components.items()},
                measured_minus_old_scores_at_measured_selected={name: float(components[name][selected]-old_components[name][selected])
                                                               for name in BLOCKS},
                score_margin_to_runner_up=float(scores[runner_up]-scores[selected]),
                component_advantage_over_runner_up={name: float(values[runner_up]-values[selected]) for name, values in components.items()},
                components_vs_observed_best={name: float(values[selected]-values[observed_best]) for name, values in components.items()},
                selected_standardized_norm=float(np.linalg.norm(group['F'][selected]/model.scale)),
                selected_outside_range_coordinates=int(outside.sum()),
                selected_outside_range_score=float(np.sum((group['F'][selected]/model.scale*model.coef)[outside])),
                selected_largest_standardized_coordinates=[dict(
                    feature=projection.READER_FEATURE_NAMES[index],
                    measured_absolute_PP=float(group['H'][group['pp'], index]),
                    measured_absolute_selected=float(group['H'][selected, index]),
                    measured_PP_difference=float(group['F'][selected, index]),
                    training_minimum=float(design[:, index].min()), training_maximum=float(design[:, index].max()),
                    training_nonzero_rows=int(np.sum(np.abs(design[:, index]) > 1e-12)),
                    RMS_scale=float(model.scale[index]), standardized_value=float(group['F'][selected, index]/model.scale[index]),
                    score_contribution=float(group['F'][selected, index]/model.scale[index]*model.coef[index]),
                    old_T_predicted_absolute_PP=float(old_absolute[group['pp'], index]),
                    old_T_predicted_absolute_selected=float(old_absolute[selected, index]),
                    old_T_training_RMS=float(old_reader.scale[index]),
                    old_T_score_contribution=float(old_F[selected, index]/old_reader.scale[index]*old_reader.coef[index]))
                    for index in np.argsort(np.abs(group['F'][selected]/model.scale))[::-1][:3]])
            held_details.append(detail)
            for name, values in components.items():
                held_blocks[name].extend(values[nonpp].tolist()); selected_blocks[name].append(float(values[selected]))
        held_design = np.concatenate(held_design)
        held_labels = np.concatenate([group['costs'][[a for a in range(8) if a != group['pp']]]-group['costs'][group['pp']]
                                      for group in held])
        folds.append(dict(outer_fold=outer, training_seeds=entry['training_seeds'], held_seeds=entry['held_out_seeds'],
            training=summarize(train_metrics), held_out=summarize(held_metrics),
            input_support=feature_support(design, held_design, model, list(projection.READER_FEATURE_NAMES)),
            components={name: dict(training_nonpp_score=quantiles(train_blocks[name]),
                held_nonpp_score=quantiles(held_blocks[name]), mean_score_at_held_selected=mean(selected_blocks[name]),
                training_score_actual_delta_correlation=correlation(train_blocks[name], labels),
                held_score_actual_delta_correlation=correlation(held_blocks[name], held_labels),
                coefficient_norm=float(np.linalg.norm(model.coef[part])),
                training_RMS=quantiles(model.scale[part]), old_T_training_RMS=quantiles(old_reader.scale[part]))
                for name, part in BLOCKS.items()}))
        training_all.extend(train_metrics); held_all.extend(held_metrics)
    if max(maximum_gradient, maximum_rms_error, maximum_score_error, maximum_old_score_error) > 1e-9:
        raise ValueError('Saved models, weighted training rows or frozen scores could not be reconstructed')
    most_extreme = max(held_details, key=lambda row: row['selected_standardized_norm'])
    other_states = [row for row in held_details if row['group_id'] != most_extreme['group_id']]
    compact = dict(status='ZERO_FIT_READER_POSTMORTEM', new_fits=0, new_native_starts=0,
        maximum_reader_normal_equation_residual=maximum_gradient, maximum_RMS_scale_difference=maximum_rms_error,
        maximum_measured_score_reconstruction_error=maximum_score_error, maximum_old_T_score_reconstruction_error=maximum_old_score_error,
        first_195_columns_identical_to_old_recipe=True,
        aggregate=dict(reader_training=summarize(training_all), outer_held=summarize(held_all)),
        chosen_component_means={name: mean(row['components_at_measured_selected'][name] for row in held_details) for name in BLOCKS},
        old_T_chosen_component_means={name: mean(row['old_T_components_at_old_selected'][name] for row in held_details) for name in BLOCKS},
        measured_vs_old_T=dict(changed_choices=sum(row['selected_action'] != row['old_T_selected_action'] for row in held_details),
            mean_actual_cost_change=mean(row['actual_delta']-row['old_T_actual_delta'] for row in held_details)),
        selected_support_diagnostic=dict(harmful_selected_inside_all_training_column_ranges=sum(
                row['actual_delta'] > 1e-12 and row['selected_outside_range_coordinates'] == 0 for row in held_details),
            most_extreme_selected_group=most_extreme['group_id'],
            remaining_nineteen_descriptive_only=summarize(other_states)),
        by_outer_fold=folds, held_out=sorted(held_details, key=lambda row: (row['seed'], row['anchor'])),
        interpretation='Training costs are reader-in-sample; eighty training state evaluations repeat twenty states. '
            'Ranges/norms, block contributions, and rank summaries are descriptive, not independent tests or causal ablations. '
            'The measured ten-second response is a future diagnostic input; no new policy, fit, threshold, or simulation is evaluated.')
    Path(output).write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=compact['status'], aggregate=compact['aggregate'],
        chosen_component_means=compact['chosen_component_means'], measured_vs_old_T=compact['measured_vs_old_T']), allow_nan=False))
    return compact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-base', type=Path, required=True)
    parser.add_argument('--readout-base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.study_base, args.readout_base, args.output)
