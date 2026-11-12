"""Zero-fit reconstruction of frozen cost-reader training and held-out rankings."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np


def _load(path, dynamics, classes):
    DynamicsModel, Scaler, ReaderModel = classes
    with np.load(path) as values:
        if dynamics:
            return DynamicsModel(Scaler(values['mean'], values['scale']), values['coef'], values['intercept'],
                                 bool(values['single_output'].item()))
        return ReaderModel(values['scale'], values['coef'])


def _quantiles(values):
    return dict(zip(('minimum', 'q25', 'median', 'q75', 'maximum'),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1])))) if values else None


def _ranks(values):
    order = np.argsort(values, kind='stable')
    ranks = np.zeros(len(values))
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and abs(values[order[end]]-values[order[start]]) <= 1e-12:
            end += 1
        ranks[order[start:end]] = (start+end-1)/2
        start = end
    return ranks


def _same_cells(groups):
    cells = {}
    for group in groups.values():
        cells.setdefault((group['tls'], group['anchor']), []).append(group)
    rows = []
    for (tls, anchor), matches in sorted(cells.items()):
        if len(matches) < 2:
            continue
        a, b = sorted(matches, key=lambda group: group['seed'])
        phase_equal = set(a['phases']) == set(b['phases'])
        if not phase_equal:
            rows.append(dict(tls=tls, anchor=anchor, seeds=[a['seed'], b['seed']], phase_roster_equal=False))
            continue
        phases = sorted(a['phases'])
        costs = [{phase: float(value) for phase, value in zip(group['phases'], group['costs'])} for group in (a, b)]
        ranks = [_ranks(np.array([values[phase] for phase in phases])) for values in costs]
        best = [sorted(phase for phase, cost in values.items() if abs(cost-min(values.values())) <= 1e-12) for values in costs]
        comparable = reversed_count = 0
        for i, phase in enumerate(phases):
            for other in phases[i+1:]:
                da, db = (values[phase]-values[other] for values in costs)
                if abs(da) > 1e-12 and abs(db) > 1e-12:
                    comparable += 1
                    reversed_count += da*db < 0
        rows.append(dict(tls=tls, anchor=anchor, seeds=[a['seed'], b['seed']], phase_roster_equal=True,
            cost_rank_spearman=float(np.corrcoef(ranks)[0, 1]) if all(values.std() > 0 for values in ranks) else None,
            observed_best_phases=best, best_phase_overlap=bool(set(best[0]) & set(best[1])),
            same_best_phase_set=best[0] == best[1], comparable_phase_pairs=comparable,
            reversed_phase_pairs=int(reversed_count), PP_phase_equal=a['phases'][a['pp']] == b['phases'][b['pp']]))
    return rows


def _state(group, scores, select):
    costs, pp = group['costs'], group['pp']
    rounded = np.round(scores, 12)
    chosen = select(scores, pp)
    different, correct, reversed_count, predicted_ties, cost_ties = 0, 0, 0, 0, 0
    for a in range(8):
        for b in range(a+1, 8):
            actual = costs[a]-costs[b]
            predicted = rounded[a]-rounded[b]
            if abs(actual) <= 1e-12:
                cost_ties += 1
            else:
                different += 1
                if predicted == 0:
                    predicted_ties += 1
                elif actual*predicted > 0:
                    correct += 1
                else:
                    reversed_count += 1
    other = [index for index in range(8) if index != pp]
    delta = costs-costs[pp]
    ordered = np.sort(rounded)
    return dict(group_id=group['id'], seed=group['seed'], anchor=group['anchor'], tls=group['tls'],
        selected_action=chosen, pp_index=pp, predicted_delta=float(scores[chosen]),
        actual_delta=float(delta[chosen]), selected_regret=float(costs[chosen]-costs.min()),
        PP_regret=float(costs[pp]-costs.min()), winner_margin=float(ordered[1]-ordered[0]),
        selected_is_observed_best=bool(abs(costs[chosen]-costs.min()) <= 1e-12),
        nonPP_delta_mae=float(np.abs(scores[other]-delta[other]).mean()),
        nonPP_delta_mse=float(np.square(scores[other]-delta[other]).mean()),
        true_comparable_pairs=different, correctly_ranked_pairs=correct, reversed_pairs=reversed_count,
        predicted_tie_pairs=predicted_ties, actual_tie_pairs=cost_ties,
        pair_accuracy=correct/different if different else None,
        predicted_benefit_nonPP=int(sum(rounded[index] < 0 for index in other)),
        harmful_predicted_benefit_nonPP=int(sum(rounded[index] < 0 and delta[index] > 1e-12 for index in other)))


def _summary(rows):
    harmful = [row for row in rows if row['actual_delta'] > 1e-12]
    beneficial = [row for row in rows if row['actual_delta'] < -1e-12]
    return dict(state_evaluations=len(rows), unique_states=len({row['group_id'] for row in rows}),
        mean_predicted_chosen_delta=mean(row['predicted_delta'] for row in rows),
        mean_actual_chosen_delta=mean(row['actual_delta'] for row in rows),
        mean_actual_minus_predicted=mean(row['actual_delta']-row['predicted_delta'] for row in rows),
        overrides=sum(row['selected_action'] != row['pp_index'] for row in rows),
        harmful=len(harmful), beneficial=len(beneficial),
        observed_best_chosen=sum(row['selected_is_observed_best'] for row in rows),
        mean_selected_regret=mean(row['selected_regret'] for row in rows),
        mean_nonPP_delta_mae=mean(row['nonPP_delta_mae'] for row in rows),
        mean_nonPP_delta_mse=mean(row['nonPP_delta_mse'] for row in rows),
        mean_state_pair_accuracy=mean(row['pair_accuracy'] for row in rows if row['pair_accuracy'] is not None),
        descriptive_pair_counts={key: sum(row[key] for row in rows) for key in
            ('true_comparable_pairs', 'correctly_ranked_pairs', 'reversed_pairs', 'predicted_tie_pairs', 'actual_tie_pairs')},
        descriptive_candidate_counts={key: sum(row[key] for row in rows) for key in
            ('predicted_benefit_nonPP', 'harmful_predicted_benefit_nonPP')},
        winner_margin=_quantiles([row['winner_margin'] for row in rows]),
        harmful_winner_margin=_quantiles([row['winner_margin'] for row in harmful]),
        beneficial_winner_margin=_quantiles([row['winner_margin'] for row in beneficial]),
        chosen_error=_quantiles([row['actual_delta']-row['predicted_delta'] for row in rows]),
        selected_regret=_quantiles([row['selected_regret'] for row in rows]))


def run(base, output):
    base = Path(base)
    sys.path.insert(0, str(base))
    import numeric_projection as projection
    from prior_ridge import DynamicsModel, Scaler, ReaderModel, predict, reader_predict, select_action
    result = json.loads((base/'nested_fit/result.json').read_text())
    dataset = json.loads((base/'training_dataset_server_only.json').read_text())
    groups = {}
    for raw in dataset['groups']:
        if raw['city'] != 'jinan':
            continue
        branches = sorted(raw['branches'], key=lambda row: row['candidate_index'])
        identity = f"jinan/seed{raw['seed']}/anchor{raw['anchor']}"
        groups[identity] = dict(id=identity, seed=raw['seed'], anchor=raw['anchor'], tls=raw['case']['tls_id'],
            pp=raw['pp_index'], costs=np.asarray([row['cost'] for row in branches]),
            phases=[row['inputs']['nominal_executor10']['snapshot']['target_green_state'] for row in branches],
            inputs=[row['inputs'] for row in branches],
            X=np.stack([projection.project_dynamics(row['inputs'])[1] for row in branches]))
    models, inner_for_seed = {}, {}
    classes = DynamicsModel, Scaler, ReaderModel
    for entry in result['fit_ledger']:
        if entry['role'] not in ('inner_target_dynamics', 'target_cost_reader'):
            continue
        key = entry['outer_fold'], entry['arm'], entry['role'], entry['inner_fold']
        models[key] = _load(base/'nested_fit'/entry['model_file'], entry['role'] == 'inner_target_dynamics', classes)
        if entry['role'] == 'inner_target_dynamics':
            for seed in entry['held_out_seeds']:
                inner_for_seed[entry['outer_fold'], entry['arm'], seed] = entry['inner_fold']
    held_rows = {arm: [] for arm in ('T', 'S', 'P')}
    fold_rows, all_training = [], {arm: [] for arm in held_rows}
    gradients, rms_differences = [], []
    for outer, held_seeds in enumerate(result['protocol']['nested_validation']['outer']):
        training = [group for group in groups.values() if group['seed'] not in held_seeds]
        for arm in held_rows:
            reader = models[outer, arm, 'target_cost_reader', None]
            training_metrics, design, labels = [], [], []
            for group in training:
                inner = inner_for_seed[outer, arm, group['seed']]
                dynamics = models[outer, arm, 'inner_target_dynamics', inner]
                delta = predict(dynamics, group['X'].reshape(-1, group['X'].shape[-1])).reshape(8, 12, 4)
                absolute = np.stack([projection.project_reader(inputs, projection.project_predicted_state(inputs, prediction))
                                     for inputs, prediction in zip(group['inputs'], delta)])
                F = absolute-absolute[group['pp']]
                scores = reader_predict(reader, F)
                training_metrics.append(_state(group, scores, select_action))
                nonPP = [index for index in range(8) if index != group['pp']]
                design.append(F[nonPP]); labels.extend((group['costs'][nonPP]-group['costs'][group['pp']]).tolist())
            F, y = np.concatenate(design), np.asarray(labels)
            weights = np.full(len(y), 1/(2*7))
            rms = np.sqrt((weights[:, None]*F**2).sum(axis=0)/weights.sum())
            rms[rms == 0] = 1
            Z = F/reader.scale
            gradient = Z.T@(weights*(Z@reader.coef-y))+reader.coef
            gradients.append(float(np.abs(gradient).max()))
            rms_differences.append(float(np.abs(rms-reader.scale).max()))
            if gradients[-1] > 1e-9 or rms_differences[-1] > 1e-12:
                raise ValueError('The saved reader does not match reconstructed inner-OOF training rows')
            held_metrics = []
            for record in result['records']:
                if record['outer_fold'] == outer and record['arm'] == arm:
                    row = _state(groups[record['group_id']], np.asarray(record['predicted_delta_costs']), select_action)
                    if row['selected_action'] != record['selected_action']:
                        raise ValueError('Saved outer choice and frozen selection rule differ')
                    held_metrics.append(row)
            held_rows[arm].extend(held_metrics); all_training[arm].extend(training_metrics)
            fold_rows.append(dict(outer_fold=outer, arm=arm, held_seeds=held_seeds,
                training=_summary(training_metrics), held_out=_summary(held_metrics)))
    truth = []
    for group in groups.values():
        ordered = np.sort(group['costs'])
        truth.append(dict(group_id=group['id'], seed=group['seed'], anchor=group['anchor'], tls=group['tls'],
            pp_index=group['pp'], costs=group['costs'].tolist(), observed_best_minus_PP=float(ordered[0]-group['costs'][group['pp']]),
            true_action_range=float(ordered[-1]-ordered[0]), best_second_gap=float(ordered[1]-ordered[0])))
    seeds = sorted({row['seed'] for row in truth})
    within_seed = [dict(seed=seed, states=[dict(row, T_choice=next(item for item in held_rows['T'] if item['group_id'] == row['group_id']))
                  for row in sorted(truth, key=lambda item: item['anchor']) if row['seed'] == seed]) for seed in seeds]
    compact = dict(status='DESCRIPTIVE_ZERO_FIT', new_fits=0, new_simulations=0,
        saved_readers_match_inner_OOF_rows=True, maximum_reader_normal_equation_residual=max(gradients),
        maximum_reader_RMS_scale_difference=max(rms_differences),
        semantics='Training rows have inner-OOF dynamics but in-sample cost-reader labels; they are not held-out cost validation. Repeated training states/pairs are descriptive, not independent units.',
        by_outer_fold=fold_rows,
        aggregate={arm: dict(reader_training=_summary(all_training[arm]), outer_held=_summary(held_rows[arm])) for arm in held_rows},
        actual_cost_geometry=dict(observed_best_minus_PP=_quantiles([row['observed_best_minus_PP'] for row in truth]),
            true_action_range=_quantiles([row['true_action_range'] for row in truth]),
            best_second_gap=_quantiles([row['best_second_gap'] for row in truth])),
        by_anchor={str(anchor): dict(truth_states=[row for row in truth if row['anchor'] == anchor],
            T_held_out=_summary([row for row in held_rows['T'] if row['anchor'] == anchor])) for anchor in (600, 1800)},
        within_seed=within_seed, same_cell_across_seeds=_same_cells(groups),
        held_out={arm: sorted(rows, key=lambda row: (row['seed'], row['anchor'])) for arm, rows in held_rows.items()},
        anchor_interpretation='Anchor and assigned focal TLS change together within each seed; these effects are not separately identified.')
    Path(output).write_text(json.dumps(compact, indent=2, allow_nan=False)+'\n')
    return compact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('cost_sorting_result.json'))
    args = parser.parse_args()
    result = run(args.base, args.output)
    print(json.dumps(dict(status=result['status'], reconstruction_residual=result['maximum_reader_normal_equation_residual'],
        aggregate=result['aggregate'], actual_cost_geometry=result['actual_cost_geometry']), allow_nan=False))
