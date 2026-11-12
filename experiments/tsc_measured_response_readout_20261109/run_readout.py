"""Five fixed whole-seed Ridge readouts using measured ten-second responses.

This is a diagnostic with future response labels in both training and held-out
features. It is neither a decision-time policy nor a source-transfer estimate.
"""
import argparse
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
T95_DF9 = 2.2621571627409915


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-base', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


# The command uses the original server study's code; tests use its local sibling.
CLI_ARGS = arguments() if __name__ == '__main__' else None
STUDY = CLI_ARGS.study_base if CLI_ARGS is not None else HERE.parent/'tsc_response_cost_readout_20261109'
sys.path.insert(0, str(STUDY))
import numeric_projection as projection
from lane_interface import TARGET_COLUMNS
from prior_ridge import fit_reader, reader_predict, select_action


@dataclass
class Group:
    city: str
    seed: int
    anchor: int
    pp: int
    features: np.ndarray
    costs: np.ndarray

    @property
    def identity(self):
        return f'{self.city}/seed{self.seed}/anchor{self.anchor}'


def _design(protocol):
    folds = [list(fold) for fold in protocol['outer_folds']]
    seeds = list(protocol['target_seeds'])
    anchors = list(protocol['anchors'])
    if (protocol['target_city'] != 'jinan' or len(folds) != 5 or
            any(len(fold) != 2 for fold in folds) or len(seeds) != 10 or
            len(set(seeds)) != 10 or sorted(sum(folds, [])) != sorted(seeds) or
            anchors != [600, 1800] or protocol['reader_columns'] != 211 or
            protocol['new_logical_fits'] != 5 or protocol['source_data_used'] is not False or
            protocol['new_dynamics_fits'] != 0 or protocol['new_source_fits'] != 0 or
            protocol['new_native_starts'] != 0):
        raise ValueError('The frozen Jinan ten-seed, five-fit measured-response design is required')
    return folds, seeds, anchors


def prepare(dataset, protocol):
    """Project only Jinan groups; no source branch or source cost is accessed."""
    unused_folds, seeds, anchors = _design(protocol)
    rows = [row for row in dataset['groups'] if row['city'] == 'jinan']
    expected = {(seed, anchor) for seed in seeds for anchor in anchors}
    roster = [(row['seed'], row['anchor']) for row in rows]
    if len(roster) != 20 or set(roster) != expected:
        raise ValueError('All twenty frozen target states are required exactly once')
    if len(projection.READER_FEATURE_NAMES) != 211:
        raise ValueError('The original 211-column reader projection must remain unchanged')
    groups = []
    for row in sorted(rows, key=lambda item: (item['seed'], item['anchor'])):
        branches = sorted(row['branches'], key=lambda branch: branch['candidate_index'])
        pp = row['pp_index']
        if [branch['candidate_index'] for branch in branches] != list(range(8)) or pp not in range(8):
            raise ValueError('Each target state needs eight canonical actions and one PP index')
        absolute, shared_lanes, shared_g0 = [], None, None
        for branch in branches:
            inputs, target = branch['inputs'], branch['targets']
            lanes = [node['lane_id'] for node in inputs['lane_nodes']]
            response = np.asarray(target['y'], dtype=float)
            g0 = projection._reader_g0(inputs)
            if (len(lanes) != 12 or len(set(lanes)) != 12 or
                    (shared_lanes is not None and lanes != shared_lanes) or
                    target['lane_ids'] != lanes or tuple(target['columns']) != tuple(TARGET_COLUMNS) or
                    response.shape != (12, 4) or not np.isfinite(response).all()):
                raise ValueError('Measured responses must align with the same twelve lanes and four outputs')
            if shared_g0 is not None and not np.array_equal(g0, shared_g0):
                raise ValueError('The t0 global/focal/receiving state changed across native actions')
            shared_lanes, shared_g0 = lanes, g0
            state = projection.project_predicted_state(inputs, response)
            absolute.append(projection.project_reader(inputs, state))
        absolute = np.asarray(absolute, dtype=float)
        if absolute.shape != (8, 211) or not np.isfinite(absolute).all():
            raise ValueError('The measured-response reader requires eight finite 211-column rows')
        features = absolute-absolute[pp]
        if not np.array_equal(features[pp], np.zeros(211)):
            raise ValueError('PP features must be exact structural zero')
        costs = np.asarray([branch['cost'] for branch in branches], dtype=float)
        if not np.isfinite(costs).all() or np.any(costs < 0):
            raise ValueError('Every target action requires its finite nonnegative native 450-second cost')
        groups.append(Group('jinan', row['seed'], row['anchor'], pp, features, costs))
    return groups


def _reader_rows(groups):
    features, labels, weights = [], [], []
    for group in groups:
        actions = [index for index in range(8) if index != group.pp]
        features.append(group.features[actions])
        labels.extend((group.costs[actions]-group.costs[group.pp]).tolist())
        weights.extend([1/14]*7)  # Equal two states and seven non-PP actions per seed.
    return np.concatenate(features), np.asarray(labels), np.asarray(weights)


def paired_summary(selected, pp, seeds):
    differences = np.asarray([selected[seed]-pp[seed] for seed in seeds], dtype=float)
    average = float(differences.mean())
    radius = T95_DF9*float(differences.std(ddof=1))/sqrt(10)
    return dict(mean_difference=average, approximate_paired_seed_95_t=[average-radius, average+radius],
                df=9, per_seed=[dict(seed=seed, difference=float(value))
                                for seed, value in zip(seeds, differences)])


def run(dataset, protocol, on_fit=None):
    """Fit exactly one target-only reader per locked outer fold, with no retry."""
    folds, seeds, unused_anchors = _design(protocol)
    groups = prepare(dataset, protocol)
    ledger, records = [], []
    for outer, held in enumerate(folds):
        training_seeds = sorted(set(seeds)-set(held))
        training = [group for group in groups if group.seed in training_seeds]
        held_groups = [group for group in groups if group.seed in held]
        features, delta, weights = _reader_rows(training)
        entry = dict(fit_index=len(ledger)+1, role='measured_response_target_cost_reader',
                     outer_fold=outer, training_seeds=training_seeds, held_out_seeds=held,
                     rows=len(features), columns=211, weight_mass=float(weights.sum()),
                     alpha=1, scaling='training-only weighted RMS; no centering', intercept=False,
                     measured_responses=True, status='STARTED')
        ledger.append(entry)
        if on_fit is not None:
            on_fit(ledger, None)
        try:
            model = fit_reader(features, delta, weights, alpha=1)
            entry['status'] = 'COMPLETE'
            if on_fit is not None:
                on_fit(ledger, model)
        except Exception as exc:
            entry.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            if on_fit is not None:
                on_fit(ledger, None)
            raise
        # Fix every held-out action before evaluating any held-out cost.
        choices = []
        for group in held_groups:
            scores = reader_predict(model, group.features)
            if scores.shape != (8,) or not np.isfinite(scores).all() or scores[group.pp] != 0:
                raise ValueError('The measured-response reader changed the finite action roster or PP zero')
            choices.append((group, scores, select_action(scores, group.pp)))
        for group, scores, selected in choices:
            actual_minimum = float(group.costs.min())
            records.append(dict(group_id=group.identity, seed=group.seed, anchor=group.anchor,
                outer_fold=outer, selected_action=selected, pp_index=group.pp,
                predicted_delta_costs=scores.tolist(), selected_cost=float(group.costs[selected]),
                phase_pressure_cost=float(group.costs[group.pp]),
                cost_minus_pp=float(group.costs[selected]-group.costs[group.pp]),
                observed_minimum_cost=actual_minimum,
                selected_cost_minus_observed_minimum=float(group.costs[selected]-actual_minimum),
                reader_training_seeds=training_seeds, held_out_seeds=held))
    if len(ledger) != 5 or len(records) != 20 or any(entry['status'] != 'COMPLETE' for entry in ledger):
        raise ValueError('The complete five-fit/twenty-state diagnostic roster changed')
    selected = {seed: float(np.mean([row['selected_cost'] for row in records if row['seed'] == seed]))
                for seed in seeds}
    pp = {seed: float(np.mean([row['phase_pressure_cost'] for row in records if row['seed'] == seed]))
          for seed in seeds}
    regret = {seed: float(np.mean([row['selected_cost_minus_observed_minimum'] for row in records
                                  if row['seed'] == seed])) for seed in seeds}
    comparison = paired_summary(selected, pp, seeds)
    passed = comparison['approximate_paired_seed_95_t'][1] < 0
    return dict(status='DIAGNOSTIC_READABILITY_SIGNAL' if passed else 'HOLD_MEASURED_RESPONSE_READOUT',
        protocol=protocol, logical_fits=5, fit_ledger=ledger, records=records,
        mean_costs=dict(measured_response_reader=float(np.mean(list(selected.values()))),
                        PP=float(np.mean(list(pp.values())))),
        per_seed_costs=dict(measured_response_reader={str(seed): value for seed, value in selected.items()},
                            PP={str(seed): value for seed, value in pp.items()}),
        comparison_measured_minus_PP=comparison, gate_passed=passed,
        mean_selected_regret=float(np.mean(list(regret.values()))),
        per_seed_selected_regret={str(seed): value for seed, value in regret.items()},
        behavior=dict(overrides=sum(row['selected_action'] != row['pp_index'] for row in records),
            harmful_overrides=sum(row['cost_minus_pp'] > 0 for row in records),
            beneficial_overrides=sum(row['cost_minus_pp'] < 0 for row in records),
            harmful_cost_sum=sum(max(row['cost_minus_pp'], 0) for row in records),
            beneficial_cost_sum=sum(min(row['cost_minus_pp'], 0) for row in records)),
        uses_measured_held_out_responses=True, deployable_policy=False,
        new_dynamics_fits=0, new_source_fits=0, new_simulations=0, source_450_costs_used=False,
        independent_confirmation=0, original_hold_unchanged=True)


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)

    def persist(ledger, model):
        if model is not None:
            name = f'fit_{ledger[-1]["fit_index"]:03d}_server_only.npz'
            np.savez_compressed(args.output/name, scale=model.scale, coef=model.coef)
            ledger[-1]['model_file'] = name
        (args.output/'fit_ledger.json').write_text(json.dumps(ledger, indent=2, allow_nan=False)+'\n')

    try:
        protocol = json.loads(args.protocol.read_text())
        dataset = json.loads((args.study_base/'training_dataset_server_only.json').read_text())
        result = run(dataset, protocol, on_fit=persist)
    except Exception as exc:
        (args.output/'failure.json').write_text(json.dumps(dict(
            status='HOLD_MEASURED_RESPONSE_READOUT_ERROR', error=f'{type(exc).__name__}: {exc}'))+'\n')
        raise
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'])


if __name__ == '__main__':
    main(CLI_ARGS)
