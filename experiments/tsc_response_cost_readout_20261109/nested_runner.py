"""One fixed 92-fit whole-seed source-prior response/cost development run."""
import argparse
from dataclasses import dataclass
import json
from math import sqrt
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import numeric_projection as projection
from lane_interface import TARGET_COLUMNS
from prior_ridge import (DynamicsModel, fit_source, fit_dynamics, predict,
                         fit_reader, reader_predict, select_action)

ARMS = ('T', 'S', 'P')
T95_DF9 = 2.2621571627409915


@dataclass
class Group:
    city: str
    seed: int
    anchor: int
    pp: int
    inputs: list
    lane_ids: list
    X: np.ndarray
    Y: np.ndarray
    costs: object

    @property
    def identity(self):
        return f'{self.city}/seed{self.seed}/anchor{self.anchor}'


def prepare(dataset, protocol):
    budget = protocol['proposed_native_budget']
    if len(budget['source_seeds']) != 4 or len(budget['target_seeds']) != 10 or len(budget['anchors']) != 2:
        raise ValueError('The fixed four-source/ten-target two-state budget is required')
    expected = {(city, seed, anchor) for city, name in (('hangzhou', 'source_seeds'), ('jinan', 'target_seeds'))
                for seed in budget[name] for anchor in budget['anchors']}
    rows = dataset['groups']
    roster = [(row['city'], row['seed'], row['anchor']) for row in rows]
    if len(roster) != len(expected) or set(roster) != expected:
        raise ValueError('The complete frozen source/target seed and anchor roster is required')
    groups = []
    for row in sorted(rows, key=lambda item: (item['city'], item['seed'], item['anchor'])):
        branches = sorted(row['branches'], key=lambda item: item['candidate_index'])
        if [branch['candidate_index'] for branch in branches] != list(range(8)) or row['pp_index'] not in range(8):
            raise ValueError('Every group requires all eight canonical actions and one PP index')
        matrices, targets, inputs, lanes, shared_g0 = [], [], [], None, None
        for branch in branches:
            g0 = projection._reader_g0(branch['inputs'])
            if shared_g0 is not None and not np.array_equal(g0, shared_g0):
                raise ValueError('The t0 reader network/focal/receiving state changed across actions')
            shared_g0 = g0
            lane_ids, X = projection.project_dynamics(branch['inputs'])
            target = branch['targets']
            Y = np.asarray(target['y'], dtype=float)
            if (len(lane_ids) != 12 or len(set(lane_ids)) != 12 or
                    (lanes is not None and lane_ids != lanes) or target['lane_ids'] != lane_ids or
                    tuple(target['columns']) != tuple(TARGET_COLUMNS) or
                    X.shape != (12, len(projection.DYNAMICS_FEATURE_NAMES)) or Y.shape != (12, 4) or
                    not np.isfinite(X).all() or not np.isfinite(Y).all()):
                raise ValueError('Lane, numeric feature or four-output target alignment changed')
            lanes = lane_ids
            matrices.append(X); targets.append(Y); inputs.append(branch['inputs'])
        # Source cost fields are never accessed, including a collector's optional None placeholder.
        costs = np.asarray([branch['cost'] for branch in branches], dtype=float) if row['city'] == 'jinan' else None
        if costs is not None and (not np.isfinite(costs).all() or np.any(costs < 0)):
            raise ValueError('Target branches need all eight finite nonnegative 450-second native costs')
        groups.append(Group(row['city'], row['seed'], row['anchor'], row['pp_index'], inputs,
                            lanes, np.stack(matrices), np.stack(targets), costs))
    return groups


def permute_source(groups, seed=20261109):
    rng, labels, mapping = np.random.default_rng(seed), {}, []
    for group in groups:
        actions = [index for index in range(8) if index != group.pp]
        donors = rng.permutation(actions)
        shuffled = group.Y.copy()
        shuffled[actions] = group.Y[donors]  # Whole action x lane x output tensor, PP unchanged.
        labels[group.identity] = shuffled
        mapping.append(dict(group_id=group.identity, pp_index=group.pp,
                            recipient_actions=actions, donor_actions=donors.tolist()))
    return labels, mapping


def _dynamics_rows(groups, labels=None):
    X, Y, weights = [], [], []
    per_seed = {seed: sum(group.seed == seed for group in groups) for seed in {group.seed for group in groups}}
    for group in groups:
        X.append(group.X.reshape(-1, group.X.shape[-1]))
        Y.append((group.Y if labels is None else labels[group.identity]).reshape(-1, 4))
        weights.extend([1/(per_seed[group.seed]*8*12)]*(8*12))
    return np.concatenate(X), np.concatenate(Y), np.asarray(weights)


def _response_features(group, predicted):
    absolute = np.stack([projection.project_reader(inputs, projection.project_predicted_state(inputs, delta))
                         for inputs, delta in zip(group.inputs, predicted)])
    if absolute.shape != (8, len(projection.READER_FEATURE_NAMES)) or not np.isfinite(absolute).all():
        raise ValueError('The declared finite reader projection is incomplete')
    features = absolute-absolute[group.pp]
    if not np.array_equal(features[group.pp], np.zeros(features.shape[1])):
        raise ValueError('The reader PP feature must be structural zero')
    return features


def _predict_group(model, group):
    return predict(model, group.X.reshape(-1, group.X.shape[-1])).reshape(8, 12, 4)


def _reader_rows(groups, features):
    F, y, weights = [], [], []
    per_seed = {seed: sum(group.seed == seed for group in groups) for seed in {group.seed for group in groups}}
    for group in groups:
        rows = [index for index in range(8) if index != group.pp]
        F.append(features[group.identity][rows])
        y.extend((group.costs[rows]-group.costs[group.pp]).tolist())
        weights.extend([1/(per_seed[group.seed]*7)]*7)
    return np.concatenate(F), np.asarray(y), np.asarray(weights)


def paired_summary(left, right, seeds):
    differences = np.asarray([left[seed]-right[seed] for seed in seeds], dtype=float)
    average = float(differences.mean())
    radius = T95_DF9*float(differences.std(ddof=1))/sqrt(10)
    return dict(mean_difference=average, approximate_paired_seed_95_t=[average-radius, average+radius], df=9,
                per_seed=[dict(seed=seed, difference=float(value)) for seed, value in zip(seeds, differences)])


def run(dataset, protocol, on_fit=None):
    groups = prepare(dataset, protocol)
    source = [group for group in groups if group.city == 'hangzhou']
    target = [group for group in groups if group.city == 'jinan']
    folds = [list(fold) for fold in protocol['nested_validation']['outer']]
    seeds = protocol['proposed_native_budget']['target_seeds']
    if len(folds) != 5 or any(len(fold) != 2 for fold in folds) or sorted(sum(folds, [])) != sorted(seeds) or len(seeds) != 10:
        raise ValueError('The locked five outer whole-seed pairs are required')
    ledger, oof_manifest, records = [], [], []

    def fitted(role, arm, training, held, X, Y, w, *, outer=None, inner=None, prior=None, reader=False):
        entry = dict(fit_index=len(ledger)+1, role=role, arm=arm, outer_fold=outer, inner_fold=inner,
                     training_seeds=sorted(set(training)), held_out_seeds=sorted(set(held)), rows=len(X),
                     weight_mass=float(w.sum()), alpha=1, status='STARTED')
        ledger.append(entry)
        if on_fit is not None:
            on_fit(ledger, None)
        try:
            model = fit_reader(X, Y, w, alpha=1) if reader else fit_source(X, Y, w) if role == 'source_prior' else fit_dynamics(X, Y, w, prior=prior, alpha=1)
        except Exception as exc:
            entry.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
            if on_fit is not None:
                on_fit(ledger, None)
            raise
        entry['status'] = 'COMPLETE'
        if on_fit is not None:
            on_fit(ledger, model)
        return model

    placebo, permutation = permute_source(source)
    priors = {}
    for arm, labels in (('S', None), ('P', placebo)):
        X, Y, w = _dynamics_rows(source, labels)
        priors[arm] = fitted('source_prior', arm, [group.seed for group in source], [], X, Y, w)
    for outer, held in enumerate(folds):
        train_seeds = sorted(set(seeds)-set(held))
        training = [group for group in target if group.seed in train_seeds]
        held_groups = [group for group in target if group.seed in held]
        features = {arm: {} for arm in ARMS}
        inner_folds = [fold for index, fold in enumerate(folds) if index != outer]
        for inner, inner_held in enumerate(inner_folds):
            inner_train_seeds = sorted(set(train_seeds)-set(inner_held))
            inner_train = [group for group in training if group.seed in inner_train_seeds]
            inner_groups = [group for group in training if group.seed in inner_held]
            X, Y, w = _dynamics_rows(inner_train)
            for arm in ARMS:
                model = fitted('inner_target_dynamics', arm, inner_train_seeds, inner_held, X, Y, w,
                               outer=outer, inner=inner, prior=priors.get(arm))
                for group in inner_groups:
                    features[arm][group.identity] = _response_features(group, _predict_group(model, group))
                    oof_manifest.append(dict(group_id=group.identity, arm=arm, outer_fold=outer, inner_fold=inner,
                        seed=group.seed, response_training_seeds=inner_train_seeds,
                        reader_training_seeds=train_seeds, outer_held_seeds=held))
        readers, dynamics = {}, {}
        X, Y, w = _dynamics_rows(training)
        for arm in ARMS:
            if set(features[arm]) != {group.identity for group in training}:
                raise ValueError('Inner OOF responses do not cover exactly the outer training groups')
            F, delta, cost_weights = _reader_rows(training, features[arm])
            readers[arm] = fitted('target_cost_reader', arm, train_seeds, held, F, delta, cost_weights,
                                 outer=outer, reader=True)
            dynamics[arm] = fitted('outer_target_dynamics', arm, train_seeds, held, X, Y, w,
                                   outer=outer, prior=priors.get(arm))
        for group in held_groups:
            predictions, scores, chosen = {}, {}, {}
            for arm in ARMS:
                predictions[arm] = _predict_group(dynamics[arm], group)
                scores[arm] = reader_predict(readers[arm], _response_features(group, predictions[arm]))
                if scores[arm][group.pp] != 0:
                    raise ValueError('Frozen reader changed the structural PP zero score')
                chosen[arm] = select_action(scores[arm], group.pp)
            # All held-out choices are fixed before actual endpoint errors or costs are evaluated.
            for arm in ARMS:
                selected = chosen[arm]
                records.append(dict(group_id=group.identity, seed=group.seed, anchor=group.anchor,
                    outer_fold=outer, arm=arm, selected_action=selected, pp_index=group.pp,
                    predicted_delta_costs=scores[arm].tolist(), selected_cost=float(group.costs[selected]),
                    phase_pressure_cost=float(group.costs[group.pp]), cost_minus_pp=float(group.costs[selected]-group.costs[group.pp]),
                    response_mae=np.abs(predictions[arm]-group.Y).mean(axis=(0, 1)).tolist(),
                    response_training_seeds=train_seeds, reader_training_seeds=train_seeds, held_out_seeds=held))
    counts = {role: sum(entry['role'] == role for entry in ledger) for role in
              ('source_prior', 'inner_target_dynamics', 'outer_target_dynamics', 'target_cost_reader')}
    expected_counts = dict(source_prior=2, inner_target_dynamics=60, outer_target_dynamics=15, target_cost_reader=15)
    if counts != expected_counts or len(ledger) != 92:
        raise ValueError('The fixed 92 logical-fit roster changed')
    costs = {arm: {seed: float(np.mean([row['selected_cost'] for row in records if row['arm'] == arm and row['seed'] == seed]))
                   for seed in seeds} for arm in ARMS}
    costs['PP'] = {seed: float(np.mean([row['phase_pressure_cost'] for row in records if row['arm'] == 'T' and row['seed'] == seed]))
                   for seed in seeds}
    comparisons = {f'S_minus_{arm}': paired_summary(costs['S'], costs[arm], seeds) for arm in ('T', 'P', 'PP')}
    passed = all(row['approximate_paired_seed_95_t'][1] < 0 for row in comparisons.values())
    behavior = {arm: dict(overrides=sum(row['selected_action'] != row['pp_index'] for row in records if row['arm'] == arm),
        harmful_overrides=sum(row['cost_minus_pp'] > 0 for row in records if row['arm'] == arm),
        harmful_cost_sum=sum(max(row['cost_minus_pp'], 0) for row in records if row['arm'] == arm)) for arm in ARMS}
    return dict(status='DEVELOPMENT_CONTROL_SIGNAL' if passed else 'HOLD_RESPONSE_COST_READOUT_DEVELOPMENT',
        protocol=protocol, logical_fits=len(ledger), fit_counts=counts, fit_ledger=ledger,
        oof_manifest=oof_manifest, source_permutation=permutation, records=records,
        mean_costs={arm: float(np.mean(list(values.values()))) for arm, values in costs.items()},
        per_seed_costs={arm: {str(seed): value for seed, value in values.items()} for arm, values in costs.items()},
        comparisons=comparisons, gate_passed=passed, behavior=behavior,
        response_mae={arm: {name: float(np.mean([np.mean([row['response_mae'][index] for row in records
            if row['arm'] == arm and row['seed'] == seed]) for seed in seeds]))
            for index, name in enumerate(TARGET_COLUMNS)} for arm in ARMS},
        independent_confirmation=0, new_simulations=0, source_450_costs_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def persist(ledger, model):
        if model is not None:
            name = f'fit_{ledger[-1]["fit_index"]:03d}_server_only.npz'
            arrays = dict(coef=model.coef)
            if isinstance(model, DynamicsModel):
                arrays.update(mean=model.scaler.mean, scale=model.scaler.scale, intercept=model.intercept,
                              single_output=np.asarray(model.single_output))
            else:
                arrays['scale'] = model.scale
            np.savez_compressed(args.output/name, **arrays)
            ledger[-1]['model_file'] = name
        (args.output/'fit_ledger.json').write_text(json.dumps(ledger, indent=2, allow_nan=False)+'\n')

    try:
        result = run(json.loads(args.dataset.read_text()), json.loads(args.protocol.read_text()), on_fit=persist)
    except Exception as exc:
        (args.output/'failure.json').write_text(json.dumps(dict(status='HOLD_NESTED_RUN_ERROR', error=f'{type(exc).__name__}: {exc}'))+'\n')
        raise
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'])
