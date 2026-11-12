"""Calibrate one source-CDF shrinkage weight on two whole held target hours."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tsc_corridor_expected_service_20261109'))
from expected_predictor import ExpectedServicePredictor
from evidence_model import EvidencePredictor
from residual_model import fit_hours, label_category, release_id, rows, update_history

DETECTOR_BOUNDS = np.arange(1., 11.)
SHRUNK_ARMS = (('S', 1), ('P', 2))


def _quadratic_fit(a, b, c):
    """A=mean(d²) is the denominator; B=mean[d(Y−T)] is the numerator."""
    unconstrained = b/a if a > 0 else None
    weight = float(np.clip(unconstrained, 0., 1.)) if a > 0 else 0.
    return dict(A=a, B=b, C=c, unconstrained_weight=unconstrained, weight=weight,
                loss_at_zero=c, loss_at_one=c-2*b+a,
                loss_at_fitted=c-2*weight*b+weight*weight*a)


def _evaluate_hour(root, city, seed, target, source, placebo):
    directory = Path(root)/'adapt'/city/str(seed)
    receipt = json.loads((directory/'result.json').read_text())
    if receipt['status'] != 'PASS_HOUR' or receipt['city'] != city or receipt['seed'] != seed:
        raise ValueError(f'Invalid held training hour: {directory}')
    labels = list(rows(directory/'labels.jsonl.gz'))
    outcomes = {release_id(item): item for item in labels}
    if len(outcomes) != len(labels) or any(item['status'] == 'unknown' for item in labels):
        raise ValueError('Unresolved or duplicate held-hour detector labels')
    predictor = EvidencePredictor(target, source, placebo)
    totals = {arm: dict(A_sum=0., B_sum=0., C_sum=0.) for arm, _ in SHRUNK_ARMS}
    samples, decisions = 0, 0
    for row in rows(directory/'decisions.jsonl.gz'):
        decisions += 1
        predictions = []
        if not row['spillback'] and not row['unmatched'] and row['t']+10 <= receipt['horizon_sec']:
            for flight in row['flights']:
                predictions.append((flight, predictor.flight_cdfs(flight, row['t'], DETECTOR_BOUNDS)))
        # Freeze every current raw forecast before consulting any current outcome.
        for flight, raw in predictions:
            category = label_category(flight, outcomes[release_id(flight)], row['t'])
            truth = ((category > 0) & (DETECTOR_BOUNDS >= category)).astype(float)
            residual = truth-raw[0]
            samples += 1
            for arm, index in SHRUNK_ARMS:
                difference = raw[index]-raw[0]
                totals[arm]['A_sum'] += float(difference @ difference)
                totals[arm]['B_sum'] += float(difference @ residual)
                totals[arm]['C_sum'] += float(residual @ residual)
        update_history(row, predictor.history)
    queries = 10*samples
    if not queries:
        raise ValueError('Held target hour has no complete observable flight queries')
    for values in totals.values():
        values.update({key: values[key+'_sum']/queries for key in ('A', 'B', 'C')})
        values['loss_at_zero'] = values['C']
        values['loss_at_one'] = values['C']-2*values['B']+values['A']
    return dict(held_seed=seed, hour_weight=.5, samples=samples, queries=queries,
                all_decisions=decisions, arms=totals)


def fit_block(root, city, seeds, source, placebo):
    """Fit S/P independently, with equal held-hour weights and no development input."""
    seeds = sorted(seeds)
    if len(seeds) != 2 or seeds[0] == seeds[1]:
        raise ValueError('Target shrinkage requires two distinct adaptation hours')
    heads = {seed: fit_hours(root, 'adapt', city, [seed]) for seed in seeds}
    folds = []
    for held_seed, training_seed in (seeds, seeds[::-1]):
        fold = _evaluate_hour(root, city, held_seed, heads[training_seed], source, placebo)
        fold.update(training_seed=training_seed, training_metadata=heads[training_seed].metadata)
        folds.append(fold)
    arms = {arm: _quadratic_fit(*(sum(fold['arms'][arm][key] for fold in folds)/2
                                 for key in ('A', 'B', 'C'))) for arm, _ in SHRUNK_ARMS}
    weights = {arm: values['weight'] for arm, values in arms.items()}
    for fold in folds:
        for arm, values in fold['arms'].items():
            weight = weights[arm]
            values['loss_at_fitted'] = values['C']-2*weight*values['B']+weight*weight*values['A']
    # Preserve the original two-hour count fit and every original metadata field.
    full_target = fit_hours(root, 'adapt', city, seeds)
    calibration = dict(city=city, target_seeds=seeds, weights=weights, folds=folds, arms=arms,
        objective='mean of two held-hour flight-by-detector-horizon (1..10) mean Brier losses',
        moment_definitions=dict(A='mean squared pooled-minus-target CDF; denominator',
                                B='mean pooled-minus-target CDF times (truth-target); numerator',
                                C='mean target Brier loss'),
        full_target_metadata=full_target.metadata)
    return full_target, calibration


class ShrinkPredictor(ExpectedServicePredictor):
    """Shrink only S/P; retain the expected-service decoder and causal history."""
    def __init__(self, target, source, placebo, weights):
        super().__init__(target, source, placebo)
        self.weights = {arm: float(weights[arm]) for arm, _ in SHRUNK_ARMS}
        if any(not 0 <= weight <= 1 for weight in self.weights.values()):
            raise ValueError('Source shrinkage weights must lie in [0,1]')
        self.raw_detector_cdfs = []

    def forecast(self, row):
        self.raw_detector_cdfs = []
        return super().forecast(row)

    def flight_cdfs(self, flight, time_s, bounds):
        count = len(bounds)
        raw = super().flight_cdfs(flight, time_s, [*bounds, *DETECTOR_BOUNDS])
        for arm, index in SHRUNK_ARMS:
            weight = self.weights[arm]
            if weight == 0:
                raw[index] = raw[0]
            elif weight != 1:
                raw[index] = raw[0]+weight*(raw[index]-raw[0])
        self.raw_detector_cdfs.append((flight, raw[:, count:].copy()))
        return raw[:, :count]
