"""Explain unchanged frozen forecasts on existing PP states; no policy updates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for sibling in ('tsc_corridor_residual_transfer_20261109', 'tsc_corridor_arrival_transfer_20261107'):
    sys.path.insert(0, str(HERE.parent / sibling))
from arrival_policy import ArrivalPolicy, fit_hours as old_fit
from model_gate import hour, load_training, release_id, rows, verify_protocol
from residual_model import ResidualPredictor, feature_context, fit_hours, fit_source, label_category
from evaluate import ARMS, PAIRS, SCOPES, choose_requests
from decision_path import DecisionPathObserver
from loss_context import LossContextObserver


class DetailedPredictor(ResidualPredictor):
    """Observe the same CDF calls, adding detector query points without feedback."""
    def forecast(self, row):
        self.items = []
        return super().forecast(row)

    def flight_cdfs(self, flight, time_s, bounds):
        context, eta = feature_context(flight, time_s, self.history)
        raw = super().flight_cdfs(flight, time_s, [*bounds, *range(1, 11)])
        original = raw[:, :11].copy()
        decoded = original.copy()
        decoded[:3] = decoded[:3] >= .5
        self.items.append(dict(flight=flight, context=context, eta=eta, raw_cdfs=original,
                               decoded=decoded, detector_cdfs=raw[:, 11:].copy(),
                               target_n=float(self.target.counts[context].sum()) if context in self.target.counts else 0,
                               source_n=float(self.source.counts[context].sum()) if context in self.source.counts else 0))
        return original.copy()


def check_hour(metrics, previous, city, seed):
    checked = {}
    for scope, actual in metrics.items():
        expected = previous['scopes'][scope]
        if actual['edge_epochs'] != expected['counts']['scored_edge_epochs']:
            raise ValueError(f'Edge denominator changed for {city}/{seed}/{scope}')
        maes = {arm: actual['error'][arm] / actual['edge_epochs'] for arm in ARMS}
        if any(abs(maes[arm]-expected['mae'][arm]) > 1e-12 for arm in ARMS):
            raise ValueError(f'Frozen predictions changed for {city}/{seed}/{scope}')
        if actual['differences'] != expected['action_differences']:
            raise ValueError(f'Frozen requests changed for {city}/{seed}/{scope}')
        checked[scope] = dict(edge_epochs=actual['edge_epochs'], mae=maes,
                              action_differences=actual['differences'], exact_reproduction=True)
    return checked


def replay_hour(directory, city, seed, block, predictor, old_policy, previous, decisions, losses):
    outcomes = {release_id(record): record for record in rows(directory / 'labels.jsonl.gz')}
    per_seed = DecisionPathObserver()
    metrics = {scope: dict(edge_epochs=0, error=Counter({arm: 0.0 for arm in ARMS}),
                           differences={f'{a}_vs_{b}': 0 for a, b in PAIRS}) for scope in SCOPES}
    weights = {scope: 1/(6*previous['scopes'][scope]['counts']['scored_edge_epochs']) for scope in SCOPES}
    for row in rows(directory / 'decisions.jsonl.gz'):
        forecasts = predictor.forecast(row)
        observable = bool(row['flights'] and not row['spillback'] and not row['unmatched'] and row['t']+10 <= 3600)
        old = old_policy.forecast(row)
        scopes = dict(original_common_scored=observable and bool(old), all_observable_risk=observable)
        actions = {scope: choose_requests(row, forecasts if enabled else None) for scope, enabled in scopes.items()}
        # Future outcomes are opened only after every model forecast and request is fixed.
        if observable:
            for item in predictor.items:
                category = label_category(item['flight'], outcomes[release_id(item['flight'])], row['t'])
                item['label_category'], item['actual10'] = category, category != 0
        per_seed.observe(dict(row, city=city, seed=seed, block=block), forecasts,
                         predictor.items, enabled=scopes)
        for scope, active in scopes.items():
            m = metrics[scope]
            for a, b in PAIRS:
                m['differences'][f'{a}_vs_{b}'] += actions[scope][a] != actions[scope][b]
            if active:
                actual = Counter()
                for item in predictor.items:
                    actual[item['flight']['edge']] += int(item['actual10'])
                for edge, values in forecasts.items():
                    m['edge_epochs'] += 1
                    for index, arm in enumerate(ARMS):
                        m['error'][arm] += abs(float(values[index, 10])-actual[edge])
        losses.add(row, predictor.items, forecasts, scopes, city=city, seed=seed,
                   block=block, edge_weights=weights)
    checked = check_hour(metrics, previous, city, seed)
    decisions.merge(per_seed)
    return dict(city=city, seed=seed, block=block, reproduction=checked, decision_path=per_seed.result())


def run(root, frozen_dir, output):
    cohort = json.loads((frozen_dir/'cohort.json').read_text())['cities']
    expected = json.loads((frozen_dir/'development/result.json').read_text())
    verify_protocol(cohort)
    source, old_source = {}, {}
    for city in cohort:
        seeds = cohort[city]['donor']
        source[city] = fit_source(root, 'donor', city, seeds)
        labels = load_training(root, 'donor', city, seeds)
        old_source[city] = old_fit(labels), old_fit(labels, permute=True)
        print(f'Restored frozen source/{city}', flush=True)
    decision_panels = {city: DecisionPathObserver() for city in cohort}
    loss_panel, seed_results = LossContextObserver(), []
    for city, donor in (('hangzhou', 'jinan'), ('jinan', 'hangzhou')):
        old_results = {record['seed']: record for record in expected['cities'][city]['by_seed']}
        for block, seeds in enumerate(cohort[city]['adapt']):
            target = fit_hours(root, 'adapt', city, seeds)
            old_target = old_fit(load_training(root, 'adapt', city, seeds))
            policy = ArrivalPolicy(old_target, *old_source[donor])
            for seed in cohort[city]['dev'][2*block:2*block+2]:
                predictor = DetailedPredictor(target, *source[donor])
                record = replay_hour(hour(root, 'dev', city, seed), city, seed, block, predictor,
                                     policy, old_results[seed], decision_panels[city], loss_panel)
                seed_results.append(record)
                print(f'Explained {city}/{seed}; all five frozen MAEs and requests reproduced', flush=True)
    losses = loss_panel.result()
    for city in cohort:
        for scope in SCOPES:
            expected_mae = expected['cities'][city]['scopes'][scope]['equal_seed_mae']
            delta = losses['by_city'][city][scope]['overall']['counts']['weighted_count_loss_delta']
            if abs(delta-(expected_mae['S']-expected_mae['T'])) > 1e-12:
                raise ValueError(f'Loss attribution changed the equal-seed estimand: {city}/{scope}')
    result = dict(status='COMPLETE_FROZEN_MODEL_DIAGNOSTIC', previous_status=expected['status'],
                  new_simulations=0, new_candidates=0, new_labels_collected=0, independent_confirmation=0,
                  city_decisions={city: panel.result() for city, panel in decision_panels.items()},
                  seed_results=seed_results, loss_context=losses,
                  reproduced_seed_hours=len(seed_results), model_changed=False)
    output.mkdir(parents=True, exist_ok=True)
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--frozen-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.frozen_dir, args.output)
