#!/usr/bin/env python3
"""Read saved group/OOF scalars only; no fitting, trajectories or gate changes."""
import argparse
import json
from pathlib import Path
import numpy as np


def macro(rows, seeds, key):
    return np.mean([np.mean([r[key] for r in rows if r['seed'] == seed], axis=0)
                    for seed in seeds], axis=0)


def excess_summary(arrays):
    values = np.concatenate([a.ravel() for a in arrays])
    positive = values[values > 0]
    return dict(movement_action_count=int(values.size), violation_count=int(positive.size),
        violation_fraction=float(np.mean(values > 0)), mean_excess=float(values.mean()),
        total_excess=float(values.sum()), maximum_excess=float(values.max()),
        violating_row_mean_excess=float(positive.mean()) if positive.size else None,
        violating_row_median_excess=float(np.median(positive)) if positive.size else None)


def signal_summary(groups, seeds, names):
    rows, deltas, measured_excess = [], [], []
    for g in groups:
        y = g['Y']; delta = (y - y[g['reference']])[np.arange(8) != g['reference']]
        nz = delta != 0; deltas.append(delta)
        rows.append(dict(seed=g['seed'], fractions=nz.mean(axis=(0, 1)),
            any_fraction=float(np.any(nz, axis=2).mean()), mae=np.abs(delta).mean(axis=(0, 1)),
            group_any=int(np.any(nz)), group_output=np.any(nz, axis=(0, 1)).astype(int),
            actions_any=int(np.count_nonzero(np.any(nz, axis=(1, 2)))),
            actions_output=np.count_nonzero(np.any(nz, axis=1), axis=0)))
        measured_excess.append(np.maximum(y[:, :, 1] - g['internal'] - y[:, :, 0], 0))
    fractions, mae = macro(rows, seeds, 'fractions'), macro(rows, seeds, 'mae')
    pooled = np.concatenate(deltas, axis=0)
    return dict(seeds=len(seeds), groups=len(groups), nonPP_actions=len(groups) * 7,
        action_movement_cells_per_output=int(pooled.shape[0] * pooled.shape[1]),
        nonzero_cells_by_output=dict(zip(names, map(int, np.count_nonzero(pooled, axis=(0, 1))))),
        nonzero_fraction_by_output=dict(zip(names, map(float, fractions))),
        zero_fraction_by_output=dict(zip(names, map(float, 1 - fractions))),
        any_output_nonzero_fraction=float(macro(rows, seeds, 'any_fraction')),
        zero_change_contrast_mae=float(mae.mean()),
        zero_change_mae_by_output=dict(zip(names, map(float, mae))),
        groups_with_any_delta=sum(r['group_any'] for r in rows),
        groups_with_any_delta_by_output=dict(zip(names, map(int, np.sum([r['group_output'] for r in rows], axis=0)))),
        nonPP_actions_with_any_delta=sum(r['actions_any'] for r in rows),
        nonPP_actions_with_any_delta_by_output=dict(zip(names, map(int, np.sum([r['actions_output'] for r in rows], axis=0)))),
        per_seed_zero_change_mae={str(s): float(np.mean([r['mae'] for r in rows if r['seed'] == s])) for s in seeds},
        measured_count_conservation=excess_summary(measured_excess))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'collection-root', 'oof'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); spec = json.loads(args.spec.read_text()); groups = {}
    for case in spec['cases']:
        path = args.collection_root / f"seed_{case['seed']}" / case['case_id'] / 'group_data_server_only.json'
        d = json.loads(path.read_text()); x = np.asarray(d['X'], dtype=float)
        groups[case['case_id']] = dict(city=case['city'], seed=case['seed'], Y=np.asarray(d['Y'], dtype=float),
            reference=int(d['reference_index']), internal=x[:, :, d['feature_names'].index('internal_vehicle_count')])
    records = json.loads(args.oof.read_text())['records']; conservation = {}
    for arm in ('target_only', 'source_target', 'source_placebo'):
        excesses = []
        for row in records:
            if row['arm'] != arm: continue
            g = groups[row['case_id']]; prediction = np.asarray(row['predicted_counts'], dtype=float)
            excesses.append(np.maximum(prediction[:, :, 1] - g['internal'] - prediction[:, :, 0], 0))
        conservation[arm] = excess_summary(excesses)
    result = dict(protocol='physical-response-saved-signal-postmortem-v1', status='DESCRIPTIVE_ONLY',
        new_fits=0, new_simulations=0, gate_recomputed=False,
        weighting='Seven nonPP actions, focal movements and two outputs equally; equal two cases within seed, then equal declared seeds. Conservation uses all eight actions; fixed equal movement counts make pooled rows equal-seed weighted.',
        zero_change_semantics='Identical predicted count response for candidate and PP gives zero contrast; post hoc diagnostic only, not a new admitted candidate.',
        conservation_semantics='Positive receiving - initial_internal - stopline; measured Y and saved predictions separately; no projection.',
        signal={city: signal_summary([g for g in groups.values() if g['city'] == city], spec[key], spec['label_names'])
                for city, key in [('cologne', 'source_seeds'), ('jinan', 'target_seeds')]},
        prediction_conservation=conservation)
    print(json.dumps(result, allow_nan=False, separators=(',', ':')))


if __name__ == '__main__':
    main()
