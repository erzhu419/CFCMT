"""Attach thirteen omitted features to the already frozen OOF neighbor pairs."""

import numpy as np


OMITTED_FEATURE_NAMES = (
    'current_green_elapsed_norm',
    *(name for parent in ('green_q_max', 'red_q_max', 'green_q_cv', 'red_q_cv')
      for name in (parent, f'reference_{parent}', f'delta_{parent}')),
)


def audit_omitted_features(contrast, previous_audit, folds, feature_names):
    names = list(feature_names)
    if names != list(OMITTED_FEATURE_NAMES):
        raise ValueError('Inspect exactly the frozen thirteen omitted features in their declared order')
    features = np.asarray(contrast.features[:, [contrast.feature_names.index(name) for name in names]], dtype=float)
    groups = np.asarray(contrast.metadata['action_group_ids'], dtype=str)
    reference_mask = np.asarray(contrast.metadata['is_reference'], dtype=bool)
    raw_delta = np.asarray(contrast.targets['interval_cost'], dtype=float)
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(raw_delta)):
        raise ValueError('Use finite original contrast features and raw cost deltas')
    rows_by_group = {group: np.flatnonzero(groups == group) for group in np.unique(groups)}
    seed_by_group = {group: int(group.split(':')[1][4:]) for group in rows_by_group}
    row_seeds = np.asarray([seed_by_group[group] for group in groups], dtype=int)

    def feature_values(record, gain_key):
        group = record['group_id']
        rows = rows_by_group[group]
        selected, reference = record['selected_index'], record['reference_index']
        if (len(rows) != 8 or selected not in range(8) or reference not in range(8) or selected == reference
                or record['seed'] != seed_by_group[group]
                or np.flatnonzero(reference_mask[rows]).tolist() != [reference]
                or abs(-raw_delta[rows[selected]] - record[gain_key]) > 1e-12):
            raise ValueError(f'{group}: frozen action index, PP reference or raw gain differs from the bank')
        return features[rows[selected]].tolist()

    queries = []
    for original in previous_audit['queries']:
        neighbors = original['neighbors']
        if (len(neighbors) != 3 or len({row['group_id'] for row in neighbors}) != 3
                or any(row['seed'] == original['seed'] for row in neighbors)):
            raise ValueError('Keep the three original different-group neighbors outside the query seed')
        queries.append({**original, 'feature_values': feature_values(original, 'actual_gain'),
            'neighbors': [{**row, 'feature_values': feature_values(row, 'raw_gain')} for row in neighbors]})
    if len(queries) != previous_audit['query_count']:
        raise ValueError('Preserve every query from the completed neighborhood audit')

    fold_diagnostics = []
    if {row['heldout_seed'] for row in folds} != set(seed_by_group.values()):
        raise ValueError('Use the original three training-seed folds')
    for fold in folds:
        seed = int(fold['heldout_seed'])
        fit_rows = np.flatnonzero((row_seeds != seed) & ~reference_mask)
        training_groups = set(groups[fit_rows])
        heldout_groups = {group for group, value in seed_by_group.items() if value == seed}
        if len(training_groups) != fold['fit_group_count'] or len(heldout_groups) != fold['heldout_group_count']:
            raise ValueError(f'{seed}: original training/heldout group count changed')
        values = features[fit_rows]
        minimum, maximum = np.min(values, axis=0), np.max(values, axis=0)
        fold_diagnostics.append({'heldout_seed': seed, 'fit_group_count': len(training_groups),
            'fit_row_count': len(fit_rows), 'training_seeds': sorted(set(int(value) for value in row_seeds[fit_rows])),
            'training_reference_rows': int(np.sum(reference_mask[fit_rows])),
            'feature_mean': np.mean(values, axis=0).tolist(), 'feature_std': np.std(values, axis=0).tolist(),
            'feature_min': minimum.tolist(), 'feature_max': maximum.tolist(),
            'constant_features': [name for name, low, high in zip(names, minimum, maximum) if low == high]})
    return {'feature_names': names, 'query_count': len(queries),
        'pair_count': sum(len(row['neighbors']) for row in queries), 'queries': queries, 'folds': fold_diagnostics,
        'checks': {'frozen_query_neighbor_roster_preserved': True, 'bank_action_and_raw_labels_exact': True,
            'fit_folds_exclude_heldout_and_reference': all(row['heldout_seed'] not in row['training_seeds']
                and row['training_reference_rows'] == 0 for row in fold_diagnostics)}}
