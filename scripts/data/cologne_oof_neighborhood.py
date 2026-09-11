"""Describe each accepted OOF action using only its fold's training neighbors."""

import numpy as np


TOLERANCE = 1e-12


def audit_neighborhood(contrast, effective_records, feature_names, folds, threshold=0.0):
    names = list(feature_names)
    features = np.asarray(contrast.features[:, [contrast.feature_names.index(name) for name in names]], dtype=float)
    groups = np.asarray(contrast.metadata['action_group_ids'], dtype=str)
    states = np.asarray(contrast.metadata['candidate_states'], dtype=str)
    reference_mask = np.asarray(contrast.metadata['is_reference'], dtype=bool)
    raw_delta = np.asarray(contrast.targets['interval_cost'], dtype=float)
    record_map = {row['group_id']: row for row in effective_records}
    if (len(record_map) != len(effective_records) or set(record_map) != set(groups)
            or reference_mask.shape != (contrast.size,)
            or not np.all(np.isfinite(features)) or not np.all(np.isfinite(raw_delta))):
        raise ValueError('Use aligned finite contrast rows and one effective OOF record per original group')
    rows_by_group = {group: np.flatnonzero(groups == group) for group in record_map}
    seed_by_group = {group: int(group.split(':')[1][4:]) for group in record_map}
    row_seeds = np.asarray([seed_by_group[group] for group in groups], dtype=int)
    candidate_indices = np.zeros(contrast.size, dtype=int)
    references = {}
    accepted = []
    for record in effective_records:
        group = record['group_id']
        rows = rows_by_group[group]
        reference, selected = int(record['reference_index']), int(record['selected_index'])
        if (len(rows) != 8 or reference not in range(8) or selected not in range(8)
                or record['seed'] != seed_by_group[group]
                or record['candidate_states'] != states[rows].tolist()
                or np.flatnonzero(reference_mask[rows]).tolist() != [reference]
                or bool(record['different_action']) != (selected != reference)):
            raise ValueError(f'{group}: OOF candidate order or reference does not match the original eight rows')
        costs = np.asarray(record['label_costs'], dtype=float)
        if (costs.shape != (8,) or not np.allclose(raw_delta[rows], costs-costs[reference], rtol=0, atol=TOLERANCE)
                or abs(-raw_delta[rows[selected]] - record['actual_advantage']) > TOLERANCE):
            raise ValueError(f'{group}: raw contrast gain differs from the cached OOF label')
        candidate_indices[rows] = np.arange(8)
        references[group] = reference
        if selected != reference and record['predicted_advantage'] > threshold + TOLERANCE:
            accepted.append(record)

    fold_map = {int(row['heldout_seed']): row for row in folds}
    if len(fold_map) != len(folds) or set(fold_map) != set(seed_by_group.values()):
        raise ValueError('Fold roster must match the original training seeds exactly')
    queries_by_group, fold_diagnostics = {}, []
    for fold in folds:
        seed = int(fold['heldout_seed'])
        training_groups = {group for group, group_seed in seed_by_group.items() if group_seed != seed}
        heldout_groups = set(record_map) - training_groups
        if len(training_groups) != fold['fit_group_count'] or len(heldout_groups) != fold['heldout_group_count']:
            raise ValueError(f'{seed}: training and held-out groups differ from the frozen fit fold')
        fit_rows = np.flatnonzero((row_seeds != seed) & ~reference_mask)
        fit_features = features[fit_rows]
        mean, std = np.mean(fit_features, axis=0), np.std(fit_features, axis=0)
        minimum, maximum = np.min(fit_features, axis=0), np.max(fit_features, axis=0)
        constant = minimum == maximum
        normalized_fit = (fit_features[:, ~constant] - mean[~constant]) / std[~constant]
        selected_records = [row for row in accepted if row['seed'] == seed]
        for record in selected_records:
            group = record['group_id']
            query_row = rows_by_group[group][record['selected_index']]
            query_features = features[query_row]
            normalized_query = (query_features[~constant] - mean[~constant]) / std[~constant]
            distances = np.linalg.norm(normalized_fit-normalized_query, axis=1)
            ordered = sorted(range(len(fit_rows)), key=lambda index: (
                float(distances[index]), str(groups[fit_rows[index]]), int(candidate_indices[fit_rows[index]])))
            neighbors, seen = [], set()
            for index in ordered:
                row = fit_rows[index]
                neighbor_group = str(groups[row])
                if neighbor_group in seen:
                    continue
                seen.add(neighbor_group)
                neighbors.append({'group_id': neighbor_group, 'seed': int(row_seeds[row]),
                    'selected_index': int(candidate_indices[row]), 'reference_index': references[neighbor_group],
                    'raw_gain': float(-raw_delta[row]), 'distance': float(distances[index])})
                if len(neighbors) == 3:
                    break
            if len(neighbors) != 3:
                raise ValueError('Each accepted query requires three different training groups')
            outside = [{'feature': name, 'value': float(query_features[index]),
                        'minimum': float(minimum[index]), 'maximum': float(maximum[index]),
                        'training_constant': bool(constant[index])}
                       for index, name in enumerate(names)
                       if query_features[index] < minimum[index] or query_features[index] > maximum[index]]
            actual_gain = float(-raw_delta[query_row])
            queries_by_group[group] = {
                'group_id': group, 'seed': seed, 'selected_index': int(record['selected_index']),
                'reference_index': int(record['reference_index']),
                'predicted_gain': float(record['predicted_advantage']), 'actual_gain': actual_gain,
                'outcome': 'beneficial' if actual_gain > TOLERANCE else 'harmful' if actual_gain < -TOLERANCE else 'tie',
                'nearest_distance': neighbors[0]['distance'], 'outside_training_range': outside, 'neighbors': neighbors,
            }
        fold_diagnostics.append({'heldout_seed': seed, 'fit_group_count': len(training_groups),
            'fit_row_count': len(fit_rows), 'query_count': len(selected_records),
            'feature_mean': mean.tolist(), 'feature_std': std.tolist(),
            'feature_min': minimum.tolist(), 'feature_max': maximum.tolist(),
            'constant_features': [name for name, is_constant in zip(names, constant) if is_constant],
            'training_seeds': sorted(set(int(value) for value in row_seeds[fit_rows])),
            'training_reference_rows': int(np.sum(reference_mask[fit_rows]))})
    queries = [queries_by_group[row['group_id']] for row in accepted]
    return {'queries': queries, 'folds': fold_diagnostics, 'query_count': len(queries), 'feature_names': names,
        'checks': {'fold_exclusions_exact': all(row['heldout_seed'] not in row['training_seeds']
                                               and row['training_reference_rows'] == 0 for row in fold_diagnostics),
                   'neighbor_groups_unique': all(len({neighbor['group_id'] for neighbor in row['neighbors']}) == 3 for row in queries),
                   'neighbors_exclude_query_seed': all(neighbor['seed'] != row['seed'] for row in queries for neighbor in row['neighbors']),
                   'candidate_order_and_raw_labels_exact': True}}
