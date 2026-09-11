"""Apply the fixed no-stay rule to existing native B100 OOF predictions."""

import numpy as np

from cf_h2o.traffic_signal.action_contrast import rule_reference_indices

if __package__:
    from .audit_cologne_local_rigid_ranking import group_record
else:
    from audit_cologne_local_rigid_ranking import group_record


TOLERANCE = 1e-12


def apply_no_stay_to_oof(records, absolute_bank, expected_groups):
    """Replace an original top-1 stay proposal with PP, without re-ranking."""
    expected = list(expected_groups)
    record_map = {row['group_id']: row for row in records}
    groups = np.asarray(absolute_bank.metadata['action_group_ids'], dtype=str)
    states = np.asarray(absolute_bank.metadata['candidate_states'], dtype=str)
    if (not expected or len(set(expected)) != len(expected)
            or len(record_map) != len(records) or set(record_map) != set(expected)
            or set(groups) != set(expected)):
        raise ValueError('OOF records and absolute bank must cover the unchanged group roster once')
    switch_column = absolute_bank.feature_names.index('switch_indicator')
    switches = np.asarray(absolute_bank.features)[:, switch_column]
    costs = np.asarray(absolute_bank.targets['prefix_mean_cost_450s'], dtype=float)
    if not np.allclose(costs, absolute_bank.targets['interval_cost'], rtol=0, atol=TOLERANCE):
        raise ValueError('Native interval costs differ from the 450-second labels')
    references = rule_reference_indices(absolute_bank, policy='phase_pressure')
    effective = []
    for group in expected:
        row = record_map[group]
        indices = np.flatnonzero(groups == group)
        if (len(indices) != 8 or row['candidate_states'] != states[indices].tolist()
                or len(set(states[indices])) != 8):
            raise ValueError(f'{group}: eight candidate states must align exactly')
        if not np.allclose(row['label_costs'], costs[indices], rtol=0, atol=TOLERANCE):
            raise ValueError(f'{group}: OOF label costs differ from the native bank')
        reference, selected = row['reference_index'], row['selected_index']
        if reference not in range(8) or selected not in range(8) or indices[reference] != references[group]:
            raise ValueError(f'{group}: OOF PP reference or selected index does not match the bank')
        if not np.all(np.isin(switches[indices], [0.0, 1.0])):
            raise ValueError(f'{group}: absolute switch_indicator must be binary')
        original = group_record(group, row['candidate_states'], row['label_costs'],
                                row['predicted_scores'], reference, selected)
        vetoed = bool(selected != reference and switches[indices[selected]] == 0.0)
        updated = group_record(group, row['candidate_states'], row['label_costs'],
                               row['predicted_scores'], reference, reference if vetoed else selected)
        effective.append({**row, **updated,
            'original_selected_index': selected,
            'original_different_action': original['different_action'],
            'original_predicted_advantage': original['predicted_advantage'],
            'model_stay_vetoed': vetoed})

    def counts(rows):
        return {'groups': len(rows), 'candidate_rows': sum(len(r['candidate_states']) for r in rows),
                'raw_proposals': sum(r['original_different_action'] for r in rows),
                'model_stay_vetoed_groups': sum(r['model_stay_vetoed'] for r in rows),
                'remaining_switch_proposals': sum(r['different_action'] for r in rows)}

    diagnostics = {**counts(effective), 'per_seed': {
        str(seed): counts([r for r in effective if r['seed'] == seed])
        for seed in sorted({r['seed'] for r in effective})}}
    return effective, diagnostics


def proposal_accounting(effective_records, threshold):
    """Count runtime decisions in their actual order: threshold, then stay veto."""
    def counts(rows):
        raw = [r for r in rows if r['original_different_action']]
        passed = [r for r in raw if threshold is not None
                  and r['original_predicted_advantage'] > threshold + TOLERANCE]
        vetoed = sum(r['model_stay_vetoed'] for r in passed)
        return {'groups': len(rows), 'raw_proposals': len(raw),
                'original_reference_actions': len(rows) - len(raw),
                'threshold_rejected': len(raw) - len(passed), 'stay_veto_count': vetoed,
                'accepted_overrides': len(passed) - vetoed}

    return {**counts(effective_records), 'per_seed': {
        str(seed): counts([r for r in effective_records if r['seed'] == seed])
        for seed in sorted({r['seed'] for r in effective_records})}}
