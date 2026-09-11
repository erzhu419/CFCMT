"""Decompose the fixed OOF policy's label cost over actions allowed by S."""

import numpy as np

from cf_h2o.traffic_signal.action_contrast import rule_reference_indices


TOLERANCE = 1e-12


def decompose_policy_loss(effective_records, absolute_bank, expected_groups, threshold=0.0):
    expected = list(expected_groups)
    records = {row['group_id']: row for row in effective_records}
    groups = np.asarray(absolute_bank.metadata['action_group_ids'], dtype=str)
    states = np.asarray(absolute_bank.metadata['candidate_states'], dtype=str)
    if (not expected or len(set(expected)) != len(expected)
            or len(records) != len(effective_records) or set(records) != set(expected)
            or set(groups) != set(expected)):
        raise ValueError('Preserve exactly one effective OOF record for every original group')
    switches = np.asarray(absolute_bank.features)[:, absolute_bank.feature_names.index('switch_indicator')]
    costs = np.asarray(absolute_bank.targets['prefix_mean_cost_450s'], dtype=float)
    if (not np.all(np.isfinite(costs))
            or not np.allclose(costs, absolute_bank.targets['interval_cost'], rtol=0, atol=TOLERANCE)):
        raise ValueError('Native interval costs must equal the finite 450-second absolute labels')
    references = rule_reference_indices(absolute_bank, policy='phase_pressure')
    rows = []
    for group in expected:
        record = records[group]
        indices = np.flatnonzero(groups == group)
        reference, selected = record['reference_index'], record['selected_index']
        if (len(indices) != 8 or len(set(states[indices])) != 8
                or record['candidate_states'] != states[indices].tolist()
                or reference not in range(8) or selected not in range(8)
                or indices[reference] != references[group]
                or record['seed'] != int(group.split(':')[1][4:])):
            raise ValueError(f'{group}: candidate order, action index or PP reference differs from the bank')
        values, indicators = costs[indices], switches[indices]
        cached = np.asarray(record['label_costs'], dtype=float)
        predicted = float(record['predicted_advantage'])
        if (cached.shape != (8,) or not np.allclose(cached, values, rtol=0, atol=TOLERANCE)
                or not np.isfinite(predicted)
                or abs(record['actual_advantage'] - (values[reference]-values[selected])) > TOLERANCE):
            raise ValueError(f'{group}: cached labels or selected action gain differs from the bank')
        if not np.all(np.isin(indicators, [0., 1.])):
            raise ValueError(f'{group}: absolute switch_indicator must be binary')
        allowed = [index for index in range(8) if index == reference or indicators[index] == 1.]
        deployed = selected if selected != reference and predicted > threshold + TOLERANCE else reference
        if deployed not in allowed:
            raise ValueError(f'{group}: deployed action violates the fixed S no-stay rule')
        reference_cost, policy_cost = float(values[reference]), float(values[deployed])
        oracle_cost = float(np.min(values[allowed]))
        oracle_indices = [index for index in allowed if abs(values[index]-oracle_cost) <= TOLERANCE]
        pp_gain, realized_gain, regret = reference_cost-oracle_cost, reference_cost-policy_cost, policy_cost-oracle_cost
        if abs(pp_gain - (realized_gain + regret)) > TOLERANCE:
            raise ValueError(f'{group}: per-group cost decomposition does not close')
        if policy_cost <= oracle_cost + TOLERANCE:
            category = 'achieves_label_optimum'
        elif reference_cost <= oracle_cost + TOLERANCE:
            category = 'unnecessary_override_when_pp_optimal'
        elif deployed != reference:
            category = 'suboptimal_switch_when_benefit_available'
        else:
            category = 'missed_beneficial_switch'
        rows.append({'group_id': group, 'seed': int(record['seed']),
            'candidate_states': list(record['candidate_states']), 'label_costs': values.tolist(),
            'switch_indicators': indicators.astype(int).tolist(), 'reference_index': reference,
            'selected_index': selected, 'deployed_index': deployed, 'predicted_advantage': predicted,
            'allowed_indices': allowed, 'oracle_indices': oracle_indices,
            'reference_cost': reference_cost, 'policy_cost': policy_cost, 'oracle_cost': oracle_cost,
            'pp_to_oracle_gain': pp_gain, 'realized_gain': realized_gain, 'regret': regret, 'category': category})
    return {'rows': rows, 'group_count': len(rows), 'checks': {
        'original_labels_and_candidate_order_exact': True, 'allowed_set_follows_s': True,
        'all_deployed_actions_allowed': True, 'per_group_cost_identity': True}}
