"""Select from S-allowed actions using the unchanged cached OOF score order."""

import numpy as np

from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import StateConditionedSourceUtilityOriginator

if __package__:
    from .audit_cologne_local_rigid_ranking import group_record
    from .cologne_no_stay_calibration import apply_no_stay_to_oof
    from .cologne_policy_loss_decomposition import decompose_policy_loss
else:
    from audit_cologne_local_rigid_ranking import group_record
    from cologne_no_stay_calibration import apply_no_stay_to_oof
    from cologne_policy_loss_decomposition import decompose_policy_loss


def pre_mask_records(raw_records, absolute_bank, expected_groups, threshold=0.0):
    baseline, _ = apply_no_stay_to_oof(raw_records, absolute_bank, expected_groups)
    original = {row['group_id']: row for row in raw_records}
    groups = np.asarray(absolute_bank.metadata['action_group_ids'], dtype=str)
    switches = absolute_bank.features[:, absolute_bank.feature_names.index('switch_indicator')]
    argmin = StateConditionedSourceUtilityOriginator._argmin_with_reference_tie_break
    selections = []
    for base in baseline:
        group = base['group_id']
        raw = original[group]
        states = np.asarray(raw['candidate_states'], dtype=str)
        scores = np.asarray(raw['predicted_scores'], dtype=float)
        reference = raw['reference_index']
        if argmin(scores, reference=reference, candidate_states=states) != raw['selected_index']:
            raise ValueError(f'{group}: cached raw top1 differs from the original score tie-break')
        indicators = switches[np.flatnonzero(groups == group)]
        allowed = [index for index in range(8) if index == reference or indicators[index] == 1.]
        selected = allowed[argmin(scores[allowed], reference=allowed.index(reference),
                                  candidate_states=states[allowed])]
        selections.append((group, selected))

    # Reconstruct evaluation records after all replacement actions are selected from scores.
    records = [{**original[group], **group_record(group, original[group]['candidate_states'],
        original[group]['label_costs'], original[group]['predicted_scores'],
        original[group]['reference_index'], selected)} for group, selected in selections]
    old_loss = decompose_policy_loss(baseline, absolute_bank, expected_groups, threshold=threshold)
    new_loss = decompose_policy_loss(records, absolute_bank, expected_groups, threshold=threshold)
    rows = []
    for base, old, new, record in zip(baseline, old_loss['rows'], new_loss['rows'], records):
        changed = old['deployed_index'] != new['deployed_index']
        if not base['model_stay_vetoed'] and changed:
            raise ValueError(f"{base['group_id']}: non-veto deployed action changed")
        comparison = {key: old[key] for key in ('group_id', 'seed', 'candidate_states', 'label_costs',
            'switch_indicators', 'reference_index', 'allowed_indices', 'oracle_indices',
            'reference_cost', 'oracle_cost')}
        comparison.update({'predicted_scores': list(record['predicted_scores']),
            'original_selected_index': base['original_selected_index'],
            'original_model_stay_vetoed': base['model_stay_vetoed'],
            'deployed_action_changed': changed,
            'realized_gain_difference': old['policy_cost']-new['policy_cost']})
        for prefix, loss in (('baseline', old), ('premask', new)):
            comparison.update({f'{prefix}_{key}': loss[key] for key in (
                'selected_index', 'deployed_index', 'predicted_advantage', 'policy_cost', 'regret', 'category')})
            comparison[f'{prefix}_actual_gain'] = loss['realized_gain']
        rows.append(comparison)
    return {'records': records, 'rows': rows, 'group_count': len(rows), 'checks': {
        'raw_top1_matches_original_tie_break': True, 'cached_scores_and_candidate_order_preserved': True,
        'allowed_set_follows_s': True, 'all_deployed_actions_allowed': True,
        'all_non_veto_deployed_actions_unchanged': True}}
