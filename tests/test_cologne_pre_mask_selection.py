from copy import deepcopy

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import StateConditionedSourceUtilityOriginator
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.data.audit_cologne_local_rigid_ranking import group_record
from scripts.data.cologne_pre_mask_selection import pre_mask_records


def selection_case():
    groups = [f'cologne1:seed2027:{index}:tls' for index in range(5)]
    states = ['stay', 'z_phase', 'a_phase', 'reference', 'phase4', 'phase5', 'phase6', 'phase7']
    scores = np.asarray([
        [-2., -1., -1., 0., 2., 3., 4., 5.],
        [-2., 0., 0., 0., 2., 3., 4., 5.],
        [2., -1., -1., 0., 2., 3., 4., 5.],
        [2., 1., 1., 0., 2., 3., 4., 5.],
        [-2., .1, .2, 0., 2., 3., 4., 5.],
    ])
    costs = np.asarray([
        [.1, 1., 2., 3., 5., 6., 7., 8.],
        [.1, 1., 2., 3., 5., 6., 7., 8.],
        [.1, 1., 2., 3., 5., 6., 7., 8.],
        [.1, 1., 2., 3., 5., 6., 7., 8.],
        [.1, 1., 2., 3., 5., 6., 7., 8.],
    ])
    records = []
    for index, group in enumerate(groups):
        selected = StateConditionedSourceUtilityOriginator._argmin_with_reference_tie_break(
            scores[index], reference=3, candidate_states=np.asarray(states))
        records.append({**group_record(group, states, costs[index], scores[index], 3, selected),
                        'heldout_seed': 2027})
    features = np.tile(np.column_stack(([1., 2., 3., 10., 4., 5., 6., 7.],
                                        [0., 1., 1., 0., 1., 1., 1., 1.])), (5, 1))
    bank = MechanismDataset(feature_names=('green_q', 'switch_indicator'), features=features,
        context_names=(), context=np.empty((40, 0)), priors={},
        targets={key: costs.ravel().copy() for key in ('interval_cost', 'prefix_mean_cost_450s')},
        domains=np.full(40, 'cologne1'), metadata={
            'action_group_ids': np.repeat(groups, 8).tolist(), 'candidate_states': states * 5})
    return records, bank, groups


def test_pre_mask_uses_original_reference_first_and_state_lexical_ties():
    raw, bank, groups = selection_case()
    saved = deepcopy(raw)
    result = pre_mask_records(raw[::-1], bank, groups)
    assert result['group_count'] == 5 and all(result['checks'].values())
    assert raw == saved
    assert [row['group_id'] for row in result['records']] == groups
    assert [row['selected_index'] for row in result['records']] == [2, 3, 2, 3, 3]
    assert [row['baseline_deployed_index'] for row in result['rows']] == [3, 3, 2, 3, 3]
    assert [row['premask_deployed_index'] for row in result['rows']] == [2, 3, 2, 3, 3]
    assert [row['deployed_action_changed'] for row in result['rows']] == [True, False, False, False, False]
    assert result['rows'][0]['allowed_indices'] == [1, 2, 3, 4, 5, 6, 7]
    # Cost-best action1 is not selected: action2 wins the cached-score lexical tie.
    assert result['rows'][0]['realized_gain_difference'] == 1.
    for old, new in zip(raw, result['records']):
        assert new['predicted_scores'] == old['predicted_scores']
        assert new['candidate_states'] == old['candidate_states']
        assert new['heldout_seed'] == old['heldout_seed']


def test_changing_consistent_labels_changes_evaluation_but_never_selection():
    raw, bank, groups = selection_case()
    before = pre_mask_records(raw, bank, groups)
    raw[0]['label_costs'][2] = 4.
    for target in bank.targets.values():
        target[2] = 4.
    after = pre_mask_records(raw, bank, groups)
    assert [row['selected_index'] for row in after['records']] == [row['selected_index'] for row in before['records']]
    assert after['rows'][0]['premask_actual_gain'] == -1.
    assert after['rows'][0]['realized_gain_difference'] == -1.


def test_reference_score_tolerance_and_threshold_still_fall_back_to_pp():
    raw, bank, groups = selection_case()
    raw[0]['predicted_scores'][1:3] = [1., -5e-13]
    assert pre_mask_records(raw, bank, groups)['records'][0]['selected_index'] == 3
    raw[0]['predicted_scores'][2] = -2e-12
    result = pre_mask_records(raw, bank, groups, threshold=2e-12)
    assert result['records'][0]['selected_index'] == 2
    assert result['rows'][0]['premask_deployed_index'] == 3


@pytest.mark.parametrize('mismatch', ['raw_top1', 'candidate_order', 'label'])
def test_raw_score_order_and_bank_mismatch_is_rejected(mismatch):
    raw, bank, groups = selection_case()
    if mismatch == 'raw_top1':
        raw[0]['selected_index'] = 1
    elif mismatch == 'candidate_order':
        raw[0]['candidate_states'][1:3] = raw[0]['candidate_states'][2:0:-1]
    else:
        raw[0]['label_costs'][1] += .2
    with pytest.raises(ValueError):
        pre_mask_records(raw, bank, groups)
