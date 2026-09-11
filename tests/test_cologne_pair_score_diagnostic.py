from copy import deepcopy

import pytest

from scripts.data.cologne_pair_score_diagnostic import diagnose_pair_scores


def score_case():
    group = 'cologne1:seed2027:6:tls'
    row = {'group_id': group, 'seed': 2027, 'reference_index': 3,
        'candidate_states': [f'phase{index}' for index in range(8)],
        'switch_indicators': [0, 1, 1, 1, 1, 1, 1, 1],
        'allowed_indices': [1, 2, 3, 4, 5, 6, 7],
        'predicted_scores': [-100., 8., 8., 10., 12., 10., 7., 10.],
        'label_costs': [-200., 7., 14., 10., 5., 4., 10., 10.]}
    return [row], [group]


def test_both_families_count_all_comparisons_once_with_exclusive_ties_and_raw_gaps():
    rows, groups = score_case()
    original = deepcopy(rows)
    result = diagnose_pair_scores(rows, groups)
    assert rows == original
    assert result['group_count'] == 1 and result['candidate_count'] == 8
    assert result['eligible_switch_count'] == 6 and all(result['checks'].values())
    row = result['rows'][0]
    assert row['eligible_switch_indices'] == [1, 2, 4, 5, 6, 7]
    pp, pairs = row['switch_vs_pp'], row['switch_pairs']
    assert pp['eligible_comparisons'] == 6
    assert pp['counts'] == {'strict_sign_agreement': 1, 'false_positive_benefit': 1,
        'strict_missed_benefit': 1, 'prediction_tie': 1, 'label_tie': 1, 'both_tie': 1}
    assert pp['absolute_label_gap_sums'] == {'strict_sign_agreement': 3., 'false_positive_benefit': 4.,
        'strict_missed_benefit': 5., 'prediction_tie': 6., 'label_tie': 0., 'both_tie': 0.}
    assert pp['total_absolute_label_gap'] == 18.
    assert pairs['eligible_comparisons'] == 15
    assert pairs['counts'] == {'strict_order_agreement': 3, 'strict_order_reversal': 9,
                              'prediction_tie': 2, 'label_tie': 1, 'both_tie': 0}
    assert pairs['absolute_label_gap_sums'] == {'strict_order_agreement': 8., 'strict_order_reversal': 47.,
                                              'prediction_tie': 13., 'label_tie': 0., 'both_tie': 0.}
    assert pairs['total_absolute_label_gap'] == 68.


def test_pp_stay_retained_as_reference_and_tolerance_ties_remain_in_denominators():
    rows, groups = score_case()
    row = rows[0]
    row['switch_indicators'] = [1, 1, 1, 0, 1, 1, 1, 1]
    row['allowed_indices'] = list(range(8))
    row['predicted_scores'] = [0.]*8
    row['label_costs'] = [0.]*8
    row['predicted_scores'][1] = 1e-12
    row['label_costs'][1] = -2e-12
    result = diagnose_pair_scores(rows, groups)['rows'][0]
    assert result['eligible_switch_count'] == 7
    assert result['switch_vs_pp']['eligible_comparisons'] == 7
    assert result['switch_vs_pp']['counts']['prediction_tie'] == 1
    assert result['switch_vs_pp']['counts']['both_tie'] == 6
    assert result['switch_pairs']['eligible_comparisons'] == 21
    assert result['switch_pairs']['counts']['prediction_tie'] == 6
    assert result['switch_pairs']['counts']['both_tie'] == 15
    assert result['switch_pairs']['absolute_label_gap_sums']['prediction_tie'] == pytest.approx(12e-12)


def test_changed_excluded_stay_and_unrelated_selected_actions_do_not_change_diagnostic():
    rows, groups = score_case()
    before = diagnose_pair_scores(rows, groups)
    rows[0]['predicted_scores'][0] = 1e6
    rows[0]['label_costs'][0] = -1e6
    rows[0]['baseline_selected_index'] = 2
    rows[0]['premask_selected_index'] = 4
    assert diagnose_pair_scores(rows, groups) == before


@pytest.mark.parametrize('mismatch', ['roster', 'missing_pp', 'switch_mask', 'label_rows'])
def test_roster_allowed_set_and_action_alignment_mismatches_are_rejected(mismatch):
    rows, groups = score_case()
    if mismatch == 'roster':
        rows.append(deepcopy(rows[0]))
    elif mismatch == 'missing_pp':
        rows[0]['allowed_indices'].remove(3)
    elif mismatch == 'switch_mask':
        rows[0]['switch_indicators'][1] = .5
    else:
        rows[0]['label_costs'].pop()
    with pytest.raises(ValueError):
        diagnose_pair_scores(rows, groups)
