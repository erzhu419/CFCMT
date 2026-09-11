"""Separate cached switch-versus-PP sign errors from switch-pair ordering errors."""

from itertools import combinations
import math


TOLERANCE = 1e-12
PP_CATEGORIES = ('strict_sign_agreement', 'false_positive_benefit', 'strict_missed_benefit',
                 'prediction_tie', 'label_tie', 'both_tie')
SWITCH_PAIR_CATEGORIES = ('strict_order_agreement', 'strict_order_reversal',
                          'prediction_tie', 'label_tie', 'both_tie')


def _summarize(comparisons, *, versus_pp):
    categories = PP_CATEGORIES if versus_pp else SWITCH_PAIR_CATEGORIES
    counts = dict.fromkeys(categories, 0)
    gaps = dict.fromkeys(categories, 0.)
    denominator, total_gap = 0, 0.
    for predicted, actual in comparisons:
        predicted_tie, label_tie = abs(predicted) <= TOLERANCE, abs(actual) <= TOLERANCE
        if predicted_tie and label_tie:
            category = 'both_tie'
        elif predicted_tie:
            category = 'prediction_tie'
        elif label_tie:
            category = 'label_tie'
        elif (predicted > 0.) == (actual > 0.):
            category = 'strict_sign_agreement' if versus_pp else 'strict_order_agreement'
        elif versus_pp:
            category = 'false_positive_benefit' if predicted > 0. else 'strict_missed_benefit'
        else:
            category = 'strict_order_reversal'
        counts[category] += 1
        gaps[category] += abs(actual)
        denominator += 1
        total_gap += abs(actual)
    return {'eligible_comparisons': denominator, 'counts': counts,
            'absolute_label_gap_sums': gaps, 'total_absolute_label_gap': total_gap}


def diagnose_pair_scores(rows, expected_groups):
    expected = list(expected_groups)
    by_group = {row['group_id']: row for row in rows}
    if (not expected or len(set(expected)) != len(expected) or len(by_group) != len(rows)
            or set(by_group) != set(expected)):
        raise ValueError('Use every original group exactly once')
    result = []
    for group in expected:
        row = by_group[group]
        costs, scores = row['label_costs'], row['predicted_scores']
        states, indicators, reference = row['candidate_states'], row['switch_indicators'], row['reference_index']
        if (any(len(values) != 8 for values in (costs, scores, states, indicators))
                or len(set(states)) != 8 or reference not in range(8)
                or row['seed'] != int(group.split(':')[1][4:])
                or not all(math.isfinite(value) for value in (*costs, *scores))
                or any(value not in (0, 1) for value in indicators)):
            raise ValueError(f'{group}: preserve eight aligned finite score/label rows and binary switches')
        allowed = [index for index in range(8) if index == reference or indicators[index] == 1]
        if row['allowed_indices'] != allowed:
            raise ValueError(f'{group}: allowed actions differ from PP plus the original switches')
        switches = [index for index in allowed if index != reference]
        pp = _summarize(((scores[reference]-scores[index], costs[reference]-costs[index])
                         for index in switches), versus_pp=True)
        pairs = _summarize(((scores[first]-scores[second], costs[first]-costs[second])
                            for first, second in combinations(switches, 2)), versus_pp=False)
        for block, denominator in ((pp, len(switches)), (pairs, len(switches)*(len(switches)-1)//2)):
            if (block['eligible_comparisons'] != denominator or sum(block['counts'].values()) != denominator
                    or abs(sum(block['absolute_label_gap_sums'].values())-block['total_absolute_label_gap']) > TOLERANCE):
                raise ValueError(f'{group}: comparison counts or raw cost-gap accounting do not close')
        result.append({'group_id': group, 'seed': row['seed'], 'reference_index': reference,
            'candidate_count': 8, 'allowed_candidate_count': len(allowed),
            'eligible_switch_indices': switches, 'eligible_switch_count': len(switches),
            'switch_vs_pp': pp, 'switch_pairs': pairs})
    return {'rows': result, 'group_count': len(result), 'candidate_count': 8*len(result),
        'eligible_switch_count': sum(row['eligible_switch_count'] for row in result), 'checks': {
            'original_group_roster_exact': True, 'allowed_nonreference_switches_only': True,
            'every_eligible_comparison_counted_once': True, 'exclusive_categories_complete': True,
            'raw_label_gap_accounting_exact': True}}
