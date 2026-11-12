"""Split detector category zero while preserving the previous state analysis."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tsc_corridor_state_stratification_20261109'))
from stratification_stats import PANELS, PAIRS, StratificationStats

OUTCOMES = ('straight_by10', 'straight_after10', 'other_by10', 'other_after10',
            'terminal_by10', 'terminal_after10', 'censored')
COHORTS = ('full_coarse', 'common_unweighted', 'standardized_within', 'composition', 'full_to_common')


def _verify_old(actual, expected, path=''):
    """Counts and raw CDFs are exact; standardized sums allow roundoff only."""
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise ValueError(f'Previous stratification structure changed: {path}')
        for key in actual:
            _verify_old(actual[key], expected[key], path+'/'+str(key))
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f'Previous stratification roster changed: {path}')
        for index, (left, right) in enumerate(zip(actual, expected)):
            _verify_old(left, right, path+'/'+str(index))
    else:
        tolerance = 1e-12 if any(word in path for word in ('standardized', 'gap', 'mean_absolute', 'full_to_common_shift')) else 0
        equal = abs(actual-expected) <= tolerance if isinstance(actual, float) and isinstance(expected, (int, float)) else actual == expected
        if not equal:
            raise ValueError(f'Previous stratification value changed: {path}')


def _rate_dict(values):
    if values is None:
        return None
    values = np.asarray(values, dtype=float)
    if not np.isclose(values.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError('Seven outcome probabilities do not sum to one')
    return {name: float(value) for name, value in zip(OUTCOMES, values)}


def _mass(counts, previous_counts):
    n = int(counts.sum())
    if (n != sum(previous_counts) or int(counts[0]) != sum(previous_counts[1:]) or
            int(counts[1:].sum()) != previous_counts[0]):
        raise ValueError('Seven outcome counts do not reproduce the previous detector categories')
    return dict(counts={name: int(value) for name, value in zip(OUTCOMES, counts)},
                rates=_rate_dict(counts/n) if n else None)


def _gap(left, right):
    return {name: left[name]-right[name] for name in OUTCOMES} if left is not None and right is not None else None


def _identity(gap):
    if gap is None:
        return None
    positive, opposite_zero = gap['straight_by10'], -sum(gap[name] for name in OUTCOMES[1:])
    if not np.isclose(positive, opposite_zero, atol=1e-12, rtol=0):
        raise ValueError('Straight-by-ten gap differs from the opposite sum of category-zero outcome gaps')
    return dict(straight_by10_gap=positive, opposite_zero_gap_sum=opposite_zero)


class OutcomeStats(StratificationStats):
    def __init__(self, frozen_contexts):
        super().__init__(frozen_contexts)
        self.outcomes = defaultdict(lambda: np.zeros(len(OUTCOMES), dtype=int))

    def update(self, panel, seed, block, context, stratum, oldcategory, outcome_name, unlocked):
        if outcome_name not in OUTCOMES or (outcome_name == 'straight_by10') != (oldcategory > 0):
            raise ValueError('Named outcome does not match the original detector category')
        kept = super().update(panel, seed, block, context, stratum, oldcategory, unlocked)
        if kept:
            self.outcomes[panel, seed, block, tuple(context), tuple(stratum), bool(unlocked)][OUTCOMES.index(outcome_name)] += 1
        return kept

    def _outcome_table(self, panel, block, context, unlocked_reference, dev_seed):
        table = defaultdict(lambda: np.zeros(len(OUTCOMES), dtype=int))
        for (part, seed, item_block, ctx, state, unlocked), counts in self.outcomes.items():
            if (part == panel and ctx == context and (panel == 'source' or block is None or item_block == block)
                    and not (panel == 'dev' and dev_seed is not None and seed != dev_seed)
                    and not (panel == 'dev' and unlocked_reference and not unlocked)):
                table[state] += counts
        return dict(table)

    def _attach_comparison(self, record, block, unlocked_reference, dev_seed=None):
        context = tuple(record['context'])
        tables = {panel: self._outcome_table(panel, block, context, unlocked_reference, dev_seed) for panel in PANELS}
        weights = {tuple(row['stratum']): row['dev_reference_weight'] for row in record['common_strata']}
        for panel, table in tables.items():
            group = record['groups'][panel]
            full = sum(table.values(), np.zeros(len(OUTCOMES), dtype=int))
            common = sum((table[state] for state in weights), np.zeros(len(OUTCOMES), dtype=int))
            standardized = sum((weight*table[state]/table[state].sum() for state, weight in weights.items()),
                               np.zeros(len(OUTCOMES))) if weights else None
            group['outcomes'] = dict(full=_mass(full, group['counts']),
                common_unweighted=_mass(common, group['common_counts']),
                standardized=dict(rates=_rate_dict(standardized)))
            for name, old_cdf in (('full', group['full_cdf']), ('common_unweighted', group['common_unweighted_cdf']),
                                  ('standardized', group['standardized_cdf'])):
                rates = group['outcomes'][name]['rates']
                if rates is not None and not np.isclose(rates['straight_by10'], old_cdf[-1], atol=1e-12, rtol=0):
                    raise ValueError('Named straight outcome differs from the original ten-second CDF')
            for row in group['strata']:
                row['outcomes'] = _mass(table[tuple(row['stratum'])], row['counts'])
        for row in record['common_strata']:
            state = tuple(row['stratum'])
            for panel, values in row['panels'].items():
                values['outcomes'] = _mass(tables[panel][state], values['counts'])
        for pair, panel in (('source_vs_dev', 'source'), ('local_vs_dev', 'local_adapt')):
            left, right = record['groups'][panel]['outcomes'], record['groups']['dev']['outcomes']
            gaps = {name: _gap(left[key]['rates'], right[key]['rates']) for name, key in
                    (('full_coarse', 'full'), ('common_unweighted', 'common_unweighted'), ('standardized_within', 'standardized'))}
            gaps['composition'] = _gap(gaps['common_unweighted'], gaps['standardized_within'])
            gaps['full_to_common'] = _gap(gaps['common_unweighted'], gaps['full_coarse'])
            record['pairs'][pair]['outcome_gaps'] = gaps
            record['pairs'][pair]['zero_category_decomposition'] = {cohort: _identity(gap) for cohort, gap in gaps.items()}

    def _attach_hours(self, result):
        for row in result['by_hour']:
            full = np.zeros(len(OUTCOMES), dtype=int)
            for state in row['strata']:
                counts = self.outcomes[row['panel'], row['seed'], row['block'], tuple(row['context']),
                                       tuple(state['stratum']), state['unlocked']]
                full += counts
                state['outcomes'] = _mass(counts, state['counts'])
            row['outcomes'] = _mass(full, row['counts'])

    def result(self, previous_target_result):
        result = super().result()
        _verify_old(result, previous_target_result)
        self._attach_hours(result)
        for scope, comparison in result['comparisons'].items():
            unlocked = scope == 'unlocked'
            for block in comparison['by_block']:
                for row in block['contexts']:
                    self._attach_comparison(row, block['block'], unlocked)
            for seed in comparison['by_dev_seed']:
                for row in seed['contexts']:
                    self._attach_comparison(row, seed['block'], unlocked, dev_seed=seed['seed'])
            for row in comparison['pooled']:
                self._attach_comparison(row, None, unlocked)
            for index, equal in enumerate(comparison['equal_block_summary']):
                for pair in PAIRS:
                    blocks = [block['contexts'][index]['pairs'][pair]['outcome_gaps'] for block in comparison['by_block']]
                    gaps = {cohort: {name: float(np.mean([block[cohort][name] for block in blocks])) for name in OUTCOMES}
                            if all(block[cohort] is not None for block in blocks) else None for cohort in COHORTS}
                    equal['pairs'][pair]['outcome_gaps'] = gaps
                    equal['pairs'][pair]['zero_category_decomposition'] = {cohort: _identity(gap) for cohort, gap in gaps.items()}
        result['outcome_order'] = list(OUTCOMES)
        result['old_reproduction'] = dict(raw_counts_and_hour_cdfs=True, state_membership_and_weights=True,
                                         block_seed_support_and_cdfs=True, derived_float_tolerance=1e-12)
        return result
