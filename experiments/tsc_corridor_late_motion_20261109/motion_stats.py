"""Split late straight arrivals by observed motion at the ten-second endpoint."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

for sibling in ('tsc_corridor_outcome_decomposition_20261109', 'tsc_corridor_state_stratification_20261109'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/sibling))
from outcome_stats import COHORTS, OUTCOMES, OutcomeStats, _verify_old
from stratification_stats import PANELS, PAIRS

LATE_MOTIONS = ('stopped_at10', 'moving_straight_at10', 'moving_other_lane_at10')
MOTIONS = ('straight_by10', *LATE_MOTIONS, *OUTCOMES[2:])
MOVING_INDEX = MOTIONS.index('moving_straight_at10')
OUTCOME_FIELDS = {'outcomes', 'outcome_gaps', 'zero_category_decomposition', 'outcome_order', 'old_reproduction'}


def _without_outcomes(value):
    if isinstance(value, dict):
        return {key: _without_outcomes(item) for key, item in value.items() if key not in OUTCOME_FIELDS}
    if isinstance(value, list):
        return [_without_outcomes(item) for item in value]
    return value


def _collapse(values):
    return {name: sum(values[motion] for motion in LATE_MOTIONS) if name == 'straight_after10' else values[name]
            for name in OUTCOMES}


def _rates(values, old_rates):
    if values is None:
        if old_rates is not None:
            raise ValueError('Motion decomposition changed outcome support')
        return None
    result = {name: float(value) for name, value in zip(MOTIONS, values)}
    if not np.isclose(sum(result.values()), 1, atol=1e-12, rtol=0):
        raise ValueError('Nine motion probabilities do not sum to one')
    collapsed = _collapse(result)
    if old_rates is None or any(not np.isclose(collapsed[name], old_rates[name], atol=1e-12, rtol=0) for name in OUTCOMES):
        raise ValueError('Nine motion probabilities do not reproduce the previous seven outcomes')
    return result


def _progress_metrics(moving, comparable, moment):
    return dict(moving_straight_probability=moving, comparable_progress_probability=comparable,
        comparable_given_moving=comparable/moving if moving else None,
        progress_m_per_100_eligible=100*moment if moment is not None else None,
        realized_speed_mps_sum_per_100_eligible=10*moment if moment is not None else None,
        conditional_mean_progress_m=moment/comparable if comparable else None,
        conditional_mean_realized_speed_mps=moment/(10*comparable) if comparable else None)


def _raw_mass(counts, progress, old_mass):
    n, comparable_n = int(counts.sum()), int(progress[0])
    values = {name: int(value) for name, value in zip(MOTIONS, counts)}
    if _collapse(values) != old_mass['counts']:
        raise ValueError('Nine motion counts do not reproduce the previous seven outcome counts')
    rates = _rates(counts/n if n else None, old_mass['rates'])
    summary = _progress_metrics(counts[MOVING_INDEX]/n if n else None, comparable_n/n if n else None,
                                float(progress[1]/n) if n else None)
    summary.update(eligible_n=n, moving_straight_n=int(counts[MOVING_INDEX]), progress_n=comparable_n,
                   progress_sum_m=float(progress[1]), realized_speed_sum_mps=float(progress[1]/10))
    return dict(counts=values, rates=rates, progress=summary)


def _gap(left, right):
    return {name: left[name]-right[name] for name in MOTIONS} if left is not None and right is not None else None


def _identities(gaps, old_gaps):
    late, zero = {}, {}
    for cohort, gap in gaps.items():
        if gap is None:
            late[cohort] = zero[cohort] = None
            continue
        three = sum(gap[name] for name in LATE_MOTIONS)
        old_late = old_gaps[cohort]['straight_after10']
        opposite = -sum(gap[name] for name in MOTIONS[1:])
        if not np.isclose(three, old_late, atol=1e-12, rtol=0) or not np.isclose(gap['straight_by10'], opposite, atol=1e-12, rtol=0):
            raise ValueError('Late-motion or category-zero gap identity failed')
        late[cohort] = dict(straight_after10_gap=old_late, three_motion_gap_sum=three)
        zero[cohort] = dict(straight_by10_gap=gap['straight_by10'], opposite_zero_gap_sum=opposite)
    return late, zero


class MotionStats(OutcomeStats):
    def __init__(self, frozen_contexts):
        super().__init__(frozen_contexts)
        self.motions = defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int))
        self.progress = defaultdict(lambda: np.zeros(2))

    def update(self, panel, seed, block, context, stratum, oldcategory, motion_name, unlocked, progress_m=None):
        if motion_name not in MOTIONS or (progress_m is not None and motion_name != 'moving_straight_at10'):
            raise ValueError('Progress is defined only for comparable moving-straight late snapshots')
        outcome = 'straight_after10' if motion_name in LATE_MOTIONS else motion_name
        kept = super().update(panel, seed, block, context, stratum, oldcategory, outcome, unlocked)
        if kept:
            key = panel, seed, block, tuple(context), tuple(stratum), bool(unlocked)
            self.motions[key][MOTIONS.index(motion_name)] += 1
            if progress_m is not None:
                self.progress[key] += (1, progress_m)
        return kept

    def _motion_tables(self, panel, block, context, unlocked_reference, dev_seed):
        counts = defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int))
        progress = defaultdict(lambda: np.zeros(2))
        for key, values in self.motions.items():
            part, seed, item_block, ctx, state, unlocked = key
            if (part == panel and ctx == context and (panel == 'source' or block is None or item_block == block)
                    and not (panel == 'dev' and dev_seed is not None and seed != dev_seed)
                    and not (panel == 'dev' and unlocked_reference and not unlocked)):
                counts[state] += values
                progress[state] += self.progress.get(key, np.zeros(2))
        return counts, progress

    def _attach_motion_comparison(self, record, block, unlocked_reference, dev_seed=None):
        weights = {tuple(row['stratum']): row['dev_reference_weight'] for row in record['common_strata']}
        tables = {}
        for panel in PANELS:
            table, progress = self._motion_tables(panel, block, tuple(record['context']), unlocked_reference, dev_seed)
            tables[panel] = table, progress
            group = record['groups'][panel]
            full = sum(table.values(), np.zeros(len(MOTIONS), dtype=int))
            common = sum((table[state] for state in weights), np.zeros(len(MOTIONS), dtype=int))
            full_progress = sum(progress.values(), np.zeros(2))
            common_progress = sum((progress[state] for state in weights), np.zeros(2))
            standardized = sum((weight*table[state]/table[state].sum() for state, weight in weights.items()),
                               np.zeros(len(MOTIONS))) if weights else None
            valid = sum(weight*progress[state][0]/table[state].sum() for state, weight in weights.items()) if weights else None
            moment = sum(weight*progress[state][1]/table[state].sum() for state, weight in weights.items()) if weights else None
            group['motions'] = dict(full=_raw_mass(full, full_progress, group['outcomes']['full']),
                common_unweighted=_raw_mass(common, common_progress, group['outcomes']['common_unweighted']),
                standardized=dict(rates=_rates(standardized, group['outcomes']['standardized']['rates']),
                    progress=_progress_metrics(float(standardized[MOVING_INDEX]) if weights else None, valid, moment)))
            for row in group['strata']:
                state = tuple(row['stratum'])
                row['motions'] = _raw_mass(table[state], progress[state], row['outcomes'])
        for row in record['common_strata']:
            state = tuple(row['stratum'])
            for panel, values in row['panels'].items():
                table, progress = tables[panel]
                values['motions'] = _raw_mass(table[state], progress[state], values['outcomes'])
        for pair, panel in (('source_vs_dev', 'source'), ('local_vs_dev', 'local_adapt')):
            left, right = record['groups'][panel]['motions'], record['groups']['dev']['motions']
            gaps = {name: _gap(left[key]['rates'], right[key]['rates']) for name, key in
                    (('full_coarse', 'full'), ('common_unweighted', 'common_unweighted'), ('standardized_within', 'standardized'))}
            gaps['composition'] = _gap(gaps['common_unweighted'], gaps['standardized_within'])
            gaps['full_to_common'] = _gap(gaps['common_unweighted'], gaps['full_coarse'])
            values = record['pairs'][pair]
            values['motion_gaps'] = gaps
            values['late_motion_decomposition'], values['motion_zero_category_decomposition'] = _identities(gaps, values['outcome_gaps'])

    def _attach_motion_hours(self, result):
        for row in result['by_hour']:
            full, progress = np.zeros(len(MOTIONS), dtype=int), np.zeros(2)
            for state in row['strata']:
                key = row['panel'], row['seed'], row['block'], tuple(row['context']), tuple(state['stratum']), state['unlocked']
                counts, current_progress = self.motions[key], self.progress.get(key, np.zeros(2))
                full += counts; progress += current_progress
                state['motions'] = _raw_mass(counts, current_progress, state['outcomes'])
            row['motions'] = _raw_mass(full, progress, row['outcomes'])

    def result(self, previous_outcome_result):
        result = super().result(_without_outcomes(previous_outcome_result))
        _verify_old(result, previous_outcome_result)
        self._attach_motion_hours(result)
        for scope, comparison in result['comparisons'].items():
            for block in comparison['by_block']:
                for row in block['contexts']:
                    self._attach_motion_comparison(row, block['block'], scope == 'unlocked')
            for seed in comparison['by_dev_seed']:
                for row in seed['contexts']:
                    self._attach_motion_comparison(row, seed['block'], scope == 'unlocked', seed['seed'])
            for row in comparison['pooled']:
                self._attach_motion_comparison(row, None, scope == 'unlocked')
            for index, equal in enumerate(comparison['equal_block_summary']):
                for pair in PAIRS:
                    blocks = [block['contexts'][index]['pairs'][pair]['motion_gaps'] for block in comparison['by_block']]
                    gaps = {cohort: {name: float(np.mean([block[cohort][name] for block in blocks])) for name in MOTIONS}
                            if all(block[cohort] is not None for block in blocks) else None for cohort in COHORTS}
                    values = equal['pairs'][pair]
                    values['motion_gaps'] = gaps
                    values['late_motion_decomposition'], values['motion_zero_category_decomposition'] = _identities(gaps, values['outcome_gaps'])
        result['motion_order'] = list(MOTIONS)
        result['motion_reproduction'] = dict(previous_seven_outcomes=True, previous_stratification=True,
                                            late_split_and_zero_gap_identities=True)
        return result
