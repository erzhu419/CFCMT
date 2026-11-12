"""Causal trend strata on the frozen motion panel, without copying old tables."""
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

for sibling in ('tsc_corridor_late_motion_20261109', 'tsc_corridor_state_stratification_20261109'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/sibling))
from motion_stats import MOTIONS, MotionStats
from stratification_stats import PANELS

TRENDS = ('slowing', 'not_slowing', 'unknown')
FEATURES = ('old_progress_rate', 'recent_progress_rate', 'delta')
STOP = MOTIONS.index('stopped_at10')
MOTION_FIELDS = {'motions', 'motion_gaps', 'late_motion_decomposition',
                 'motion_zero_category_decomposition', 'motion_order', 'motion_reproduction'}


def _strip_motion(value):
    if isinstance(value, dict):
        return {key: _strip_motion(item) for key, item in value.items() if key not in MOTION_FIELDS}
    if isinstance(value, list):
        return [_strip_motion(item) for item in value]
    return value


def _verify(actual, expected, path=''):
    if isinstance(actual, dict):
        if not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise ValueError(f'Previous motion structure changed: {path}')
        for key in actual:
            _verify(actual[key], expected[key], path+'/'+str(key))
    elif isinstance(actual, list):
        if not isinstance(expected, list) or len(actual) != len(expected):
            raise ValueError(f'Previous motion roster changed: {path}')
        for index, (left, right) in enumerate(zip(actual, expected)):
            _verify(left, right, path+'/'+str(index))
    elif isinstance(actual, float) and isinstance(expected, (int, float)):
        if not np.isclose(actual, expected, atol=1e-12, rtol=0):
            raise ValueError(f'Previous motion value changed: {path}')
    elif actual != expected:
        raise ValueError(f'Previous motion value changed: {path}')


def _rates(counts):
    n = int(counts.sum())
    return {name: float(value/n) for name, value in zip(MOTIONS, counts)} if n else None


def _rate(counts):
    return float(counts[STOP]/counts.sum()) if counts.sum() else None


def _difference(left, right):
    return left-right if left is not None and right is not None else None


def _sum(table, keys):
    return sum((table[key] for key in keys), np.zeros(len(MOTIONS), dtype=int))


def _summary(counts, reasons, features, denominator):
    n = int(counts.sum())
    observed = {name: {feature: dict(n=int(features[index, j, 0]), sum=float(features[index, j, 1]),
                    mean=float(features[index, j, 1]/features[index, j, 0]) if features[index, j, 0] else None)
                for j, feature in enumerate(FEATURES)}
                for index, name in enumerate(MOTIONS) if counts[index]}
    return dict(n=n, counts={name: int(value) for name, value in zip(MOTIONS, counts)},
                mixture_fraction=n/denominator if denominator else None, stopped_at10_rate=_rate(counts),
                reason_counts=dict(reasons), progress_features=observed)


class SlowingStats:
    """Use only the passed predecision trend for stratification; motion is a label."""
    def __init__(self, frozen_contexts):
        self.baseline = MotionStats(frozen_contexts)
        self.counts = defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int))
        self.reasons = defaultdict(Counter)
        self.features = defaultdict(lambda: np.zeros((len(MOTIONS), len(FEATURES), 2)))

    def register_hour(self, panel, seed, block):
        self.baseline.register_hour(panel, seed, block)

    def update(self, panel, seed, block, context, stratum, oldcategory, motion_name, unlocked,
               progress_m=None, slowing=None):
        feature = slowing if slowing is not None else dict(group='unknown', reason='missing_feature')
        trend = feature['group']
        if trend not in TRENDS:
            raise ValueError('Expected slowing, not_slowing or unknown predecision trend')
        kept = self.baseline.update(panel, seed, block, context, stratum, oldcategory, motion_name, unlocked, progress_m)
        if kept:
            key = panel, seed, block, tuple(context), tuple(stratum), bool(unlocked), trend
            index = MOTIONS.index(motion_name)
            self.counts[key][index] += 1
            self.reasons[key][feature['reason']] += 1
            for j, name in enumerate(FEATURES):
                if feature.get(name) is not None:
                    self.features[key][index, j] += (1, float(feature[name]))
        return kept

    def _tables(self, panel, block, context, unlocked_reference, dev_seed):
        counts = defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int))
        reasons, features = defaultdict(Counter), defaultdict(lambda: np.zeros((len(MOTIONS), len(FEATURES), 2)))
        for key, values in self.counts.items():
            part, seed, item_block, ctx, state, unlocked, trend = key
            if (part == panel and ctx == context and (panel == 'source' or block is None or item_block == block)
                    and not (panel == 'dev' and dev_seed is not None and seed != dev_seed)
                    and not (panel == 'dev' and unlocked_reference and not unlocked)):
                counts[state, trend] += values
                reasons[state, trend].update(self.reasons[key])
                features[state, trend] += self.features[key]
        return counts, reasons, features

    @staticmethod
    def _combined(table, reasons, features, keys, denominator):
        reason_counts = Counter()
        for key in keys:
            reason_counts.update(reasons[key])
        moments = sum((features[key] for key in keys), np.zeros((len(MOTIONS), len(FEATURES), 2)))
        return _summary(_sum(table, keys), reason_counts, moments, denominator)

    def _compare(self, old, block, unlocked_reference, dev_seed=None):
        tables = {panel: self._tables(panel, block, tuple(old['context']), unlocked_reference, dev_seed) for panel in PANELS}
        states = {tuple(row['stratum']) for row in old['common_strata']}
        joint = set.intersection(*(set(key for key in table if key[0] in states and key[1] != 'unknown')
                                   for table, reasons, features in tables.values()))
        dev = tables['dev'][0]
        dev_joint_n = int(_sum(dev, joint).sum())
        weights = {key: int(dev[key].sum())/dev_joint_n for key in joint} if dev_joint_n else {}
        state_weights = {state: sum(weight for (item_state, trend), weight in weights.items() if item_state == state)
                         for state, trend in joint}
        groups = {}
        by_state = {state: dict(stratum=list(state), dev_reference_weight=row['dev_reference_weight'], panels={})
                    for row in old['common_strata'] for state in (tuple(row['stratum']),)}
        for panel, (table, reasons, features) in tables.items():
            original = old['groups'][panel]
            c0keys = [key for key in table if key[0] in states]
            c0 = _sum(table, c0keys)
            c1 = _sum(table, joint)
            if int(c0.sum()) != original['common_n']:
                raise ValueError('Trend strata changed the original common cohort')
            known_n = int(_sum(table, [key for key in c0keys if key[1] != 'unknown']).sum())
            standard = sum((weight*table[key]/table[key].sum() for key, weight in weights.items()), np.zeros(len(MOTIONS))) if weights else None
            state_only = sum(weight*_rate(_sum(table, [key for key in joint if key[0] == state]))
                             for state, weight in state_weights.items()) if weights else None
            unknown_keys = [key for key in c0keys if key[1] == 'unknown']
            old_rates = original['motions']['standardized']['rates']
            groups[panel] = dict(full_n=original['n'], original_common_n=int(c0.sum()),
                original_common_coverage=original['common_support_coverage'],
                original_common_stopped_at10_rate=_rate(c0),
                original_standardized_stopped_at10_rate=old_rates['stopped_at10'] if old_rates is not None else None,
                known_trend_n=known_n, unknown=self._combined(table, reasons, features, unknown_keys, int(c0.sum())),
                joint_common_n=int(c1.sum()), joint_coverage_full=int(c1.sum())/original['n'] if original['n'] else None,
                joint_coverage_original_common=int(c1.sum())/int(c0.sum()) if c0.sum() else None,
                joint_counts={name: int(value) for name, value in zip(MOTIONS, c1)},
                joint_unweighted_stopped_at10_rate=_rate(c1),
                joint_state_only_standardized_stopped_at10_rate=state_only,
                joint_standardized_rates={name: float(value) for name, value in zip(MOTIONS, standard)} if standard is not None else None)
            for state in states:
                state_keys = [key for key in c0keys if key[0] == state]
                total = int(_sum(table, state_keys).sum())
                by_state[state]['panels'][panel] = dict(n=total, stopped_at10_rate=_rate(_sum(table, state_keys)),
                    trends={trend: self._combined(table, reasons, features, [key for key in state_keys if key[1] == trend], total)
                            for trend in TRENDS})
        pairs = {}
        for pair, panel in (('source_vs_dev', 'source'), ('local_vs_dev', 'local_adapt')):
            a, b = groups[panel], groups['dev']
            gap = {name: _difference(a[key], b[key]) for name, key in (
                ('original_common', 'original_common_stopped_at10_rate'),
                ('original_standardized', 'original_standardized_stopped_at10_rate'),
                ('joint_common_unweighted', 'joint_unweighted_stopped_at10_rate'),
                ('joint_state_only_standardized', 'joint_state_only_standardized_stopped_at10_rate'))}
            gap['joint_standardized'] = _difference(a['joint_standardized_rates']['stopped_at10'] if a['joint_standardized_rates'] else None,
                                                   b['joint_standardized_rates']['stopped_at10'] if b['joint_standardized_rates'] else None)
            gap['cohort_change'] = _difference(gap['joint_common_unweighted'], gap['original_common'])
            gap['trend_composition_fixed_joint_cohort'] = _difference(gap['joint_state_only_standardized'], gap['joint_standardized'])
            pairs[pair] = gap
        return dict(context=old['context'], groups=groups, pairs=pairs,
            original_common_states=[by_state[state] for state in sorted(states)],
            joint_common_strata=[dict(stratum=list(state), trend=trend, dev_reference_weight=weights[state, trend])
                                  for state, trend in sorted(joint)])

    def _hours(self):
        result = []
        for panel, seed, block in sorted(self.baseline.hours, key=repr):
            for context in self.baseline.contexts:
                entries = []
                totals = Counter()
                for key, counts in self.counts.items():
                    part, item_seed, item_block, ctx, state, unlocked, trend = key
                    if (part, item_seed, item_block, ctx) == (panel, seed, block, context):
                        entries.append((key, counts))
                        totals[state, unlocked] += int(counts.sum())
                rows = [dict(stratum=list(key[4]), unlocked=key[5], trend=key[6],
                        **_summary(counts, self.reasons[key], self.features[key], totals[key[4], key[5]]))
                        for key, counts in entries]
                result.append(dict(panel=panel, seed=seed, block=block, context=list(context), strata=rows))
        return result

    def result(self, previous_target):
        old = self.baseline.result(_strip_motion(previous_target))
        _verify(old, previous_target)
        comparisons = {}
        for scope, prior in old['comparisons'].items():
            unlocked = scope == 'unlocked'
            comparisons[scope] = dict(
                by_block=[dict(block=block['block'], contexts=[self._compare(row, block['block'], unlocked)
                    for row in block['contexts']]) for block in prior['by_block']],
                by_dev_seed=[dict(seed=seed['seed'], block=seed['block'], contexts=[self._compare(row, seed['block'], unlocked, seed['seed'])
                    for row in seed['contexts']]) for seed in prior['by_dev_seed']],
                pooled=[self._compare(row, None, unlocked) for row in prior['pooled']])
        return dict(frozen_contexts=[list(context) for context in self.baseline.contexts], trends=list(TRENDS),
            motion_order=list(MOTIONS), reproduction=dict(previous_motion_panel=True, float_tolerance=1e-12),
            observation_unit='correlated flight_decision_snapshot', by_hour=self._hours(), comparisons=comparisons)
