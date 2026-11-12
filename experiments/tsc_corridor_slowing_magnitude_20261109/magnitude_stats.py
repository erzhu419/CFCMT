"""Training-only magnitude cuts and fixed-cohort descriptive risk comparisons."""
from bisect import bisect_left
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

for sibling in ('tsc_corridor_predecision_slowing_20261109', 'tsc_corridor_late_motion_20261109',
                'tsc_corridor_outcome_decomposition_20261109', 'tsc_corridor_state_stratification_20261109'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/sibling))
from slowing_stats import SlowingStats, _verify
from motion_stats import MOTIONS, MotionStats
from outcome_stats import OutcomeStats
from stratification_stats import KNOWN_STATES, PANELS, StratificationStats

STOP = MOTIONS.index('stopped_at10')
QUANTILES = (.25, .5, .75)


def _sum(table, keys):
    return sum((table[key] for key in keys), np.zeros(len(MOTIONS), dtype=int))


def _risk(counts):
    return float(counts[STOP]/counts.sum()) if counts.sum() else None


def _difference(a, b):
    return a-b if a is not None and b is not None else None


def _balanced_cuts(source, local):
    """Exact empirical quartiles of the equally weighted two-panel mixture."""
    ns, nl = sum(source.values()), sum(local.values())
    if not ns or not nl:
        return None
    cumulative_source = cumulative_local = 0
    cuts, next_quantile = [], 1
    for value in sorted(source.keys() | local.keys()):
        cumulative_source += source[value]
        cumulative_local += local[value]
        while next_quantile <= 3 and 2*(cumulative_source*nl+cumulative_local*ns) >= next_quantile*ns*nl:
            if not cuts or cuts[-1] != value:
                cuts.append(value)
            next_quantile += 1
    return cuts


class MagnitudeStats:
    def __init__(self, contexts):
        self.contexts = tuple(tuple(context) for context in contexts)
        self.baseline = SlowingStats(self.contexts)
        self.frozen = False
        self.training_records = []
        self.cuts = {}
        self.counts = defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int))
        self.moments = defaultdict(lambda: np.zeros(2))
        self.reasons = defaultdict(Counter)

    def register_hour(self, panel, seed, block):
        if panel == 'dev' and not self.frozen:
            raise ValueError('Development records require frozen training cuts')
        self.baseline.register_hour(panel, seed, block)

    def update(self, panel, seed, block, context, stratum, oldcategory, motion_name, unlocked,
               progress_m=None, slowing=None, magnitude=None):
        if panel == 'dev' and not self.frozen:
            raise ValueError('Development records require frozen training cuts')
        if panel != 'dev' and self.frozen:
            raise ValueError('Training records cannot change after cuts are frozen')
        kept = self.baseline.update(panel, seed, block, context, stratum, oldcategory, motion_name,
                                    unlocked, progress_m, slowing)
        if kept:
            feature = magnitude if magnitude is not None else dict(value=None, reason='missing_feature')
            record = (panel, seed, block, tuple(context), tuple(stratum), bool(unlocked),
                      slowing['group'] if slowing is not None else 'unknown', MOTIONS.index(motion_name),
                      feature['value'], feature['reason'])
            if panel == 'dev':
                self._add(record, block)
            else:
                self.training_records.append(record)
        return kept

    def _bin(self, trend, value, block):
        if trend == 'not_slowing':
            return 'N/A'
        if trend == 'unknown' or value is None or self.cuts[block] is None:
            return 'unknown'
        return bisect_left(self.cuts[block], value)

    def _add(self, record, binning_block):
        panel, seed, block, context, state, unlocked, trend, outcome, value, reason = record
        bin_id = self._bin(trend, value, binning_block)
        key = binning_block, panel, seed, block, context, state, unlocked, trend, bin_id
        self.counts[key][outcome] += 1
        if value is not None and trend == 'slowing':
            self.moments[key] += (1, value)
        self.reasons[key][reason] += 1

    def freeze_training(self):
        if self.frozen:
            raise ValueError('Training cuts are already frozen')
        training_n = {}
        for block in range(3):
            source, local = Counter(), Counter()
            for panel, seed, item_block, context, state, unlocked, trend, outcome, value, reason in self.training_records:
                if trend == 'slowing' and state in KNOWN_STATES and value is not None:
                    if panel == 'source':
                        source[value] += 1
                    elif item_block == block:
                        local[value] += 1
            self.cuts[block] = _balanced_cuts(source, local)
            training_n[str(block)] = dict(source=sum(source.values()), local_adapt=sum(local.values()))
        for record in self.training_records:
            for block in range(3) if record[0] == 'source' else (record[2],):
                self._add(record, block)
        self.training_records.clear()
        self.frozen = True
        curves = {}
        for block in range(3):
            curves[str(block)] = {}
            for panel in ('source', 'local_adapt'):
                keys = [key for key in self.counts if key[0] == block and key[1] == panel and
                        key[5] in KNOWN_STATES and key[7] == 'slowing']
                bins = sorted({key[8] for key in keys}, key=repr)
                curves[str(block)][panel] = [dict(bin=bin_id, **self._cell_summary(
                    [key for key in keys if key[8] == bin_id])) for bin_id in bins]
        self.training_artifact = dict(quantiles=list(QUANTILES), panel_weights=dict(source=.5, local_adapt=.5),
            cuts_by_block={str(block): self.cuts[block] for block in range(3)}, train_n_by_panel=training_n,
            bin_rule='value <= cut belongs to the lower bin; duplicate cuts merged', training_curves=curves)
        return self.training_artifact

    def _cell_summary(self, keys):
        counts = _sum(self.counts, keys)
        n = int(counts.sum())
        moment = sum((self.moments[key] for key in keys), np.zeros(2))
        reasons = Counter()
        for key in keys:
            reasons.update(self.reasons[key])
        return dict(n=n, stopped_at10_n=int(counts[STOP]), stopped_at10_rate=_risk(counts),
            magnitude_n=int(moment[0]), magnitude_sum=float(moment[1]),
            magnitude_mean=float(moment[1]/moment[0]) if moment[0] else None, reason_counts=dict(reasons))

    def _tables(self, block, context, unlocked_reference, dev_seed):
        tables = {panel: defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int)) for panel in PANELS}
        keys = {panel: defaultdict(list) for panel in PANELS}
        for key, values in self.counts.items():
            cut_block, panel, seed, item_block, ctx, state, unlocked, trend, bin_id = key
            if (cut_block == block and ctx == context and
                    not (panel == 'dev' and dev_seed is not None and seed != dev_seed) and
                    not (panel == 'dev' and unlocked_reference and not unlocked)):
                tables[panel][state, trend, bin_id] += values
                keys[panel][state, trend, bin_id].append(key)
        return tables, keys

    def _compare(self, prior, block, unlocked_reference, dev_seed=None):
        tables, keys = self._tables(block, tuple(prior['context']), unlocked_reference, dev_seed)
        c1 = {(tuple(row['stratum']), row['trend']) for row in prior['joint_common_strata']}
        c2 = set.intersection(*(set(key for key in table if key[:2] in c1 and key[2] != 'unknown')
                                for table in tables.values()))
        dev_n = int(_sum(tables['dev'], c2).sum())
        weights = {key: int(tables['dev'][key].sum())/dev_n for key in c2} if dev_n else {}
        trend_weights = {base: sum(weight for key, weight in weights.items() if key[:2] == base)
                         for base in {key[:2] for key in c2}}
        groups = {}
        all_c1_bins = set.union(*(set(key for key in table if key[:2] in c1) for table in tables.values()))
        for panel, table in tables.items():
            c1_keys = [key for key in table if key[:2] in c1]
            c1_counts, c2_counts = _sum(table, c1_keys), _sum(table, c2)
            previous = prior['groups'][panel]
            if int(c1_counts.sum()) != previous['joint_common_n']:
                raise ValueError('Magnitude bins changed the original binary common cohort')
            state_trend = sum(weight*_risk(_sum(table, [key for key in c2 if key[:2] == base]))
                              for base, weight in trend_weights.items()) if weights else None
            refined = sum(weight*_risk(table[key]) for key, weight in weights.items()) if weights else None
            unknown = [raw for key in c1_keys if key[2] == 'unknown' for raw in keys[panel][key]]
            groups[panel] = dict(C1_n=int(c1_counts.sum()), C1_stopped_at10_rate=_risk(c1_counts),
                C1_standardized_stopped_at10_rate=previous['joint_standardized_rates']['stopped_at10']
                    if previous['joint_standardized_rates'] is not None else None,
                C1_counts={name: int(value) for name, value in zip(MOTIONS, c1_counts)},
                magnitude_unknown=self._cell_summary(unknown), C2_n=int(c2_counts.sum()),
                C2_coverage_C1=int(c2_counts.sum())/int(c1_counts.sum()) if c1_counts.sum() else None,
                C2_coverage_full=int(c2_counts.sum())/previous['full_n'] if previous['full_n'] else None,
                C2_counts={name: int(value) for name, value in zip(MOTIONS, c2_counts)},
                C2_unweighted_stopped_at10_rate=_risk(c2_counts),
                C2_trend_standardized_stopped_at10_rate=state_trend,
                C2_magnitude_standardized_stopped_at10_rate=refined)
        pairs = {}
        for pair, panel in (('source_vs_dev', 'source'), ('local_vs_dev', 'local_adapt')):
            a, b = groups[panel], groups['dev']
            gap = {name: _difference(a[key], b[key]) for name, key in (
                ('C1_unweighted', 'C1_stopped_at10_rate'), ('C1_standardized', 'C1_standardized_stopped_at10_rate'),
                ('C2_unweighted', 'C2_unweighted_stopped_at10_rate'),
                ('C2_trend_standardized', 'C2_trend_standardized_stopped_at10_rate'),
                ('C2_magnitude_standardized', 'C2_magnitude_standardized_stopped_at10_rate'))}
            gap['cohort_change_unweighted'] = _difference(gap['C2_unweighted'], gap['C1_unweighted'])
            gap['cohort_and_reference_change_standardized'] = _difference(gap['C2_trend_standardized'], gap['C1_standardized'])
            gap['magnitude_composition_fixed_C2'] = _difference(gap['C2_trend_standardized'], gap['C2_magnitude_standardized'])
            pairs[pair] = gap
        return dict(context=prior['context'], groups=groups, pairs=pairs,
            bins=[dict(stratum=list(state), trend=trend, bin=bin_id, common_C2=(state, trend, bin_id) in c2,
                dev_reference_weight=weights.get((state, trend, bin_id)), panels={panel: self._cell_summary(keys[panel][state, trend, bin_id])
                    for panel in PANELS}) for state, trend, bin_id in sorted(all_c1_bins, key=repr)])

    @staticmethod
    def _equal(records, expected):
        result = []
        for index in range(len(records[0]['contexts'])):
            rows = [record['contexts'][index] for record in records]
            pairs = {}
            for pair in ('source_vs_dev', 'local_vs_dev'):
                pairs[pair] = {metric: float(np.mean([row['pairs'][pair][metric] for row in rows]))
                    if len(rows) == expected and all(row['pairs'][pair][metric] is not None for row in rows) else None
                    for metric in rows[0]['pairs'][pair]}
                pairs[pair]['complete_support_panels'] = sum(row['pairs'][pair]['C2_magnitude_standardized'] is not None for row in rows)
            result.append(dict(context=rows[0]['context'], pairs=pairs, expected_panels=expected, observed_panels=len(rows)))
        return result

    def _binary_result(self):
        motion = self.baseline.baseline
        seven = OutcomeStats.result(motion, StratificationStats.result(motion))
        nine = MotionStats.result(motion, seven)
        return self.baseline.result(nine)

    def result(self, previous_binary):
        if not self.frozen:
            raise ValueError('Training cuts must be frozen before comparisons')
        prior = self._binary_result()
        _verify(prior, previous_binary)
        comparisons = {}
        for scope, old in prior['comparisons'].items():
            blocks = [dict(block=block['block'], contexts=[self._compare(row, block['block'], scope == 'unlocked')
                       for row in block['contexts']]) for block in old['by_block']]
            seeds = [dict(seed=seed['seed'], block=seed['block'], contexts=[self._compare(row, seed['block'], scope == 'unlocked', seed['seed'])
                      for row in seed['contexts']]) for seed in old['by_dev_seed']]
            comparisons[scope] = dict(by_block=blocks, by_dev_seed=seeds,
                equal_block_summary=self._equal(blocks, 3), equal_seed_summary=self._equal(seeds, 6) if seeds else [])
        hours = []
        for cut_block in range(3):
            for panel, seed, block in sorted(self.baseline.baseline.hours, key=repr):
                if panel != 'source' and block != cut_block:
                    continue
                for context in self.contexts:
                    selected = [key for key in self.counts if key[:5] == (cut_block, panel, seed, block, context)]
                    hours.append(dict(binning_block=cut_block, panel=panel, seed=seed, block=block, context=list(context),
                        strata=[dict(stratum=list(key[5]), unlocked=key[6], trend=key[7], bin=key[8], **self._cell_summary([key]))
                                for key in selected]))
        return dict(frozen_contexts=[list(context) for context in self.contexts], training=self.training_artifact,
            reproduction=dict(previous_binary_panel=True, float_tolerance=1e-12),
            observation_unit='correlated flight_decision_snapshot', by_hour=hours, comparisons=comparisons)
