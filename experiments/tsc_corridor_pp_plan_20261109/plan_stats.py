"""PP-plan strata on frozen magnitude bins and a fixed common cohort."""
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

for sibling in ('tsc_corridor_slowing_magnitude_20261109', 'tsc_corridor_predecision_slowing_20261109',
                'tsc_corridor_late_motion_20261109', 'tsc_corridor_state_stratification_20261109'):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/sibling))
from magnitude_stats import MagnitudeStats
from slowing_stats import _verify
from motion_stats import MOTIONS
from stratification_stats import PANELS

STOP = MOTIONS.index('stopped_at10')


def _sum(table, keys):
    return sum((table[key] for key in keys), np.zeros(len(MOTIONS), dtype=int))


def _risk(counts):
    return float(counts[STOP]/counts.sum()) if counts.sum() else None


def _difference(a, b):
    return a-b if a is not None and b is not None else None


class PlanStats:
    def __init__(self, contexts):
        self.contexts = tuple(tuple(context) for context in contexts)
        self.baseline = MagnitudeStats(self.contexts)
        self.training_records = []
        self.counts = defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int))
        self.moments = defaultdict(lambda: np.zeros(2))

    def register_hour(self, panel, seed, block):
        self.baseline.register_hour(panel, seed, block)

    def update(self, panel, seed, block, context, stratum, oldcategory, motion_name, unlocked,
               progress_m=None, slowing=None, magnitude=None, plan=None):
        seconds = plan['protected_seconds'] if plan is not None else None
        if seconds is not None and seconds not in range(11):
            raise ValueError('Nominal PP protection must be zero to ten recorded seconds')
        kept = self.baseline.update(panel, seed, block, context, stratum, oldcategory, motion_name,
                                    unlocked, progress_m, slowing, magnitude)
        if kept:
            record = (panel, seed, block, tuple(context), tuple(stratum), bool(unlocked),
                slowing['group'] if slowing is not None else 'unknown', MOTIONS.index(motion_name),
                magnitude['value'] if magnitude is not None else None, seconds if seconds is not None else 'unknown')
            if panel == 'dev':
                self._add(record, block)
            else:
                self.training_records.append(record)
        return kept

    def _add(self, record, cut_block):
        panel, seed, block, context, state, unlocked, trend, outcome, value, seconds = record
        bin_id = self.baseline._bin(trend, value, cut_block)
        key = cut_block, panel, seed, block, context, state, unlocked, trend, bin_id, seconds
        self.counts[key][outcome] += 1
        if trend == 'slowing' and value is not None:
            self.moments[key] += (1, value)

    def freeze_training(self, expected_prior_training):
        artifact = self.baseline.freeze_training()
        _verify(artifact, expected_prior_training)
        for record in self.training_records:
            for block in range(3) if record[0] == 'source' else (record[2],):
                self._add(record, block)
        self.training_records.clear()
        return dict(previous_training_reproduced=True, training=artifact)

    def _summary(self, keys, denominator=None):
        counts = _sum(self.counts, keys)
        moments = sum((self.moments[key] for key in keys), np.zeros(2))
        n = int(counts.sum())
        return dict(n=n, share=n/denominator if denominator else None,
            magnitude_n=int(moments[0]), magnitude_sum=float(moments[1]),
            magnitude_mean=float(moments[1]/moments[0]) if moments[0] else None,
            stopped_at10_n=int(counts[STOP]), stopped_at10_rate=_risk(counts))

    def _tables(self, block, context, unlocked_reference, dev_seed):
        tables = {panel: defaultdict(lambda: np.zeros(len(MOTIONS), dtype=int)) for panel in PANELS}
        keys = {panel: defaultdict(list) for panel in PANELS}
        for key, values in self.counts.items():
            cut_block, panel, seed, item_block, ctx, state, unlocked, trend, bin_id, seconds = key
            if (cut_block == block and ctx == context and
                    not (panel == 'dev' and dev_seed is not None and seed != dev_seed) and
                    not (panel == 'dev' and unlocked_reference and not unlocked)):
                tables[panel][state, trend, bin_id, seconds] += values
                keys[panel][state, trend, bin_id, seconds].append(key)
        return tables, keys

    def _compare(self, prior, block, unlocked_reference, dev_seed=None):
        tables, keys = self._tables(block, tuple(prior['context']), unlocked_reference, dev_seed)
        c2 = {(tuple(row['stratum']), row['trend'], row['bin']) for row in prior['bins'] if row['common_C2']}
        c3 = set.intersection(*(set(key for key in table if key[:3] in c2 and key[3] != 'unknown')
                                for table in tables.values()))
        dev_n = int(_sum(tables['dev'], c3).sum())
        weights = {key: int(tables['dev'][key].sum())/dev_n for key in c3} if dev_n else {}
        cell_weights = {cell: sum(weight for key, weight in weights.items() if key[:3] == cell)
                        for cell in {key[:3] for key in c3}}
        all_plans = set.union(*(set(key for key in table if key[:3] in c2) for table in tables.values()))
        groups = {}
        cell_ns = {}
        for panel, table in tables.items():
            c2_keys = [key for key in table if key[:3] in c2]
            c2_counts, c3_counts = _sum(table, c2_keys), _sum(table, c3)
            previous = prior['groups'][panel]
            if (int(c2_counts.sum()) != previous['C2_n'] or
                    {name: int(value) for name, value in zip(MOTIONS, c2_counts)} != previous['C2_counts']):
                raise ValueError('PP-plan strata changed the previous magnitude common cohort')
            cell_ns[panel] = {cell: int(_sum(table, [key for key in c2_keys if key[:3] == cell]).sum()) for cell in c2}
            cell_standard = sum(weight*_risk(_sum(table, [key for key in c3 if key[:3] == cell]))
                                for cell, weight in cell_weights.items()) if weights else None
            plan_standard = sum(weight*_risk(table[key]) for key, weight in weights.items()) if weights else None
            coverage = int(c3_counts.sum())/int(c2_counts.sum()) if c2_counts.sum() else None
            unknown_keys = [raw for key in c2_keys if key[3] == 'unknown' for raw in keys[panel][key]]
            groups[panel] = dict(C2_n=int(c2_counts.sum()), C2_coverage_full=previous['C2_coverage_full'],
                C2_unweighted_stopped_at10_rate=_risk(c2_counts),
                C2_standardized_stopped_at10_rate=previous['C2_magnitude_standardized_stopped_at10_rate'],
                C3_n=int(c3_counts.sum()), C3_coverage_C2=coverage,
                C3_coverage_full=coverage*previous['C2_coverage_full']
                    if coverage is not None and previous['C2_coverage_full'] is not None else None,
                unknown_plan=self._summary(unknown_keys, int(c2_counts.sum())),
                C3_unweighted_stopped_at10_rate=_risk(c3_counts),
                C3_cell_standardized_stopped_at10_rate=cell_standard,
                C3_plan_standardized_stopped_at10_rate=plan_standard)
        pairs = {}
        for pair, panel in (('source_vs_dev', 'source'), ('local_vs_dev', 'local_adapt')):
            a, b = groups[panel], groups['dev']
            gap = {name: _difference(a[key], b[key]) for name, key in (
                ('C2_unweighted', 'C2_unweighted_stopped_at10_rate'), ('C2_standardized', 'C2_standardized_stopped_at10_rate'),
                ('C3_unweighted', 'C3_unweighted_stopped_at10_rate'), ('C3_cell_standardized', 'C3_cell_standardized_stopped_at10_rate'),
                ('C3_plan_standardized', 'C3_plan_standardized_stopped_at10_rate'))}
            gap['cohort_change_unweighted'] = _difference(gap['C3_unweighted'], gap['C2_unweighted'])
            gap['cohort_and_reference_change_standardized'] = _difference(gap['C3_cell_standardized'], gap['C2_standardized'])
            gap['plan_composition_fixed_C3'] = _difference(gap['C3_cell_standardized'], gap['C3_plan_standardized'])
            pairs[pair] = gap
        return dict(context=prior['context'], groups=groups, pairs=pairs,
            plans=[dict(stratum=list(state), trend=trend, bin=bin_id, protected_seconds=seconds,
                common_C3=(state, trend, bin_id, seconds) in c3,
                dev_reference_weight=weights.get((state, trend, bin_id, seconds)),
                panels={panel: self._summary(keys[panel][state, trend, bin_id, seconds], cell_ns[panel][state, trend, bin_id])
                        for panel in PANELS}) for state, trend, bin_id, seconds in sorted(all_plans, key=repr)])

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
                pairs[pair]['complete_support_panels'] = sum(row['pairs'][pair]['C3_plan_standardized'] is not None for row in rows)
            result.append(dict(context=rows[0]['context'], pairs=pairs, expected_panels=expected, observed_panels=len(rows)))
        return result

    def result(self, previous_magnitude):
        old = self.baseline.result(self.baseline._binary_result())
        _verify(old, previous_magnitude)
        comparisons = {}
        for scope, prior in old['comparisons'].items():
            blocks = [dict(block=block['block'], contexts=[self._compare(row, block['block'], scope == 'unlocked')
                       for row in block['contexts']]) for block in prior['by_block']]
            seeds = [dict(seed=seed['seed'], block=seed['block'], contexts=[self._compare(row, seed['block'], scope == 'unlocked', seed['seed'])
                      for row in seed['contexts']]) for seed in prior['by_dev_seed']]
            comparisons[scope] = dict(by_block=blocks, by_dev_seed=seeds,
                equal_block_summary=self._equal(blocks, 3), equal_seed_summary=self._equal(seeds, 6) if seeds else [])
        hours = []
        for cut_block in range(3):
            for panel, seed, block in sorted(self.baseline.baseline.baseline.hours, key=repr):
                if panel != 'source' and block != cut_block:
                    continue
                for context in self.contexts:
                    selected = [key for key in self.counts if key[:5] == (cut_block, panel, seed, block, context)]
                    denominators = {key[5:9]: sum(int(self.counts[other].sum()) for other in selected if other[5:9] == key[5:9])
                                    for key in selected}
                    hours.append(dict(binning_block=cut_block, panel=panel, seed=seed, block=block, context=list(context),
                        strata=[dict(stratum=list(key[5]), unlocked=key[6], trend=key[7], bin=key[8], protected_seconds=key[9],
                            **self._summary([key], denominators[key[5:9]])) for key in selected]))
        return dict(frozen_contexts=[list(context) for context in self.contexts],
            reproduction=dict(previous_training=True, previous_magnitude_panel=True, float_tolerance=1e-12),
            observation_unit='correlated flight_decision_snapshot', by_hour=hours, comparisons=comparisons)
