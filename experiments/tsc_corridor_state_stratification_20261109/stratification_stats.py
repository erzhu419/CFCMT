"""Descriptive detector CDFs on fixed contexts and identifiable current states."""
from collections import defaultdict

import numpy as np

PANELS = ('source', 'local_adapt', 'dev')
KNOWN_STATES = {(phase, queue) for phase in ('protected', 'not_protected')
                for queue in ('zero', 'positive')}
PAIRS = ('source_vs_dev', 'local_vs_dev')


def _cdf(counts):
    n = int(np.sum(counts))
    return (np.cumsum(counts[1:])/n).tolist() if n else None


def _difference(left, right):
    return (np.asarray(left)-np.asarray(right)).tolist() if left is not None and right is not None else None


def _mean_absolute(values):
    return float(np.mean(np.abs(values))) if values is not None else None


def _count_sum(table, states=None):
    return sum((counts for state, counts in table.items() if states is None or state in states), np.zeros(11, dtype=int))


def _state_record(state, counts):
    return dict(stratum=list(state), counts=counts.tolist(), n=int(counts.sum()), cdf=_cdf(counts))


class StratificationStats:
    """Keep source, local adaptation and development hours as separate panels.

    category=0 means no straight detection in the next ten seconds; categories
    1..10 are detector lags. Counts describe correlated flight snapshots.
    """
    def __init__(self, frozen_contexts):
        self.contexts = tuple(tuple(context) for context in frozen_contexts)
        self.counts = defaultdict(lambda: np.zeros(11, dtype=int))
        self.hours = set()

    def register_hour(self, panel, seed, block):
        self.hours.add((panel, seed, block))

    def update(self, panel, seed, block, context, stratum, category, unlocked):
        self.register_hour(panel, seed, block)
        context, stratum = tuple(context), tuple(stratum)
        if context not in self.contexts:
            return False
        if panel not in PANELS or category not in range(11):
            raise ValueError('Expected a fixed source/local_adapt/dev panel and detector category 0..10')
        self.counts[panel, seed, block, context, stratum, bool(unlocked)][category] += 1
        return True

    def _table(self, panel, block, context, unlocked_reference=False, dev_seed=None):
        table = defaultdict(lambda: np.zeros(11, dtype=int))
        for (part, seed, item_block, ctx, state, unlocked), counts in self.counts.items():
            if (part == panel and ctx == context and (panel == 'source' or block is None or item_block == block)
                    and not (panel == 'dev' and dev_seed is not None and seed != dev_seed)
                    and not (panel == 'dev' and unlocked_reference and not unlocked)):
                table[state] += counts
        return dict(table)

    def _compare(self, block, context, unlocked_reference, dev_seed=None):
        tables = {panel: self._table(panel, block, context, unlocked_reference, dev_seed) for panel in PANELS}
        common = KNOWN_STATES.intersection(*(set(table) for table in tables.values()))
        ref_common = int(_count_sum(tables['dev'], common).sum())
        weights = {state: int(tables['dev'][state].sum())/ref_common for state in common} if ref_common else {}
        groups = {}
        for panel, table in tables.items():
            full, supported = _count_sum(table), _count_sum(table, common)
            n, support_n = int(full.sum()), int(supported.sum())
            standard = sum((weights[state]*np.asarray(_cdf(table[state])) for state in common), np.zeros(10)) if weights else None
            full_cdf, common_cdf = _cdf(full), _cdf(supported)
            groups[panel] = dict(n=n, counts=full.tolist(), full_cdf=full_cdf,
                known_n=int(_count_sum(table, KNOWN_STATES).sum()), common_n=support_n,
                common_counts=supported.tolist(), common_support_coverage=support_n/n if n else None,
                outside_support_n=n-support_n,
                unknown_state_n=int(_count_sum(table, set(table)-KNOWN_STATES).sum()),
                common_unweighted_cdf=common_cdf,
                standardized_cdf=standard.tolist() if standard is not None else None,
                full_to_common_shift=_difference(common_cdf, full_cdf),
                strata=[_state_record(state, table[state]) for state in sorted(table)])
        pairs = {}
        for name, panel in (('source_vs_dev', 'source'), ('local_vs_dev', 'local_adapt')):
            left, right = groups[panel], groups['dev']
            full = _difference(left['full_cdf'], right['full_cdf'])
            unweighted = _difference(left['common_unweighted_cdf'], right['common_unweighted_cdf'])
            within = _difference(left['standardized_cdf'], right['standardized_cdf'])
            composition = _difference(unweighted, within)
            pairs[name] = dict(full_coarse_gap=full, common_unweighted_gap=unweighted,
                standardized_within_gap=within, composition_gap=composition,
                full_to_common_gap_shift=_difference(unweighted, full),
                mean_absolute_full_coarse_gap=_mean_absolute(full),
                mean_absolute_common_unweighted_gap=_mean_absolute(unweighted),
                mean_absolute_standardized_within_gap=_mean_absolute(within))
        return dict(context=list(context), groups=groups, pairs=pairs,
            common_strata=[dict(stratum=list(state), dev_reference_weight=weights[state],
                panels={panel: _state_record(state, table[state]) for panel, table in tables.items()})
                for state in sorted(common)])

    def _by_hour(self):
        hours = defaultdict(lambda: defaultdict(lambda: np.zeros(11, dtype=int)))
        for panel, seed, block in self.hours:
            for context in self.contexts:
                hours[panel, seed, block, context]
        for (panel, seed, block, context, state, unlocked), counts in self.counts.items():
            hours[panel, seed, block, context][state, unlocked] += counts
        result = []
        for (panel, seed, block, context), strata in sorted(hours.items(), key=lambda item: repr(item[0])):
            full = sum(strata.values(), np.zeros(11, dtype=int))
            result.append(dict(panel=panel, seed=seed, block=block, context=list(context),
                n=int(full.sum()), counts=full.tolist(), cdf=_cdf(full),
                strata=[dict(_state_record(state, counts), unlocked=unlocked)
                        for (state, unlocked), counts in sorted(strata.items())]))
        return result

    @staticmethod
    def _equal_blocks(blocks):
        result = []
        for index, first in enumerate(blocks[0]['contexts']):
            records = [block['contexts'][index] for block in blocks]
            pairs = {}
            for pair in PAIRS:
                pairs[pair] = {}
                for metric in first['pairs'][pair]:
                    values = [record['pairs'][pair][metric] for record in records]
                    pairs[pair][metric] = np.mean(values, axis=0).tolist() if all(value is not None for value in values) else None
                pairs[pair]['blocks_with_common_support'] = sum(record['pairs'][pair]['standardized_within_gap'] is not None for record in records)
            result.append(dict(context=first['context'], pairs=pairs,
                common_support_coverage={panel: float(np.mean([record['groups'][panel]['common_support_coverage']
                    for record in records])) if all(record['groups'][panel]['common_support_coverage'] is not None for record in records) else None
                    for panel in PANELS}))
        return result

    def result(self):
        comparisons = {}
        for scope in ('all_observable', 'unlocked'):
            unlocked = scope == 'unlocked'
            blocks = [dict(block=block, contexts=[self._compare(block, context, unlocked) for context in self.contexts])
                      for block in range(3)]
            comparisons[scope] = dict(by_block=blocks,
                by_dev_seed=[dict(seed=seed, block=block,
                    contexts=[self._compare(block, context, unlocked, dev_seed=seed) for context in self.contexts])
                    for panel, seed, block in sorted(self.hours, key=repr) if panel == 'dev'],
                pooled=[self._compare(None, context, unlocked) for context in self.contexts],
                equal_block_summary=self._equal_blocks(blocks))
        return dict(observation_unit='correlated flight_decision_snapshot', detector_seconds=list(range(1, 11)),
            frozen_contexts=[list(context) for context in self.contexts], by_hour=self._by_hour(), comparisons=comparisons)
