"""Exact loss partitions for a frozen source/target arrival predictor.

Per-flight mistakes are diagnostics. Only edge count errors enter the MAE
attribution; multiple active context groups remain a separate joint category.
"""
from collections import Counter, defaultdict
import numpy as np

FEATURES = ('current_lane', 'history_eta_bin', 'speed_ratio', 'entry_queue')


def context_label(context):
    straight, eta_bin, speed, queued = context
    return f'{"straight" if straight else "other"}|eta={eta_bin}|speed={speed}|queue={str(queued).lower()}'


def feature_labels(context):
    return ('straight' if context[0] else 'other', str(context[1]), context[2], str(context[3]).lower())


class Cell:
    def __init__(self):
        self.counts = Counter()
        self.target_n_min = self.target_n_max = None
        self.source_n_min = self.source_n_max = None
        self.examples = []

    def flight(self, item, weight):
        c = self.counts
        t, s = (int(item['decoded'][i, 10]) for i in (0, 1))
        y = int(item['actual10'])
        c['flight_snapshots'] += 1
        c['actual_arrivals10'] += y
        c['T_predicted10'] += t
        c['S_predicted10'] += s
        c['decoded10_changes'] += t != s
        c['T_flight_fp'] += t > y
        c['T_flight_fn'] += t < y
        c['S_flight_fp'] += s > y
        c['S_flight_fn'] += s < y
        c['flight_absolute_error_delta'] += abs(s-y) - abs(t-y)
        c['weighted_flight_error_delta'] += weight * (abs(s-y) - abs(t-y))
        nt, ns = int(item['target_n']), int(item['source_n'])
        self.target_n_min = nt if self.target_n_min is None else min(self.target_n_min, nt)
        self.target_n_max = nt if self.target_n_max is None else max(self.target_n_max, nt)
        c['target_physical_prior_sum'] += 20 / (20 + nt)
        c['target_cdf10_sum'] += float(item['raw_cdfs'][0, 10])
        if ns:
            self.source_n_min = ns if self.source_n_min is None else min(self.source_n_min, ns)
            self.source_n_max = ns if self.source_n_max is None else max(self.source_n_max, ns)
            target = float(item['raw_cdfs'][0, 10])
            source = 2 * float(item['raw_cdfs'][1, 10]) - target
            difference = source - target
            c['source_supported_flights'] += 1
            c['source_cdf10_sum'] += source
            c['supported_target_cdf10_sum'] += target
            c['source_minus_target_cdf10_sum'] += difference
            c['source_physical_prior_sum'] += 20 / (20 + ns)
            c['source_cdf10_above_target'] += difference > 1e-12
            c['source_cdf10_below_target'] += difference < -1e-12
            if t != s:
                c['decoded_change_source_direction_agrees'] += (s-t) * difference > 0
            if nt:
                baseline = float(item['raw_cdfs'][3, 10])
                pt = ((20+nt)*target - 20*baseline) / nt
                ph = ((20+ns)*source - 20*baseline) / ns
                lt, lh = nt/(20+nt), ns/(20+ns)
                distribution = .5*(lh+lt)*(ph-pt)
                shrinkage = (lh-lt)*(.5*(ph+pt)-baseline)
                c['cdf_decomposition_samples'] += 1
                c['target_empirical_arrival10_sum'] += pt
                c['source_empirical_arrival10_sum'] += ph
                c['empirical_distribution_difference_sum'] += distribution
                c['physical_prior_strength_difference_sum'] += shrinkage
                c['cdf_decomposition_residual_sum'] += difference-distribution-shrinkage
        else:
            c['source_missing_flights'] += 1

    def edge(self, t, s, y, weight):
        c = self.counts
        lt, ls = abs(t-y), abs(s-y)
        c['edge_epochs'] += 1
        c['T_count_absolute_error'] += lt
        c['S_count_absolute_error'] += ls
        c['count_loss_delta'] += ls-lt
        c['weighted_T_count_loss'] += weight * lt
        c['weighted_S_count_loss'] += weight * ls
        c['weighted_count_loss_delta'] += weight * (ls-lt)
        c['improved_edge_epochs'] += ls < lt
        c['worsened_edge_epochs'] += ls > lt

    def attribute(self, delta, weight, example):
        self.counts['attributed_edge_epochs'] += 1
        self.counts['attributed_count_loss_delta'] += delta
        self.counts['weighted_attributed_count_loss_delta'] += weight * delta
        self.counts['attributed_improved_edges'] += delta < 0
        self.counts['attributed_worsened_edges'] += delta > 0
        if delta:
            self.examples.append(example)
            self.examples.sort(key=lambda x: (-abs(x['weighted_loss_delta']), x['seed'], x['t'], x['edge']))
            del self.examples[3:]

    def result(self):
        c, n, supported = self.counts, self.counts['flight_snapshots'], self.counts['source_supported_flights']
        return dict(counts=dict(c),
                    pooled_T_count_mae=c['T_count_absolute_error']/c['edge_epochs'] if c['edge_epochs'] else None,
                    pooled_S_count_mae=c['S_count_absolute_error']/c['edge_epochs'] if c['edge_epochs'] else None,
                    target_n_range=[self.target_n_min, self.target_n_max] if self.target_n_min is not None else None,
                    source_n_range=[self.source_n_min, self.source_n_max] if self.source_n_min is not None else None,
                    mean_target_cdf10=c['target_cdf10_sum']/n if n else None,
                    mean_source_cdf10=c['source_cdf10_sum']/supported if supported else None,
                    mean_supported_target_cdf10=c['supported_target_cdf10_sum']/supported if supported else None,
                    mean_source_minus_target_cdf10=c['source_minus_target_cdf10_sum']/supported if supported else None,
                    mean_target_physical_prior=c['target_physical_prior_sum']/n if n else None,
                    mean_source_physical_prior=c['source_physical_prior_sum']/supported if supported else None,
                    mean_distribution_difference_component=c['empirical_distribution_difference_sum']/c['cdf_decomposition_samples'] if c['cdf_decomposition_samples'] else None,
                    mean_prior_strength_difference_component=c['physical_prior_strength_difference_sum']/c['cdf_decomposition_samples'] if c['cdf_decomposition_samples'] else None,
                    examples=self.examples)


class Panel:
    def __init__(self):
        self.overall = Cell()
        self.tables = {name: defaultdict(Cell) for name in ('seed', 'block', 'edge', 'context', 'block_context', *FEATURES)}
        self.context_attribution = defaultdict(Cell)
        self.feature_attribution = {name: defaultdict(Cell) for name in FEATURES}

    def add(self, row, items, forecasts, *, city, seed, block, weight):
        structural = [self.overall, self.tables['seed'][str(seed)], self.tables['block'][str(block)]]
        by_edge = defaultdict(list)
        for item in items:
            context = tuple(item['context'])
            edge = item['flight']['edge']
            label = context_label(context)
            cells = structural + [self.tables['edge'][edge], self.tables['context'][label],
                                   self.tables['block_context'][f'{block}::{label}']]
            cells += [self.tables[name][value] for name, value in zip(FEATURES, feature_labels(context))]
            for cell in cells:
                cell.flight(item, weight)
            by_edge[edge].append(item)
        for edge, group in by_edge.items():
            t, s = float(forecasts[edge][0, 10]), float(forecasts[edge][1, 10])
            y = sum(int(item['actual10']) for item in group)
            delta = abs(s-y) - abs(t-y)
            for cell in structural + [self.tables['edge'][edge]]:
                cell.edge(t, s, y, weight)
            example = dict(city=city, seed=seed, block=block, t=row['t'], tls=row['tls'], edge=edge,
                           T_count=t, S_count=s, actual_count=y, loss_delta=delta, weighted_loss_delta=weight*delta)
            changes = defaultdict(int)
            features = {name: defaultdict(int) for name in FEATURES}
            for item in group:
                difference = int(item['decoded'][1, 10]) - int(item['decoded'][0, 10])
                changes[context_label(item['context'])] += difference
                for name, label in zip(FEATURES, feature_labels(item['context'])):
                    features[name][label] += difference
            if sum(changes.values()) != s-t:
                raise ValueError('Context changes do not reconstruct the frozen edge prediction')
            example['changed_contexts'] = {key: value for key, value in changes.items() if value}
            active = [key for key, value in changes.items() if value]
            label = active[0] if len(active) == 1 else '__joint__' if active else '__no_net_change__'
            self.context_attribution[label].attribute(delta, weight, example)
            self.tables['context'][label].attribute(delta, weight, example)
            self.tables['block_context'][f'{block}::{label}'].attribute(delta, weight, example)
            if len(active) > 1:
                self.overall.counts['multiple_active_context_edges'] += 1
                self.overall.counts['group_net_cancellation_edges'] += s == t
            for name, values in features.items():
                active_values = [key for key, value in values.items() if value]
                key = active_values[0] if len(active_values) == 1 else '__joint__' if active_values else '__no_net_change__'
                self.feature_attribution[name][key].attribute(delta, weight, example)
                self.tables[name][key].attribute(delta, weight, example)

    def result(self):
        total = self.overall.counts['weighted_count_loss_delta']
        partitions = dict(context=self.context_attribution, **self.feature_attribution)
        checks = {}
        for name, table in partitions.items():
            value = sum(cell.counts['weighted_attributed_count_loss_delta'] for cell in table.values())
            checks[name] = dict(weighted_delta=value, total=total, difference=value-total)
            if abs(value-total) > 1e-10:
                raise ValueError(f'{name} attribution failed to reconstruct the equal-seed loss')
        return dict(overall=self.overall.result(),
                    tables={name: {key: cell.result() for key, cell in sorted(table.items())} for name, table in self.tables.items()},
                    attribution={name: {key: cell.result() for key, cell in sorted(table.items())} for name, table in partitions.items()},
                    attribution_checks=checks)


class LossContextObserver:
    def __init__(self):
        self.panels = defaultdict(Panel)

    def add(self, row, items, forecasts, scopes, *, city, seed, block, edge_weights=None):
        """Items carry pre-update contexts/predictions; actual10 is evaluation-only.

        ``edge_weights[scope] = 1/(6*N_seed_scope)`` preserves the declared city
        equal-seed MAE. Each feature partition is additive on its own, never
        summed with another feature partition.
        """
        for scope, enabled in scopes.items():
            if enabled:
                weight = edge_weights[scope] if edge_weights is not None else 1.0
                self.panels[city, scope].add(row, items, forecasts, city=city, seed=seed, block=block, weight=weight)

    def result(self):
        cities = defaultdict(dict)
        for (city, scope), panel in sorted(self.panels.items()):
            cities[city][scope] = panel.result()
        return dict(by_city=dict(cities), definitions=dict(
            identity='Current moving flight snapshots; not distinct vehicles across road links.',
            contribution='Positive S-minus-T count loss is worse; weighted contributions use 1/(6*N_seed_scope).',
            attribution='Unique net-changing context receives its whole edge loss change; multiple contexts remain joint. Each partition separately reconstructs total.',
            probability='Source CDF recovered as 2*S_raw-T_raw only where source_n>0; differences describe fitted predictive heads, not physical causality.',
            prior='20/(20+n) is the physical prior fraction; source/target head differences can include data-distribution and prior-strength differences.'))
