"""Descriptive probability calibration and signed service-score attribution.

Repeated queries of one flight are correlated observations. These accumulators
report descriptive sums, never iid confidence intervals or effective sample sizes.
"""
from copy import deepcopy

import numpy as np

ARMS = ('T', 'S', 'P')
COLUMNS = tuple(f'service_{second}' for second in range(1, 14)) + ('detector_10',)
BIN_EDGES = np.array([index/10 for index in range(11)])


class ProbabilityCalibration:
    """Keep each query separate; input endpoint probabilities remain unthresholded."""
    def __init__(self):
        self.sums = {key: np.zeros((3, 14), dtype=int if key == 'n' else float)
                     for key in ('n', 'p_sum', 'y_sum', 'brier_sum')}
        self.sums.update({key: np.zeros((3, 14, 10), dtype=int if key == 'bin_n' else float)
                          for key in ('bin_n', 'bin_p_sum', 'bin_y_sum', 'bin_brier_sum')})

    def update(self, probabilities, truth, mask=None):
        probabilities, truth = np.asarray(probabilities, dtype=float), np.asarray(truth, dtype=float)
        selected = np.ones(14, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        if probabilities.shape != (3, 14) or truth.shape != (14,) or selected.shape != (14,):
            raise ValueError('Calibration requires three raw probability rows, fourteen truths and query mask')
        if (not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)) or
                not np.all((truth == 0) | (truth == 1))):
            raise ValueError('Calibration requires probabilities in [0,1] and binary CDF outcomes')
        indices = np.flatnonzero(selected)
        for arm in range(3):
            p, y = probabilities[arm, indices], truth[indices]
            bins = np.minimum(np.searchsorted(BIN_EDGES, p, side='right')-1, 9)
            squared_error = (p-y)**2
            self.sums['n'][arm, indices] += 1
            self.sums['p_sum'][arm, indices] += p
            self.sums['y_sum'][arm, indices] += y
            self.sums['brier_sum'][arm, indices] += squared_error
            np.add.at(self.sums['bin_n'][arm], (indices, bins), 1)
            for key, values in (('bin_p_sum', p), ('bin_y_sum', y), ('bin_brier_sum', squared_error)):
                np.add.at(self.sums[key][arm], (indices, bins), values)
        return self

    def merge(self, other):
        sums = other.sums if isinstance(other, ProbabilityCalibration) else other
        for key in self.sums:
            self.sums[key] += np.asarray(sums[key], dtype=self.sums[key].dtype)
        return self

    @staticmethod
    def _statistics(n, p_sum, y_sum, brier_sum):
        n = int(n)
        return dict(n=n, p_mean=float(p_sum/n) if n else None,
                    y_mean=float(y_sum/n) if n else None,
                    bias=float((p_sum-y_sum)/n) if n else None,
                    brier=float(brier_sum/n) if n else None)

    def result(self):
        arms = {}
        for index, arm in enumerate(ARMS):
            arms[arm] = {}
            for column, name in enumerate(COLUMNS):
                stats = self._statistics(*(self.sums[key][index, column]
                    for key in ('n', 'p_sum', 'y_sum', 'brier_sum')))
                stats['bins'] = [dict(lower=float(BIN_EDGES[bin_index]), upper=float(BIN_EDGES[bin_index+1]),
                    **self._statistics(*(self.sums[key][index, column, bin_index]
                        for key in ('bin_n', 'bin_p_sum', 'bin_y_sum', 'bin_brier_sum')))) for bin_index in range(10)]
                arms[arm][name] = stats
        return dict(observation_unit='flight_decision_snapshot_at_one_query',
                    uncertainty='descriptive_only; repeated flights and queries are correlated',
                    columns=list(COLUMNS), bin_edges=BIN_EDGES.tolist(), arms=arms,
                    raw_sums={key: value.tolist() for key, value in self.sums.items()})


def _direction():
    return dict(queries=0, flights=0, expected_credits={arm: 0.0 for arm in ARMS},
                observed_credits=0.0, errors={arm: 0.0 for arm in ARMS},
                signed_errors={arm: 0.0 for arm in ARMS}, source_correction_sum=0.0,
                generic_count_component=0.0, conditional_source_component=0.0,
                source_empirical_credits=0.0, source_empirical_queries=0,
                source_weight_sum=0.0, n_target_sum=0.0, n_source_sum=0.0)


def _components():
    return dict(source_correction_sum=0.0, baseline_error_sum=0.0, source_error_sum=0.0,
                positive_correction_sum=0.0, negative_correction_sum=0.0,
                generic_count_component=0.0, conditional_source_component=0.0,
                directions={name: _direction() for name in ('w_positive', 'w_negative')})


def _add(target, source):
    """Add the raw, numeric metric tree used by row and context summaries."""
    for key, value in source.items():
        if isinstance(value, dict):
            _add(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0)+value


def decision_decomposition(aligned, row_flights, captures, actions, label_scores, contrast=('S', 'T')):
    """Attribute one fixed S-versus-baseline request pair using signed masks.

    The two requests are held fixed under all probability arms. Positive and
    negative mask directions retain unsigned credits and their signed errors.
    """
    if len(row_flights) != len(captures):
        raise ValueError('Every current flight needs its own captured forecast and truth')
    source_arm, baseline_arm = contrast
    source_index, baseline_index = ARMS.index(source_arm), ARMS.index(baseline_arm)
    a, b = actions[source_arm], actions[baseline_arm]
    base = float(aligned['base'][a]-aligned['base'][b])
    label_margin = float(label_scores[a]-label_scores[b])
    predicted = {arm: base for arm in ARMS}
    observed_margin = base
    weights = {edge: np.array([int(bool(int(masks[a]) & (1 << bit)))-int(bool(int(masks[b]) & (1 << bit)))
                               for bit in range(13)], dtype=int)
               for edge, masks in aligned['protected'].items()}
    totals, contexts = _components(), {}
    for flight, capture in zip(row_flights, captures):
        raw, truth = np.asarray(capture['raw']), np.asarray(capture['truth'])
        if raw.shape != (5, 14) or truth.shape != (14,):
            raise ValueError('Decomposition needs per-flight five-arm raw CDFs and fourteen truths')
        w = weights[flight['edge']]
        values, y = raw[:3, :13], truth[:13]
        margins = values @ w
        for arm, margin in zip(ARMS, margins):
            predicted[arm] += float(margin)
        observed_margin += float(y @ w)
        correction = w*(values[source_index]-values[baseline_index])
        components = _components()
        components.update(source_correction_sum=float(correction.sum()),
                          baseline_error_sum=float(w @ (values[baseline_index]-y)),
                          source_error_sum=float(w @ (values[source_index]-y)),
                          positive_correction_sum=float(np.maximum(correction, 0).sum()),
                          negative_correction_sum=float(np.minimum(correction, 0).sum()),
                          generic_count_component=float(margins[2]-margins[0]),
                          conditional_source_component=float(margins[1]-margins[2]))
        weight = float(capture['source_weight'])
        empirical = capture.get('source_empirical')
        if empirical is None and weight > 0:
            empirical = (raw[1]-(1-weight)*raw[0])/weight
        empirical = None if empirical is None else np.asarray(empirical)[:13]
        for name, selected, sign in (('w_positive', w > 0, 1), ('w_negative', w < 0, -1)):
            direction = components['directions'][name]
            n = int(selected.sum())
            expected = values[:, selected].sum(axis=1)
            observed = float(y[selected].sum())
            direction.update(queries=n, flights=int(n > 0),
                expected_credits={arm: float(value) for arm, value in zip(ARMS, expected)},
                observed_credits=observed,
                errors={arm: float(value-observed) for arm, value in zip(ARMS, expected)},
                signed_errors={arm: float(sign*(value-observed)) for arm, value in zip(ARMS, expected)},
                source_correction_sum=float(correction[selected].sum()),
                generic_count_component=float(sign*(expected[2]-expected[0])),
                conditional_source_component=float(sign*(expected[1]-expected[2])),
                source_empirical_credits=float(empirical[selected].sum()) if empirical is not None else 0.0,
                source_empirical_queries=n if empirical is not None else 0,
                source_weight_sum=weight*n, n_target_sum=float(capture['n_target'])*n,
                n_source_sum=float(capture['n_source'])*n)
        _add(totals, components)
        context = tuple(capture['context'])
        entry = contexts.setdefault(context, dict(flights=0, service_queries=0, active_queries=0,
            eta_sum=0.0, n_target_sum=0.0, n_source_sum=0.0, source_weight_sum=0.0, **_components()))
        _add(entry, components)
        entry['flights'] += 1
        entry['service_queries'] += 13
        entry['active_queries'] += int(np.count_nonzero(w))
        for key in ('eta', 'n_target', 'n_source', 'source_weight'):
            entry[key+'_sum'] += float(capture[key])
    checks = ((observed_margin, label_margin),
              (totals['source_correction_sum']+totals['baseline_error_sum'], totals['source_error_sum']),
              (predicted[source_arm]-predicted[baseline_arm], totals['source_correction_sum']),
              (predicted[source_arm]-label_margin, totals['source_error_sum']),
              (totals['generic_count_component']+totals['conditional_source_component'], predicted['S']-predicted['T']))
    if any(not np.isclose(left, right, atol=1e-9, rtol=1e-10) for left, right in checks):
        raise ValueError('Signed forecast/label decomposition does not reproduce the fixed request scores')
    return dict(contrast=list(contrast), actions={source_arm: int(a), baseline_arm: int(b)},
                queue_base_margin=base, predicted_margins=predicted, label_margin=label_margin,
                signed_masks={edge: values.tolist() for edge, values in weights.items()},
                contexts=[dict(context=list(context), **values) for context, values in contexts.items()], **totals)


class DecompositionAccumulator:
    """Sum row diagnostics, keeping label-loss strata and full context attribution."""
    def __init__(self):
        self.decisions = 0
        self.totals = {}
        self.groups = {name: dict(decisions=0, totals={}) for name in ('negative', 'zero', 'positive')}
        self.contexts = {}
        self.contexts_by_label_sign = {name: {} for name in self.groups}

    def update(self, result):
        metrics = {key: value for key, value in result.items()
                   if key not in ('contrast', 'actions', 'signed_masks', 'contexts')}
        self.decisions += 1
        _add(self.totals, metrics)
        sign = 'positive' if result['label_margin'] > 0 else 'negative' if result['label_margin'] < 0 else 'zero'
        self.groups[sign]['decisions'] += 1
        _add(self.groups[sign]['totals'], metrics)
        for row in result['contexts']:
            context = tuple(row['context'])
            for table in (self.contexts, self.contexts_by_label_sign[sign]):
                target = table.setdefault(context, dict(decisions=0))
                target['decisions'] += 1
                _add(target, {key: value for key, value in row.items() if key != 'context'})
        return self

    def merge(self, other):
        record = other.result() if isinstance(other, DecompositionAccumulator) else other
        self.decisions += record['decisions']
        _add(self.totals, record['totals'])
        for sign in self.groups:
            self.groups[sign]['decisions'] += record['by_label_sign'][sign]['decisions']
            _add(self.groups[sign]['totals'], record['by_label_sign'][sign]['totals'])
            for row in record['by_label_sign'][sign]['contexts']:
                _add(self.contexts_by_label_sign[sign].setdefault(tuple(row['context']), {}),
                     {key: value for key, value in row.items() if key != 'context'})
        for row in record['contexts']:
            _add(self.contexts.setdefault(tuple(row['context']), {}),
                 {key: value for key, value in row.items() if key != 'context'})
        return self

    def result(self):
        groups = {sign: dict(values, contexts=[dict(context=list(context), **row)
                    for context, row in self.contexts_by_label_sign[sign].items()])
                  for sign, values in self.groups.items()}
        return deepcopy(dict(decisions=self.decisions, totals=self.totals, by_label_sign=groups,
            contexts=[dict(context=list(context), **values) for context, values in self.contexts.items()]))
