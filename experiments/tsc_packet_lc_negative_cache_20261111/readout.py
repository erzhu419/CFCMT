"""Streaming four-point likelihood and fixed-neighbourhood LC readouts."""

from collections import Counter, defaultdict
from math import fsum, isclose

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import extended, paired_difference
from experiments.tsc_packet_lc_discrimination_20261111.readout import (
    DIRECTIONS, METRICS, direction_readout, rank, score, training_summary,
)
from experiments.tsc_packet_lc_refit_20261111.readout import (
    L2, _lc_gradient, _mixed_point, _points as coefficient_points,
)
from experiments.tsc_packet_mixture_confirmation_20261111.readout import (
    MixtureParticleLaw, _proposal_components,
)
from experiments.tsc_packet_positive_covered_cache_20261111.readout import _exact_point, _point_aggregate


POINTS = ('zero', 'original', 'frequency_bias', 'refitted')


def _points(points):
    points = coefficient_points(points)
    if set(points) != set(POINTS):
        raise ValueError('The four previously frozen coefficient points are required')
    return points


def _identity(row):
    reference = row.get('reference', row)
    return int(reference['seed']), int(reference['time_sec']), reference['packet_id']


def read_case(law, points, task):
    """Read one cache; return JSON scalars and separate JSON gradient arrays."""
    points = _points(points)
    y = int(task['y'])
    if y not in (0, 1, 2):
        raise ValueError('The frozen native LC label must be left0, stay1 or right2')
    mixed = isinstance(law, MixtureParticleLaw)
    if not mixed and not isinstance(law, CachedLaw):
        raise TypeError('The registered exact or fixed-anchor mixture law is required')
    if task['proposed_method'] != ('mixture256' if mixed else 'exact'):
        raise ValueError('The frozen numerical method changed')
    if mixed:
        _proposal_components(law, None)
    row = dict(task, reference=dict(zip(('seed', 'time_sec', 'packet_id'), _identity(task))),
               readouts={}, points={})
    gradients = {}
    for view, n in (('primary', 256), ('128', 128)) if mixed else (('primary', None),):
        values, private = {}, {}
        for point, theta in points.items():
            values[point], gradient = (_mixed_point(law, theta, y, n) if mixed
                                       else _exact_point(law, theta, y))
            private[point] = gradient.tolist() if gradient is not None else None
        row['readouts'][view], gradients[view] = dict(points=values), private
    row['native_support_passed'] = all(
        value['native']['defined'] and value['native']['supported']
        and value['native']['loss']['state'] == 'finite' and gradients[view][point] is not None
        for view, data in row['readouts'].items() for point, value in data['points'].items())
    if row['native_support_passed']:
        directions = DIRECTIONS if y == 1 else (y,)
        row['points'] = {point:{str(direction):direction_readout(law, theta, direction)
                               for direction in directions} for point, theta in points.items()}
    return row, gradients


def _native_aggregate(records, gradients, points):
    """Same pure aggregation as lc_refit.training_readouts, without laws."""
    result = {}
    for name, theta in points.items():
        penalty = float(L2*np.sum(theta[3:, 1:]**2))
        views = {}
        for view in ('primary', '128'):
            scalar = _point_aggregate(records, gradients, view, name)
            scalar['lc_gradient'] = _lc_gradient(records, gradients, view, name, theta)
            loss = scalar['weighted_nll_contribution']
            scalar['penalized_objective'] = (extended(loss['value']+penalty)
                                               if loss['state'] == 'finite' else dict(loss))
            views[view] = scalar
        result[name] = dict(views, lc_regularization=penalty)
    comparisons = {baseline:{name:paired_difference(
        value['primary']['weighted_nll_contribution'],
        result[baseline]['primary']['weighted_nll_contribution'])
        for name, value in result.items() if name != baseline}
        for baseline in ('original', 'frequency_bias')}
    return dict(reference_count=len(records), objective_weight_sum=fsum(r['objective_weight'] for r in records),
        selected_class_counts=dict(Counter(r['y'] for r in records)), points=result,
        mixed_size_sensitivity={name:paired_difference(value['128']['weighted_nll_contribution'],
            value['primary']['weighted_nll_contribution']) for name, value in result.items()},
        weighted_nll_differences=comparisons)


def _matched_group(positives, neighbors, point, direction):
    groups = [(positive, neighbors[_identity(positive)]) for positive in positives]
    covered = [(positive, stays) for positive, stays in groups if stays]
    metrics = {}
    for metric in METRICS:
        positive_means, comparisons, numerator, denominator, defined_denominator = [], Counter(), [], [], []
        for positive, stays in groups:
            wins, weights, undefined = [], [], False
            for stay in stays:
                outcome = rank(score(positive['points'][point][str(direction)], metric),
                               score(stay['points'][point][str(direction)], metric))
                weight = stay['objective_weight']
                comparisons[outcome] += 1
                denominator.append(weight)
                weights.append(weight)
                if outcome == 'undefined':
                    undefined = True
                else:
                    contribution = weight*(1. if outcome == 'wins' else .5 if outcome == 'ties' else 0.)
                    wins.append(contribution)
                    numerator.append(contribution)
                    defined_denominator.append(weight)
            positive_means.append(fsum(wins)/fsum(weights) if weights and not undefined else None)
        valid = [value for value in positive_means if value is not None]
        covered_means = [value for value, (_, stays) in zip(positive_means, groups, strict=True) if stays]
        pair_weight = fsum(denominator)
        metrics[metric] = dict(
            equal_positive_mean_of_weighted_neighbor_concordance=
                fsum(valid)/len(positives) if positives and len(valid) == len(positives) else None,
            covered_positive_mean_of_weighted_neighbor_concordance=
                fsum(valid)/len(covered) if covered and len(valid) == len(covered) else None,
            explicitly_defined_positive_mean_of_weighted_neighbor_concordance=fsum(valid)/len(valid) if valid else None,
            explicitly_defined_positive_count=len(valid),
            covered_positive_with_undefined_comparisons=len(covered)-len(valid),
            pair_weighted_concordance_ratio=fsum(numerator)/pair_weight
                if pair_weight and not comparisons['undefined'] else None,
            pair_weight_denominator=pair_weight, defined_pair_weight_denominator=fsum(defined_denominator),
            explicitly_defined_pair_weighted_concordance_ratio=fsum(numerator)/fsum(defined_denominator)
                if defined_denominator else None,
            pair_comparison_counts={name:comparisons[name] for name in ('wins','ties','losses','undefined')})
    return dict(positive_count=len(positives), positive_with_selected_neighbors=len(covered),
        zero_neighbor_positive_count=len(positives)-len(covered),
        registered_positive_stay_edges=sum(len(stays) for _, stays in groups), metrics=metrics)


def matched_summary(records, point):
    """Registered neighbours, weighted within positives and equal across them.

    This is an internal conditional contrast, not a finite-population unbiased
    ratio. The separate pair ratio weights every edge by its stay objective
    weight, including repeated use of a stay for several registered positives.
    """
    records = list(records)
    positives = {_identity(row):row for row in records if row['y'] != 1}
    neighbors = defaultdict(list)
    for stay in records:
        if stay['y'] != 1:
            continue
        for direction, name in DIRECTIONS.items():
            for key in stay['directions'][name]['matched_positive_keys']:
                key = tuple(key)
                if key not in positives or positives[key]['y'] != direction:
                    raise ValueError('A registered neighbor points outside the fixed training positives')
                neighbors[key].append(stay)
    result = {}
    for direction, name in DIRECTIONS.items():
        selected = [row for row in positives.values() if row['y'] == direction]
        groups = dict(all=selected)
        groups.update({str(seed):[row for row in selected if _identity(row)[0] == seed]
                       for seed in sorted({_identity(row)[0] for row in records})})
        result[name] = {group:_matched_group(rows, neighbors, point, direction) for group, rows in groups.items()}
    return result


def aggregate_records(records, gradients, points, *, expected_references=1000):
    """Aggregate the complete fixed cohort after streaming case construction."""
    records, points = list(records), _points(points)
    if (len(records) != expected_references
            or len({r['case_index'] for r in records}) != len(records)
            or not isclose(fsum(r['objective_weight'] for r in records), 1., abs_tol=1e-12)):
        raise ValueError('All fixed naturally weighted training references are required')
    arrays = {row['case_index']:{view:{point:np.asarray(value, dtype=float) if value is not None else None
              for point, value in entries.items()} for view, entries in gradients[row['case_index']].items()}
              for row in records}
    native = _native_aggregate(records, arrays, points)
    supported = all(row['native_support_passed'] for row in records)
    return dict(native=native, native_support_passed=supported,
        direction_readout_status='COMPLETE' if supported else 'HOLD_NATIVE_SUPPORT_OR_GRADIENT',
        unpaired={name:training_summary(records, name) for name in points} if supported else None,
        matched={name:matched_summary(records, name) for name in points} if supported else None,
        direction_weighting='Registered objective weights; matched neighbours may cross training seeds. Each positive has raw weight1.',
        matched_estimand='Weighted selected-neighbor concordance within each positive, then equal across positives; zero-neighbor and undefined cases remain explicit. Separate pair ratios use repeated-edge stay objective weights.')
