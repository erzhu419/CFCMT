"""Two-point native LC readouts with the registered natural-frequency weights."""

from collections import Counter
from math import fsum, isfinite

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw, _vectors
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import (
    extended, paired_difference,
)
from experiments.tsc_packet_joint_law_definition_20261110.law import CHOICES
from experiments.tsc_packet_mixture_confirmation_20261111.readout import (
    MixtureParticleLaw, _point, _proposal_components,
)


POINTS = ('zero', 'candidate')
TOLERANCE = 1e-10


def _gradient(raw, shape):
    value = raw.get('gradient')
    if value is None:
        return None
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        return None
    return value.copy()


def _native(raw, gradient):
    loss = extended(raw.get('nll'))
    defined = bool(raw['defined'])
    supported = bool(raw.get('supported', False))
    failure = ('NO_SURVIVOR_MASS' if not defined else 'ZERO_NUMERATOR' if not supported else
               'NONFINITE_NLL' if loss['state'] != 'finite' else
               'MISSING_OR_NONFINITE_GRADIENT' if gradient is None else None)
    return dict(defined=defined, supported=supported, likelihood=raw.get('likelihood'),
        log_likelihood=extended(raw.get('log_likelihood')), loss=loss, nll=loss['value'],
        failure=failure, gradient_l2=float(np.linalg.norm(gradient)) if gradient is not None else None)


def _exact_point(law, theta, y):
    raw = law.marginal(theta, 'lane_change', y)
    gradient = _gradient(raw, theta.shape)
    summary = law.summary(theta)
    weighted_score = summary.pop('weighted_score')
    errors = [abs(summary['total_probability']-1.),
              abs(summary['survivor_probability']+summary['end_probability']-1.),
              float(np.max(np.abs(weighted_score)))]
    probabilities = [law.marginal(theta, 'lane_change', label)['likelihood'] for label in range(3)]
    if all(value is not None for value in probabilities):
        errors.append(abs(sum(probabilities)-1.))
    if max(errors) > TOLERANCE:
        raise ArithmeticError('The exact native LC law failed normalization')
    outcomes = law._leaf_outcomes()
    summary.update(normalization_error=max(errors), class_probabilities=probabilities,
        leaf_count=len(outcomes), surviving_leaves=sum(outcome['survived'] for outcome in outcomes),
        native_positive_leaves=sum(outcome['survived'] and outcome['lane_change'] == CHOICES[y]
                                   for outcome in outcomes))
    return dict(summary=summary, native=_native(raw, gradient)), gradient


def read_case(law, candidate, y):
    """Return JSON scalars and a separate private gradient map for aggregation."""
    candidate = _vectors(candidate)
    if y not in (0, 1, 2):
        raise ValueError('The archived LC label must be left0, stay1 or right2')
    targets = dict(zero=np.zeros_like(candidate), candidate=candidate)
    readouts, gradients = {}, {}
    if isinstance(law, MixtureParticleLaw):
        components, ratio_error = _proposal_components(law, None)
        for view, n in (('primary', 256), ('128', 128)):
            law._size(n)
            count = int(np.count_nonzero(components[:n]))
            if count != n//2:
                raise ValueError('The native LC mixed prefix must retain both components equally')
            points, private = {}, {}
            for point, theta in targets.items():
                scalar, _ = _point(law, theta, [{'head':'lane_change', 'y':y}], n, y, None)
                raw = law.marginal(theta, 'lane_change', y, n=n)
                private[point] = _gradient(raw, theta.shape)
                native = dict(scalar['observed_heads'][0])
                if native['failure'] is None and private[point] is None:
                    native['failure'] = 'MISSING_OR_NONFINITE_GRADIENT'
                points[point] = dict(summary=scalar['summary'], native=native)
            readouts[view] = dict(method='mixture256', particle_count=n,
                component_counts={'0':n-count, 'f':count}, maximum_component_log_ratio_error=ratio_error,
                points=points,
                candidate_minus_zero=paired_difference(points['candidate']['native']['loss'],
                                                       points['zero']['native']['loss']))
            gradients[view] = private
    elif isinstance(law, CachedLaw):
        points, private = {}, {}
        for point, theta in targets.items():
            points[point], private[point] = _exact_point(law, theta, y)
        readouts['primary'] = dict(method='exact', leaf_count=len(law.data['leaves']), points=points,
            candidate_minus_zero=paired_difference(points['candidate']['native']['loss'],
                                                   points['zero']['native']['loss']))
        gradients['primary'] = private
    else:
        raise TypeError('Native LC readout requires an exact law or a corrected mixture law')
    return readouts, gradients


def _weighted_loss(losses, weights):
    states = {loss['state'] for loss in losses}
    if not losses or 'undefined' in states or {'positive_infinity','negative_infinity'} <= states:
        return extended(None)
    if 'positive_infinity' in states:
        return extended(float('inf'))
    if 'negative_infinity' in states:
        return extended(-float('inf'))
    return extended(fsum(weight*loss['value'] for loss, weight in zip(losses, weights, strict=True)))


def _divide(value, denominator):
    if denominator <= 0.:
        return extended(None)
    return extended(value['value']/denominator) if value['state'] == 'finite' else value


def _view(record, view):
    views = record.get('readouts', {})
    if view == '128' and record['proposed_method'] == 'exact':
        view = 'primary'
    return views.get(view)


def _native_for(record, view, point):
    item = _view(record, view)
    if item is None:
        return dict(defined=False, supported=False, loss=extended(None), failure='MISSING_CASE_READOUT')
    return item['points'][point]['native']


def _weighted_gradient(records, gradients, view, point):
    total = None
    missing = 0
    for record in records:
        name = 'primary' if view == '128' and record['proposed_method'] == 'exact' else view
        gradient = gradients.get(record['case_index'], {}).get(name, {}).get(point)
        if gradient is None:
            missing += 1
            continue
        gradient = np.asarray(gradient, dtype=float)
        if not np.isfinite(gradient).all():
            missing += 1
            continue
        if total is None:
            total = np.zeros_like(gradient)
        if total.shape != gradient.shape:
            raise ValueError('Native LC gradients must retain the common coefficient layout')
        total += record['objective_weight']*gradient
    return dict(defined=bool(records and not missing and total is not None),
        missing_references=missing,
        weighted_gradient_l2=float(np.linalg.norm(total)) if records and not missing and total is not None else None)


def _subset(records, gradients, view, point):
    weights = [record['objective_weight'] for record in records]
    weight_sum = fsum(weights)
    natives = [_native_for(record, view, point) for record in records]
    contribution = _weighted_loss([native['loss'] for native in natives], weights)
    summaries = [_view(record, view)['points'][point]['summary']
                 for record in records if _view(record, view) is not None]
    mixed = [summary for record, summary in zip(
        [r for r in records if _view(r, view) is not None], summaries, strict=True)
        if record['proposed_method'] == 'mixture256']
    supported = sum(native['defined'] and native['supported'] for native in natives)
    failures = Counter(native.get('failure') for native in natives if native.get('failure') is not None)
    native_mixed = [_native_for(record, view, point) for record in records
                    if record['proposed_method'] == 'mixture256' and _view(record, view) is not None]
    return dict(references=len(records), objective_weight=weight_sum,
        weighted_nll_contribution=contribution, conditional_weighted_mean_nll=_divide(contribution, weight_sum),
        weighted_gradient=_weighted_gradient(records, gradients, view, point),
        supported_references=supported, unsupported_references=sum(
            native['defined'] and not native['supported'] for native in natives),
        undefined_references=sum(not native['defined'] for native in natives), failures=dict(failures),
        all_references_supported=supported == len(records),
        maximum_normalization_error=max((summary['normalization_error'] for summary in summaries), default=None),
        mixture_diagnostics=dict(
            references=len(mixed),
            minimum_global_ess=min((s['global_ess'] for s in mixed), default=None),
            minimum_survivor_ess=min((s['survivor_ess'] for s in mixed), default=None),
            minimum_posterior_ess=min((native['posterior_ess'] for native in native_mixed), default=None),
            maximum_raw_weight=max((s['max_raw_weight'] for s in mixed), default=None),
            maximum_normalized_weight=max((s['max_normalized_weight'] for s in mixed), default=None),
            maximum_survivor_normalized_weight=max((s['max_survivor_normalized_weight']
                for s in mixed if s['max_survivor_normalized_weight'] is not None), default=None),
            maximum_posterior_normalized_weight=max((native['max_posterior_normalized_weight']
                for native in native_mixed if native['max_posterior_normalized_weight'] is not None), default=None)))


def _point_aggregate(records, gradients, view, point):
    result = _subset(records, gradients, view, point)
    result['by_class'] = {str(y):_subset([r for r in records if r['y'] == y], gradients, view, point)
                          for y in (0, 1, 2)}
    result['by_seed'] = {str(seed):_subset([r for r in records if int(r.get('seed', r.get('reference', {}).get('seed'))) == seed],
                                         gradients, view, point)
                         for seed in sorted({int(r.get('seed', r.get('reference', {}).get('seed'))) for r in records})}
    result['by_method'] = {method:_subset([r for r in records if r['proposed_method'] == method],
                                         gradients, view, point) for method in ('exact', 'mixture256')}
    return result


def aggregate_records(records, gradients, expected_references=424):
    """Retain every registered row in weighted losses, including failed cases."""
    records = list(records)
    if len({r['case_index'] for r in records}) != len(records):
        raise ValueError('Native LC aggregation requires unique case indices')
    if len(records) != expected_references:
        raise ValueError('Every registered reference must remain present in native LC aggregation')
    if any(not isfinite(r['objective_weight']) or r['objective_weight'] <= 0. for r in records):
        raise ValueError('Registered native LC objective weights must be finite and positive')
    results = {}
    for view in ('primary', '128'):
        points = {point:_point_aggregate(records, gradients, view, point) for point in POINTS}
        results[view] = dict(reference_count=len(records), expected_references=expected_references,
            complete_reference_target=True,
            interpretation='Mixed128 with exact cases retained' if view == '128' else 'Exact plus mixed256',
            points=points,
            candidate_minus_zero=paired_difference(points['candidate']['weighted_nll_contribution'],
                                                   points['zero']['weighted_nll_contribution']))
    return dict(views=results,
        mixed_size_sensitivity={point:paired_difference(
            results['128']['points'][point]['weighted_nll_contribution'],
            results['primary']['points'][point]['weighted_nll_contribution']) for point in POINTS},
        objective_weight_sum=fsum(r['objective_weight'] for r in records),
        selected_class_counts=dict(Counter(r['y'] for r in records)))
