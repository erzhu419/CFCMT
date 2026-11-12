"""Three frozen-model likelihood, calibration and discrimination readouts."""

from collections import Counter
from math import exp, fsum, isclose, isfinite

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import extended, paired_difference
from experiments.tsc_packet_lane_change_holdout_20261111.readout import _focal_branch_factors
from experiments.tsc_packet_lc_discrimination_20261111.readout import weighted_auc
from experiments.tsc_packet_lc_refit_20261111.readout import (
    L2, _lc_gradient, _mixed_point, _points as coefficient_points,
)
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw, _proposal_components
from experiments.tsc_packet_positive_covered_cache_20261111.readout import _exact_point, _point_aggregate


POINTS = ('old_lc', 'new_lc', 'frequency_bias')
DIRECTIONS = {0:'left', 2:'right'}
TOLERANCE = 1e-10


def _points(points):
    points = coefficient_points(points)
    if set(points) != set(POINTS):
        raise ValueError('The three frozen904 coefficient points are required')
    return points


def _probability_readout(law, theta, y, n):
    mixed = isinstance(law, MixtureParticleLaw)
    raw = [law.marginal(theta, 'lane_change', label, n=n) if mixed
           else law.marginal(theta, 'lane_change', label) for label in range(3)]
    probabilities = [row['likelihood'] for row in raw]
    logs = [extended(row['log_likelihood']) for row in raw]
    valid = all(p is not None and isfinite(p) and -TOLERANCE <= p <= 1+TOLERANCE for p in probabilities)
    error = abs(fsum(probabilities)-1.) if valid else None
    normalized = bool(valid and error <= TOLERANCE)
    brier = fsum((p-float(label == y))**2 for label,p in enumerate(probabilities)) if normalized else None
    logw = law._evaluate(theta)[0]
    if mixed:
        size = law._size(n)
        logw, survived = logw[:size], law.survived[:size]
        groups = [law.data['focal_groups'][index] for index in law.focal_groups[:size][survived]]
        masks = np.zeros((len(groups),3), dtype=bool)
        for index,group in enumerate(groups):
            masks[index,group['allowed']] = True
    else:
        survived = np.asarray([row['survived'] for row in law._leaf_outcomes()], dtype=bool)
        factors, _, allowed = _focal_branch_factors(law, theta)
        if not np.array_equal(factors >= 0, survived):
            raise ArithmeticError('Focal LC factors and saved survival disagree')
        masks = allowed[survived]
    survivor_weights = logw[survived]
    denominator = float(logsumexp(survivor_weights))
    permission, conditional = {}, {}
    for direction in DIRECTIONS:
        permitted = masks[:,direction]
        allowed_log = float(logsumexp(survivor_weights[permitted])-denominator) if np.isfinite(denominator) else None
        permission[str(direction)] = exp(allowed_log) if allowed_log is not None else None
        conditional[str(direction)] = extended(raw[direction]['log_likelihood']-allowed_log
            if np.any(permitted) and allowed_log is not None and raw[direction]['log_likelihood'] is not None else None)
        if normalized and allowed_log is not None and probabilities[direction] > permission[str(direction)]+TOLERANCE:
            raise ArithmeticError('Direction probability exceeds its saved permission mass')
    return dict(class_probabilities=probabilities, class_log_probabilities=logs, allowed_mass=permission,
        conditional_log_probability=conditional, brier=brier, normalization_error=error,
        normalization_passed=normalized)


def read_case(law, points, task):
    """Read one saved law, returning scalars and separate JSON gradient arrays."""
    points = _points(points)
    y = int(task['y'])
    if y not in (0,1,2):
        raise ValueError('The frozen native LC label must be left0, stay1 or right2')
    mixed = isinstance(law, MixtureParticleLaw)
    if not mixed and not isinstance(law, CachedLaw):
        raise TypeError('A registered exact or fixed-anchor mixture law is required')
    if task['proposed_method'] != ('mixture256' if mixed else 'exact'):
        raise ValueError('The frozen numerical method changed')
    if mixed:
        _proposal_components(law, None)
    row, gradients = dict(task, readouts={}), {}
    for view,n in (('primary',256),('128',128)) if mixed else (('primary',None),):
        values,private = {},{}
        for point,theta in points.items():
            values[point],gradient = (_mixed_point(law,theta,y,n) if mixed else _exact_point(law,theta,y))
            values[point]['probability'] = _probability_readout(law,theta,y,n)
            private[point] = gradient.tolist() if gradient is not None else None
        row['readouts'][view],gradients[view] = dict(points=values),private
    row['probability_normalization_passed'] = all(value['probability']['normalization_passed']
        for view in row['readouts'].values() for value in view['points'].values())
    row['native_support_passed'] = row['probability_normalization_passed'] and all(
        value['native']['defined'] and value['native']['supported'] and value['native']['loss']['state'] == 'finite'
        and gradients[view][point] is not None
        for view,data in row['readouts'].items() for point,value in data['points'].items())
    return row,gradients


def _probability(record, point, view):
    if record['proposed_method'] == 'exact':
        view = 'primary'
    return record['readouts'][view]['points'][point]['probability']


def _log_score(value):
    return value['value'] if value['state'] == 'finite' else -float('inf') if value['state'] == 'negative_infinity' else None


def _direction_auc(records, probabilities, direction):
    positive = [(row,value) for row,value in zip(records,probabilities,strict=True) if row['y'] == direction]
    stays = [(row,value) for row,value in zip(records,probabilities,strict=True) if row['y'] == 1]
    pw,nw = ([row['objective_weight'] for row,value in group] for group in (positive,stays))
    p,n = ([_log_score(value['class_log_probabilities'][direction]) for row,value in group] for group in (positive,stays))
    cp,cn = ([_log_score(value['conditional_log_probability'][str(direction)]) for row,value in group] for group in (positive,stays))
    vp = [(v,w) for v,w in zip(cp,pw,strict=True) if v is not None]
    vn = [(v,w) for v,w in zip(cn,nw,strict=True) if v is not None]
    return dict(positive_count=len(positive), stay_count=len(stays), positive_weight=fsum(pw), stay_weight=fsum(nw),
        weighted_auc=weighted_auc(p,pw,n,nw), conditional_weighted_auc=weighted_auc(cp,pw,cn,nw),
        explicitly_defined_conditional_auc=dict(positive_count=len(vp),stay_count=len(vn),
            excluded_positive_count=len(cp)-len(vp),excluded_stay_count=len(cn)-len(vn),
            excluded_positive_weight=fsum(pw)-fsum(w for v,w in vp),excluded_stay_weight=fsum(nw)-fsum(w for v,w in vn),
            auc=weighted_auc([v for v,w in vp],[w for v,w in vp],[v for v,w in vn],[w for v,w in vn])),
        zero_permission={name:dict(count=sum(value['allowed_mass'][str(direction)] == 0 for row,value in group),
            objective_weight=fsum(row['objective_weight'] for row,value in group if value['allowed_mass'][str(direction)] == 0))
            for name,group in (('positive',positive),('stay',stays))},
        zero_direction_probability={name:sum(value['class_probabilities'][direction] == 0 for row,value in group)
                                    for name,group in (('positive',positive),('stay',stays))})


def aggregate_probability(records, point, view='primary'):
    """Natural-frequency Brier, raw predicted totals and weighted direction AUC."""
    records = list(records)
    values = [_probability(row,point,view) for row in records]
    brier = [value['brier'] for value in values]
    totals = {str(label):fsum(row['raw_weight']*value['class_probabilities'][label]
        for row,value in zip(records,values,strict=True))
        if all(value['class_probabilities'][label] is not None for value in values) else None for label in range(3)}
    return dict(reference_count=len(records), brier=fsum(row['objective_weight']*value
        for row,value in zip(records,brier,strict=True)) if all(value is not None for value in brier) else None,
        predicted_class_totals=totals, predicted_class_total_sum=fsum(totals.values()) if all(v is not None for v in totals.values()) else None,
        observed_class_totals={str(label):fsum(row['raw_weight'] for row in records if row['y'] == label) for label in range(3)},
        all_probability_normalization_passed=all(value['normalization_passed'] for value in values),
        maximum_probability_normalization_error=max((value['normalization_error'] for value in values
            if value['normalization_error'] is not None), default=None),
        discrimination={name:_direction_auc(records,values,direction) for direction,name in DIRECTIONS.items()})


def aggregate_records(records, gradients, points, *, expected_references=340):
    records,points = list(records),_points(points)
    if (len(records) != expected_references or len({row['case_index'] for row in records}) != len(records)
            or not isclose(fsum(row['objective_weight'] for row in records),1.,abs_tol=1e-12)
            or not isclose(fsum(row['raw_weight'] for row in records),10000.,abs_tol=1e-8)
            or any(not isclose(row['objective_weight'],row['raw_weight']/10000.,rel_tol=0.,abs_tol=1e-15) for row in records)):
        raise ValueError('All fixed904 references and denominator10000 weights are required')
    if expected_references == 340 and (Counter(row['y'] for row in records) != {0:56,1:223,2:61}
            or any(not isclose(fsum(row['raw_weight'] for row in records if row['y'] == label),expected,abs_tol=1e-8)
                   for label,expected in ((0,56.),(1,9883.),(2,61.)))):
        raise ValueError('The frozen904 natural class totals changed')
    arrays = {row['case_index']:{view:{point:np.asarray(value,dtype=float) if value is not None else None
        for point,value in entries.items()} for view,entries in gradients[row['case_index']].items()} for row in records}
    native,probability,sensitivity = {},{},{}
    for point,theta in points.items():
        views = {}
        probability[point] = {}
        for view in ('primary','128'):
            scalar = _point_aggregate(records,arrays,view,point)
            scalar['lc_gradient'] = _lc_gradient(records,arrays,view,point,theta)
            views[view] = scalar
            probability[point][view] = aggregate_probability(records,point,view)
        native[point] = dict(views,lc_regularization=float(L2*np.sum(theta[3:,1:]**2)))
        a,b = probability[point]['128'],probability[point]['primary']
        sensitivity[point] = dict(weighted_nll_difference=paired_difference(
            views['128']['weighted_nll_contribution'],views['primary']['weighted_nll_contribution']),
            brier_difference=a['brier']-b['brier'] if a['brier'] is not None and b['brier'] is not None else None,
            predicted_class_total_differences={label:a['predicted_class_totals'][label]-b['predicted_class_totals'][label]
                if a['predicted_class_totals'][label] is not None and b['predicted_class_totals'][label] is not None else None
                for label in ('0','1','2')})
    return dict(native=dict(reference_count=len(records),objective_weight_sum=fsum(r['objective_weight'] for r in records),
            raw_weight_sum=fsum(r['raw_weight'] for r in records),selected_class_counts=dict(Counter(r['y'] for r in records)),points=native),
        probability=probability,prefix_sensitivity=sensitivity,
        all_native_support_passed=all(row['native_support_passed'] for row in records),
        all_probability_normalization_passed=all(row['probability_normalization_passed'] for row in records),
        brier_definition='sum(objective_weight_i * sum_class(P_i-class - onehot(y_i))^2); no division by class count',
        discrimination_scope='Natural-weight direction-positive versus stay groups; conditional undefined and zero probabilities retained explicitly. Historical62 matched pairs are separate.')
