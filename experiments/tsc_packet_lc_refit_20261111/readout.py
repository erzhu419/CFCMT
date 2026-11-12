"""Scalar training and fixed-pair readouts for the cached LC refit."""

from collections import Counter, defaultdict
from math import exp, fsum, isclose

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw, _vectors
from experiments.tsc_packet_conditional_particles_20261110.verify import consistency
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import (
    extended, paired_difference, strict_mean,
)
from experiments.tsc_packet_lane_change_holdout_20261111.readout import (
    _compare_ranks, _focal_branch_factors, _rank, _rank_summary,
)
from experiments.tsc_packet_mixture_confirmation_20261111.readout import (
    MixtureParticleLaw, _max_normalized, _proposal_components, _query,
)
from experiments.tsc_packet_positive_covered_cache_20261111.readout import (
    _exact_point, _gradient, _native, _point_aggregate,
)


L2 = .001
DIRECTIONS = {0: 'left', 2: 'right'}


def _points(points):
    result = {name: _vectors(theta) for name, theta in points.items() if theta is not None}
    if not result:
        raise ValueError('At least one frozen coefficient point is required')
    return result


def _mixed_point(law, theta, y, n):
    """Read arbitrary target coefficients with the original proposal anchor."""
    error, summary = consistency(law, theta, law.data['eligibility'], n)
    summary.pop('weighted_score')
    logw = law._evaluate(theta)[0][:n]
    survived = law.survived[:n]
    summary.update(normalization_error=error,
        survivor_samples=int(survived.sum()), max_raw_weight=exp(float(np.max(logw))),
        max_survivor_normalized_weight=_max_normalized(logw[survived]))
    raw = law.marginal(theta, 'lane_change', y, n=n)
    gradient = _gradient(raw, theta.shape)
    native = _query(law, theta, 'lane_change', y, n)
    if native['failure'] is None and gradient is None:
        native['failure'] = 'MISSING_OR_NONFINITE_GRADIENT'
    return dict(summary=summary, native=native), gradient


def _lc_gradient(records, gradients, view, point, theta):
    total = np.zeros_like(theta)
    for record in records:
        actual_view = 'primary' if record['proposed_method'] == 'exact' else view
        gradient = gradients[record['case_index']][actual_view][point]
        if gradient is None:
            return dict(defined=False, unpenalized_l2=None, penalized_l2=None)
        total += record['objective_weight'] * gradient
    if not np.isfinite(total).all():
        return dict(defined=False, unpenalized_l2=None, penalized_l2=None)
    optimized = total[3:].copy()
    penalized = optimized.copy()
    penalized[:, 1:] += 2*L2*theta[3:, 1:]
    return dict(defined=True, unpenalized_l2=float(np.linalg.norm(optimized)),
                penalized_l2=float(np.linalg.norm(penalized)))


def training_readouts(laws, references, points, *, expected_references=424):
    """Evaluate each supplied point on the complete registered weighted cohort."""
    references = list(references)
    points = _points(points)
    if (len(references) != expected_references
            or len({r['case_index'] for r in references}) != len(references)
            or not isclose(fsum(r['objective_weight'] for r in references), 1., abs_tol=1e-12)):
        raise ValueError('Training readouts require the full registered objective')
    records, gradients = [], {}
    for ref in references:
        index, y = ref['case_index'], int(ref['y'])
        law = laws[index]
        if y not in (0, 1, 2):
            raise ValueError('Native LC labels must be left0, stay1 or right2')
        mixed = isinstance(law, MixtureParticleLaw)
        if not mixed and not isinstance(law, CachedLaw):
            raise TypeError('Training needs exact laws or corrected fixed-anchor mixtures')
        method = 'mixture256' if mixed else 'exact'
        if ref['proposed_method'] != method:
            raise ValueError('The registered numerical method changed')
        if mixed:
            _proposal_components(law, None)
        row = dict(ref, readouts={})
        private = {}
        for view, n in (('primary', 256), ('128', 128)) if mixed else (('primary', None),):
            values, derivatives = {}, {}
            for name, theta in points.items():
                values[name], derivatives[name] = (_mixed_point(law, theta, y, n) if mixed
                                                   else _exact_point(law, theta, y))
            row['readouts'][view] = dict(points=values)
            private[view] = derivatives
        records.append(row)
        gradients[index] = private

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

    comparisons = {}
    for baseline in ('original', 'frequency_bias'):
        if baseline not in result:
            continue
        comparisons[baseline] = {name: paired_difference(
            value['primary']['weighted_nll_contribution'],
            result[baseline]['primary']['weighted_nll_contribution'])
            for name, value in result.items() if name != baseline}
    return dict(reference_count=len(records), objective_weight_sum=fsum(r['objective_weight'] for r in records),
        selected_class_counts=dict(Counter(r['y'] for r in records)), points=result,
        mixed_size_sensitivity={name: paired_difference(
            value['128']['weighted_nll_contribution'], value['primary']['weighted_nll_contribution'])
            for name, value in result.items()}, weighted_nll_differences=comparisons)


def _regression_case(law, case, theta, masks):
    direction, label = int(case['direction']), int(case['y'])
    summary = law.summary(theta)
    score = summary.pop('weighted_score')
    raw_direction = law.marginal(theta, 'lane_change', direction)
    raw_actual = (raw_direction if direction == label else law.marginal(theta, 'lane_change', label))
    directional = _native(raw_direction, _gradient(raw_direction, theta.shape))
    actual = (directional if direction == label else
              _native(raw_actual, _gradient(raw_actual, theta.shape)))
    probabilities = [law.marginal(theta, 'lane_change', y)['likelihood'] for y in range(3)]
    errors = [abs(summary['total_probability']-1.),
        abs(summary['survivor_probability']+summary['end_probability']-1.),
        float(np.max(np.abs(score)))]
    if all(p is not None for p in probabilities):
        errors.append(abs(sum(probabilities)-1.))
    normalization = max(errors)
    if normalization > 1e-10:
        raise ArithmeticError('The exact regression law failed normalization')
    logw = law._evaluate(theta)[0]
    survived, allowed = masks
    denominator = float(logsumexp(logw[survived]))
    directional['mask_availability'] = (exp(float(logsumexp(logw[survived][allowed[:, direction]])
        - denominator)) if np.isfinite(denominator) else None)
    return dict(direction=directional, actual=actual, normalization_error=normalization,
                survivor_probability=summary['survivor_probability'])


def _regression_group(pairs, name):
    positives = [row['positive']['points'][name] for row in pairs]
    stays = [row['stay']['points'][name] for row in pairs]
    ranks = [_rank(a['direction']['loss'], b['direction']['loss'])
             for a, b in zip(positives, stays, strict=True)]
    blocks = defaultdict(list)
    for index, row in enumerate(pairs):
        t = row['positive'].get('time_sec', row['positive']['reference']['time_sec'])
        blocks[str(int(t)//300)].append(index)
    return dict(
        pairs=len(pairs),
        mean_positive_direction_probability=strict_mean([extended(r['direction']['likelihood']) for r in positives]),
        mean_stay_direction_probability=strict_mean([extended(r['direction']['likelihood']) for r in stays]),
        mean_positive_actual_nll=strict_mean([r['actual']['loss'] for r in positives]),
        mean_stay_actual_nll=strict_mean([r['actual']['loss'] for r in stays]),
        mean_pair_balanced_actual_nll=strict_mean([r['actual']['loss'] for r in positives+stays]),
        mean_pair_log_probability_margin=strict_mean([paired_difference(b['direction']['loss'], a['direction']['loss'])
            for a, b in zip(positives, stays, strict=True)]),
        ranking=_rank_summary(ranks),
        positive_direction_support=dict(supported=sum(r['direction']['supported'] for r in positives),
            undefined=sum(not r['direction']['defined'] for r in positives)),
        stay_actual_support=dict(supported=sum(r['actual']['supported'] for r in stays),
            undefined=sum(not r['actual']['defined'] for r in stays)),
        maximum_normalization_error=max(r['normalization_error'] for r in positives+stays),
        positive_time_blocks={block: dict(pairs=len(indices), ranking=_rank_summary([ranks[i] for i in indices]))
                              for block, indices in sorted(blocks.items())}), ranks


def regression_readouts(laws, cases, points, *, expected_pairs=62):
    """Evaluate all fixed 904 pairs under each supplied frozen point."""
    points = _points(points)
    cases = list(cases)
    if len(cases) != 2*expected_pairs or len({r['case_index'] for r in cases}) != len(cases):
        raise ValueError('All fixed regression cases are required')
    pairs = {}
    for case in cases:
        law = laws[case['case_index']]
        if not isinstance(law, CachedLaw):
            raise TypeError('The fixed 904 regression readout requires its exact cache')
        if case['direction'] not in DIRECTIONS or case['y'] not in (1, case['direction']):
            raise ValueError('Regression case direction or native label changed')
        survived = np.asarray([outcome['survived'] for outcome in law._leaf_outcomes()], dtype=bool)
        factors, _, allowed = _focal_branch_factors(law, np.zeros_like(next(iter(points.values()))))
        if not np.array_equal(factors >= 0, survived):
            raise ArithmeticError('Focal LC factor and survival disagree')
        row = dict(case, points={name:_regression_case(law, case, theta, (survived, allowed[survived]))
                                 for name, theta in points.items()})
        pair = pairs.setdefault(case['pair_index'], dict(pair_index=case['pair_index'], direction=case['direction']))
        if pair['direction'] != case['direction'] or case['role'] in pair:
            raise ValueError('A frozen pair has duplicate roles or inconsistent direction')
        pair[case['role']] = row
    if len(pairs) != expected_pairs or any(not {'positive','stay'} <= row.keys() for row in pairs.values()):
        raise ValueError('Every fixed positive/control pair must be present')
    grouped = {name:[row for row in pairs.values() if row['direction'] == label]
               for label, name in DIRECTIONS.items()}
    summaries, ranks = {}, {}
    for name in points:
        summaries[name], ranks[name] = {}, {}
        for direction, rows in grouped.items():
            if rows:
                summaries[name][direction], ranks[name][direction] = _regression_group(rows, name)
    comparisons = {}
    for baseline in ('original', 'frequency_bias'):
        if baseline not in summaries:
            continue
        comparisons[baseline] = {}
        for name, groups in summaries.items():
            if name == baseline:
                continue
            comparisons[baseline][name] = {direction: dict(
                positive_nll=paired_difference(value['mean_positive_actual_nll'],
                    summaries[baseline][direction]['mean_positive_actual_nll']),
                stay_nll=paired_difference(value['mean_stay_actual_nll'],
                    summaries[baseline][direction]['mean_stay_actual_nll']),
                ranking_pairs=_compare_ranks(ranks[baseline][direction], ranks[name][direction]))
                for direction, value in groups.items()}
    return dict(pair_count=len(pairs), case_count=len(cases), points=summaries, comparisons=comparisons)
