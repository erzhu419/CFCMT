"""Read exact fixed lane-change probabilities on the matched 904 cohort."""

from collections import Counter, defaultdict
from math import exp, log

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_packet_conditional_cache_20261110.cache import _vectors
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import (
    extended, paired_difference, strict_mean,
)
from experiments.tsc_packet_focal_latent_decomposition_20261111.readout import (
    COMBINATIONS, external_receipt_check,
)
from experiments.tsc_packet_joint_law_definition_20261110.law import CHOICES


DIRECTIONS = {0: 'left', 2: 'right'}
TOLERANCE = 1e-10


def _focal_branch_factors(cache, candidate):
    """Link every leaf to its focal categorical decision through the prefix tree."""
    data = cache.data
    delta = np.zeros(len(data['factors'])+1)
    allowed = np.zeros((len(delta), len(CHOICES)), dtype=bool)
    focal_ids = set()
    for i, factor in enumerate(data['factors']):
        if factor['kind'] == 'categorical' and factor['actor'] == cache.focal_id:
            phi = cache.features[factor['feature_id']]
            logits = np.array([candidate[3] @ phi, 0., candidate[4] @ phi])
            choices = factor['allowed']
            delta[i] = logits[factor['choice']]-logsumexp(logits[choices])+log(len(choices))
            allowed[i, choices] = True
            focal_ids.add(i)

    prefix_factors = np.full(len(data['prefixes']), -1, dtype=int)
    for i, node in enumerate(data['prefixes']):
        factor = node['factor_id']
        if factor in focal_ids:
            prefix_factors[i] = factor
        elif node['parent'] >= 0:
            prefix_factors[i] = prefix_factors[node['parent']]
    leaf_factors = prefix_factors[[leaf['prefix_id'] for leaf in data['leaves']]]
    return leaf_factors, delta[leaf_factors], allowed[leaf_factors]


def _native_readout(label, log_probabilities, mask, survivor_weights, log_survival):
    defined = np.isfinite(log_survival)
    logp = log_probabilities[label] if defined else None
    return dict(
        defined=bool(defined), supported=bool(defined and np.isfinite(logp)),
        native_probability=exp(logp) if defined else None,
        native_nll=extended(-logp if defined else None),
        native_mask_availability=(exp(float(logsumexp(survivor_weights[mask])-log_survival))
                                  if defined else None),
        native_compatible_leaves=int(mask.sum()),
    )


def _exact_combination(weights, survived, class_masks, availability, direction, actual_label):
    survivor_weights = weights[survived]
    log_survival = float(logsumexp(survivor_weights))
    total = exp(float(logsumexp(weights)))
    survival = exp(log_survival)
    end = exp(float(logsumexp(weights[~survived])))
    errors = [abs(total-1.), abs(survival+end-1.)]
    log_probabilities = ([(float(logsumexp(weights[mask]))-log_survival) for mask in class_masks]
                         if np.isfinite(log_survival) else [None]*len(CHOICES))
    probabilities = [exp(p) if p is not None else None for p in log_probabilities]
    if np.isfinite(log_survival):
        errors.append(abs(sum(probabilities)-1.))
    summary = dict(total_probability=total, survivor_probability=survival, end_probability=end,
                   normalization_error=max(errors), leaf_count=len(weights),
                   surviving_leaves=int(survived.sum()), class_probabilities=probabilities)
    if summary['normalization_error'] > TOLERANCE:
        raise ArithmeticError('An exact crossed lane-change law failed normalization')
    direction_row = _native_readout(direction, log_probabilities, availability[:, direction],
                                    survivor_weights, log_survival)
    actual_row = (direction_row if direction == actual_label else
                  _native_readout(actual_label, log_probabilities, availability[:, actual_label],
                                  survivor_weights, log_survival))
    return summary, direction_row, actual_row


def read_case(cache, theta, *, direction, actual_label):
    """Use the pair direction for ranking and the native label for loss."""
    if direction not in DIRECTIONS or actual_label not in (1, direction):
        raise ValueError('The frozen positive must change in its pair direction and the control must stay')
    candidate = _vectors(theta)
    outcomes = cache._leaf_outcomes()
    survived = np.asarray([row['survived'] for row in outcomes], dtype=bool)
    class_masks = [np.asarray([row['lane_change'] == label for row in outcomes], dtype=bool)
                   for label in CHOICES]
    focal, delta, allowed = _focal_branch_factors(cache, candidate)
    if not np.array_equal(focal >= 0, survived):
        raise ArithmeticError('Focal categorical decisions must correspond to surviving leaves')
    availability = allowed[survived]

    numeric, endpoints = {}, {}
    for point, coefficients in (('0', np.zeros_like(candidate)), ('f', candidate)):
        numeric[point] = cache._evaluate(coefficients)[0]
        endpoints[point] = {label: cache.marginal(coefficients, 'lane_change', label)['nll']
                            for label in {direction, actual_label}}

    summaries, directional, actual = {}, {}, {}
    for key in COMBINATIONS:
        latent, focal_point = key
        weights = numeric[latent]
        if latent != focal_point:
            weights = weights+delta if focal_point == 'f' else weights-delta
        summaries[key], directional[key], actual[key] = _exact_combination(
            weights, survived, class_masks, availability, direction, actual_label,
        )
    for endpoint, crossed in (('00', '0f'), ('ff', 'f0')):
        if abs(summaries[endpoint]['survivor_probability'] -
               summaries[crossed]['survivor_probability']) > TOLERANCE:
            raise ArithmeticError('Changing the focal class probabilities changed survival mass')

    points, errors = {}, []
    for point, name in (('0', 'zero'), ('f', 'candidate')):
        key = point*2
        for label, row in ((direction, directional[key]), (actual_label, actual[key])):
            check = external_receipt_check(row['native_nll'], endpoints[point][label], tol=TOLERANCE)
            if not check['matched']:
                raise ArithmeticError('Exact crossed endpoint differs from the cached marginal')
            if check['absolute_error'] is not None:
                errors.append(check['absolute_error'])
        points[name] = dict(summary=summaries[key], observed_heads=[dict(head='lane_change', y=actual_label,
            nll=actual[key]['native_nll']['value'], loss=actual[key]['native_nll'],
            likelihood=actual[key]['native_probability'], defined=actual[key]['defined'],
            supported=actual[key]['supported'])])

    return {'exact': dict(points=points, direction_combinations=directional,
                          actual_label_combinations=actual, combination_normalization=summaries,
                          maximum_endpoint_nll_error=max(errors, default=None))}


def _rank(positive_nll, stay_nll):
    """A smaller direction NLL means that the change receives more probability."""
    first, second = positive_nll['state'], stay_nll['state']
    if 'undefined' in (first, second):
        return 'undefined'
    if first == second == 'positive_infinity':
        return 'ties'
    if first == 'positive_infinity':
        return 'losses'
    if second == 'positive_infinity':
        return 'wins'
    difference = positive_nll['value']-stay_nll['value']
    if abs(difference) <= TOLERANCE:
        return 'ties'
    return 'wins' if difference < 0. else 'losses'


def _rank_summary(ranks):
    counts = Counter(ranks)
    score = (counts['wins'] + .5*counts['ties'])/len(ranks) if ranks and not counts['undefined'] else None
    return dict(wins=counts['wins'], ties=counts['ties'], losses=counts['losses'],
                undefined=counts['undefined'], score=score)


def _compare_ranks(zero, candidate):
    ordering = {'wins': 1, 'ties': 0, 'losses': -1}
    counts = Counter()
    for a, b in zip(zero, candidate, strict=True):
        if 'undefined' in (a, b):
            counts['undefined'] += 1
            continue
        difference = ordering[b]-ordering[a]
        counts['improved' if difference > 0 else 'worsened' if difference < 0 else 'unchanged'] += 1
    return {key: counts[key] for key in ('improved', 'unchanged', 'worsened', 'undefined')}


def _support(rows):
    return dict(
        zero_numerator=sum(r['defined'] and not r['supported'] for r in rows),
        undefined=sum(not r['defined'] for r in rows),
        compatible_leaves_min=min(r['native_compatible_leaves'] for r in rows),
    )


def _group(rows):
    positive = [r['positive']['readouts']['exact'] for r in rows]
    stays = [r['stay']['readouts']['exact'] for r in rows]
    combinations, ranks = {}, {}
    for key in COMBINATIONS:
        p = [r['direction_combinations'][key] for r in positive]
        s = [r['direction_combinations'][key] for r in stays]
        stay_actual = [r['actual_label_combinations'][key] for r in stays]
        stats = [r['combination_normalization'][key] for r in positive + stays]
        ranks[key] = [_rank(a['native_nll'], b['native_nll']) for a, b in zip(p, s, strict=True)]
        combinations[key] = dict(
            mean_positive_direction_probability=strict_mean([extended(r['native_probability']) for r in p]),
            mean_stay_direction_probability=strict_mean([extended(r['native_probability']) for r in s]),
            mean_positive_direction_availability=strict_mean([extended(r['native_mask_availability']) for r in p]),
            mean_stay_direction_availability=strict_mean([extended(r['native_mask_availability']) for r in s]),
            mean_positive_actual_nll=strict_mean([r['native_nll'] for r in p]),
            mean_stay_actual_nll=strict_mean([r['native_nll'] for r in stay_actual]),
            mean_pair_balanced_actual_nll=strict_mean([r['native_nll'] for r in p + stay_actual]),
            mean_pair_log_probability_margin=strict_mean([
                paired_difference(b['native_nll'], a['native_nll']) for a, b in zip(p, s, strict=True)
            ]),
            ranking=_rank_summary(ranks[key]),
            both_direction_zero=sum(a['native_nll']['state'] == b['native_nll']['state'] == 'positive_infinity'
                                    for a, b in zip(p, s, strict=True)),
            maximum_normalization_error=max(r['normalization_error'] for r in stats),
            positive_direction_support=_support(p), stay_direction_support=_support(s),
            stay_actual_support=_support(stay_actual),
        )

    zero, candidate = combinations['00'], combinations['ff']
    changes = {metric: paired_difference(candidate[metric], zero[metric])
               for metric in zero if metric.startswith('mean_')}
    changes['ranking_score'] = paired_difference(
        extended(candidate['ranking']['score']), extended(zero['ranking']['score']),
    )
    changes['ranking_pairs'] = _compare_ranks(ranks['00'], ranks['ff'])
    changes['positive_direction_probability_pairs'] = _rank_summary([
        _rank(r['direction_combinations']['ff']['native_nll'], r['direction_combinations']['00']['native_nll'])
        for r in positive
    ])
    blocks = defaultdict(list)
    for index, row in enumerate(rows):
        if 'time_sec' in row['positive']:
            blocks[str(int(row['positive']['time_sec'])//300)].append(index)
    time_blocks = {block: dict(
        pairs=len(indices),
        zero=_rank_summary([ranks['00'][i] for i in indices]),
        candidate=_rank_summary([ranks['ff'][i] for i in indices]),
    ) for block, indices in sorted(blocks.items())}

    points = {}
    for name in ('zero', 'candidate'):
        stats = [r['points'][name]['summary'] for r in positive + stays]
        points[name] = dict(
            minimum_survivor_probability=min(r['survivor_probability'] for r in stats),
            maximum_normalization_error=max(r['normalization_error'] for r in stats),
        )
    return dict(pairs=len(rows), combinations=combinations, candidate_minus_zero=changes,
                positive_time_blocks=time_blocks, point_diagnostics=points)


def aggregate_pairs(rows):
    """Compare complete exact matched pairs within each direction."""
    rows = list(rows)
    if any(row['direction'] not in DIRECTIONS for row in rows):
        raise ValueError('Every pair must have its frozen left or right direction')
    by_direction = {name: [r for r in rows if r['direction'] == y] for y, name in DIRECTIONS.items()}
    return {'exact': {direction: _group(group) for direction, group in by_direction.items() if group}}
