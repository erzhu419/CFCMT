"""Evaluate fixed parameter points on a balanced proposal mixture."""

from math import exp, log

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_packet_conditional_cache_20261110.cache import _vectors
from experiments.tsc_packet_conditional_particles_20261110.particles import ParticleLaw
from experiments.tsc_packet_conditional_particles_20261110.verify import consistency
from experiments.tsc_packet_fitted_cache_stability_20261111.readout import (
    extended,
    paired_difference,
)
from experiments.tsc_packet_focal_latent_decomposition_20261111.readout import (
    nll_decomposition,
)


LOG_TWO = log(2.)
TOLERANCE = 1e-10


def mixture_log_offset(log_anchor_ratio):
    """Return log(q_mix / q0) for the fixed equal mixture."""
    return np.logaddexp(-LOG_TWO, np.asarray(log_anchor_ratio, dtype=float) - LOG_TWO)


class MixtureParticleLaw(ParticleLaw):
    """Apply the marginal proposal denominator to the saved prefix weights."""

    def __init__(self, data, anchor):
        super().__init__(data)
        self.anchor_log_ratio = super()._evaluate(anchor)[0]
        self.log_offset = mixture_log_offset(self.anchor_log_ratio)

    def _evaluate(self, theta):
        logw, scores, focal_prob, focal_score = super()._evaluate(theta)
        return logw - self.log_offset, scores, focal_prob, focal_score


def _max_normalized(log_weights):
    if not len(log_weights):
        return None
    return exp(float(np.max(log_weights) - logsumexp(log_weights)))


def _query(pool, theta, head, observed, n, *, class_label=None):
    if head == 'lane_change':
        class_label = int(observed)
    if head == 'motion' and class_label is not None:
        result = pool.class_speed(theta, class_label, observed, n=n)
    else:
        result = pool.marginal(theta, head, observed, n=n)
    logw, _, focal_prob, _ = pool._evaluate(theta)
    logw = logw[:n]
    survived = pool.survived[:n]
    posterior_max = None
    if result['supported']:
        values = pool._base_values(head, observed)[:n].copy()
        if class_label is not None:
            values[survived] *= focal_prob[pool.focal_groups[:n][survived], class_label]
        selected = survived if result['survivor_conditioned'] else np.ones(n, dtype=bool)
        positive = selected & (values > 0.)
        posterior_max = _max_normalized(logw[positive] + np.log(values[positive]))
    loss = extended(result['nll'])
    return dict(
        defined=result['defined'],
        supported=result['supported'],
        likelihood=result['likelihood'],
        log_likelihood=extended(result['log_likelihood']),
        loss=loss,
        nll=loss['value'],
        failure=('NO_SURVIVOR_SAMPLES' if not result['defined'] else
                 'ZERO_NUMERATOR' if not result['supported'] else None),
        posterior_ess=result['posterior_ess'],
        max_posterior_normalized_weight=posterior_max,
        gradient_l2=(float(np.linalg.norm(result['gradient']))
                     if result['gradient'] is not None else None),
    )


def _point(pool, theta, observations, n, native_lc_label, native_speed_mps):
    error, summary = consistency(pool, theta, pool.data['eligibility'], n)
    summary.pop('weighted_score')
    logw, _, focal_prob, _ = pool._evaluate(theta)
    logw = logw[:n]
    survived = pool.survived[:n]
    max_logw = float(np.max(logw))
    if isinstance(pool, MixtureParticleLaw) and max_logw > LOG_TWO + TOLERANCE:
        raise ArithmeticError('A fixed-target raw mixture weight exceeded two')
    summary.update(
        normalization_error=error,
        survivor_samples=int(survived.sum()),
        max_raw_weight=exp(max_logw),
        max_survivor_normalized_weight=_max_normalized(logw[survived]),
    )
    observed_heads = []
    for obs in observations:
        head, y = obs['head'], obs['y']
        observed_heads.append(dict(
            head=head, y=y, manifest_ordinal=obs.get('manifest_ordinal'),
            **_query(pool, theta, head, y, n),
        ))
    row = dict(summary=summary, observed_heads=observed_heads)
    if native_speed_mps is not None:
        row['saved_native_speed'] = _query(pool, theta, 'motion', native_speed_mps, n)
        row['saved_native_class_speed'] = _query(
            pool, theta, 'motion', native_speed_mps, n, class_label=native_lc_label,
        )
    return row, (logw, focal_prob)


def _class_readout(logw, table, masks, native_label):
    if not len(logw):
        return dict(
            defined=False, supported=False, probabilities=[None]*3, probability_sum=None,
            normalization_error=None, native_probability=None, native_nll=extended(None),
            native_mask_availability=None, native_compatible_samples=0, posterior_ess=0.,
            max_posterior_normalized_weight=None,
        )
    denominator = float(logsumexp(logw))
    probabilities, log_probabilities, posterior_terms = [], [], []
    for values in table.T:
        positive = values > 0.
        terms = logw[positive] + np.log(values[positive])
        logp = float(logsumexp(terms)) - denominator
        probabilities.append(exp(logp))
        log_probabilities.append(logp)
        posterior_terms.append(terms)
    masks_mass = [exp(float(logsumexp(logw[mask])) - denominator) for mask in masks.T]
    error = abs(sum(probabilities) - 1.)
    if error > TOLERANCE or any(p > m + TOLERANCE for p, m in zip(probabilities, masks_mass)):
        raise ArithmeticError('Focal probabilities failed normalization or mask availability')
    native_terms = posterior_terms[native_label]
    return dict(
        defined=True,
        supported=bool(len(native_terms)),
        probabilities=probabilities,
        probability_sum=sum(probabilities),
        normalization_error=error,
        native_probability=probabilities[native_label],
        native_nll=extended(-log_probabilities[native_label]),
        native_mask_availability=masks_mass[native_label],
        native_compatible_samples=int(masks[:, native_label].sum()),
        posterior_ess=ParticleLaw._ess(native_terms),
        max_posterior_normalized_weight=_max_normalized(native_terms),
    )


def _class_combinations(pool, numeric, n, native_label):
    survived = pool.survived[:n]
    groups = pool.focal_groups[:n][survived]
    masks = np.zeros((len(groups), 3), dtype=bool)
    for i, group in enumerate(groups):
        masks[i, pool.data['focal_groups'][group]['allowed']] = True
    return {
        key: _class_readout(numeric[latent][0][survived], numeric[focal][1][groups],
                            masks, native_label)
        for key, latent, focal in (
            ('00', 'zero', 'zero'), ('0f', 'zero', 'candidate'),
            ('f0', 'candidate', 'zero'), ('ff', 'candidate', 'candidate'),
        )
    }


def _compare_points(first, second):
    observed = []
    for a, b in zip(first['observed_heads'], second['observed_heads'], strict=True):
        identity = {key: a[key] for key in ('head', 'y', 'manifest_ordinal')}
        if any(b[key] != value for key, value in identity.items()):
            raise ValueError('Comparison must retain the identical original head reference')
        observed.append(dict(identity, nll_difference=paired_difference(a['loss'], b['loss'])))
    result = dict(observed_heads=observed)
    for key in ('saved_native_speed', 'saved_native_class_speed'):
        if key in first:
            result[key] = paired_difference(first[key]['loss'], second[key]['loss'])
    return result


def _proposal_components(pool, components):
    if components is None:
        components = [row['proposal_component'] for row in pool.rows]
    values = np.asarray(components)
    if not np.array_equal(values, np.arange(len(pool.rows)) % 2):
        raise ValueError('The mixed pool must interleave component 0 and component f')
    recorded = np.asarray([row['proposal_log_ratio'] for row in pool.rows])
    expected = np.where(values == 1, pool.anchor_log_ratio, 0.)
    error = float(np.max(np.abs(recorded - expected)))
    if not np.isfinite(error) or error > TOLERANCE:
        raise ArithmeticError('The recorded component density differs from the prefix factor product')
    return values, error


def evaluate_pool(pool, candidate, observations, *, sizes, native_lc_label=None,
                  native_speed_mps=None, components=None):
    """Read the same original observations from each requested pool prefix."""
    candidate = _vectors(candidate)
    observations = list(observations)
    labels = {int(obs['y']) for obs in observations if obs['head'] == 'lane_change'}
    if native_lc_label is not None:
        labels.add(int(native_lc_label))
    if len(labels) > 1 or labels - {0, 1, 2}:
        raise ValueError('The original LC receipts must agree on one valid class')
    native_label = next(iter(labels), None)
    if native_speed_mps is not None and native_label is None:
        raise ValueError('The saved native class-speed observation requires its LC label')
    mixed = isinstance(pool, MixtureParticleLaw)
    if mixed:
        components, path_error = _proposal_components(pool, components)
    readouts = {}
    for n in sizes:
        pool._size(n)
        if mixed and n % 2:
            raise ValueError('Every mixed readout must use a balanced prefix')
        points, numeric = {}, {}
        for name, theta in (('zero', np.zeros_like(candidate)), ('candidate', candidate)):
            points[name], numeric[name] = _point(
                pool, theta, observations, n, native_label, native_speed_mps,
            )
        combinations = (_class_combinations(pool, numeric, n, native_label)
                        if native_label is not None else None)
        counts = int(np.count_nonzero(components[:n])) if mixed else 0
        readouts[str(n)] = dict(
            particle_count=n,
            component_counts={'0': n-counts, 'f': counts},
            maximum_component_log_ratio_error=path_error if mixed else None,
            points=points,
            candidate_minus_zero=_compare_points(points['candidate'], points['zero']),
            lane_change_combinations=combinations,
            lane_change_decomposition=nll_decomposition(combinations) if combinations else None,
        )
    return readouts


def compare_readouts(mixture, old):
    """Pair the new and saved sizes without dropping any support failures."""
    comparisons = {}
    for new_size, new in mixture.items():
        for old_size, saved in old.items():
            pairs = {
                point: _compare_points(new['points'][point], saved['points'][point])
                for point in ('zero', 'candidate')
            }
            classes = None
            if new['lane_change_combinations'] is not None:
                classes = {
                    name: paired_difference(row['native_nll'],
                                             saved['lane_change_combinations'][name]['native_nll'])
                    for name, row in new['lane_change_combinations'].items()
                }
            comparisons[f'{new_size}_vs_{old_size}'] = dict(
                points=pairs, lane_change_nll_differences=classes,
            )
    return comparisons
