"""Four-head finite-SNIS objective on fixed cached particles, without physics."""
from collections import Counter

import numpy as np
from scipy.special import logsumexp

from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


HEADS = ('motion', 'start', 'yield_entry', 'lane_change')
TRAIN_SEEDS = (76116901, 76116902, 76116903)
DEVELOPMENT_SEED = 76116904
SHAPE = (5, len(FEATURE_NAMES))
DIMENSION = int(np.prod(SHAPE))
L2 = 1e-3


def coefficients(value):
    theta = np.asarray(value, dtype=float).reshape(SHAPE)
    if not np.isfinite(theta).all():
        raise ValueError('Nonfinite coefficients terminate the single fit attempt')
    return theta


def split_of(reference):
    seed = int(reference['seed'])
    if seed in TRAIN_SEEDS:
        result = 'train'
    elif seed == DEVELOPMENT_SEED:
        result = 'development'
    else:
        raise ValueError('Only the four frozen training trajectories are allowed')
    if reference.get('split', result) != result:
        raise ValueError('Coefficient-training/internal-development seed split changed')
    return result


def weight_maxima(pool, theta, head, observed):
    """Read the same finite law's survivor/posterior weight maxima."""
    logw, unused_scores, focal_prob, unused_focal_score = pool._evaluate(theta)
    survived = pool.survived
    survivor_max = (float(np.exp(np.max(logw[survived])-logsumexp(logw[survived])))
                    if np.any(survived) else None)
    values = np.array(pool._base_values(head, observed), dtype=float, copy=True)
    selected = survived if head in ('motion', 'start', 'lane_change') else np.ones(len(logw), dtype=bool)
    if head == 'lane_change':
        values[~survived] = 0.
        values[survived] *= focal_prob[pool.focal_groups[survived], int(observed)]
    positive = selected & (values > 0.)
    if not np.any(positive):
        raise ValueError('Zero posterior mass after supported marginal readout')
    terms = logw[positive]+np.log(values[positive])
    posterior_max = float(np.exp(np.max(terms)-logsumexp(terms)))
    return survivor_max, posterior_max


def diagnostic_extrema(rows):
    if not rows:
        raise ValueError('A requested split must retain every head reference')
    survivors = [row['max_survivor_normalized_weight'] for row in rows
                 if row['max_survivor_normalized_weight'] is not None]
    return dict(
        min_global_ess=min(row['global_ess'] for row in rows),
        min_survivor_ess=min(row['survivor_ess'] for row in rows),
        min_posterior_ess=min(row['posterior_ess'] for row in rows),
        max_global_normalized_weight=max(row['max_normalized_weight'] for row in rows),
        max_survivor_normalized_weight=max(survivors) if survivors else None,
        max_posterior_normalized_weight=max(row['max_posterior_normalized_weight'] for row in rows),
        no_survivor_references=sum(row['max_survivor_normalized_weight'] is None for row in rows),
        min_nll=min(row['nll'] for row in rows), max_nll=max(row['nll'] for row in rows),
        max_query_gradient_l2=max(row['gradient_l2'] for row in rows))


class CompositeObjective:
    """Preserve references/multiplicities; share only cached case computation."""

    def __init__(self, pools, references, record=None):
        self.pools = pools
        self.references = [dict(ref, split=split_of(ref)) for ref in references]
        self.trace = []
        self.record = record
        for ref in self.references:
            if ref['head'] not in HEADS or ref['case_index'] not in pools:
                raise ValueError('Every reference needs its declared head and current-case pool')

    def evaluate(self, value, split='train', details=False):
        theta = coefficients(value)
        references = [ref for ref in self.references if ref['split'] == split]
        counts = Counter(ref['head'] for ref in references)
        if any(counts[head] == 0 for head in HEADS):
            raise ValueError('All four original heads must remain in each requested split')
        losses = dict.fromkeys(HEADS, 0.)
        gradients = {head: np.zeros(SHAPE) for head in HEADS}
        zero_signal = Counter()
        rows = []
        for ref in references:
            head, observed = ref['head'], ref['y']
            pool = self.pools[ref['case_index']]
            result = pool.marginal(theta, head, observed)
            gradient = result.get('gradient')
            if (not result['defined'] or not result.get('supported', False)
                    or result.get('nll') is None or not np.isfinite(result['nll'])
                    or gradient is None or np.shape(gradient) != SHAPE
                    or not np.isfinite(gradient).all()):
                raise ValueError('Unsupported or nonfinite selected query: '+repr(ref))
            losses[head] += float(result['nll'])
            gradients[head] += gradient
            norm = float(np.linalg.norm(gradient))
            zero_signal[head] += int(norm == 0.)
            survivor_max, posterior_max = weight_maxima(pool, theta, head, observed)
            row = dict(ref, defined=True, supported=True, nll=float(result['nll']),
                       gradient_l2=norm, zero_signal=norm == 0.,
                       max_survivor_normalized_weight=survivor_max,
                       max_posterior_normalized_weight=posterior_max,
                       **{key:float(result[key]) for key in ('global_ess', 'survivor_ess',
                           'posterior_ess', 'max_normalized_weight')})
            numeric = [row[key] for key in ('global_ess', 'survivor_ess', 'posterior_ess',
                       'max_normalized_weight', 'max_posterior_normalized_weight')]
            if survivor_max is not None:
                numeric.append(survivor_max)
            if not np.isfinite(numeric).all():
                raise ValueError('Nonfinite particle-weight diagnostic: '+repr(ref))
            rows.append(row)
        means = {head:losses[head]/counts[head] for head in HEADS}
        unpenalized = float(sum(means.values()))
        gradient = sum((gradients[head]/counts[head] for head in HEADS), np.zeros(SHAPE))
        penalty = float(L2*np.sum(theta[:, 1:]**2))
        if split == 'train':
            gradient[:, 1:] += 2*L2*theta[:, 1:]
        total = unpenalized+penalty if split == 'train' else unpenalized
        if not np.isfinite(total) or not np.isfinite(gradient).all():
            raise ValueError('Nonfinite aggregate loss or gradient terminates the single attempt')
        result = dict(split=split, reference_count=len(references), head_counts=dict(counts),
                      head_means=means, unpenalized_sum=unpenalized,
                      regularization=penalty, regularization_in_objective=split == 'train',
                      objective=total, gradient=gradient,
                      gradient_l2=float(np.linalg.norm(gradient)),
                      zero_signal_references=sum(zero_signal.values()),
                      zero_signal_by_head={head:zero_signal[head] for head in HEADS},
                      all_selected_queries_finite_and_supported=True,
                      diagnostics=diagnostic_extrema(rows))
        if details:
            result['references'] = rows
        return result

    def fun(self, value):
        row = self.evaluate(value, split='train')
        receipt = dict(evaluation=len(self.trace)+1,
                       **{key:row[key] for key in ('head_means', 'unpenalized_sum',
                           'regularization', 'objective', 'gradient_l2',
                           'zero_signal_references', 'diagnostics')})
        self.trace.append(receipt)
        if self.record is not None:
            self.record(receipt)
        return row['objective'], row['gradient'].reshape(-1)
