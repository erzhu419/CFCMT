"""Natural-frequency LC loss with the three longitudinal rows held fixed."""

from collections import defaultdict
from math import exp, fsum, isclose

import numpy as np

from experiments.tsc_packet_conditional_cache_20261110.cache import CachedLaw, _vectors
from experiments.tsc_packet_mixture_confirmation_20261111.readout import MixtureParticleLaw
from experiments.tsc_packet_particle_first_fit_20261110.model import weight_maxima
from experiments.tsc_target_packet_kernel_20261109.local_model import FEATURE_NAMES


SHAPE = (5, len(FEATURE_NAMES))
LC_SHAPE = (2, len(FEATURE_NAMES))
DIMENSION = int(np.prod(LC_SHAPE))
L2 = 1e-3


def _stratum(rows):
    weight = fsum(row['objective_weight'] for row in rows)
    loss = fsum(row['weighted_nll'] for row in rows)
    return dict(references=len(rows), objective_weight=weight,
                weighted_nll_contribution=loss,
                conditional_weighted_mean_nll=loss/weight if weight else None)


def _mixed_diagnostics(rows):
    if not rows:
        return dict(references=0)
    return dict(
        references=len(rows),
        minimum_global_ess=min(row['global_ess'] for row in rows),
        minimum_survivor_ess=min(row['survivor_ess'] for row in rows),
        minimum_posterior_ess=min(row['posterior_ess'] for row in rows),
        maximum_raw_weight=max(row['raw_weight'] for row in rows),
        maximum_normalized_weight=max(row['max_normalized_weight'] for row in rows),
        maximum_survivor_normalized_weight=max(row['survivor_max'] for row in rows),
        maximum_posterior_normalized_weight=max(row['posterior_max'] for row in rows),
    )


class LCObjective:
    """Evaluate one fixed-cache objective and its 76-variable Jacobian."""

    def __init__(self, laws, references, anchor, record=None):
        self.laws = laws
        self.references = [dict(row) for row in references]
        self.anchor = _vectors(anchor).copy()
        self.initial_vector = self.anchor[3:5].reshape(-1).copy()
        self.record = record
        self.trace = []
        ids = [row['case_index'] for row in self.references]
        weights = [row['objective_weight'] for row in self.references]
        if (not self.references or len(ids) != len(set(ids))
                or any(index not in laws for index in ids)
                or any(not np.isfinite(weight) or weight <= 0. for weight in weights)
                or not isclose(fsum(weights), 1., rel_tol=0., abs_tol=1e-12)):
            raise ValueError('The complete LC references and normalized objective weights are required')
        if any(not isinstance(laws[index], (CachedLaw, MixtureParticleLaw)) for index in ids):
            raise TypeError('LC fitting requires exact laws or corrected fixed-anchor mixtures')
        if any(row['y'] not in (0, 1, 2) for row in self.references):
            raise ValueError('Every registered LC label must be left0, stay1 or right2')

    def coefficients(self, value):
        value = np.asarray(value, dtype=float)
        if value.size != DIMENSION or not np.isfinite(value).all():
            raise ValueError('The LC attempt requires 76 finite coefficients')
        theta = self.anchor.copy()
        theta[3:5] = value.reshape(LC_SHAPE)
        return theta

    def evaluate(self, value, diagnostics=True):
        theta = self.coefficients(value)
        gradient = np.zeros(LC_SHAPE)
        rows, mixed = [], []
        for ref in self.references:
            law = self.laws[ref['case_index']]
            observed = ref['y']
            seed = int(ref.get('seed', ref.get('reference', {}).get('seed')))
            result = law.marginal(theta, 'lane_change', observed)
            full_gradient = result.get('gradient')
            if (not result['defined'] or not result.get('supported', False)
                    or result.get('nll') is None or not np.isfinite(result['nll'])
                    or full_gradient is None or np.shape(full_gradient) != SHAPE
                    or not np.isfinite(full_gradient).all()):
                raise ValueError(f'Unsupported or nonfinite LC case {ref["case_index"]}, seed {seed}, label {observed}')
            weight = ref['objective_weight']
            gradient += weight*full_gradient[3:5]
            rows.append(dict(seed=seed, y=observed, objective_weight=weight,
                             weighted_nll=weight*float(result['nll'])))
            if diagnostics and isinstance(law, MixtureParticleLaw):
                survivor_max, posterior_max = weight_maxima(law, theta, 'lane_change', observed)
                logw = law._evaluate(theta)[0]
                row = dict(raw_weight=exp(float(np.max(logw))), survivor_max=survivor_max,
                           posterior_max=posterior_max,
                           **{name:float(result[name]) for name in ('global_ess', 'survivor_ess',
                                                                  'posterior_ess', 'max_normalized_weight')})
                if any(value is None or not np.isfinite(value) for value in row.values()):
                    raise ValueError(f'Nonfinite mixture diagnostic for LC case {ref["case_index"]}')
                mixed.append(row)

        unpenalized = fsum(row['weighted_nll'] for row in rows)
        unpenalized_gradient_l2 = float(np.linalg.norm(gradient))
        regularization = float(L2*np.sum(theta[3:5, 1:]**2))
        gradient[:, 1:] += 2*L2*theta[3:5, 1:]
        objective = unpenalized+regularization
        if not np.isfinite(objective) or not np.isfinite(gradient).all():
            raise ValueError('Nonfinite objective or Jacobian terminates the LC attempt')
        result = dict(
            reference_count=len(rows), objective_weight_sum=fsum(row['objective_weight'] for row in rows),
            objective=objective, unpenalized=unpenalized, regularization=regularization,
            gradient=gradient.reshape(-1), gradient_l2=float(np.linalg.norm(gradient)),
            gradient_infinity=float(np.max(np.abs(gradient))),
            unpenalized_gradient_l2=unpenalized_gradient_l2,
            all_references_supported=True,
        )
        if diagnostics:
            by_seed = defaultdict(list)
            for row in rows:
                by_seed[row['seed']].append(row)
            result.update(
                by_class={str(label):_stratum([row for row in rows if row['y'] == label])
                          for label in (0, 1, 2)},
                by_seed={str(seed):_stratum(group) for seed, group in sorted(by_seed.items())},
                mixed_diagnostics=_mixed_diagnostics(mixed),
            )
        return result

    def fun(self, value):
        evaluation = len(self.trace)+1
        try:
            result = self.evaluate(value)
        except Exception as error:
            receipt = dict(evaluation=evaluation, status='FAILED_OBJECTIVE_EVALUATION',
                           error=f'{type(error).__name__}: {error}')
            self.trace.append(receipt)
            if self.record is not None:
                self.record(receipt)
            raise
        receipt = dict(evaluation=evaluation, status='FINITE_SUPPORTED_OBJECTIVE',
                       **{key:value for key, value in result.items() if key != 'gradient'})
        self.trace.append(receipt)
        if self.record is not None:
            self.record(receipt)
        return result['objective'], result['gradient']
