"""Fixed-reference paired likelihood summaries; no estimation or filtering."""
from math import log, pi

import numpy as np
from scipy.special import logsumexp


HEADS = ('motion', 'start', 'yield_entry')
ENVELOPE_TOL_MPS = 1e-8


def log_probability(head, y, prior, components):
    """Stable original-prior mixture for Gaussian or zero-logit binary heads."""
    prior = np.asarray(prior, dtype=float)
    if head == 'motion':
        means = np.asarray([c['mean_mps'] for c in components], dtype=float)
        component = -.5*(float(y)-means)**2-.5*log(2*pi)
    elif head in ('start', 'yield_entry'):
        risk_key = 'start_risk' if head == 'start' else 'yield_risk'
        risk = np.asarray([c[risk_key] for c in components], dtype=bool)
        component = np.where(risk, -log(2.), 0. if y == 0 else -np.inf)
    else:
        raise ValueError('Only the three registered heads are evaluated')
    if component.shape != prior.shape:
        raise ValueError('Each frozen prior component needs its own response')
    with np.errstate(divide='ignore'):
        return float(logsumexp(np.log(prior)+component))


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, minimum=None, maximum=None, mean=None, p50=None, p90=None, p99=None)
    return dict(count=len(values), minimum=float(values.min()), maximum=float(values.max()),
        mean=float(values.mean()), p50=float(np.quantile(values, .5)),
        p90=float(np.quantile(values, .9)), p99=float(np.quantile(values, .99)))


def _arm(logp, retained):
    finite = np.isfinite(logp)
    n = len(logp)
    return dict(records=n, supported=int(finite.sum()), unsupported=int((~finite).sum()),
        full_mean_nll=float(-logp.mean()) if n and finite.all() else None,
        finite_nll_count=int(finite.sum()), finite_nll_sum=float(-logp[finite].sum()),
        retained=int(retained.sum()), excluded=int((~retained).sum()),
        retained_unsupported=int((retained & ~finite).sum()),
        excluded_unsupported=int((~retained & ~finite).sum()),
        retained_finite_nll_count=int((retained & finite).sum()),
        retained_finite_nll_sum=float(-logp[retained & finite].sum()),
        retained_mean_nll=float(-logp[retained].mean()) if retained.any() and finite[retained].all() else None,
        full_objective_status='FINITE' if n and finite.all() else 'ZERO_LABEL_LIKELIHOOD' if n else 'EMPTY')


def _transition(before, after, selected):
    return dict(records=int(selected.sum()), gained=int((selected & ~before & after).sum()),
        lost=int((selected & before & ~after).sum()), both_true=int((selected & before & after).sum()),
        both_false=int((selected & ~before & ~after).sum()))


def _paired(before, after):
    finite = np.isfinite(before) & np.isfinite(after)
    # Candidate minus baseline NLL; a negative delta is an improvement.
    delta = before[finite]-after[finite]
    return dict(finite_in_both=int(finite.sum()),
        delta_nll_sum=float(delta.sum()), delta_nll_mean=float(delta.mean()) if len(delta) else None,
        improved=int((delta < 0).sum()), worsened=int((delta > 0).sum()), unchanged=int((delta == 0).sum()),
        delta_nll_distribution=distribution(delta), baseline_nll_distribution=distribution(-before[finite]),
        candidate_nll_distribution=distribution(-after[finite]))


def summarize_head(head, y, baseline_logp, candidate_logp, baseline_retained, candidate_retained,
                   *, baseline_max_mean=None, candidate_max_mean=None,
                   baseline_max_physical=None, candidate_max_physical=None):
    """Summarize all original rows, including unsupported and excluded labels.

    Retained means some positive-prior component has the head's physical risk;
    it is not a label-support indicator. Optional motion maxima must already
    exclude zero-prior intents. Null means an infinite/empty full objective,
    paired with explicit unsupported counts and finite-only denominators.
    """
    if head not in HEADS:
        raise ValueError('Only the three registered heads are evaluated')
    y = np.asarray(y, dtype=float)
    before, after = np.asarray(baseline_logp, dtype=float), np.asarray(candidate_logp, dtype=float)
    old_risk, new_risk = np.asarray(baseline_retained, dtype=bool), np.asarray(candidate_retained, dtype=bool)
    if any(a.shape != y.shape for a in (before, after, old_risk, new_risk)) or y.ndim != 1:
        raise ValueError('Paired values must cover the same fixed reference rows')
    if any(np.isnan(a).any() or np.isposinf(a).any() for a in (before, after)):
        raise ValueError('NaN or positive-infinite likelihood is a calculation failure')
    old_support, new_support = np.isfinite(before), np.isfinite(after)
    all_rows = np.ones(len(y), dtype=bool)
    result = dict(head=head, records=len(y), baseline=_arm(before, old_risk), candidate=_arm(after, new_risk),
        support_transition=_transition(old_support, new_support, all_rows),
        retention_transition=_transition(old_risk, new_risk, all_rows),
        paired=_paired(before, after), delta_sign='candidate minus baseline NLL; negative is better')
    if head != 'motion':
        result['by_label'] = {}
        for label in (0, 1):
            selected = y == label
            result['by_label'][str(label)] = dict(
                baseline=_arm(before[selected], old_risk[selected]),
                candidate=_arm(after[selected], new_risk[selected]),
                support_transition=_transition(old_support, new_support, selected),
                permission_transition=_transition(old_risk, new_risk, selected),
                paired=_paired(before[selected], after[selected]))
    envelopes = (baseline_max_mean, candidate_max_mean, baseline_max_physical, candidate_max_physical)
    if head == 'motion' and any(a is not None for a in envelopes):
        if any(a is None or np.asarray(a).shape != y.shape for a in envelopes):
            raise ValueError('Both arms need accepted-mean and physical-proposal maxima on all rows')
        result['motion_envelope'] = dict(tolerance_mps=ENVELOPE_TOL_MPS)
        for name, mean, physical in (('baseline', baseline_max_mean, baseline_max_physical),
                                     ('candidate', candidate_max_mean, candidate_max_physical)):
            mean_deficit, physical_deficit = y-np.asarray(mean), y-np.asarray(physical)
            result['motion_envelope'][name] = dict(
                observed_above_max_accepted_mean=int((mean_deficit > ENVELOPE_TOL_MPS).sum()),
                observed_above_max_base_physical_proposal=int((physical_deficit > ENVELOPE_TOL_MPS).sum()),
                accepted_mean_signed_deficit=distribution(mean_deficit),
                base_physical_signed_deficit=distribution(physical_deficit))
    return result
