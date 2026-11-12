"""Likelihood support is not a changed risk denominator or an exp underflow."""
import json

import numpy as np
import pytest

from experiments.tsc_packet_shared_population_20261110.stats import log_probability, summarize_head


def test_gaussian_mixture_stays_finite_far_above_physical_mean():
    # Computing component densities with exp would underflow every component.
    logp = log_probability('motion', 1000., [0., .25, .75],
        [dict(mean_mps=1000.), dict(mean_mps=0.), dict(mean_mps=1.)])
    assert np.isfinite(logp)
    assert logp == pytest.approx(-.5*999**2-.5*np.log(2*np.pi)+np.log(.75))
    result = summarize_head('motion', [1000.], [logp], [logp-1.], [True], [True],
        baseline_max_mean=[1.], candidate_max_mean=[0.],
        baseline_max_physical=[1001.], candidate_max_physical=[1001.])
    assert result['candidate']['unsupported'] == 0
    assert result['paired']['worsened'] == 1
    assert result['motion_envelope']['candidate']['observed_above_max_accepted_mean'] == 1
    assert result['motion_envelope']['candidate']['observed_above_max_base_physical_proposal'] == 0


def test_binary_lost_positive_support_cannot_be_hidden_by_retention_or_y0_permissions():
    # Candidate opens a y0 permission while removing the only permission for y1.
    priors = [.25, .75]
    no_risk = [dict(yield_risk=False), dict(yield_risk=False)]
    partial_risk = [dict(yield_risk=False), dict(yield_risk=True)]
    before = [log_probability('yield_entry', 0, priors, no_risk),
              log_probability('yield_entry', 1, priors, partial_risk)]
    after = [log_probability('yield_entry', 0, priors, partial_risk),
             log_probability('yield_entry', 1, priors, no_risk)]
    result = summarize_head('yield_entry', [0, 1], before, after, [False, True], [True, False])
    assert result['candidate']['records'] == 2
    assert result['candidate']['full_mean_nll'] is None
    assert result['candidate']['unsupported'] == 1
    assert result['candidate']['excluded_unsupported'] == 1
    assert result['candidate']['finite_nll_count'] == 1
    assert result['candidate']['retained_mean_nll'] == pytest.approx(-np.log(.625))
    assert result['support_transition']['lost'] == 1
    assert result['by_label']['0']['support_transition']['gained'] == 0
    assert result['by_label']['0']['permission_transition']['gained'] == 1
    assert result['by_label']['1']['support_transition']['lost'] == 1
    assert result['paired']['finite_in_both'] == 1
    assert result['paired']['worsened'] == 1
    json.dumps(result, allow_nan=False)


def test_zero_prior_risk_does_not_supply_binary_support():
    assert log_probability('start', 1, [0., 1.],
        [dict(start_risk=True), dict(start_risk=False)]) == -np.inf
    assert log_probability('start', 0, [0., 1.],
        [dict(start_risk=True), dict(start_risk=False)]) == 0.
