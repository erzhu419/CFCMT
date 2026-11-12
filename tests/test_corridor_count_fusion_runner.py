"""The single repair must preserve controls and report its sparse-cell mechanism."""
from copy import deepcopy

import numpy as np
import pytest

from experiments.tsc_corridor_count_fusion_20261109 import run_development as r
from experiments.tsc_corridor_count_fusion_20261109.evidence_model import EvidencePredictor
from experiments.tsc_corridor_residual_transfer_20261109.residual_model import HistogramHead


def record():
    pairs = {f'{a}_vs_{b}': 0 for a, b in r.PAIRS}
    scope = dict(counts={'scored_edge_epochs': 123},
                 mae={arm: .2 for arm in r.ARMS}, action_differences=pairs,
                 action_differences_unlocked=deepcopy(pairs))
    return dict(city='jinan', seed=1, scopes={name: deepcopy(scope) for name in r.SCOPES})


def test_control_gate_allows_source_changes_and_rejects_control_or_unlocked_drift():
    old, new = record(), record()
    for scope in r.SCOPES:
        new['scopes'][scope]['mae'].update(S=.1, P=.3)
        new['scopes'][scope]['action_differences']['S_vs_T'] = 9
    assert r.verify_controls(new, old)['all_scopes_reproduced']
    drift = deepcopy(new)
    drift['scopes']['all_observable_risk']['mae']['T'] = .21
    with pytest.raises(ValueError, match='MAE'):
        r.verify_controls(drift, old)
    drift = deepcopy(new)
    drift['scopes']['original_common_scored']['action_differences_unlocked']['T_vs_GP'] = 1
    with pytest.raises(ValueError, match='request differences'):
        r.verify_controls(drift, old)


def test_sparse_context_report_restores_positive_evidence_direction():
    target, source = {}, {}
    for queued, nt, positive, ns, source_positive in ((False, 58, 44, 14, 11), (True, 105, 80, 3, 3)):
        context = True, 11, '<=0.5', queued
        ct, ch = np.zeros(11), np.zeros(11)
        ct[0], ct[10] = nt-positive, positive
        ch[0], ch[10] = ns-source_positive, source_positive
        target[context], source[context] = ct, ch
    target, source = HistogramHead(target), HistogramHead(source)
    predictor = EvidencePredictor(target, source, source)
    cases = r.sparse_contexts(target, source, predictor, 2)
    worst = next(case for case in cases if case['queued_at_entry'])
    assert worst['source_n'] == 3 and worst['target_n'] == 105
    assert worst['source_empirical_arrival10'] == 1
    assert worst['old_source_cdf10'] == pytest.approx(3/23)
    assert worst['old_equal_mixture_cdf10'] == pytest.approx(.5*(80/125+3/23))
    assert worst['new_count_fusion_cdf10'] == pytest.approx(83/128)
    assert worst['new_count_fusion_cdf10'] > worst['target_cdf10']
    assert worst['effective_source_weight'] == pytest.approx(3/128)
    assert worst['new_source_decoded10'] and not worst['old_source_mixture_decoded10']
