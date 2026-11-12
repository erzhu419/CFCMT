"""Separate strict nominal label-score headroom from tie-only request changes."""
import copy

import pytest

from experiments.tsc_corridor_service_alignment_20261109 import oracle_screen as o


def test_oracle_gain_counts_strict_improvement_and_tie_changes_separately():
    assert o.oracle_gain([2,2,1],0)==dict(strict_score_gain_sum=0.0,positive_epochs=0,tie_only_change=0)
    assert o.oracle_gain([2,2,1],1)==dict(strict_score_gain_sum=0.0,positive_epochs=0,tie_only_change=1)
    assert o.oracle_gain([2,2,1],2)==dict(strict_score_gain_sum=1.0,positive_epochs=1,tie_only_change=0)


def test_logged_actions_only_restore_source_and_placebo_after_controls_match():
    base=dict(T=0,S=0,P=0,history=1,analytic=2,GP=3)
    assert o.recover_actions(base,None)==base
    logged=dict(base,S=1,P=2)
    assert o.recover_actions(base,logged)==logged
    with pytest.raises(ValueError,match='controls'):
        o.recover_actions(base,dict(logged,T=1))


def test_reproduction_requires_all_pairs_and_original_label_score_differences():
    panel=o._panel()
    expected=dict(counts={'scored_decisions':0,'scored_unlocked_decisions':0},action_differences=panel['differences'].copy(),
        action_differences_unlocked=panel['unlocked'].copy(),label_score_difference_sums=dict(panel['source_delta']))
    o.check_reproduction(panel,expected,'case')
    changed=copy.deepcopy(panel);changed['unlocked']['S_vs_T']=1
    with pytest.raises(ValueError,match='request pairs'):
        o.check_reproduction(changed,expected,'case')
    changed=copy.deepcopy(panel);changed['source_delta']['S_vs_T']=1
    with pytest.raises(ValueError,match='score differences'):
        o.check_reproduction(changed,expected,'case')
