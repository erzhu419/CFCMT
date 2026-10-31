from dataclasses import replace
import importlib
from pathlib import Path

import pytest

from test_fraction_target_rigid import setup_case
from test_thresholded_fraction_rigid import contract
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardAudit, ContrastProposal, _coordinate_contrast_proposals,
)
from cf_h2o.traffic_signal.thresholded_fraction_rigid import ThresholdedFractionRigidOriginator


@pytest.fixture
def apply_branch_policy(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts/data"))
    return importlib.import_module("run_cologne_continuation_counterfactual").apply_branch_policy


def _decision_case(tmp_path):
    payload, context, contrast, preparation = setup_case(tmp_path, (0., -1., 2.))
    originator = ThresholdedFractionRigidOriginator(
        runtime_payload=payload, network_context=context, frozen_threshold=contract(threshold=0.))
    originator.prepare_interval(**preparation)
    return originator.select(contrast, reference_index=0), contrast.metadata["candidate_states"]


@pytest.mark.parametrize("arm,expected_indices", [
    ("model_once_then_pp", (1, 1, 0, 0)),
    ("pp_only", (1, 0, 0, 0)),
    ("model_repeated", (1, 1, 1, 1)),
])
def test_branch_boundaries_execute_through_v3_coordinator(
        tmp_path, apply_branch_policy, arm, expected_indices):
    original, states = _decision_case(tmp_path)
    for time_sec, expected_index in zip((26910., 26920., 26930., 27360.), expected_indices):
        decision = apply_branch_policy(arm, time_sec, original)
        proposal = ContrastProposal(
            prior_candidate=states[decision.reference_index],
            selected_candidate=states[decision.selected_index],
            learned_differs=decision.learned_differs,
            eligible=decision.eligible,
            priority=decision.priority,
            rejection=decision.rejection,
            originator_diagnostics=decision.to_dict())
        audit = ContrastGuardAudit()
        selected = _coordinate_contrast_proposals(
            {"A": proposal}, cooldowns={}, cooldown_intervals=0,
            max_simultaneous_overrides=None, audit=audit,
            current_green_states={"A": states[1]})
        assert selected["A"] == states[expected_index]
        assert audit.proposed_overrides == int(expected_index != original.reference_index)
        if expected_index == original.reference_index:
            assert decision == replace(
                original, selected_index=original.reference_index, learned_differs=False,
                eligible=False, priority=0., predicted_score=original.reference_score, rejection=None)
        else:
            assert decision == original


def test_pp_policy_clears_old_rejection_and_priority(tmp_path, apply_branch_policy):
    original, _ = _decision_case(tmp_path)
    rejected = replace(original, rejection="minimum_priority", eligible=False)
    decision = apply_branch_policy("pp_only", 26920., rejected)
    assert decision.rejection is None
    assert decision.priority == 0.
    assert not decision.eligible and not decision.learned_differs
    assert decision.selected_index == decision.reference_index
    assert decision.predicted_score == decision.reference_score


def test_forced_pp_preserves_original_model_prediction(tmp_path, apply_branch_policy):
    original, _ = _decision_case(tmp_path)
    before = original.to_dict()
    decision = apply_branch_policy("model_once_then_pp", 26930., original)
    assert original.to_dict() == before
    assert decision.proposed_index == decision.rigid_selected_index == 1
    assert decision.rigid_predicted_score == original.rigid_predicted_score == -1.
    assert decision.reference_score == original.reference_score == 0.
