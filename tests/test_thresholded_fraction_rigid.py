import json
import pickle

import pytest

from test_fraction_target_rigid import setup_case
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardAudit, ContrastProposal, _coordinate_contrast_proposals,
)
from cf_h2o.traffic_signal.thresholded_fraction_rigid import (
    COMPARISON, ORIGINATOR_PROTOCOL, THRESHOLD_PROTOCOL, ThresholdedFractionRigidOriginator,
)


THRESHOLD = 0.3580414874633295


def contract(path="model.pkl", threshold=THRESHOLD):
    return {"protocol": THRESHOLD_PROTOCOL, "model_path": str(path),
            "threshold": threshold, "comparison": COMPARISON,
            "mode": "minimum_advantage" if threshold is not None else "phase_pressure_only"}


@pytest.mark.parametrize("advantage,accepted", [
    (THRESHOLD, False), (THRESHOLD + 1e-12, False), (THRESHOLD + 2e-12, True),
])
def test_frozen_strict_threshold_controls_v3_execution(tmp_path, advantage, accepted):
    payload, context, contrast, preparation = setup_case(tmp_path, (0., -advantage, 2.))
    originator = ThresholdedFractionRigidOriginator(
        runtime_payload=payload, network_context=context, frozen_threshold=contract())
    originator.prepare_interval(**preparation)
    decision = originator.select(contrast, reference_index=0)
    assert decision.proposed_index == decision.rigid_selected_index == 1
    assert decision.selected_index == (1 if accepted else 0)
    assert decision.eligible == decision.learned_differs == accepted
    assert decision.priority == (advantage if accepted else 0.)
    assert decision.predicted_score == (-advantage if accepted else 0.)
    assert decision.rigid_predicted_score == -advantage
    assert decision.rejection == (None if accepted else "minimum_priority")
    states = contrast.metadata["candidate_states"]
    proposal = ContrastProposal(
        prior_candidate=states[0], selected_candidate=states[decision.selected_index],
        learned_differs=decision.learned_differs, eligible=decision.eligible,
        priority=decision.priority, rejection=decision.rejection,
        originator_diagnostics=decision.to_dict())
    audit = ContrastGuardAudit()
    selected = _coordinate_contrast_proposals(
        {"A": proposal}, cooldowns={}, cooldown_intervals=0,
        max_simultaneous_overrides=None, audit=audit, current_green_states={"A": states[0]})
    assert selected["A"] == states[1 if accepted else 0]
    assert audit.proposed_overrides == int(accepted)
    assert payload["rigid_model"].audit.prediction_calls == 1
    diagnostics = originator.diagnostics()
    assert diagnostics["protocol"] == ORIGINATOR_PROTOCOL
    assert diagnostics["decision_count"] == diagnostics["ungated_rigid_proposal_count"] == 1
    assert diagnostics["phase_pressure_override_count"] == int(accepted)
    assert diagnostics["threshold_rejected_proposal_count"] == int(not accepted)


@pytest.mark.parametrize("scores,threshold,proposed,selected", [
    ((0., 0., 2.), THRESHOLD, 0, 0),
    ((0., -1., -1.), THRESHOLD, 2, 2),
    ((0., -1., -1.), None, 2, 0),
    ((0., -1., -1.), 0., 2, 2),
    ((0., 0., 2.), 0., 0, 0),
])
def test_original_tie_break_and_explicit_pp_only(tmp_path, scores, threshold, proposed, selected):
    payload, context, contrast, preparation = setup_case(tmp_path, scores)
    originator = ThresholdedFractionRigidOriginator(
        runtime_payload=payload, network_context=context, frozen_threshold=contract(threshold=threshold))
    originator.prepare_interval(**preparation)
    decision = originator.select(contrast, reference_index=0)
    assert decision.proposed_index == proposed
    assert decision.selected_index == selected
    assert originator.diagnostics()["phase_pressure_override_count"] == int(selected != 0)
    assert originator.diagnostics()["ungated_rigid_proposal_count"] == int(proposed != 0)


def test_load_binds_existing_model_and_frozen_threshold(tmp_path):
    payload, context, contrast, preparation = setup_case(tmp_path)
    path, threshold_path = tmp_path / "model.pkl", tmp_path / "frozen_threshold.json"
    path.write_bytes(pickle.dumps(payload))
    threshold_path.write_text(json.dumps(contract(path)))
    originator = ThresholdedFractionRigidOriginator.load(
        path, frozen_threshold_path=threshold_path, expected_city="cologne", network_context=context)
    originator.prepare_interval(**preparation)
    assert originator.select(contrast, reference_index=0).selected_index == 1
    with pytest.raises(ValueError, match="another city"):
        ThresholdedFractionRigidOriginator.load(
            path, frozen_threshold_path=threshold_path, expected_city="atlanta", network_context=context)
    threshold_path.write_text(json.dumps(contract(tmp_path / "different.pkl")))
    with pytest.raises(ValueError, match="different B100 model path"):
        ThresholdedFractionRigidOriginator.load(
            path, frozen_threshold_path=threshold_path, expected_city="cologne", network_context=context)


@pytest.mark.parametrize("changes", [
    {"protocol": "different"}, {"comparison": "predicted_advantage >= threshold"},
    {"threshold": -0.1}, {"mode": "phase_pressure_only"},
])
def test_threshold_contract_is_not_silently_changed(tmp_path, changes):
    payload, context, _, _ = setup_case(tmp_path)
    with pytest.raises(ValueError):
        ThresholdedFractionRigidOriginator(
            runtime_payload=payload, network_context=context,
            frozen_threshold={**contract(), **changes})
