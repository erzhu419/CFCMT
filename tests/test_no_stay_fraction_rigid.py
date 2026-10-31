from types import SimpleNamespace

import pytest

from test_fraction_target_rigid import setup_case
from test_thresholded_fraction_rigid import contract
from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
from cf_h2o.traffic_signal.no_stay_fraction_rigid import (
    ORIGINATOR_PROTOCOL, NoStayFractionRigidOriginator,
)
from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor


def prepare_case(tmp_path, scores, current_index, threshold=0.):
    payload, context, contrast, preparation = setup_case(tmp_path, scores)
    states = contrast.metadata["candidate_states"]
    executor = SafePhaseExecutor(
        sumo_api=SimpleNamespace(trafficlight=SimpleNamespace(
            setRedYellowGreenState=lambda *args: None)),
        tls_id="A", timings=[PhaseTiming(state, 10., 3., 2.) for state in states],
        initial_state=states[current_index], initial_green_elapsed_sec=20.,
    )
    preparation["executors"] = {"A": executor}
    originator = NoStayFractionRigidOriginator(
        runtime_payload=payload, network_context=context,
        frozen_threshold=contract(threshold=threshold),
    )
    originator.prepare_interval(**preparation)
    return originator, executor, contrast, preparation


def coordinate(proposal, executor):
    audit = v3.ContrastGuardAudit()
    action = v3._coordinate_contrast_proposals(
        {"A": proposal}, cooldowns={}, cooldown_intervals=0,
        max_simultaneous_overrides=None, audit=audit,
        current_green_states={"A": executor.current_green_state},
    )["A"]
    return action, audit


@pytest.mark.parametrize("scores,current_index,expected,veto,accepted", [
    ((0., -1., 2.), 1, 0, 1, 0),  # A learned hold must not cancel the PP switch.
    ((0., -1., 2.), 0, 1, 0, 1),  # PP holds; the learned switch remains available.
    ((0., -1., 2.), 2, 1, 0, 1),  # Different learned and PP switches remain available.
    ((0., 0., 2.), 1, 0, 0, 0),   # Original tie breaking selects PP.
])
def test_no_stay_decisions_execute_through_v3_coordinator(
        tmp_path, scores, current_index, expected, veto, accepted):
    originator, executor, contrast, _ = prepare_case(tmp_path, scores, current_index)
    decision = originator.select(contrast, reference_index=0)
    states = contrast.metadata["candidate_states"]
    proposal = v3.ContrastProposal(
        prior_candidate=SimpleNamespace(state=states[0]),
        selected_candidate=SimpleNamespace(state=states[decision.selected_index]),
        learned_differs=decision.learned_differs, eligible=decision.eligible,
        priority=decision.priority, rejection=decision.rejection,
        originator_diagnostics=decision.to_dict(),
    )
    action, audit = coordinate(proposal, executor)
    assert action.state == states[expected]
    assert audit.accepted_overrides == accepted
    assert audit.accepted_stay_overrides == 0
    assert audit.accepted_switch_overrides == accepted
    executor.request(action.state)
    assert executor.target_green_state == states[expected]
    diagnostics = originator.diagnostics()
    assert diagnostics["protocol"] == ORIGINATOR_PROTOCOL
    assert diagnostics["model_stay_veto_count"] == veto
    assert diagnostics["phase_pressure_override_count"] == accepted
    assert diagnostics["threshold_rejected_proposal_count"] == 0
    assert diagnostics["ungated_rigid_proposal_count"] == accepted + veto
    if veto:
        assert decision.proposed_index == decision.rigid_selected_index == 1
        assert decision.rigid_predicted_score == -1.
        assert decision.predicted_score == decision.reference_score == 0.
        assert decision.priority == 0.
        assert not decision.learned_differs and not decision.eligible
        assert decision.rejection == "model_stay_veto"


def test_threshold_rejection_is_not_counted_as_stay_veto(tmp_path):
    originator, _, contrast, _ = prepare_case(tmp_path, (0., -.4, 2.), 1, threshold=.5)
    decision = originator.select(contrast, reference_index=0)
    assert decision.rejection == "minimum_priority"
    diagnostics = originator.diagnostics()
    assert diagnostics["model_stay_veto_count"] == diagnostics["phase_pressure_override_count"] == 0
    assert diagnostics["threshold_rejected_proposal_count"] == diagnostics["ungated_rigid_proposal_count"] == 1


@pytest.mark.parametrize("elapsed,mode", [(0., "yellow"), (3., "all_red")])
def test_v3_clearance_keeps_single_target_without_model_selection(tmp_path, monkeypatch, elapsed, mode):
    originator, executor, contrast, preparation = prepare_case(tmp_path, (0., -1., 2.), 0)
    assert executor.request("rrG")
    executor.advance(elapsed)
    assert executor.mode == mode
    originator.prepare_interval(**preparation)
    # V3's existing candidate builder uses this exact feasible state set.
    candidates = tuple(SimpleNamespace(state=state) for state in executor.feasible_states_now())
    assert len(candidates) == 1
    monkeypatch.setattr(v3, "_absolute_target_candidates", lambda *args, **kwargs: (candidates, contrast))
    proposal = v3._contrast_proposal(
        model=object(), family="cfcmt_mechanism",
        models=SimpleNamespace(action_originator=originator, prediction_horizon_sec=450, prior_spec=object()),
        state=object(), executor=executor, control_interval_sec=10, guarded=True, regularized=True,
    )
    action, audit = coordinate(proposal, executor)
    before = executor.snapshot()
    assert not executor.request(action.state)
    assert action.state == before["target_green_state"] == executor.target_green_state
    assert executor.mode == mode
    assert audit.accepted_overrides == 0
    assert originator.diagnostics()["decision_count"] == originator.diagnostics()["model_stay_veto_count"] == 0
