from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import (
    CONTEXT_NAMES_V2,
    FEATURE_NAMES_V2,
    OUTPUT_NAMES_V2,
    TSC_MECHANISM_DEFINITIONS,
    GuardAuditV2,
    GuardConfigV2,
    _guarded_candidate,
    _candidate_aggregates,
    LocalTransitionState,
    candidate_features_v2,
    uncalibrated_mechanism_priors,
)
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import PhaseCandidate, TlsPhaseInfo, _phase_score
from cf_h2o.traffic_signal.safe_phase_controller import PhaseTiming, SafePhaseExecutor


class _TrafficLight:
    def setRedYellowGreenState(self, tls_id, state):
        pass


def _state_and_executor():
    links = ((('in0', 'out0', ''),), (('in1', 'out1', ''),))
    candidates = (
        PhaseCandidate(0, "Gr", 10.0, 1),
        PhaseCandidate(1, "rG", 10.0, 1),
    )
    info = TlsPhaseInfo("J0", links, candidates, ("in0", "in1"))
    state = LocalTransitionState(
        tls_id="J0",
        info=info,
        q_by_lane={"in0": 10.0, "in1": 4.0},
        veh_by_lane={"in0": 12.0, "in1": 6.0},
        speed_by_lane={"in0": 4.0, "in1": 8.0},
        occ_by_lane={"in0": 0.45, "in1": 0.25},
        down_q_by_lane={"in0": 2.0, "in1": 1.0},
        down_occ_by_lane={"in0": 0.20, "in1": 0.10},
        context=np.ones(len(CONTEXT_NAMES_V2), dtype=float),
        sim_time=3600.0,
    )
    api = SimpleNamespace(trafficlight=_TrafficLight())
    executor = SafePhaseExecutor(
        sumo_api=api,
        tls_id="J0",
        timings=(PhaseTiming("Gr", 5.0, 3.0, 1.0), PhaseTiming("rG", 5.0, 3.0, 1.0)),
        initial_state="Gr",
        initial_green_elapsed_sec=5.0,
    )
    return state, executor


def test_v2_candidate_features_include_execution_semantics_without_ids():
    state, executor = _state_and_executor()
    stay = candidate_features_v2(state, state.info.candidates[0], executor, control_interval_sec=10)
    switch = candidate_features_v2(state, state.info.candidates[1], executor, control_interval_sec=10)
    index = {name: idx for idx, name in enumerate(FEATURE_NAMES_V2)}

    assert stay.shape == (len(FEATURE_NAMES_V2),)
    assert stay[index["switch_indicator"]] == 0.0
    assert switch[index["switch_indicator"]] == 1.0
    assert switch[index["clearance_fraction"]] == 0.4
    assert not any(name.endswith("_id") for name in FEATURE_NAMES_V2)


def test_uncalibrated_priors_cover_every_mechanism_and_are_finite():
    state, executor = _state_and_executor()
    features = np.vstack(
        [candidate_features_v2(state, candidate, executor, control_interval_sec=10) for candidate in state.info.candidates]
    )
    context = np.repeat(state.context[None, :], features.shape[0], axis=0)
    priors = uncalibrated_mechanism_priors(features, context, control_interval_sec=10)

    assert tuple(priors) == OUTPUT_NAMES_V2
    assert all(values.shape == (2,) for values in priors.values())
    assert all(np.isfinite(values).all() for values in priors.values())
    assert all((values >= 0.0).all() for values in priors.values())
    assert np.all(priors["next_downstream_occupancy"] <= 1.0)


def test_production_pressure_features_and_spillback_use_fraction_units():
    state, _ = _state_and_executor()
    state.down_occ_by_lane["in0"] = 0.8
    candidate = state.info.candidates[0]
    aggregate = _candidate_aggregates(state, candidate)
    assert aggregate["service_pressure"] == pytest.approx((10.0 - 2.0) / 3.0)
    assert aggregate["movement_service_pressure"] == pytest.approx((10.0 - 2.0) / 3.0)
    api = SimpleNamespace(lane=SimpleNamespace(
        getLastStepHaltingNumber=lambda lane: 10.0,
        getLastStepVehicleNumber=lambda lane: 12.0,
        getLastStepOccupancy=lambda lane: 0.45 if lane == "in0" else 0.8,
    ))
    queue_proxy = 10.0 + 0.35 * 12.0 + 0.025 * 0.45
    assert _phase_score(api, state.info, candidate, spillback=False) == pytest.approx(queue_proxy)
    assert _phase_score(api, state.info, candidate, spillback=True) == pytest.approx(queue_proxy / 3.0)


def test_every_v2_mechanism_has_multiple_auditable_parent_variants():
    feature_names = set(FEATURE_NAMES_V2)
    assert len(TSC_MECHANISM_DEFINITIONS) == len(OUTPUT_NAMES_V2)
    for definition in TSC_MECHANISM_DEFINITIONS:
        assert set(definition.parent_variants) == {"minimal", "physical"}
        for parents in definition.parent_variants.values():
            assert set(parents) <= feature_names


class _PreferenceModel:
    def __init__(self, *, trust=1.0, uncertainty=0.0):
        self.trust = float(trust)
        self.uncertainty = float(uncertainty)

    def predict(self, dataset):
        green_q_idx = FEATURE_NAMES_V2.index("green_q")
        # Prefer the phase serving the smaller current queue so the learned
        # proposal deliberately disagrees with pressure control in this test.
        learned_cost = dataset.features[:, green_q_idx]
        outputs = {}
        for definition in TSC_MECHANISM_DEFINITIONS:
            mean = learned_cost.copy() if definition.name == "control_cost" else np.zeros(dataset.size)
            outputs[definition.name] = {
                "mean": mean,
                "uncertainty": np.full(dataset.size, self.uncertainty),
                "context_trust": np.full(dataset.size, self.trust),
            }
        return outputs


def test_guard_reports_and_controls_learned_overrides():
    state, executor = _state_and_executor()
    audit = GuardAuditV2()
    candidate = _guarded_candidate(
        _PreferenceModel(),
        state,
        executor,
        control_interval_sec=10,
        guard=GuardConfigV2(risk_multiplier=0.0, margin=0.0),
        audit=audit,
    )

    assert candidate.state == "rG"
    assert audit.proposed_overrides == 1
    assert audit.accepted_overrides == 1

    rejected_audit = GuardAuditV2()
    candidate = _guarded_candidate(
        _PreferenceModel(trust=0.0),
        state,
        executor,
        control_interval_sec=10,
        guard=GuardConfigV2(risk_multiplier=0.0, margin=0.0, min_context_trust=0.1),
        audit=rejected_audit,
    )
    assert candidate.state == "Gr"
    assert rejected_audit.rejected_context == 1
