from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardAudit,
    _contrast_proposal,
    _coordinate_contrast_proposals,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.waiting_aligned_action_originator import (
    WaitingAlignedActionOriginator,
    WaitingAlignedActionOriginatorConfig,
)


def _dataset() -> MechanismDataset:
    return MechanismDataset(
        feature_names=("action",),
        features=np.asarray([[0.0], [1.0], [2.0]], dtype=float),
        context_names=("demand",),
        context=np.zeros((3, 1), dtype=float),
        priors={"interval_cost": np.zeros(3, dtype=float)},
        targets={"interval_cost": np.zeros(3, dtype=float)},
        domains=np.asarray(["target"] * 3),
        metadata={
            "action_group_ids": ["online", "online", "online"],
            "is_reference": [True, False, False],
            "candidate_states": ["phase0", "phase1", "phase2"],
            "row_tls": ["tls0"] * 3,
            "row_times": [300.0] * 3,
        },
    )


class _PredictionModel:
    def __init__(self, mean, trust, uncertainty=None, benefit_probability=None):
        self.mean = np.asarray(mean, dtype=float)
        self.trust = np.asarray(trust, dtype=float)
        self.uncertainty = np.asarray(
            np.zeros_like(self.mean) if uncertainty is None else uncertainty,
            dtype=float,
        )
        self.benefit_probability = (
            None
            if benefit_probability is None
            else np.asarray(benefit_probability, dtype=float)
        )

    def predict(self, dataset):
        assert dataset.size == self.mean.size
        objective = {
                "mean": self.mean,
                "uncertainty": self.uncertainty,
                "context_trust": self.trust,
                "latent_level": np.asarray([1.0, 3.0, 3.0]),
                "latent_support_count": np.asarray([1.0, 80.0, 80.0]),
        }
        if self.benefit_probability is not None:
            objective["benefit_probability_calibrated"] = self.benefit_probability
        return {"control_cost": objective}


def _originator(mean, trust, *, risk=0.0):
    return WaitingAlignedActionOriginator(
        model=_PredictionModel(mean, trust, uncertainty=[0.0, 0.2, 0.2]),
        config=WaitingAlignedActionOriginatorConfig(
            risk_multiplier=risk,
            minimum_context_trust=0.75,
        ),
    )


def test_originator_selects_supported_negative_action() -> None:
    decision = _originator([0.0, -0.4, 0.2], [1.0, 0.8, 0.9]).select(
        _dataset(), reference_index=0
    )

    assert decision.proposed_index == 1
    assert decision.selected_index == 1
    assert decision.eligible is True
    assert decision.rejection is None
    assert decision.priority == 0.4
    assert decision.latent_level == 3
    assert decision.support_count == 80


def test_originator_falls_back_when_context_trust_is_below_frozen_gate() -> None:
    decision = _originator([0.0, -0.4, 0.2], [1.0, 0.74, 0.9]).select(
        _dataset(), reference_index=0
    )

    assert decision.proposed_index == 1
    assert decision.selected_index == 0
    assert decision.learned_differs is True
    assert decision.eligible is False
    assert decision.rejection == "local_support"


def test_originator_tie_returns_phase_pressure_reference() -> None:
    decision = _originator([0.0, 0.0, 0.2], [1.0, 1.0, 1.0]).select(
        _dataset(), reference_index=0
    )

    assert decision.proposed_index == 0
    assert decision.selected_index == 0
    assert decision.learned_differs is False
    assert decision.eligible is False


def test_originator_rejects_low_calibrated_benefit_probability() -> None:
    originator = WaitingAlignedActionOriginator(
        model=_PredictionModel(
            [0.0, -0.4, 0.2],
            [1.0, 0.8, 0.9],
            benefit_probability=[1.0, 0.59, 0.8],
        ),
        config=WaitingAlignedActionOriginatorConfig(
            minimum_context_trust=0.75,
            minimum_benefit_probability=0.60,
            benefit_probability_kind="calibrated",
        ),
    )

    decision = originator.select(_dataset(), reference_index=0)

    assert decision.selected_index == 0
    assert decision.eligible is False
    assert decision.rejection == "benefit_probability"
    assert decision.benefit_probability == 0.59


def test_runtime_proposal_stays_inside_existing_coordinator(monkeypatch) -> None:
    dataset = _dataset()
    candidates = tuple(
        SimpleNamespace(state=state)
        for state in dataset.metadata["candidate_states"]
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3._absolute_target_candidates",
        lambda *args, **kwargs: (candidates, dataset),
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3.make_target_action_contrasts",
        lambda *args, **kwargs: (dataset, 0),
    )
    proposal = _contrast_proposal(
        model=object(),
        family="cfcmt_mechanism",
        models=SimpleNamespace(
            action_originator=_originator(
                [0.0, -0.4, 0.2], [1.0, 0.8, 0.9]
            ),
            prediction_horizon_sec=450,
            prior_spec=object(),
        ),
        state=object(),
        executor=object(),
        control_interval_sec=10,
        guarded=True,
        regularized=True,
    )

    audit = ContrastGuardAudit()
    coordinated = _coordinate_contrast_proposals(
        {"tls0": proposal},
        cooldowns={"tls0": 0},
        cooldown_intervals=2,
        max_simultaneous_overrides=1,
        audit=audit,
    )

    assert proposal.selection_layer == "target_spatiotemporal_latent"
    assert proposal.originator_diagnostics["candidate_state"] == "phase1"
    assert coordinated["tls0"].state == "phase1"
    assert audit.target_originator_decisions == 1
    assert audit.selected_target_originators == 1
    assert audit.hierarchical_decisions == 0
