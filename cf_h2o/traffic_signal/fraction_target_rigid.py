"""Target-only deployment of the original rigid blend under fractional occupancy."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np

from cf_h2o.eval import traffic_signal_state_conditioned_source_closed_loop as closed_loop
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import FrozenAnchoredBlendModel
from cf_h2o.traffic_signal.occupancy_equations import OCCUPANCY_EQUATION_PROTOCOL


MODEL_PROTOCOL = "tsc-v154j-fraction-target-rigid-model-v1"
ORIGINATOR_PROTOCOL = "tsc-v154j-fraction-target-rigid-originator-v1"


class FractionTargetRigidOriginator(closed_loop.StateConditionedSourceUtilityOriginator):
    """Keep the existing context, rigid scores and tie-break without source fitting."""

    def __init__(self, *, runtime_payload: Mapping[str, Any], network_context: Any):
        super().__init__(runtime_payload=runtime_payload, network_context=network_context,
                         arm="rigid_target_only", expected_runtime_protocol=MODEL_PROTOCOL,
                         originator_protocol=ORIGINATOR_PROTOCOL)

    @classmethod
    def load(cls, path: Path, *, expected_city: str, network_context: Any):
        payload = pickle.loads(Path(path).read_bytes())
        if not isinstance(payload, Mapping) or payload.get("city") != expected_city:
            raise ValueError("Fraction target-only model belongs to another city")
        return cls(runtime_payload=payload, network_context=network_context)

    def _validate_payload(self):
        if self.payload.get("protocol") != MODEL_PROTOCOL:
            raise ValueError("Fraction target-only model protocol changed")
        if self.payload.get("occupancy_equation_protocol") != OCCUPANCY_EQUATION_PROTOCOL:
            raise ValueError("Fraction target-only model occupancy equation contract mismatch")
        if not self.payload.get("city") or self.payload.get("target_budget") != 100:
            raise ValueError("Fraction target-only model requires its city and the frozen B100 budget")
        if not isinstance(self.payload.get("rigid_model"), FrozenAnchoredBlendModel):
            raise ValueError("Fraction target-only model must use the original frozen rigid blend")

    def select(self, contrast, *, reference_index: int):
        reference = int(reference_index)
        augmented = self._augment_contrast(contrast, reference_index=reference)
        score = np.asarray(closed_loop._rigid_score(self.payload["rigid_model"], augmented), dtype=float)
        proposed = self._argmin_with_reference_tie_break(
            score, reference=reference,
            candidate_states=np.asarray(contrast.metadata["candidate_states"], dtype=str))
        differs = proposed != reference
        self._decision_count += 1
        self._override_count += int(differs)
        return closed_loop.StateConditionedSourceDecision(
            reference_index=reference, proposed_index=proposed, selected_index=proposed,
            rigid_selected_index=proposed, learned_differs=differs, eligible=differs,
            priority=max(float(score[reference] - score[proposed]), 0.0), rejection=None,
            arm=self.arm, source_city_admitted=False, utility_gate_selected=False,
            selected_candidate_key=None, source_changed_rigid_action=False,
            predicted_score=float(score[proposed]), reference_score=float(score[reference]),
            rigid_predicted_score=float(score[proposed]), candidate_score_constraint=None,
            candidate_changed_rigid_group_count=0, candidate_pressure_aligned_change_group_count=0,
            candidate_off_reference_rejected_group_count=0,
            source_changed_to_reference_action=False, source_changed_off_reference_action=False)

    def diagnostics(self):
        return {
            "protocol": self.originator_protocol, "occupancy_equation_protocol": OCCUPANCY_EQUATION_PROTOCOL,
            "arm": self.arm, "city": self.payload["city"], "target_budget": 100,
            "interval_count": self._interval_count, "decision_count": self._decision_count,
            "phase_pressure_override_count": self._override_count,
            "source_city_admitted": False, "utility_gate_selected_count": 0,
            "source_changed_rigid_action_count": 0, "source_changed_to_reference_action_count": 0,
            "source_changed_off_reference_action_count": 0, "candidate_score_constraint": None,
            "candidate_changed_rigid_group_count": 0, "candidate_pressure_aligned_change_group_count": 0,
            "candidate_off_reference_rejected_group_count": 0, "selected_candidate_counts": {},
            "mean_neighbor_observation_ratio": float(np.mean(self._neighbor_observation_ratios))
                if self._neighbor_observation_ratios else 0.0,
            "target_labels_used_at_fit": True, "target_outcomes_used_online": False,
            "source_identity_used_by_utility_gate": False,
        }
