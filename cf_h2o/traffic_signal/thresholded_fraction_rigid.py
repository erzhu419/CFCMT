"""Apply the frozen V154L PP fallback threshold to the unchanged B100 model."""

from dataclasses import replace
import json
import math
from pathlib import Path
import pickle
from typing import Any, Mapping

from cf_h2o.traffic_signal.fraction_target_rigid import FractionTargetRigidOriginator


THRESHOLD_PROTOCOL = "tsc-v154l-cologne-rigid-oof-threshold-v1"
ORIGINATOR_PROTOCOL = "tsc-v154m-thresholded-fraction-rigid-originator-v1"
COMPARISON = "predicted_advantage > threshold + 1e-12"
TOLERANCE = 1e-12


class ThresholdedFractionRigidOriginator(FractionTargetRigidOriginator):
    def __init__(self, *, runtime_payload: Mapping[str, Any], network_context: Any,
                 frozen_threshold: Mapping[str, Any]):
        super().__init__(runtime_payload=runtime_payload, network_context=network_context)
        self.frozen_threshold = dict(frozen_threshold)
        contract = self.frozen_threshold
        if (contract.get("protocol") != THRESHOLD_PROTOCOL
                or contract.get("comparison") != COMPARISON):
            raise ValueError("Use the frozen V154L threshold protocol and comparison")
        self.threshold = contract.get("threshold")
        self.threshold_mode = contract.get("mode")
        if self.threshold_mode == "minimum_advantage":
            if (self.threshold is None or not math.isfinite(float(self.threshold))
                    or float(self.threshold) < 0):
                raise ValueError("Minimum advantage threshold must be finite and nonnegative")
            self.threshold = float(self.threshold)
        elif self.threshold_mode != "phase_pressure_only" or self.threshold is not None:
            raise ValueError("Frozen threshold mode and value disagree")
        self.originator_protocol = ORIGINATOR_PROTOCOL
        self._ungated_proposal_count = 0
        self._threshold_rejected_count = 0

    @classmethod
    def load(cls, path: Path, *, frozen_threshold_path: Path,
             expected_city: str, network_context: Any):
        frozen = json.loads(Path(frozen_threshold_path).read_text())
        if frozen.get("model_path") != str(Path(path)):
            raise ValueError("Frozen threshold belongs to a different B100 model path")
        payload = pickle.loads(Path(path).read_bytes())
        if not isinstance(payload, Mapping) or payload.get("city") != expected_city:
            raise ValueError("Fraction target-only model belongs to another city")
        return cls(runtime_payload=payload, network_context=network_context,
                   frozen_threshold=frozen)

    def select(self, contrast, *, reference_index: int):
        decision = super().select(contrast, reference_index=reference_index)
        differs = decision.proposed_index != decision.reference_index
        advantage = decision.reference_score - decision.predicted_score
        accepted = bool(differs and self.threshold is not None
                        and advantage > self.threshold + TOLERANCE)
        rejected = differs and not accepted
        self._ungated_proposal_count += int(differs)
        self._threshold_rejected_count += int(rejected)
        # The parent counts its unfiltered proposal; expose the accepted count.
        self._override_count -= int(rejected)
        return replace(
            decision,
            selected_index=decision.proposed_index if accepted else decision.reference_index,
            learned_differs=accepted, eligible=accepted,
            priority=advantage if accepted else 0.0,
            predicted_score=decision.predicted_score if accepted else decision.reference_score,
            rejection="minimum_priority" if rejected else None,
        )

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "frozen_threshold_protocol": THRESHOLD_PROTOCOL,
            "threshold": self.threshold, "threshold_mode": self.threshold_mode,
            "threshold_comparison": COMPARISON,
            "threshold_model_path": self.frozen_threshold.get("model_path"),
            "ungated_rigid_proposal_count": self._ungated_proposal_count,
            "threshold_rejected_proposal_count": self._threshold_rejected_count,
        }
