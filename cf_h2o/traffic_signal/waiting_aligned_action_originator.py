"""Target-adapted action originator for waiting-aligned phase control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


ORIGINATOR_PROTOCOL = "waiting-aligned-action-originator-v1"


@dataclass(frozen=True)
class WaitingAlignedActionOriginatorConfig:
    enabled: bool = True
    risk_multiplier: float = 0.0
    minimum_context_trust: float = 0.75
    minimum_benefit_probability: float = 0.0
    benefit_probability_kind: str = "calibrated"
    rollout_value_horizon_sec: int = 450
    objective_name: str = "control_cost"

    def __post_init__(self) -> None:
        if float(self.risk_multiplier) < 0.0:
            raise ValueError("originator risk multiplier must be nonnegative")
        if not 0.0 <= float(self.minimum_context_trust) <= 1.0:
            raise ValueError("originator context trust must be in [0, 1]")
        if not 0.0 <= float(self.minimum_benefit_probability) <= 1.0:
            raise ValueError("originator benefit probability must be in [0, 1]")
        if self.benefit_probability_kind not in {"raw", "calibrated"}:
            raise ValueError("unknown originator benefit probability kind")
        if int(self.rollout_value_horizon_sec) <= 0:
            raise ValueError("originator rollout horizon must be positive")
        if not str(self.objective_name):
            raise ValueError("originator objective name must be nonempty")


@dataclass(frozen=True)
class WaitingAlignedActionDecision:
    reference_index: int
    proposed_index: int
    selected_index: int
    learned_differs: bool
    eligible: bool
    priority: float
    rejection: str | None
    predicted_delta: float
    uncertainty: float
    context_trust: float
    predicted_upper: float
    benefit_probability: float
    candidate_state: str
    reference_state: str
    latent_level: int
    support_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WaitingAlignedActionOriginator:
    """Originate a phase override from a fitted target action-value latent.

    The originator only ranks the feasible candidates supplied by the runtime.
    Coordination, cooldowns, and legal phase transitions remain the executor's
    responsibility.
    """

    def __init__(
        self,
        *,
        model: Any,
        config: WaitingAlignedActionOriginatorConfig,
    ) -> None:
        self.model = model
        self.config = config

    def select(
        self,
        dataset: MechanismDataset,
        *,
        reference_index: int,
    ) -> WaitingAlignedActionDecision:
        reference = int(reference_index)
        if not 0 <= reference < dataset.size:
            raise IndexError("originator reference index is outside the candidate set")
        groups = np.asarray(
            dataset.metadata.get("action_group_ids", ()), dtype=str
        )
        candidate_states = np.asarray(
            dataset.metadata.get("candidate_states", ()), dtype=str
        )
        is_reference = np.asarray(
            dataset.metadata.get("is_reference", ()), dtype=bool
        )
        if not all(
            values.shape == (dataset.size,)
            for values in (groups, candidate_states, is_reference)
        ):
            raise ValueError("originator requires row-aligned action metadata")
        if np.unique(groups).size != 1 or np.flatnonzero(is_reference).tolist() != [
            reference
        ]:
            raise ValueError("originator requires one action group and one reference")

        prediction = self.model.predict(dataset)
        objective = self._objective(prediction)
        mean = self._array(objective, "mean", dataset.size)
        uncertainty = self._array(objective, "uncertainty", dataset.size)
        trust = self._array(objective, "context_trust", dataset.size)
        level = self._array(
            objective, "latent_level", dataset.size, default=np.zeros(dataset.size)
        )
        support = self._array(
            objective,
            "latent_support_count",
            dataset.size,
            default=np.zeros(dataset.size),
        )
        probability_key = (
            f"benefit_probability_{self.config.benefit_probability_kind}"
        )
        benefit_probability = self._array(
            objective,
            probability_key,
            dataset.size,
            default=(
                np.ones(dataset.size, dtype=float)
                if float(self.config.minimum_benefit_probability) == 0.0
                else None
            ),
        )
        if not all(
            np.isfinite(values).all()
            for values in (
                mean,
                uncertainty,
                trust,
                level,
                support,
                benefit_probability,
            )
        ):
            raise ValueError("originator model emitted non-finite predictions")

        proposed = self._argmin_with_reference_tie_break(
            mean,
            reference=reference,
            candidate_states=candidate_states,
        )
        predicted_delta = float(mean[proposed] - mean[reference])
        selected_uncertainty = float(uncertainty[proposed])
        selected_trust = float(trust[proposed])
        selected_benefit_probability = float(benefit_probability[proposed])
        predicted_upper = predicted_delta + float(
            self.config.risk_multiplier
        ) * selected_uncertainty
        differs = proposed != reference
        rejection = None
        eligible = False
        if differs and not bool(self.config.enabled):
            rejection = "disabled"
        elif differs and selected_trust < float(self.config.minimum_context_trust):
            rejection = "local_support"
        elif differs and selected_benefit_probability < float(
            self.config.minimum_benefit_probability
        ):
            rejection = "benefit_probability"
        elif differs and predicted_upper >= 0.0:
            rejection = "confidence"
        elif differs:
            eligible = True
        selected = proposed if eligible else reference
        return WaitingAlignedActionDecision(
            reference_index=reference,
            proposed_index=proposed,
            selected_index=selected,
            learned_differs=differs,
            eligible=eligible,
            priority=max(-predicted_upper, 0.0) if eligible else 0.0,
            rejection=rejection,
            predicted_delta=predicted_delta,
            uncertainty=selected_uncertainty,
            context_trust=selected_trust,
            predicted_upper=predicted_upper,
            benefit_probability=selected_benefit_probability,
            candidate_state=str(candidate_states[proposed]),
            reference_state=str(candidate_states[reference]),
            latent_level=int(round(float(level[proposed]))),
            support_count=int(round(float(support[proposed]))),
        )

    def diagnostics(self) -> dict[str, Any]:
        model_diagnostics = getattr(self.model, "diagnostics", None)
        return {
            "protocol": ORIGINATOR_PROTOCOL,
            "config": asdict(self.config),
            "model": model_diagnostics() if callable(model_diagnostics) else None,
            "uses_target_labels_at_fit": True,
            "uses_future_outcomes_at_prediction": False,
        }

    def _objective(self, prediction: Mapping[str, Any]) -> Mapping[str, Any]:
        name = str(self.config.objective_name)
        if name not in prediction or not isinstance(prediction[name], Mapping):
            raise KeyError(f"originator prediction lacks objective {name!r}")
        return prediction[name]

    @staticmethod
    def _array(
        objective: Mapping[str, Any],
        key: str,
        size: int,
        *,
        default: np.ndarray | None = None,
    ) -> np.ndarray:
        if key not in objective:
            if default is None:
                raise KeyError(f"originator prediction lacks {key!r}")
            values = np.asarray(default, dtype=float)
        else:
            values = np.asarray(objective[key], dtype=float)
        if values.shape != (size,):
            raise ValueError(f"originator prediction {key!r} is not row-aligned")
        return values

    @staticmethod
    def _argmin_with_reference_tie_break(
        values: np.ndarray,
        *,
        reference: int,
        candidate_states: np.ndarray,
    ) -> int:
        rows = np.arange(values.size, dtype=int)
        minimum = float(np.min(values))
        tied = rows[np.isclose(values, minimum, rtol=0.0, atol=1e-12)]
        if reference in {int(row) for row in tied}:
            return reference
        return min(
            (int(row) for row in tied),
            key=lambda row: (str(candidate_states[row]), row),
        )
