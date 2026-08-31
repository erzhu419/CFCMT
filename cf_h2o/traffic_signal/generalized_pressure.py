"""Generalized pressure priors shared by source calibration and target control.

The policy is deliberately small and auditable.  Its parameters have the same
meaning on every network and use only local queue, downstream occupancy, and
safe-executor switching information.  Source networks may select a parameter
setting, but target transition labels never enter that selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PressurePolicySpec:
    """Network-invariant generalized-pressure scoring parameters."""

    downstream_queue_weight: float = 0.0
    downstream_occupancy_weight: float = 0.0
    switch_penalty: float = 0.0
    mode: str = "linear"

    def __post_init__(self) -> None:
        if self.mode not in {"linear", "spillback"}:
            raise ValueError(f"unknown pressure mode: {self.mode!r}")
        for name in (
            "downstream_queue_weight",
            "downstream_occupancy_weight",
            "switch_penalty",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def key(self) -> str:
        if self == PHASE_PRESSURE_SPEC:
            return "phase_pressure"
        if self == MAX_PRESSURE_SPEC:
            return "max_pressure"
        if self == SPILLBACK_PRESSURE_SPEC:
            return "spillback_pressure"
        return "gp_d{}_o{}_s{}".format(
            _encode_float(self.downstream_queue_weight),
            _encode_float(self.downstream_occupancy_weight),
            _encode_float(self.switch_penalty),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, **asdict(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PressurePolicySpec":
        return cls(
            downstream_queue_weight=float(values.get("downstream_queue_weight", 0.0)),
            downstream_occupancy_weight=float(values.get("downstream_occupancy_weight", 0.0)),
            switch_penalty=float(values.get("switch_penalty", 0.0)),
            mode=str(values.get("mode", "linear")),
        )


PHASE_PRESSURE_SPEC = PressurePolicySpec()
MAX_PRESSURE_SPEC = PressurePolicySpec(downstream_queue_weight=1.0)
SPILLBACK_PRESSURE_SPEC = PressurePolicySpec(mode="spillback")

LEGACY_PRESSURE_SPECS = {
    spec.key: spec
    for spec in (PHASE_PRESSURE_SPEC, MAX_PRESSURE_SPEC, SPILLBACK_PRESSURE_SPEC)
}


def generalized_pressure_grid(
    *,
    downstream_queue_weights: Sequence[float] = (0.0, 0.25, 0.45, 0.65, 1.0),
    downstream_occupancy_weights: Sequence[float] = (0.0, 0.02, 0.04),
    switch_penalties: Sequence[float] = (0.0, 1.0),
    include_spillback: bool = True,
) -> tuple[PressurePolicySpec, ...]:
    """Return a deterministic, duplicate-free source-calibration grid."""

    values = [
        PressurePolicySpec(
            downstream_queue_weight=float(downstream),
            downstream_occupancy_weight=float(occupancy),
            switch_penalty=float(switch),
        )
        for downstream in downstream_queue_weights
        for occupancy in downstream_occupancy_weights
        for switch in switch_penalties
    ]
    if include_spillback:
        values.append(SPILLBACK_PRESSURE_SPEC)
    result: list[PressurePolicySpec] = []
    seen: set[str] = set()
    for spec in values:
        if spec.key not in seen:
            result.append(spec)
            seen.add(spec.key)
    return tuple(result)


def pressure_spec(value: str | PressurePolicySpec | Mapping[str, Any]) -> PressurePolicySpec:
    """Resolve a legacy name, serialized mapping, or already-built spec."""

    if isinstance(value, PressurePolicySpec):
        return value
    if isinstance(value, Mapping):
        return PressurePolicySpec.from_dict(value)
    if value in LEGACY_PRESSURE_SPECS:
        return LEGACY_PRESSURE_SPECS[value]
    raise ValueError(f"unknown pressure policy specification: {value!r}")


def pressure_scores_from_feature_matrix(
    features: np.ndarray,
    feature_names: Sequence[str],
    spec: str | PressurePolicySpec | Mapping[str, Any],
) -> np.ndarray:
    """Score candidate rows using only portable local feature semantics."""

    resolved = pressure_spec(spec)
    x = np.asarray(features, dtype=float)
    index = {name: idx for idx, name in enumerate(feature_names)}
    required = {"green_q"}
    if resolved.downstream_queue_weight > 0.0:
        required.add("green_down_q")
    if resolved.downstream_occupancy_weight > 0.0:
        required.add("green_down_occ")
    if resolved.switch_penalty > 0.0:
        required.add("switch_indicator")
    if resolved.mode == "spillback":
        required.add("service_pressure")
    missing = required - set(index)
    if missing:
        raise KeyError(f"missing generalized-pressure features: {sorted(missing)}")
    green_q = x[:, index["green_q"]]
    downstream_q = x[:, index["green_down_q"]] if "green_down_q" in index else np.zeros(x.shape[0])
    downstream_occ = x[:, index["green_down_occ"]] if "green_down_occ" in index else np.zeros(x.shape[0])
    switch = x[:, index["switch_indicator"]] if "switch_indicator" in index else np.zeros(x.shape[0])
    if resolved.mode == "spillback":
        if "service_pressure" not in index:
            raise KeyError("spillback pressure requires service_pressure")
        score = x[:, index["service_pressure"]]
    else:
        score = (
            green_q
            - resolved.downstream_queue_weight * downstream_q
            - resolved.downstream_occupancy_weight * downstream_occ
        )
    return np.asarray(score - resolved.switch_penalty * switch, dtype=float)


def _encode_float(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p") or "0"
