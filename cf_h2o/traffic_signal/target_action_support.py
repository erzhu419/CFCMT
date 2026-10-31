"""Local support for target-adapted signal actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.spatial import cKDTree

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


TARGET_SUPPORT_FEATURES = (
    "total_q",
    "total_veh",
    "mean_speed",
    "mean_occ",
    "green_q",
    "red_q",
    "green_down_q",
    "green_down_occ",
    "delta_green_q",
    "delta_red_q",
    "delta_green_down_q",
    "delta_green_down_occ",
    "delta_service_pressure",
    "delta_switch_indicator",
    "delta_clearance_fraction",
    "delta_current_phase_overlap",
)

LABEL_FREE_SUPPORT_PROTOCOL = "target-action-coverage-nearest-prototype-v1"
OOF_VALIDATED_SUPPORT_PROTOCOL = "oof-safe-action-nearest-prototype-v1"


@dataclass
class TargetActionSupport:
    """Nearest-prototype support in deterministic local state-action space."""

    feature_names: tuple[str, ...]
    center: np.ndarray
    scale: np.ndarray
    prototypes: np.ndarray
    support_protocol: str = LABEL_FREE_SUPPORT_PROTOCOL
    label_free: bool = True
    minimum_total_vehicles: float = 0.0
    _prototype_tree: Any = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def fit(
        cls,
        dataset: MechanismDataset,
        *,
        feature_names: Sequence[str] = TARGET_SUPPORT_FEATURES,
    ) -> "TargetActionSupport":
        if dataset.size <= 0:
            raise ValueError("target action support requires at least one row")
        index = {name: idx for idx, name in enumerate(dataset.feature_names)}
        selected_names = tuple(name for name in feature_names if name in index)
        if not selected_names:
            raise ValueError("target action support has no available local features")
        selected = np.asarray([index[name] for name in selected_names], dtype=int)
        values = np.asarray(dataset.features[:, selected], dtype=float)
        return cls._fit_values(
            feature_names=selected_names,
            values=values,
            support_protocol=LABEL_FREE_SUPPORT_PROTOCOL,
            label_free=True,
            minimum_total_vehicles=0.0,
        )

    @classmethod
    def fit_validated_records(
        cls,
        records: Sequence[dict[str, object]],
        *,
        feature_names: Sequence[str] = TARGET_SUPPORT_FEATURES,
        minimum_total_vehicles: float = 1.0,
    ) -> "TargetActionSupport":
        """Fit support only to OOF actions with independently observed gains."""

        if not records:
            raise ValueError("validated action support requires at least one record")
        requested = tuple(str(name) for name in feature_names)
        rows = []
        for record in records:
            raw = record.get("candidate_features")
            if not isinstance(raw, dict):
                raise ValueError("validated support record lacks candidate_features")
            missing = [name for name in requested if name not in raw]
            if missing:
                raise KeyError(f"validated support features missing: {missing}")
            rows.append([float(raw[name]) for name in requested])
        values = np.asarray(rows, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("validated support features must be finite")
        return cls._fit_values(
            feature_names=requested,
            values=values,
            support_protocol=OOF_VALIDATED_SUPPORT_PROTOCOL,
            label_free=False,
            minimum_total_vehicles=float(minimum_total_vehicles),
        )

    @classmethod
    def _fit_values(
        cls,
        *,
        feature_names: tuple[str, ...],
        values: np.ndarray,
        support_protocol: str,
        label_free: bool,
        minimum_total_vehicles: float,
    ) -> "TargetActionSupport":
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[0] <= 0 or values.shape[1] != len(
            feature_names
        ):
            raise ValueError("target action support feature matrix is invalid")
        center = np.median(values, axis=0)
        q25, q75 = np.quantile(values, (0.25, 0.75), axis=0)
        robust_scale = (q75 - q25) / 1.349
        standard_scale = np.std(values, axis=0)
        magnitude_floor = 0.05 * np.maximum(np.median(np.abs(values), axis=0), 1.0)
        scale = np.maximum.reduce(
            [robust_scale, 0.25 * standard_scale, magnitude_floor, np.full(values.shape[1], 0.05)]
        )
        prototypes = (values - center[None, :]) / scale[None, :]
        return cls(
            feature_names=feature_names,
            center=center,
            scale=scale,
            prototypes=prototypes,
            support_protocol=str(support_protocol),
            label_free=bool(label_free),
            minimum_total_vehicles=float(minimum_total_vehicles),
        )

    def _standardized_values(
        self, dataset: MechanismDataset
    ) -> tuple[np.ndarray, dict[str, int]]:
        index = {name: idx for idx, name in enumerate(dataset.feature_names)}
        missing = [name for name in self.feature_names if name not in index]
        if missing:
            raise KeyError(f"target support features missing at prediction: {missing}")
        selected = np.asarray([index[name] for name in self.feature_names], dtype=int)
        values = np.asarray(dataset.features[:, selected], dtype=float)
        standardized = (values - self.center[None, :]) / self.scale[None, :]
        return standardized, index

    def _nearest_rms_bruteforce(
        self, standardized: np.ndarray, *, chunk_size: int = 256
    ) -> np.ndarray:
        """Reference implementation retained for numerical equivalence audits."""

        values = np.asarray(standardized, dtype=float)
        nearest_squared = np.full(values.shape[0], np.inf, dtype=float)
        for start in range(0, int(self.prototypes.shape[0]), int(chunk_size)):
            prototypes = self.prototypes[start : start + int(chunk_size)]
            squared = (values[:, None, :] - prototypes[None, :, :]) ** 2
            nearest_squared = np.minimum(
                nearest_squared, np.min(np.mean(squared, axis=2), axis=1)
            )
        return np.sqrt(nearest_squared)

    def _nearest_rms_exact_tree(self, standardized: np.ndarray) -> np.ndarray:
        tree = getattr(self, "_prototype_tree", None)
        if tree is None:
            tree = cKDTree(
                np.asarray(self.prototypes, dtype=float),
                compact_nodes=True,
                balanced_tree=True,
            )
            self._prototype_tree = tree
        distance, _ = tree.query(
            np.asarray(standardized, dtype=float),
            k=1,
            eps=0.0,
            p=2.0,
            workers=1,
        )
        return np.asarray(distance, dtype=float) / np.sqrt(len(self.feature_names))

    def support_bruteforce(self, dataset: MechanismDataset) -> np.ndarray:
        """Compute the pre-optimization support for a frozen equivalence audit."""

        standardized, index = self._standardized_values(dataset)
        support = np.exp(-self._nearest_rms_bruteforce(standardized))
        return self._apply_minimum_vehicle_gate(support, dataset, index)

    def _apply_minimum_vehicle_gate(
        self,
        support: np.ndarray,
        dataset: MechanismDataset,
        index: dict[str, int],
    ) -> np.ndarray:
        support = np.asarray(support, dtype=float).copy()
        minimum_total_vehicles = float(getattr(self, "minimum_total_vehicles", 0.0))
        if minimum_total_vehicles > 0.0:
            if "total_veh" not in index:
                raise KeyError("minimum-vehicle support requires total_veh")
            total_vehicles = np.asarray(
                dataset.features[:, index["total_veh"]], dtype=float
            )
            support[total_vehicles < minimum_total_vehicles] = 0.0
        return support

    def support(self, dataset: MechanismDataset) -> np.ndarray:
        standardized, index = self._standardized_values(dataset)
        support = np.exp(-self._nearest_rms_exact_tree(standardized))
        return self._apply_minimum_vehicle_gate(support, dataset, index)

    def diagnostics(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "feature_count": len(self.feature_names),
            "prototype_count": int(self.prototypes.shape[0]),
            "distance": "nearest_standardized_rms",
            "support": "exp(-distance)",
            "support_protocol": str(
                getattr(self, "support_protocol", LABEL_FREE_SUPPORT_PROTOCOL)
            ),
            "label_free": bool(getattr(self, "label_free", True)),
            "minimum_total_vehicles": float(
                getattr(self, "minimum_total_vehicles", 0.0)
            ),
        }


@dataclass
class HierarchicalTargetActionSupport:
    """Keep pressure-coverage and OOF-safe support as separate frozen gates."""

    pressure_support: TargetActionSupport
    conformal_support: TargetActionSupport
    conformal_guard_enabled: bool

    def component_support(
        self, dataset: MechanismDataset
    ) -> dict[str, np.ndarray]:
        pressure = self.pressure_support.support(dataset)
        conformal = (
            self.conformal_support.support(dataset)
            if self.conformal_guard_enabled
            else np.ones(dataset.size, dtype=float)
        )
        return {"pressure": pressure, "conformal": conformal}

    def support(self, dataset: MechanismDataset) -> np.ndarray:
        components = self.component_support(dataset)
        return np.minimum(components["pressure"], components["conformal"])

    def diagnostics(self) -> dict[str, object]:
        return {
            "support_protocol": "pressure-and-oof-safe-intersection-v1",
            "conformal_guard_enabled": bool(self.conformal_guard_enabled),
            "pressure_support": self.pressure_support.diagnostics(),
            "conformal_support": self.conformal_support.diagnostics(),
            "combination": (
                "minimum_component_support"
                if self.conformal_guard_enabled
                else "pressure_support_only"
            ),
        }
