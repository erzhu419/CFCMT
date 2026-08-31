"""City-balanced context geometry with network-level support anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONTEXT_SUPPORT_PROTOCOL = "network_anchors_city_balanced_scale_v1"


@dataclass(frozen=True)
class ContextReferenceGeometry:
    domain_names: tuple[str, ...]
    domain_centroids: np.ndarray
    support_anchors: np.ndarray
    mean: np.ndarray
    scale: np.ndarray

    def diagnostics(self) -> dict[str, object]:
        return {
            "protocol": CONTEXT_SUPPORT_PROTOCOL,
            "domain_count": len(self.domain_names),
            "support_anchor_count": int(self.support_anchors.shape[0]),
        }


def fit_context_reference_geometry(
    context: np.ndarray,
    domains: np.ndarray,
    *,
    min_scale: float = 0.01,
) -> ContextReferenceGeometry:
    """Fit equal-city normalization without collapsing network support."""

    values = np.asarray(context, dtype=float)
    labels = np.asarray(domains, dtype=str).reshape(-1)
    if values.ndim != 2 or values.shape[0] != labels.size or not labels.size:
        raise ValueError("context geometry requires aligned non-empty rows")
    if not np.all(np.isfinite(values)):
        raise ValueError("context geometry contains non-finite values")
    domain_names = tuple(str(value) for value in np.unique(labels))
    domain_centroids = np.vstack(
        [np.mean(values[labels == domain], axis=0) for domain in domain_names]
    )

    # Context is static per network in this benchmark. Rounding removes only
    # serialization noise while preserving the first deterministic occurrence.
    rounded = np.round(values, decimals=12)
    _, first_indices = np.unique(rounded, axis=0, return_index=True)
    support_anchors = values[np.sort(first_indices)]
    mean = np.mean(domain_centroids, axis=0)
    scale = np.maximum(np.std(domain_centroids, axis=0), float(min_scale))
    return ContextReferenceGeometry(
        domain_names=domain_names,
        domain_centroids=domain_centroids,
        support_anchors=support_anchors,
        mean=mean,
        scale=scale,
    )
