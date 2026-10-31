"""Shared numerical scales for matched traffic-signal action contrasts."""

from __future__ import annotations

import numpy as np


ACTION_TARGET_SCALE_PROTOCOL = (
    "median-positive-within-domain-action-range-relative-floor-v3-consistent-selectors"
)
GROUP_ACTION_REGRET_SCALE_PROTOCOL = "within-group-action-range-relative-floor-v1"
ACTION_TARGET_SCALE_ABSOLUTE_FLOOR = 1e-8
ACTION_TARGET_SCALE_RELATIVE_FLOOR = 1e-6


def action_scale_floor(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    level = float(np.quantile(np.abs(values), 0.75)) if values.size else 0.0
    return max(
        ACTION_TARGET_SCALE_ABSOLUTE_FLOOR,
        ACTION_TARGET_SCALE_RELATIVE_FLOOR * level,
    )


def action_group_range(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not values.size:
        raise ValueError("action group range requires at least one value")
    return max(float(np.ptp(values)), action_scale_floor(values))


def domain_action_scale(target: np.ndarray, groups: np.ndarray) -> float:
    target = np.asarray(target, dtype=float)
    groups = np.asarray(groups)
    if target.shape != groups.shape:
        raise ValueError("domain action scale requires row-aligned targets and groups")
    floor = action_scale_floor(target)
    ranges = [
        float(np.ptp(target[groups == group]))
        for group in np.unique(groups)
    ]
    positive = np.asarray([value for value in ranges if value > floor], dtype=float)
    if positive.size:
        return max(float(np.median(positive)), floor)
    fallback = float(np.quantile(np.abs(target), 0.75)) if target.size else 0.0
    return max(fallback, floor)
