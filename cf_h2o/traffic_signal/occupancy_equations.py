"""Lane-occupancy equations whose inputs and occupancy outputs are fractions.

This contract requires new feature caches and fitted models before deployment.
Historical equations remain reproducible from the frozen experiment snapshots.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


OCCUPANCY_EQUATION_PROTOCOL = "sumo-lane-occupancy-fraction-equations-v1"


def pressure_receiving_factor(
    downstream_occupancy: float | np.ndarray, *, minimum_factor: float = 0.0
) -> np.ndarray:
    """Return the 120-percent saturation factor for fractional occupancy.

The service-pressure features use a zero floor. The spillback-pressure baseline
uses its existing 0.20 floor.
    """

    return np.maximum(
        float(minimum_factor),
        1.0 - np.asarray(downstream_occupancy, dtype=float) / 1.2,
    )


def analytic_mechanism_priors_fraction(
    features: np.ndarray,
    context: np.ndarray,
    *,
    feature_names: Sequence[str],
    control_interval_sec: int,
) -> dict[str, np.ndarray]:
    """Express the complete analytic prior in a consistent fraction contract.

The coefficients are algebraically equivalent to the original intended
percentage equations, with occupancy inputs multiplied by 100 there and its
predicted occupancy divided by 100 on return. Queue, speed and cost units stay
unchanged.
    """

    x = np.asarray(features, dtype=float)
    z = np.asarray(context, dtype=float)
    idx = {name: index for index, name in enumerate(feature_names)}
    total_q = x[:, idx["total_q"]]
    total_veh = x[:, idx["total_veh"]]
    mean_speed = x[:, idx["mean_speed"]]
    mean_occ = x[:, idx["mean_occ"]]
    green_q = x[:, idx["green_q"]]
    red_q = x[:, idx["red_q"]]
    green_veh = x[:, idx["green_veh"]]
    green_down = x[:, idx["green_down_occ"]]
    green_down_q = x[:, idx["green_down_q"]]
    green_link_ratio = x[:, idx["green_link_ratio"]]
    green_lane_ratio = x[:, idx["green_lane_ratio"]]
    red_pressure = x[:, idx["red_pressure"]]
    clearance_fraction = np.clip(x[:, idx["clearance_fraction"]], 0.0, 1.0)

    arrival = z[:, 0] + 0.18 * z[:, 1] + 0.025 * total_veh + 0.008 * red_pressure
    available_green = np.clip(1.0 - clearance_fraction, 0.0, 1.0)
    downstream_factor = np.clip(1.0 - green_down / 1.35, 0.15, 1.0)
    capacity = (
        0.09
        * float(control_interval_sec)
        * (1.0 + 3.0 * green_lane_ratio + 1.4 * green_link_ratio)
        * available_green
        * downstream_factor
    )
    service = np.minimum(
        np.maximum(green_q - 0.25 * green_down_q, 0.0) + 0.3 * green_veh,
        capacity,
    )
    next_green = np.clip(green_q + arrival * green_lane_ratio - service, 0.0, 500.0)
    next_red = np.clip(
        red_q + arrival * (1.0 - green_lane_ratio) - 0.015 * red_q * available_green,
        0.0,
        500.0,
    )
    next_total = np.clip(next_green + next_red, 0.0, 500.0)
    next_down = np.clip(
        green_down
        + 0.0045 * service
        - 0.001 * mean_speed
        - 0.025 * np.maximum(1.0 - mean_occ, 0.0),
        0.0,
        1.0,
    )
    next_speed = np.clip(
        mean_speed
        + 0.12 * (13.0 - mean_speed)
        - 0.035 * total_q / np.maximum(x[:, idx["lane_count_norm"]] * 12.0, 1.0)
        - 1.2 * green_down,
        0.0,
        30.0,
    )
    interval_cost = np.clip(
        0.5 * (total_q + next_total)
        + 0.08 * red_pressure
        + 4.0 * next_down
        + 0.8 * float(control_interval_sec) * clearance_fraction,
        0.0,
        1000.0,
    )
    return {
        "next_total_queue": next_total,
        "next_green_queue": next_green,
        "next_red_queue": next_red,
        "next_downstream_occupancy": next_down,
        "next_mean_speed": next_speed,
        "interval_cost": interval_cost,
    }
