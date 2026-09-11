import numpy as np
import pytest

from cf_h2o.traffic_signal.occupancy_equations import (
    analytic_mechanism_priors_fraction,
    pressure_receiving_factor,
)


@pytest.mark.parametrize("floor", [0.0, 0.20])
def test_service_and_spillback_pressure_match_percentage_units(floor: float) -> None:
    # The measured Cologne value represents 28.158 percent lane occupancy.
    occupancy = np.asarray([0.0, 0.28158186943028773, 0.8, 1.0])
    percentage_factor = np.maximum(floor, 1.0 - occupancy * 100.0 / 120.0)

    np.testing.assert_allclose(
        pressure_receiving_factor(occupancy, minimum_factor=floor),
        percentage_factor,
        rtol=0.0,
        atol=1e-15,
    )
    assert pressure_receiving_factor(0.8, minimum_factor=floor) == pytest.approx(
        1.0 / 3.0
    )


def test_complete_analytic_prior_preserves_physical_units() -> None:
    names = (
        "total_q", "total_veh", "mean_speed", "mean_occ", "green_q", "red_q",
        "green_veh", "green_down_occ", "green_down_q", "green_link_ratio",
        "green_lane_ratio", "red_pressure", "clearance_fraction", "lane_count_norm",
    )
    # Include a busy receiving lane and an empty receiving lane whose prediction
    # reaches the physical zero bound. Every occupancy-dependent equation is used.
    features = np.asarray([
        [20.0, 30.0, 4.0, 0.4, 12.0, 8.0, 18.0, 0.8, 3.0, 0.5, 0.5, 6.0, 0.2, 0.25],
        [5.0, 8.0, 15.0, 0.1, 0.0, 5.0, 0.0, 0.0, 0.0, 0.5, 0.5, 3.75, 0.0, 0.25],
    ])
    context = np.asarray([[1.0, 2.0], [0.5, 1.0]])
    duration = 30
    actual = analytic_mechanism_priors_fraction(
        features, context, feature_names=names, control_interval_sec=duration
    )

    q, vehicles, speed, occupancy, green_q, red_q, green_vehicles, downstream, down_q, link_ratio, lane_ratio, red_pressure, clearance, lanes = features.T
    occupancy_pct = 100.0 * occupancy
    downstream_pct = 100.0 * downstream
    arrival = context[:, 0] + 0.18 * context[:, 1] + 0.025 * vehicles + 0.008 * red_pressure
    capacity = 0.09 * duration * (1.0 + 3.0 * lane_ratio + 1.4 * link_ratio) * (1.0 - clearance) * np.clip(1.0 - downstream_pct / 135.0, 0.15, 1.0)
    service = np.minimum(np.maximum(green_q - 0.25 * down_q, 0.0) + 0.3 * green_vehicles, capacity)
    expected_green = np.clip(green_q + arrival * lane_ratio - service, 0.0, 500.0)
    expected_red = np.clip(red_q + arrival * (1.0 - lane_ratio) - 0.015 * red_q * (1.0 - clearance), 0.0, 500.0)
    expected_total = np.clip(expected_green + expected_red, 0.0, 500.0)
    expected_down_pct = np.clip(downstream_pct + 0.45 * service - 0.10 * speed - 0.025 * np.maximum(100.0 - occupancy_pct, 0.0), 0.0, 100.0)
    expected_speed = np.clip(speed + 0.12 * (13.0 - speed) - 0.035 * q / np.maximum(lanes * 12.0, 1.0) - 0.012 * downstream_pct, 0.0, 30.0)
    expected_cost = np.clip(0.5 * (q + expected_total) + 0.08 * red_pressure + 0.04 * expected_down_pct + 0.8 * duration * clearance, 0.0, 1000.0)
    expected = {
        "next_total_queue": expected_total,
        "next_green_queue": expected_green,
        "next_red_queue": expected_red,
        "next_downstream_occupancy": expected_down_pct / 100.0,
        "next_mean_speed": expected_speed,
        "interval_cost": expected_cost,
    }
    for name, values in expected.items():
        np.testing.assert_allclose(actual[name], values, rtol=0.0, atol=1e-12)
    assert actual["next_downstream_occupancy"][0] > 0.7
    assert actual["next_downstream_occupancy"][1] == 0.0
