from __future__ import annotations

import numpy as np

from cf_h2o.traffic_signal.historical_local_demand_context import (
    HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES,
    HISTORICAL_LOCAL_DEMAND_PROTOCOL,
    HistoricalLocalDemandContext,
)


def _context(*, tls_a: tuple[float, ...], tls_b: tuple[float, ...]):
    return HistoricalLocalDemandContext(
        protocol=HISTORICAL_LOCAL_DEMAND_PROTOCOL,
        horizon_sec=1200,
        begin_sec=0.0,
        bin_sec=100,
        input_sha256="a" * 64,
        feature_names=HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES,
        tls_arrivals={"a": tls_a, "b": tls_b},
        incoming_lane_count={"a": 2, "b": 1},
        tls_neighbors={"a": ("b",), "b": ("a",)},
    )


def test_future_arrivals_cannot_change_past_feature_vector() -> None:
    prefix_a = (1.0, 2.0, 3.0, 4.0, 0.0, 0.0)
    prefix_b = (0.0, 1.0, 0.0, 2.0, 0.0, 0.0)
    left = _context(
        tls_a=prefix_a + (0.0,) * 6,
        tls_b=prefix_b + (0.0,) * 6,
    )
    right = _context(
        tls_a=prefix_a + (1000.0,) * 6,
        tls_b=prefix_b + (500.0,) * 6,
    )
    np.testing.assert_array_equal(left.vector_at("a", 600.0), right.vector_at("a", 600.0))


def test_past_arrivals_change_historical_feature_vector() -> None:
    left = _context(tls_a=(1.0,) * 12, tls_b=(1.0,) * 12)
    right = _context(tls_a=(1.0, 1.0, 1.0, 4.0, 4.0, 4.0) + (1.0,) * 6, tls_b=(1.0,) * 12)
    assert not np.array_equal(left.vector_at("a", 600.0), right.vector_at("a", 600.0))


def test_historical_feature_contract_contains_no_future_window() -> None:
    forbidden = ("next", "remaining", "future")
    assert all(
        marker not in name
        for name in HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES
        for marker in forbidden
    )
