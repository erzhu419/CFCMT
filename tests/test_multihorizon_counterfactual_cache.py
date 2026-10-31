from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3_suite as suite
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    normalize_rollout_prefix_horizons,
    rollout_prefix_cost_targets,
    rollout_prefix_target_name,
)


def test_rollout_prefix_cost_targets_use_exact_prefix_means() -> None:
    costs = np.arange(1.0, 11.0)

    targets = rollout_prefix_cost_targets(costs, (10, 2, 5, 2))

    assert targets == {
        "prefix_mean_cost_2s": 1.5,
        "prefix_mean_cost_5s": 3.0,
        "prefix_mean_cost_10s": 5.5,
    }


def test_rollout_prefix_horizons_reject_out_of_rollout_values() -> None:
    assert normalize_rollout_prefix_horizons(
        (30, 10, 30), rollout_horizon_sec=60
    ) == (10, 30)
    with pytest.raises(ValueError, match="positive"):
        normalize_rollout_prefix_horizons((0, 10), rollout_horizon_sec=60)
    with pytest.raises(ValueError, match="exceeds"):
        normalize_rollout_prefix_horizons((61,), rollout_horizon_sec=60)
    with pytest.raises(ValueError, match="positive"):
        rollout_prefix_target_name(-1)


def test_multihorizon_identity_is_disjoint_from_v19(monkeypatch) -> None:
    monkeypatch.setattr(suite, "libsumo_version", lambda: "test-sumo")
    monkeypatch.setattr(suite, "_scenario_input_fingerprint", lambda _: "inputs")
    common = {
        "sumocfg": Path("network.sumocfg"),
        "scenario": "city",
        "duration_sec": 3600.0,
        "control_interval_sec": 10,
        "warmup_sec": 60.0,
        "seed": 1,
        "max_focal_tls": 4,
        "counterfactual_horizon_intervals": 45,
        "behavior_policy": "phase_pressure",
        "collection_shard_index": 0,
        "collection_shard_count": 32,
        "counterfactual_cost_mode": "halted_queue",
    }

    legacy = suite._counterfactual_cache_identity(**common)
    multihorizon = suite._counterfactual_cache_identity(
        **common,
        rollout_prefix_horizons_sec=(450, 10, 30, 60, 120),
    )

    assert legacy["version"] == suite.COUNTERFACTUAL_CACHE_VERSION
    assert multihorizon["version"] == suite.MULTIHORIZON_COUNTERFACTUAL_CACHE_VERSION
    assert multihorizon["rollout_prefix_horizons_sec"] == [10, 30, 60, 120, 450]
    assert multihorizon != legacy
