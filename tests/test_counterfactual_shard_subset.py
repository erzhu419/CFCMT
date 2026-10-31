from __future__ import annotations

from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_counterfactual_shard_subset import (
    _payload,
    parse_seed_shards,
)


def test_parse_seed_shards_preserves_explicit_nonrectangular_assignment() -> None:
    assert parse_seed_shards(
        ["2027:0", "3037:5", "4047:15"], collection_shards=16
    ) == ((2027, 0), (3037, 5), (4047, 15))
    with pytest.raises(ValueError, match="duplicates"):
        parse_seed_shards(["2027:0", "2027:0"], collection_shards=16)
    with pytest.raises(ValueError, match="outside"):
        parse_seed_shards(["2027:16"], collection_shards=16)


def test_payload_uses_global_shard_count_in_cache_identity(monkeypatch) -> None:
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_counterfactual_shard_subset._counterfactual_cache_identity",
        lambda **values: dict(values),
    )
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_counterfactual_shard_subset._counterfactual_cache_path",
        lambda cache_root, **values: Path(cache_root)
        / f"{values['seed']}_{values['collection_shard_index']}.npz",
    )
    item = _payload(
        scenario="manhattan_28x7",
        sumocfg=Path("manhattan.sumocfg"),
        seed=3037,
        shard_index=7,
        collection_shards=16,
        duration_sec=3600,
        control_interval_sec=10,
        warmup_sec=60,
        max_focal_tls=4,
        counterfactual_horizon_intervals=45,
        behavior_policy="phase_pressure",
        counterfactual_cost_mode="halted_queue",
        rollout_prefix_horizons_sec=(450,),
        cache_root=Path("cache"),
    )
    assert item["collection_shard_index"] == 7
    assert item["collection_shard_count"] == 16
    assert item["cache_identity"]["collection_shard_count"] == 16
    assert item["cache_path"].endswith("3037_7.npz")
