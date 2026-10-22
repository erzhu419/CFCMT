"""Check the frozen source allocation and omit only the unavailable causal rate."""

import json
import h5py
import numpy as np

from source_pool import (CANONICAL_RATE_COLUMNS, RAW_RATE_COLUMNS,
                         assemble_source_pool, selected_rows)


def test_frozen_240_shard_quota_and_original_endpoints():
    ids = np.repeat(np.arange(240), 101)
    indices, reports = selected_rows(ids, 16384)
    assert len(np.unique(indices)) == 16384
    assert [r["selected_rows"] for r in reports] == [69] * 64 + [68] * 176
    for group, report in enumerate(reports):
        kept = indices[ids[indices] == group]
        assert kept[0] == group * 101
        assert kept[-1] == group * 101 + 100
        assert report["terminal_source_row"] == kept[-1]


def test_pool_keeps_original_pairs_and_zeros_all_source_rate_slots(tmp_path):
    source_path = tmp_path / "source.h5"
    rows = 40
    raw = np.arange(rows * 245, dtype=np.float32).reshape(rows, 245)
    canonical = np.arange(rows * 229, dtype=np.float32).reshape(rows, 229)
    terminals = np.zeros(rows, np.int8)
    terminals[[19, 39]] = 1
    arrays = dict(observations=canonical, next_observations=canonical + 1,
        raw_observations=raw, raw_next_observations=raw + 2,
        actions=np.arange(rows, dtype=np.float32)[:, None], rewards=-np.arange(rows),
        delta_t_seconds=np.arange(rows) % 3, terminals=terminals,
        discounts=1-terminals, split_id=np.zeros(rows, np.int8),
        trajectory_id=np.repeat([0, 1], 20), source_shard_id=np.repeat([0, 1], 20))
    with h5py.File(source_path, "w") as data:
        for name, value in arrays.items():
            data.create_dataset(name, data=value)
        data.attrs["schema_version"] = "transit-smdp-v7"
        data.attrs["config"] = json.dumps({"horizon_seconds": 24000})
        data.attrs["station_max_by_line"] = json.dumps([38, 36])
    summary = assemble_source_pool(source_path, tmp_path / "pool", count=12, expected_shards=2)
    assert summary["source_arrival_rate_omitted"]
    with h5py.File(tmp_path / "pool/source_pool.h5") as data:
        selected = data["source_row_index"][:]
        for name, original in arrays.items():
            expected = original[selected].copy()
            if name in ("observations", "next_observations"):
                expected[:, CANONICAL_RATE_COLUMNS] = 0
            elif name in ("raw_observations", "raw_next_observations"):
                expected[:, RAW_RATE_COLUMNS] = 0
            np.testing.assert_array_equal(data[name][:], expected)
        assert data.attrs["schema_version"] == "transit-smdp-v7"
        assert json.loads(data.attrs["config"])["horizon_seconds"] == 24000
        assert np.sum(data["terminals"][:]) == 2
        assert summary["encoder_transfer_reset_columns"] == list(range(12)) + [19]
