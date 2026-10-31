"""Verify dense snapshots preserve consumed transitions and stream state arrays."""

import json

import h5py
import numpy as np
import pytest

from dense_source_ablation import freeze_dense
from source_pool import (CANONICAL_RATE_COLUMNS, RAW_RATE_COLUMNS,
                         assemble_source_pool)


def source_fixture(tmp_path):
    rows = 40
    canonical = np.arange(rows * 229, dtype=np.float32).reshape(rows, 229)
    raw = np.arange(rows * 245, dtype=np.float32).reshape(rows, 245)
    terminals = np.zeros(rows, dtype=np.uint8)
    terminals[[19, 39]] = 1
    arrays = dict(
        observations=canonical, next_observations=canonical + 3,
        raw_observations=raw, raw_next_observations=raw + 5,
        actions=np.arange(rows, dtype=np.float32)[:, None],
        rewards=-np.arange(rows, dtype=np.float64),
        delta_t_seconds=np.arange(rows, dtype=np.float64) % 3,
        discounts=(1 - terminals).astype(np.float32), terminals=terminals,
        split_id=np.zeros(rows, dtype=np.uint8),
        trajectory_id=np.repeat([0, 1], 20),
        source_shard_id=np.repeat([0, 1], 20),
    )
    source = tmp_path / "source.h5"
    with h5py.File(source, "w") as data:
        for name, values in arrays.items():
            data[name] = values
        data.attrs["schema_version"] = "transit-smdp-v7"
        data.attrs["config"] = json.dumps(dict(horizon_seconds=24000))
        data.attrs["station_max_by_line"] = json.dumps([38, 36])
    assemble_source_pool(source, tmp_path / "compact", count=12, expected_shards=2)
    return source, tmp_path / "compact/source_pool.h5", arrays


def test_dense_snapshot_streams_original_pairs_and_only_omits_rates(tmp_path, monkeypatch):
    source, compact, arrays = source_fixture(tmp_path)
    state_names = ("observations", "next_observations",
                   "raw_observations", "raw_next_observations")
    reads = {name: [] for name in state_names}
    getitem = h5py.Dataset.__getitem__

    def track_state_reads(dataset, key):
        name = dataset.name.lstrip("/")
        if dataset.file.filename == str(source) and name in reads and isinstance(key, slice):
            reads[name].append((key.start, key.stop, key.step))
        return getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track_state_reads)
    output = tmp_path / "dense/source_dense.h5"
    report = freeze_dense(source, compact, output, rows=40, shards=2, chunk=7)
    expected_reads = [(start, min(start + 7, 40), None) for start in range(0, 40, 7)]
    assert all(slices == expected_reads for slices in reads.values())
    with h5py.File(output, "r") as data:
        for name, original in arrays.items():
            expected = original.copy()
            if name in ("observations", "next_observations"):
                expected[:, CANONICAL_RATE_COLUMNS] = 0
            elif name in ("raw_observations", "raw_next_observations"):
                expected[:, RAW_RATE_COLUMNS] = 0
            np.testing.assert_array_equal(data[name][:], expected)
            assert data[name].dtype == original.dtype
        assert data.attrs["schema_version"] == "transit-smdp-v7"
        assert json.loads(data.attrs["config"]) == dict(horizon_seconds=24000)
        assert data.attrs["source_arrival_rate_omitted"]
        assert data.attrs["n_transitions"] == 40
    assert report["retained_rows"] == 40
    assert report["compact_reference_rows_verified"] == 12
    assert report["terminal_rows"] == 2


@pytest.mark.parametrize("field,column", [("rewards", None), ("next_observations", 17)])
def test_dense_snapshot_refuses_changed_consumed_transition(tmp_path, field, column):
    source, compact, _ = source_fixture(tmp_path)
    with h5py.File(compact, "r") as data:
        row = int(data["source_row_index"][3])
    with h5py.File(source, "r+") as data:
        if column is None:
            data[field][row] += 1
        else:
            data[field][row, column] += 1
    output = tmp_path / "dense/source_dense.h5"
    with pytest.raises(ValueError, match="Original source changed in consumed field: " + field):
        freeze_dense(source, compact, output, rows=40, shards=2, chunk=7)
    assert not output.exists()
