"""Freeze a position-selected source pool without source future-arrival features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SOURCE_ROWS = 16384
SOURCE_SHARDS = 240
CANONICAL_RATE_COLUMNS = np.arange(14, 229, 14)
RAW_RATE_COLUMNS = np.arange(14, 245, 15)
TRANSFER_RESET_COLUMNS = list(range(12)) + [19]


def selected_rows(shard_ids, count):
    """Allocate sorted shards equally, then keep equally spaced original pairs."""
    groups, group_counts = np.unique(shard_ids, return_counts=True)
    base, remainder = divmod(count, len(groups))
    quotas = np.full(len(groups), base, dtype=np.int64)
    quotas[:remainder] += 1
    if base < 2 or np.any(group_counts < quotas):
        raise ValueError("the frozen source allocation cannot retain each shard's endpoints")
    chosen, reports = [], []
    for group, size, quota in zip(groups, group_counts, quotas):
        rows = np.flatnonzero(shard_ids == group)
        index = rows[np.linspace(0, size - 1, quota, dtype=np.int64)]
        chosen.append(index)
        reports.append(dict(source_shard_id=int(group), original_rows=int(size),
            selected_rows=int(quota), first_source_row=int(index[0]),
            terminal_source_row=int(index[-1])))
    return np.sort(np.concatenate(chosen)), reports


def assemble_source_pool(source_path, output_root, count=SOURCE_ROWS,
                         expected_shards=SOURCE_SHARDS):
    """Read only one full narrow id column; read states only at chosen rows."""
    source_path, output_root = Path(source_path), Path(output_root)
    with h5py.File(source_path, "r") as source:
        if source.attrs["schema_version"] != "transit-smdp-v7":
            raise ValueError("source initialization requires the native v7 schema")
        shard_ids = source["source_shard_id"][:]
        if len(np.unique(shard_ids)) != expected_shards:
            raise ValueError("source shard inventory differs from the frozen selector")
        indices, groups = selected_rows(shard_ids, count)
        selected = {name: dataset[indices] for name, dataset in source.items()}
        if np.any(selected["split_id"] != 0):
            raise ValueError("the source initializer includes a non-training row")
        for name in ("observations", "next_observations"):
            if selected[name].shape != (count, 229):
                raise ValueError("source canonical state width differs from v7")
            selected[name][:, CANONICAL_RATE_COLUMNS] = 0
        for name in ("raw_observations", "raw_next_observations"):
            if selected[name].shape != (count, 245):
                raise ValueError("source raw state width differs from v7")
            selected[name][:, RAW_RATE_COLUMNS] = 0
        for report in groups:
            tail = np.searchsorted(indices, report["terminal_source_row"])
            if selected["terminals"][tail] != 1:
                raise ValueError("a selected source shard does not retain its terminal tail")
        attrs = dict(source.attrs)
        original_rows = len(shard_ids)
    selected["source_row_index"] = indices
    selected["global_trajectory_id"] = selected["trajectory_id"].copy()
    summary = dict(complete=True, decision="READY", source_city="Changsha",
        source_original_rows=original_rows, source_selected_rows=count,
        source_shards=expected_shards, source_file=str(source_path),
        dataset_file="source_pool.h5", split_id=0, groups=groups,
        source_arrival_rate_omitted=True,
        canonical_rate_columns=CANONICAL_RATE_COLUMNS.tolist(),
        raw_rate_columns=RAW_RATE_COLUMNS.tolist(),
        encoder_transfer_reset_columns=TRANSFER_RESET_COLUMNS,
        selection="Sorted shard ids; equal quotas with the remainder assigned to the first ids; equally spaced original row positions including both endpoints",
        transition_pairs="Selected rows retain their original action, reward, duration, discount and adjacent next state; arrival-rate features alone are omitted",
        budget="Source history is separate from the fixed target union; no source reward or policy-performance selection")
    output_root.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_root / "source_pool.h5", "w") as destination:
        for name, value in selected.items():
            destination.create_dataset(name, data=value, compression="gzip")
        for name, value in attrs.items():
            if name not in {"content_signature", "derivation_signature", "created_utc", "build_provenance"}:
                destination.attrs[name] = value
        if "build_provenance" in attrs:
            destination.attrs["source_original_build_provenance"] = attrs["build_provenance"]
        destination.attrs["n_transitions"] = count
        destination.attrs["source_pool_selection"] = json.dumps(summary)
        destination.attrs["source_arrival_rate_omitted"] = True
    (output_root / "source_pool_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = assemble_source_pool(args.source, args.output)
    print(json.dumps({name: summary[name] for name in
        ("complete", "decision", "source_original_rows", "source_selected_rows", "source_shards", "source_arrival_rate_omitted")}))


if __name__ == "__main__":
    main()
