"""Source compression ablation: server-owned full history, fixed IQL budget."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from diagnose_values import (actor_summary, learner_from_payload, read_roles,
                             read_source_samples, tensors, value_summary)
from source_pool import CANONICAL_RATE_COLUMNS, RAW_RATE_COLUMNS
from train_native import HERE, runtime_api, update_loop, write


def omit_rates(name, values):
    if name in ("observations", "next_observations"):
        values[:, CANONICAL_RATE_COLUMNS] = 0
    elif name in ("raw_observations", "raw_next_observations"):
        values[:, RAW_RATE_COLUMNS] = 0
    return values


def freeze_dense(source_path, compact_path, output, *, rows, shards, chunk=8192):
    """Match the consumed source rows, then stream original transition pairs."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Dense source snapshot already exists")
    with h5py.File(source_path, "r") as source, h5py.File(compact_path, "r") as compact:
        if source.attrs["schema_version"] != "transit-smdp-v7":
            raise ValueError("Source schema differs from the existing pilot")
        if len(source["split_id"]) != rows:
            raise ValueError("Source row count differs from the pilot history")
        if json.loads(source.attrs["config"]) != json.loads(compact.attrs["config"]):
            raise ValueError("Source contract changed since compact-pool creation")
        ids = compact["source_row_index"][:]
        for name, dataset in source.items():
            expected = omit_rates(name, dataset[ids])
            if not np.array_equal(expected, compact[name][:]):
                raise ValueError("Original source changed in consumed field: " + name)
        print(json.dumps(dict(compact_reference_rows_verified=len(ids))), flush=True)
        shard_ids = source["source_shard_id"][:]
        if len(np.unique(shard_ids)) != shards or np.any(source["split_id"][:] != 0):
            raise ValueError("Source shard inventory or training role changed")
        terminal_count = int(np.count_nonzero(source["terminals"][:]))
        if terminal_count != shards:
            raise ValueError("Source terminal inventory differs from the pilot")
        with h5py.File(output, "w") as destination:
            for name, dataset in source.items():
                copied = destination.create_dataset(name, shape=dataset.shape,
                    dtype=dataset.dtype, compression="gzip")
                for start in range(0, rows, chunk):
                    end = min(start+chunk, rows)
                    copied[start:end] = omit_rates(name, dataset[start:end])
                print(json.dumps(dict(frozen_field=name, rows=rows)), flush=True)
            for name, value in source.attrs.items():
                if name not in {"content_signature", "derivation_signature", "created_utc", "build_provenance"}:
                    destination.attrs[name] = value
            destination.attrs["n_transitions"] = rows
            destination.attrs["source_arrival_rate_omitted"] = True
            destination.attrs["source_dense_ablation"] = "All original rows; no resampling or next-state reconstruction"
    report = dict(complete=True, original_rows=rows, retained_rows=rows,
        source_shards=shards, terminal_rows=terminal_count,
        compact_reference_rows_verified=len(ids), source_file=str(source_path),
        frozen_dataset=str(output), frozen_bytes=output.stat().st_size,
        source_arrival_rate_omitted=True, chunk_rows=chunk,
        target_rows_read=0, transition_pairs="Original s/a/r/ns/dt/discount/terminal pairs retained")
    write(output.parent / "snapshot_summary.json", report)
    return report


def diagnose(seed, folder, trainer_class, config_class):
    roles, contract, _ = read_roles(HERE / "target_pool.h5")
    roles = {name: tensors(raw) for name, raw in roles.items()}
    source = {name: tensors(raw) for name, raw in read_source_samples(HERE / "source_pool.h5").items()}
    payload = torch.load(folder / "initializer.pt", map_location="cpu", weights_only=False)
    stages, source_report = {}, {}
    for reset in (False, True):
        learner = learner_from_payload(trainer_class, config_class, contract, payload, seed, reset_columns=reset)
        stages["transferred" if reset else "native"] = {
            role: dict(values=value_summary(learner, batch, seed+80000+10000*index),
                       actor=actor_summary(learner.actor, batch))
            for index, (role, batch) in enumerate(roles.items())}
        if not reset:
            source_report = {name: dict(rows=len(batch["actions"]),
                values=value_summary(learner, batch, seed+90000),
                actor=actor_summary(learner.actor, batch)) for name, batch in source.items()}
    write(folder / "diagnostic_summary.json", dict(complete=True, seed=seed,
        initialization_updates=payload["total_updates"], target_test_rows_read=0,
        role_rows={name: len(batch["actions"]) for name, batch in roles.items()},
        source_fit=source_report, stages=stages,
        comparison="Same existing source256, all240 source terminal tails, target stop/calibrate4096; no target adaptation or final evaluation"))


def train(seed, *, admission=False):
    protocol = json.loads((HERE / "dense_source_protocol.json").read_text())
    training = json.loads((HERE / "training_protocol.json").read_text())["training"]
    torch.set_num_threads(1)
    trainer_class, config_class, replay_class, _ = runtime_api()
    config = replace(config_class(**training["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    root = HERE / "source_dense_ablation"
    folder = root / ("admission" if admission else f"seed{seed}")
    replay = replay_class(root / "source_dense.h5", device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    if replay.offline_size != protocol["source_rows"]:
        raise ValueError("Full source history was not loaded")
    learner = trainer_class(replay, config=config, device="cpu", seed=seed)
    count = 3 if admission else protocol["initialization_updates"]
    update_loop(learner, count, "dense_source_initialization", folder,
        online=False, log_every=training["log_every"])
    if not admission:
        learner.save_offline_initializer(folder / "initializer.pt", extra=dict(
            seed=seed, domain="source", fresh_initialization=True,
            initialization_updates=count, source_compression_ablation=True))
    report = dict(complete=True, seed=seed, source_rows=replay.offline_size,
        initialization_updates=count, device="cpu", target_training_rows_read=0,
        target_test_rows_read=0, admission=admission, config=asdict(config))
    # Release the full replay before the small diagnostic forward passes.
    del learner, replay
    if not admission:
        diagnose(seed, folder, trainer_class, config_class)
    write(folder / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("complete", "seed", "source_rows", "initialization_updates", "admission")}), flush=True)


def compact_result(seed):
    from collect_values import compact_actor, compact_values
    root = HERE / "source_dense_ablation" / f"seed{seed}"
    if not (root / "summary.json").exists():
        return None
    new = json.loads((root / "diagnostic_summary.json").read_text())
    old = json.loads((HERE / "diagnostics" / f"seed{seed}" / "summary.json").read_text())
    def compact(row):
        result = compact_values(row)
        for name in ("q_recorded_min", "value", "following_value", "iql_target"):
            result[name+"_range"] = {key: row[name][key] for key in ("min", "max")}
        return result
    def action_value(row):
        return dict(values=compact(row["values"]), actor=compact_actor(row["actor"]))
    return dict(seed=seed, complete=new["complete"],
        training=json.loads((root / "summary.json").read_text()),
        role_rows=new["role_rows"], target_test_rows_read=0,
        dense={stage: {role: action_value(row) for role, row in roles.items()} for stage, roles in new["stages"].items()},
        sparse={stage: {role: action_value(row) for role, row in old["stages"]["source_initializer_"+stage]["roles"].items()} for stage in ("native", "transferred")},
        source_fit_dense={name: dict(rows=row["rows"], values=compact(row["values"])) for name, row in new["source_fit"].items()},
        source_fit_sparse={name: dict(rows=row["rows"], values=compact(row["values"])) for name, row in old["source_initializer_on_source_fit"]["samples"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train", "admission"))
    parser.add_argument("--seed", type=int, default=75101)
    args = parser.parse_args()
    protocol = json.loads((HERE / "dense_source_protocol.json").read_text())
    if args.seed not in protocol["training_seeds"]:
        parser.error("Seed must belong to the frozen three-seed protocol")
    if args.mode == "prepare":
        freeze_dense(protocol["original_source"], HERE / "source_pool.h5",
            HERE / "source_dense_ablation/source_dense.h5",
            rows=protocol["source_rows"], shards=protocol["source_shards"])
        train(args.seed, admission=True)
    else:
        train(args.seed, admission=args.mode == "admission")
