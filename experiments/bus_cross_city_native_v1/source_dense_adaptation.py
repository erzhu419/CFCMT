"""Original target IQL adaptation from the fixed full-source initializers."""

import argparse
from dataclasses import asdict, replace
import json

import h5py
import numpy as np
import torch

from collect_values import compact_actor, compact_values
from diagnose_values import (ARRAYS, actor_summary, learner_from_payload,
    read_roles, read_source_samples, tensors, value_summary)
from train_native import (HERE, continuation_trainer, runtime_api, seed_all,
    transfer_parameters, update_loop, write)


def fit_terminal_rows():
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        ids = np.flatnonzero((data["split_id"][:] == 0) & (data["terminals"][:] > .5))
        if len(ids) != 6:
            raise ValueError("Target fit terminal inventory differs from the original six trajectories")
        return tensors({key: data[key][ids] for key in ARRAYS})


def diagnostics(seed, folder, trainer_class, config_class):
    raw_roles, contract, _ = read_roles(HERE / "target_pool.h5")
    roles = {name: tensors(raw) for name, raw in raw_roles.items()}
    source = {name: tensors(raw) for name, raw in read_source_samples(HERE / "source_pool.h5").items()}
    fit_tails = fit_terminal_rows()
    initial = HERE / "source_dense_ablation" / f"seed{seed}" / "initializer.pt"
    stages = {}
    for stage, path, reset in (("before", initial, True), ("after", folder / "adapted.pt", False)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        learner = learner_from_payload(trainer_class, config_class, contract, payload, seed, reset_columns=reset)
        stages[stage] = dict(
            target_roles={name: dict(values=value_summary(learner, batch, seed+80000+10000*index),
                actor=actor_summary(learner.actor, batch)) for index, (name, batch) in enumerate(roles.items())},
            target_fit_terminals=dict(rows=len(fit_tails["actions"]), values=value_summary(learner, fit_tails, seed+95000)),
            source_numerical_probes={name: dict(rows=len(batch["actions"]),
                values=value_summary(learner, batch, seed+90000)) for name, batch in source.items()})
    write(folder / "diagnostic_summary.json", dict(complete=True, seed=seed,
        role_rows={name: len(batch["actions"]) for name, batch in roles.items()},
        target_fit_terminal_rows=6, target_test_rows_read=0, stages=stages,
        interpretation="Both stages use reset city/rate columns. Source rows are numerical probes, not source-city calibration-retention evidence. Six fit terminal rows are training diagnostics; the two held-out terminal rows remain in stop/calibrate."))


def adapt(seed, *, admission=False):
    protocol = json.loads((HERE / "source_dense_adaptation_protocol.json").read_text())
    p = json.loads((HERE / "training_protocol.json").read_text())["training"]
    torch.set_num_threads(1)
    trainer_class, config_class, replay_class, _ = runtime_api()
    config = replace(config_class(**p["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    folder = HERE / "source_dense_adaptation" / ("admission" if admission else f"seed{seed}")
    replay = replay_class(HERE / "target_pool.h5", device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    if replay.offline_size != protocol["target_fit_rows"]:
        raise ValueError("Target adaptation must use exactly the original fit rows")
    learner = continuation_trainer(trainer_class)(replay, config=config, device="cpu", seed=seed)
    path = HERE / "source_dense_ablation" / f"seed{seed}" / "initializer.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["total_updates"] != protocol["source_initialization_updates"]:
        raise ValueError("Full-source initializer update count differs from the frozen protocol")
    transfer_parameters(learner, payload, reset_line_columns=True)
    del payload
    seed_all(seed+p["adaptation_rng_offset"])
    learner.rng = np.random.default_rng(seed+p["adaptation_rng_offset"])
    count = 3 if admission else protocol["adaptation_updates"]
    update_loop(learner, count, "dense_source_target_adaptation", folder,
        online=False, log_every=p["log_every"], offset=protocol["source_initialization_updates"])
    report = dict(complete=True, seed=seed, domain="source", device="cpu",
        initialization_updates=protocol["source_initialization_updates"], adaptation_updates=count,
        target_fit_rows=replay.offline_size, target_union_rows=16384,
        target_test_rows_read=0, optimizer_transfer=False, source_line_onehot_transfer=False,
        source_arrival_rate_transfer=False, admission=admission, config=asdict(config))
    if not admission:
        learner.save(folder / "adapted.pt", extra=report, include_online_replay=False)
        diagnostics(seed, folder, trainer_class, config_class)
    write(folder / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("complete", "seed", "adaptation_updates", "target_fit_rows", "admission")}), flush=True)


def compact_values_with_ranges(values):
    result = compact_values(values)
    for key in ("q_recorded_min", "value", "following_value", "iql_target"):
        result[key+"_range"] = {name: values[key][name] for name in ("min", "max")}
    return result


def compact_result(seed):
    root = HERE / "source_dense_adaptation" / f"seed{seed}"
    if not (root / "summary.json").exists():
        return None
    new = json.loads((root / "diagnostic_summary.json").read_text())
    old = json.loads((HERE / "diagnostics" / f"seed{seed}" / "summary.json").read_text())
    def compact_stage(stage):
        result = {key: {role: dict(rows=sample["rows"], values=compact_values_with_ranges(sample["values"]))
            for role, sample in stage[key].items()} for key in ("source_numerical_probes",)}
        result["target_roles"] = {name: dict(values=compact_values_with_ranges(row["values"]),
            actor=compact_actor(row["actor"])) for name, row in stage["target_roles"].items()}
        result["target_fit_terminals"] = dict(rows=stage["target_fit_terminals"]["rows"],
            values=compact_values_with_ranges(stage["target_fit_terminals"]["values"]))
        return result
    return dict(seed=seed, complete=new["complete"], role_rows=new["role_rows"],
        training=json.loads((root / "summary.json").read_text()),
        dense={name: compact_stage(stage) for name, stage in new["stages"].items()},
        references={stage: {role: dict(values=compact_values_with_ranges(row["values"]),
            actor=compact_actor(row["actor"])) for role, row in old["stages"][stage]["roles"].items()}
            for stage in ("source_adapted", "target_adapted")},
        target_test_rows_read=0, interpretation=new["interpretation"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--admission", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "source_dense_adaptation_protocol.json").read_text())
    if args.seed not in protocol["training_seeds"]:
        parser.error("Seed must belong to the existing frozen roster")
    adapt(args.seed, admission=args.admission)
