"""Repeat the original target adaptation using only corrected terminal labels."""

import argparse
from dataclasses import asdict, replace
import json
import subprocess
import sys

import h5py
import numpy as np
import torch

from diagnose_values import ARRAYS, tensors
from train_native import (HERE, continuation_trainer, runtime_api, seed_all,
    transfer_parameters, update_loop, write)


PREFIX = "native_terminal_adaptation"


def terminal_diagnostic(learner, batch):
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        target = learner._target(batch, backup="iql").reshape(-1)
        expected = learner.config.reward_scale * batch["rewards"].reshape(-1)
        if not torch.equal(target, expected):
            raise RuntimeError("Actual terminal IQL targets do not equal corrected scaled rewards")
        q = learner.critic(batch["observations"], batch["actions"]).min(0).values.reshape(-1)
    return dict(rows=len(target), targets=target.tolist(), recorded_min_q=q.tolist(),
        terminal_target_exact=True, mae=float((q-target).abs().mean()))


def run(seed, *, admission=False):
    protocol_path = HERE / (PREFIX + "_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    p = json.loads((HERE / "training_protocol.json").read_text())["training"]
    runtime = json.loads((HERE / "runtime.json").read_text())
    if seed not in protocol["training_seeds"]:
        raise ValueError("Training seed is outside the fixed roster")
    pool = HERE / protocol["target_pool"]
    with h5py.File(pool, "r") as data:
        if not json.loads(data.attrs["pool_summary"])["terminal_horizon_corrected"]:
            raise RuntimeError("Target replay is not the corrected full-horizon pool")
        ids = np.flatnonzero((data["split_id"][:] == 0) & (data["terminals"][:] > .5))
        if len(ids) != 6:
            raise RuntimeError("The six fit terminal rows changed")
        tails = tensors({key: data[key][ids] for key in ARRAYS})
    torch.set_num_threads(1)
    trainer_class, config_class, replay_class, _ = runtime_api()
    config = replace(config_class(**p["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    folder = HERE / PREFIX / ("admission" if admission else f"seed{seed}")
    replay = replay_class(pool, device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    if replay.offline_size != protocol["target_fit_rows"]:
        raise RuntimeError("Corrected target fit budget differs from the original")
    learner = continuation_trainer(trainer_class)(replay, config=config, device="cpu", seed=seed)
    source = HERE / "source_dense_ablation" / f"seed{seed}" / "initializer.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload["total_updates"] != protocol["source_initialization_updates"]:
        raise RuntimeError("Source initializer is not the fixed final50000 checkpoint")
    transfer_parameters(learner, payload, reset_line_columns=True)
    del payload
    before = terminal_diagnostic(learner, tails)
    seed_all(seed+p["adaptation_rng_offset"])
    learner.rng = np.random.default_rng(seed+p["adaptation_rng_offset"])
    count = 3 if admission else protocol["adaptation_updates"]
    update_loop(learner, count, "corrected_terminal_target_adaptation", folder,
        online=False, log_every=p["log_every"], offset=protocol["source_initialization_updates"])
    report = dict(complete=True, protocol=protocol["name"], seed=seed, admission=admission,
        initialization_updates=protocol["source_initialization_updates"], adaptation_updates=count,
        historical_policy_updates=protocol["source_initialization_updates"]+count,
        target_fit_rows=replay.offline_size, target_union_rows=protocol["target_union_rows"],
        target_pool=str(pool), source_initializer=str(source), config=asdict(config),
        optimizer_transfer=False, source_line_onehot_transfer=False, source_arrival_rate_transfer=False,
        world_model_updates=0, continuation_updates=0, target_online_training_events=0,
        fit_terminal_before=before, actor_export_equal=None,
        interpretation=protocol["interpretation"])
    if not admission:
        learner.save(folder / "adapted.pt", extra=report, include_online_replay=False)
        artifact = folder / "policy.pt"
        learner.save_policy_artifact(artifact, extra=report)
        from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
        actor, _ = load_actor_checkpoint_v7(artifact, "cpu", require_policy_artifact=True)
        expected, actual = learner.actor.state_dict(), actor.state_dict()
        if expected.keys() != actual.keys() or any(not torch.equal(actual[k],v) for k,v in expected.items()):
            raise RuntimeError("Exported deployment actor differs from the final adapted actor")
        report["actor_export_equal"] = True
        del actor
    report["fit_terminal_after"] = terminal_diagnostic(learner, tails)
    write(folder / "train_summary.json", report)
    if admission:
        write(folder / "summary.json", report)
        print(json.dumps(dict(complete=True, seed=seed, adaptation_updates=count,
            actual_terminal_targets_verified=True)), flush=True)
        return
    del learner, replay
    for case in protocol["evaluation"]["cases"]:
        subprocess.run([sys.executable, "-u", str(HERE / "evaluate_native.py"),
            "--method", "cross_city", "--seed", str(seed), "--case", str(case["case"]),
            "--assets", runtime["assets_root"]+f"/{protocol['asset_folder']}/case{case['case']:02d}/asset_manifest.json",
            "--checkpoint", str(artifact), "--output", str(folder / "evaluation" / f"case{case['case']:02d}"),
            "--evaluation-protocol", str(protocol_path), "--actor-action-space", "executed"], check=True)
    print(json.dumps(dict(complete=True, seed=seed, adaptation_updates=count, evaluations=3)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--admission", action="store_true")
    args = parser.parse_args()
    run(args.seed, admission=args.admission)
