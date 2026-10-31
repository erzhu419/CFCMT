"""Fresh target initialization and adaptation on the corrected native pool."""

import argparse
from dataclasses import asdict, replace
import json
import subprocess
import sys

import h5py
import numpy as np
import torch

from diagnose_values import ARRAYS, tensors
from native_terminal_adaptation import terminal_diagnostic
from train_native import (HERE, continuation_trainer, runtime_api, seed_all,
    transfer_parameters, update_loop, write)

PREFIX = "native_terminal_target_only"


def run(seed, *, admission=False):
    protocol_path = HERE / (PREFIX + "_protocol.json")
    protocol = json.loads(protocol_path.read_text())
    settings = protocol["training_settings"]
    runtime = json.loads((HERE / "runtime.json").read_text())
    if seed not in protocol["training_seeds"]:
        raise ValueError("Training seed is outside the fixed roster")
    pool = HERE / protocol["target_pool"]
    with h5py.File(pool, "r") as data:
        if not json.loads(data.attrs["pool_summary"])["terminal_horizon_corrected"]:
            raise RuntimeError("Target replay is not the corrected full-horizon pool")
        if len(data["rewards"]) != protocol["target_union_rows"]:
            raise RuntimeError("Corrected target union differs from the frozen budget")
        ids = np.flatnonzero((data["split_id"][:] == 0) & (data["terminals"][:] > .5))
        if len(ids) != 6:
            raise RuntimeError("The six fit terminal rows changed")
        tails = tensors({key: data[key][ids] for key in ARRAYS})
    torch.set_num_threads(1)
    base, config_class, replay_class, _ = runtime_api()
    config = replace(config_class(**settings["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    folder = HERE / PREFIX / ("admission" if admission else f"seed{seed}")
    initialization = folder / "initialization"
    initialization_count = 3 if admission else protocol["initialization_updates"]
    adaptation_count = 3 if admission else protocol["adaptation_updates"]

    replay = replay_class(pool, device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    if replay.offline_size != protocol["target_fit_rows"]:
        raise RuntimeError("Corrected target fit budget differs from the original")
    initializer = base(replay, config=config, device="cpu", seed=seed)
    initial_terminal = terminal_diagnostic(initializer, tails)
    update_loop(initializer, initialization_count, "target_initialization", initialization,
        online=False, log_every=settings["log_every"])
    initialized_terminal = terminal_diagnostic(initializer, tails)
    initializer_path = initialization / "initializer.pt"
    initializer.save_offline_initializer(initializer_path, extra=dict(protocol=protocol["name"],
        seed=seed, domain="target", fresh_initialization=True,
        initialization_updates=initialization_count, target_pool=str(pool)))
    write(initialization / "summary.json", dict(complete=True, protocol=protocol["name"],
        seed=seed, admission=admission, initialization_updates=initialization_count,
        target_fit_rows=replay.offline_size, fit_terminal_initial=initial_terminal,
        fit_terminal_initialized=initialized_terminal))
    del initializer, replay

    # Match the original target adaptation: new learner/optimizers and replay,
    # network-only transfer, then the fixed adaptation RNG offset.
    replay = replay_class(pool, device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    learner = continuation_trainer(base)(replay, config=config, device="cpu", seed=seed)
    payload = torch.load(initializer_path, map_location="cpu", weights_only=False)
    if payload["total_updates"] != initialization_count:
        raise RuntimeError("Fresh target initializer has the wrong update count")
    transfer_parameters(learner, payload, reset_line_columns=False)
    del payload
    adaptation_before = terminal_diagnostic(learner, tails)
    seed_all(seed + settings["adaptation_rng_offset"])
    learner.rng = np.random.default_rng(seed + settings["adaptation_rng_offset"])
    update_loop(learner, adaptation_count, "target_adaptation", folder, online=False,
        log_every=settings["log_every"], offset=initialization_count)
    adapted_terminal = terminal_diagnostic(learner, tails)
    report = dict(complete=True, protocol=protocol["name"], seed=seed, admission=admission,
        initialization_updates=initialization_count, adaptation_updates=adaptation_count,
        historical_policy_updates=initialization_count+adaptation_count,
        target_fit_rows=replay.offline_size, target_union_rows=protocol["target_union_rows"],
        target_pool=str(pool), initializer=str(initializer_path), config=asdict(config),
        fresh_initialization=True, optimizer_transfer=False, encoder_columns_reset=False,
        source_rows=0, world_model_updates=0, continuation_updates=0, target_online_training_events=0,
        fit_terminal_initial=initial_terminal, fit_terminal_initialized=initialized_terminal,
        fit_terminal_adaptation_before=adaptation_before, fit_terminal_adapted=adapted_terminal,
        actor_export_equal=None, interpretation=protocol["interpretation"])
    if admission:
        write(folder / "train_summary.json", report)
        write(folder / "summary.json", report)
        print(json.dumps(dict(complete=True, seed=seed, admission=True,
            initialization_updates=initialization_count, adaptation_updates=adaptation_count)), flush=True)
        return

    learner.save(folder / "adapted.pt", extra=report, include_online_replay=False)
    artifact = folder / "policy.pt"
    learner.save_policy_artifact(artifact, extra=report)
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    actor, _ = load_actor_checkpoint_v7(artifact, "cpu", require_policy_artifact=True)
    expected, actual = learner.actor.state_dict(), actor.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(actual[key], value)
            for key, value in expected.items()):
        raise RuntimeError("Exported deployment actor differs from the final target actor")
    report["actor_export_equal"] = True
    write(folder / "train_summary.json", report)
    del actor, learner, replay
    for case in protocol["evaluation"]["cases"]:
        subprocess.run([sys.executable, "-u", str(HERE / "evaluate_native.py"),
            "--method", "target_only", "--seed", str(seed), "--case", str(case["case"]),
            "--assets", runtime["assets_root"] +
                f"/{protocol['asset_folder']}/case{case['case']:02d}/asset_manifest.json",
            "--checkpoint", str(artifact), "--output", str(folder / "evaluation" / f"case{case['case']:02d}"),
            "--evaluation-protocol", str(protocol_path), "--actor-action-space", "executed"], check=True)
    print(json.dumps(dict(complete=True, seed=seed, initialization_updates=initialization_count,
        adaptation_updates=adaptation_count, evaluations=len(protocol["evaluation"]["cases"]))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--admission", action="store_true")
    args = parser.parse_args()
    run(args.seed, admission=args.admission)
