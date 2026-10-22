"""Matched continuation with only the actor objective replaced by recorded IQL."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from canonical_execution_projection import project_canonical_executed_action
from iql_actor_update import iql_actor_update_trainer
from projected_actor_sampling import install_projected_sampling
from train_native import ModelContinuation, continuation_trainer, runtime_api, seed_all, update_loop, write

HERE = Path(__file__).resolve().parent
PREFIX = "iql_actor_continuation"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def evaluate(branch, seed, *, protocol=None, prefix=PREFIX):
    p = P if protocol is None else protocol
    spec = p["branches"][branch] if branch != "strong_rule" else p["rule"]
    for case in p["evaluation"]["cases"]:
        folder = HERE / prefix / branch
        command = [sys.executable, "-u", str(HERE / "evaluate_native.py"),
            "--method", spec["method"], "--evaluation-protocol", str(HERE / (prefix + "_protocol.json")),
            "--case", str(case["case"]), "--assets", R["assets_root"] +
            f"/{p['asset_folder']}/case{case['case']:02d}/asset_manifest.json"]
        if seed is not None:
            folder = folder / f"seed{seed}"
            command += ["--seed", str(seed), "--actor-action-space", "executed", "--checkpoint",
                str(HERE / spec["artifact_folder"] / f"seed{seed}" / "policy.pt")]
        command += ["--output", str(folder / f"case{case['case']:02d}")]
        subprocess.run(command, check=True)


def build_learner(branch, seed, *, iql_actor=True, protocol=None, backup_entropy=None,
                  free_execution_actor=False):
    p = P if protocol is None else protocol
    original = json.loads((HERE / "training_protocol.json").read_text())
    settings = original["training"]
    torch.set_num_threads(1)
    base, config_class, replay_class, encoder_class = runtime_api()
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    config = replace(config_class(**settings["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    replay = replay_class(HERE / "target_pool.h5", device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    if replay.offline_size != p["target_fit_rows"]:
        raise ValueError("Continuation must use the original target-fit replay")
    trainer = continuation_trainer(base)
    if iql_actor:
        trainer = iql_actor_update_trainer(trainer)
    if free_execution_actor:
        from free_execution_actor_update import free_execution_actor_trainer
        trainer = free_execution_actor_trainer(trainer)
    learner = trainer(replay, config=config, device="cpu", seed=seed)
    initializer = HERE / "source_dense_adaptation" / f"seed{seed}" / "adapted.pt"
    learner.load(initializer, load_optimizers=True)
    if backup_entropy is not None:
        learner.config = replace(learner.config, backup_entropy=backup_entropy)
    if learner.total_updates != p["adaptation_updates"]:
        raise ValueError("Wrong adapted learner update count")
    install_projected_sampling(learner.actor,
        lambda states, actions: project_canonical_executed_action(states, actions, DEFAULT_CONFIG_V7))
    learner.augmentation = ModelContinuation(original, seed,
        HERE / "training/models" / f"seed{seed}", encoder_class, "cpu")
    learner.model_coefficient = p["branches"][branch]["model_td_coefficient"]
    learner.model_seed = seed + 60000
    learner.trust_update_every = settings["trust_update_every"]
    seed_all(seed + settings["continuation_rng_offset"])
    learner.rng = np.random.default_rng(seed + settings["continuation_rng_offset"])
    return learner, replay, original


def train(branch, seed, *, protocol=None, prefix=PREFIX, backup_entropy=None,
          free_execution_actor=False):
    p = P if protocol is None else protocol
    learner, replay, original = build_learner(branch, seed, protocol=p, backup_entropy=backup_entropy,
        free_execution_actor=free_execution_actor)
    before = learner.online_critic_updates
    log_alpha_before = float(learner.log_alpha.detach())
    folder = HERE / prefix / branch / f"seed{seed}"
    count = p["continuation_updates"]
    print(json.dumps(dict(phase="initialized", protocol=p["name"], branch=branch,
        seed=seed, loaded_learner_updates=learner.total_updates,
        target_fit_rows=replay.offline_size, continuation_updates=count,
        model_td_coefficient=learner.model_coefficient,
        actor_objective="recorded_iql", backup_entropy=learner.config.backup_entropy,
        free_execution_actor_mask=free_execution_actor,
        alpha=float(learner.alpha.detach()))), flush=True)
    update_loop(learner, count, ("free_execution_actor_" if free_execution_actor else "iql_actor_") + branch, folder,
        online=True, log_every=original["training"]["log_every"],
        offset=p["initialization_updates"] + p["adaptation_updates"])
    if learner.total_updates != p["adaptation_updates"] + count or learner.online_critic_updates - before != count:
        raise ValueError("Wrong fixed continuation budget")
    report = dict(complete=True, protocol=p["name"], branch=branch, seed=seed,
        initializer=str(HERE / "source_dense_adaptation" / f"seed{seed}" / "adapted.pt"),
        initialization_updates=p["initialization_updates"], adaptation_updates=p["adaptation_updates"],
        continuation_updates=count,
        total_policy_updates=p["initialization_updates"] + p["adaptation_updates"] + count,
        target_unique_rows=p["target_union_rows"], actor_target_fit_rows=replay.offline_size,
        source_rows=p["source_rows"], target_online_training_events=0,
        continuation_model_td_coefficient=learner.model_coefficient,
        continuation_real_td_coefficient=1 - learner.model_coefficient,
        continuation_kernel=("iql_actor_with_original_sac_critic_and_temperature" if learner.config.backup_entropy
            else "iql_actor_with_entropy_free_h2oplus_critic_and_original_temperature"),
        actor_objective=("free_execution_recorded_iql_full_batch_denominator" if free_execution_actor
            else "existing_recorded_action_iql_weighted_likelihood"),
        free_execution_actor_mask=free_execution_actor,
        effective_critic_backup=learner.config.online_backup, backup_entropy=learner.config.backup_entropy,
        online_critic_updates_added=learner.online_critic_updates-before,
        log_alpha_before=log_alpha_before, log_alpha_after=float(learner.log_alpha.detach()),
        world_model_artifact=str(HERE / "training/models" / f"seed{seed}" / "model.pt"),
        world_model_refit_updates=0, model_generation_and_trust_enabled_in_both_arms=True,
        sampled_action_space="projected_execution", entropy_space="latent_desired_action",
        replay_label_projection=False, config=asdict(learner.config), checkpoint_selection="fixed final")
    learner.save_policy_artifact(folder / "policy.pt", extra=report)
    learner.save(folder / "final_learner.pt", extra=report, include_online_replay=False)
    write(folder / "train_summary.json", report)
    del learner, replay
    evaluate(branch, seed, protocol=p, prefix=prefix)
    write(folder / "summary.json", dict(report, evaluation_cases=len(p["evaluation"]["cases"])))
    print(json.dumps(dict(complete=True, branch=branch, seed=seed, continuation_updates=count)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--branch", required=True, choices=(*P["branches"], "strong_rule"))
    parser.add_argument("--seed", type=int, choices=P["training_seeds"])
    args = parser.parse_args()
    if args.mode == "train" and args.branch not in ("td0", "td025"):
        parser.error("Only the two IQL actor arms train")
    if (args.branch == "strong_rule") != (args.seed is None):
        parser.error("Learned actors require their fixed seed; the rule has no training seed")
    {"train":train, "evaluate":evaluate}[args.mode](args.branch, args.seed)
