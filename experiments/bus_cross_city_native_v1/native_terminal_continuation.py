"""Three fixed continuation arms from the repaired source-adapted learners."""

import argparse
from dataclasses import asdict, replace
import json

import h5py
import numpy as np
import torch

from canonical_execution_projection import project_canonical_executed_action
from diagnose_values import ARRAYS, tensors
from free_execution_actor_update import free_execution_actor_trainer
from iql_actor_update import iql_actor_update_trainer
from projected_actor_sampling import install_projected_sampling
from train_native import (HERE, ModelContinuation, continuation_trainer,
    runtime_api, seed_all, update_loop, write)

PREFIX = "native_terminal_continuation"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


class UniformModelContinuation(ModelContinuation):
    @torch.no_grad()
    def generate(self, actor):
        batch, weights = super().generate(actor)
        return batch, torch.ones_like(weights)


def fit_terminal_rows(pool):
    with h5py.File(pool, "r") as data:
        if not json.loads(data.attrs["pool_summary"])["terminal_horizon_corrected"]:
            raise RuntimeError("Continuation replay is not the corrected full-horizon pool")
        if len(data["rewards"]) != P["target_union_rows"]:
            raise RuntimeError("Corrected target union differs from the frozen budget")
        ids = np.flatnonzero((data["split_id"][:] == 0) & (data["terminals"][:] > .5))
        if len(ids) != 6:
            raise RuntimeError("The six fit terminal rows changed")
        return tensors({key: data[key][ids] for key in ARRAYS})


def terminal_diagnostic(learner, batch):
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        targets = learner._target(batch, backup="h2oplus").reshape(-1)
        expected = learner.config.reward_scale * batch["rewards"].reshape(-1)
        if not torch.equal(targets, expected):
            raise RuntimeError("Terminal H2O+ targets do not equal corrected scaled rewards")
        q = learner.critic(batch["observations"], batch["actions"]).min(0).values.reshape(-1)
    return dict(rows=len(targets), backup="h2oplus", terminal_target_exact=True,
        targets=targets.tolist(), recorded_min_q=q.tolist(), mae=float((q-targets).abs().mean()))


def build_learner(branch, seed):
    if branch not in P["branches"] or seed not in P["training_seeds"]:
        raise ValueError("Branch or seed is outside the fixed roster")
    settings, spec = P["training_settings"], P["branches"][branch]
    original = dict(training=settings)
    torch.set_num_threads(1)
    base, config_class, replay_class, encoder_class = runtime_api()
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    config = replace(config_class(**settings["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    pool = HERE / P["target_pool"]
    tails = fit_terminal_rows(pool)
    replay = replay_class(pool, device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    if replay.offline_size != P["target_fit_rows"]:
        raise RuntimeError("Continuation must use the corrected fixed target-fit replay")
    trainer = free_execution_actor_trainer(iql_actor_update_trainer(continuation_trainer(base)))
    learner = trainer(replay, config=config, device="cpu", seed=seed)
    initializer = HERE / P["initializer_folder"] / f"seed{seed}" / "adapted.pt"
    learner.load(initializer, load_optimizers=True)
    learner.config = replace(learner.config, backup_entropy=False)
    if learner.config != replace(config, backup_entropy=False):
        raise RuntimeError("Loaded learner settings differ from the frozen continuation kernel")
    if learner.total_updates != P["adaptation_updates"]:
        raise RuntimeError("Initializer is not the fixed repaired adapted learner")
    install_projected_sampling(learner.actor,
        lambda states, actions: project_canonical_executed_action(states, actions, DEFAULT_CONFIG_V7))
    augmentation = UniformModelContinuation if branch == "wm_uniform" else ModelContinuation
    learner.augmentation = augmentation(original, seed,
        HERE / P["world_model_folder"] / f"seed{seed}", encoder_class, "cpu")
    learner.augmentation.weight_mode = spec["model_weight_mode"]
    learner.model_coefficient = spec["model_td_coefficient"]
    learner.model_seed = seed + 60000
    learner.trust_update_every = settings["trust_update_every"]
    learner.fit_terminal_before = terminal_diagnostic(learner, tails)
    seed_all(seed + settings["continuation_rng_offset"])
    learner.rng = np.random.default_rng(seed + settings["continuation_rng_offset"])
    return learner, replay, original


def count_optimizer_steps(learner):
    counts = {}
    for name in ("actor", "critic", "value", "alpha"):
        optimizer = getattr(learner, name + "_optimizer")
        original = optimizer.step
        counts[name] = 0
        def step(*args, _original=original, _name=name, **kwargs):
            value = _original(*args, **kwargs)
            counts[_name] += 1
            return value
        optimizer.step = step
    return counts


def train(branch, seed):
    learner, replay, original = build_learner(branch, seed)
    initial_updates, initial_critic = learner.total_updates, learner.online_critic_updates
    initial_log_alpha = float(learner.log_alpha.detach())
    counts = count_optimizer_steps(learner)
    settings, spec = original["training"], P["branches"][branch]
    count = P["continuation_updates"]
    folder = HERE / spec["artifact_folder"] / f"seed{seed}"
    print(json.dumps(dict(phase="initialized", protocol=P["name"], branch=branch, seed=seed,
        stored_updates=initial_updates, historical_updates=P["initialization_updates"]+initial_updates,
        target_fit_rows=replay.offline_size, backup_entropy=False,
        model_td_coefficient=learner.model_coefficient, model_weight_mode=learner.augmentation.weight_mode,
        alpha=float(learner.alpha.detach()))), flush=True)
    update_loop(learner, count, branch, folder, online=True, log_every=settings["log_every"],
        offset=P["initialization_updates"]+P["adaptation_updates"])
    if (learner.total_updates != initial_updates+count or learner.online_critic_updates-initial_critic != count
            or counts != {name: count for name in counts}):
        raise RuntimeError("Actual continuation optimizer/update counts differ from the frozen budget")
    report = dict(complete=True, protocol=P["name"], branch=branch, seed=seed,
        initializer=str(HERE / P["initializer_folder"] / f"seed{seed}" / "adapted.pt"),
        optimizer_transfer=True, target_pool=str(HERE / P["target_pool"]), target_fit_rows=replay.offline_size,
        target_union_rows=P["target_union_rows"], source_rows=P["source_rows"],
        initialization_updates=P["initialization_updates"], adaptation_updates=P["adaptation_updates"],
        continuation_updates=count, historical_policy_updates=P["initialization_updates"]+learner.total_updates,
        loaded_stored_updates=initial_updates, stored_total_updates=learner.total_updates,
        optimizer_steps=counts, online_critic_updates_added=learner.online_critic_updates-initial_critic,
        config=asdict(learner.config), actor_objective=P["actor_objective"], free_execution_actor_mask=True,
        sampled_action_space=P["sampled_action_space"], entropy_space=P["entropy_space"],
        temperature_algorithm="Original SAC latent-temperature update retained",
        backup_entropy=False, effective_critic_backup="h2oplus", mixed_backup_lambda=learner.config.mixed_backup_lambda,
        passenger_discount_gamma=1., terminal_discount=0.,
        fit_terminal_before=learner.fit_terminal_before,
        fit_terminal_after=terminal_diagnostic(learner, fit_terminal_rows(HERE / P["target_pool"])),
        continuation_model_td_coefficient=learner.model_coefficient,
        continuation_real_td_coefficient=1-learner.model_coefficient,
        model_weight_mode=learner.augmentation.weight_mode,
        model_td_weight_reduction="Mean of weight times elementwise Huber; no sum-of-weights division",
        confidence_definition=P["confidence_definition"], comparison_boundary=P["comparison_boundary"],
        trust_discount_gamma=float(learner.augmentation.trust.gamma),
        model_generation_and_trust_enabled_in_all_arms=True, shadow_model_enabled=branch == "real",
        world_model_artifact=str(HERE / P["world_model_folder"] / f"seed{seed}" / "model.pt"),
        world_model_data_pool=str(HERE / P["world_model_data_pool"]), world_model_refit_updates=0,
        trust_update_every=learner.trust_update_every, model_seed=learner.model_seed,
        continuation_rng_seed=seed+settings["continuation_rng_offset"], target_online_training_events=0,
        log_alpha_before=initial_log_alpha, log_alpha_after=float(learner.log_alpha.detach()),
        actor_export_equal=None, checkpoint_selection="fixed final", interpretation=P["interpretation"])
    artifact = folder / "policy.pt"
    learner.save_policy_artifact(artifact, extra=report)
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    actor, _ = load_actor_checkpoint_v7(artifact, "cpu", require_policy_artifact=True)
    expected, actual = learner.actor.state_dict(), actor.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(actual[key], value)
            for key, value in expected.items()):
        raise RuntimeError("Exported deployment actor differs from the final continuation actor")
    report["actor_export_equal"] = True
    learner.save(folder / "final_learner.pt", extra=report, include_online_replay=False)
    write(folder / "train_summary.json", report)
    del actor, learner, replay
    from iql_actor_continuation import evaluate
    evaluate(branch, seed, protocol=P, prefix=PREFIX)
    write(folder / "summary.json", dict(report, evaluation_cases=len(P["evaluation"]["cases"])))
    print(json.dumps(dict(complete=True, branch=branch, seed=seed, continuation_updates=count)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train",))
    parser.add_argument("--branch", required=True, choices=P["branches"])
    parser.add_argument("--seed", type=int, required=True, choices=P["training_seeds"])
    args = parser.parse_args()
    train(args.branch, args.seed)
