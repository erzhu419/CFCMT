"""Eight actual updates admitting the actor-only frozen55k weight intervention."""

import copy
from dataclasses import asdict
import json
from types import MethodType

import numpy as np
import torch

from native_terminal_actor_unmasked import build_learner as build_dynamic
from native_terminal_fixed_actor_weights import (
    P, PREFIX, TEMPLATE, build_learner, scorer_unchanged,
)
from native_terminal_continuation import count_optimizer_steps, fit_terminal_rows
from native_terminal_continuation_admission import clone_batch, require, terminal_targets
from train_native import HERE, write


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(value, b[key]) for key, value in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def parameters(learner, modules):
    return {module + "/" + name: value.detach().clone()
        for module in modules for name, value in getattr(learner, module).state_dict().items()}


def run_arm(fixed):
    seed = P["training_seeds"][0]
    learner, replay, original = (build_learner if fixed else build_dynamic)(seed)
    require(original == {"training": TEMPLATE["training_settings"]}, "Frozen training settings changed")
    config = asdict(learner.config)
    require(config["mixed_backup_lambda"] == 1. and learner.model_coefficient == 0
        and config["online_fraction"] == 0 and not config["backup_entropy"]
        and not config["use_ratio_correction"], "Common unmasked V-only continuation changed")
    initial_parameters = parameters(learner, ("actor", "critic", "target_critic", "value", "ratio"))
    initial_parameters["log_alpha"] = learner.log_alpha.detach().clone()
    initial_optimizers = {name: copy.deepcopy(getattr(learner, name + "_optimizer").state_dict())
        for name in ("actor", "critic", "value", "alpha")}
    initial_optimizers["ratio"] = copy.deepcopy(learner.ratio.optimizer.state_dict())
    tails = fit_terminal_rows(HERE / TEMPLATE["target_pool"])
    before = terminal_targets(learner, tails)
    initial, initial_online = learner.total_updates, learner.online_critic_updates
    steps = count_optimizer_steps(learner)
    owner = next(cls for cls in type(learner).__mro__
        if "_update_iql_actor" in cls.__dict__ and cls.__module__.startswith("SimpleSAC."))
    native = owner.__dict__["_update_iql_actor"]
    native_delegate = learner._fixed_actor_native_update if fixed else learner._update_iql_actor
    require(native_delegate.__func__ is native, "Actor delegate is not the original native all-fit method")
    actor_original, value_original = learner._update_iql_actor, learner._iql_value_terms
    sample_original, generate_original = replay.sample_mixed, learner.augmentation.generate
    current, first, real_batches, requests, rng_steps, paths, losses = {}, {}, [], [], [], [], []
    counts = dict(real=0, model=0, actor=0)
    actor_weight_differences = []
    if fixed:
        for name in ("target_critic", "value"):
            require(equal(learner.fixed_actor_scorer_initial[name], getattr(learner, name).state_dict()),
                "Frozen scorer differs from the loaded initial55k function")

    def sample(self, *args, **kwargs):
        offline, online = sample_original(*args, **kwargs)
        require(offline is not None and online is None and kwargs["rng"] is learner.rng,
            "Replay source or RNG changed")
        real_batches.append(clone_batch(offline))
        requests.append(dict(args=args, kwargs={key: value for key, value in kwargs.items() if key != "rng"}))
        counts["real"] += 1
        return offline, online

    def generate(self, actor):
        batch, weights = generate_original(actor)
        if counts["model"] == 0:
            first["model"], first["model_weights"] = clone_batch(batch), weights.detach().clone()
        counts["model"] += 1
        return batch, weights

    def value_terms(self, batch):
        result = value_original(batch)
        require(not result[2].requires_grad, "Original dynamic IQL actor weights are not detached")
        current.update(batch=batch, weights=result[2].detach().clone())
        return result

    def checked_native(self, batch, weights, *, collect_metrics=True):
        require(batch is current["batch"], "Native actor changed the recorded batch")
        with torch.no_grad():
            if fixed:
                q = self.fixed_actor_target_critic(batch["observations"], batch["actions"]).min(dim=0).values
                value = self.fixed_actor_value(batch["observations"])
                expected = torch.exp((q - value).detach() * config["iql_advantage_temperature"]).clamp_max(
                    config["iql_max_weight"])
            else:
                expected = current["weights"]
            require(torch.equal(weights, expected) and not weights.requires_grad,
                "Actual actor weights differ from the registered native formula")
            log_probability = self.actor.action_log_prob(batch["observations"], batch["actions"])
            require(weights.ndim == log_probability.ndim == 1 and len(weights) == len(batch["observations"]),
                "Actor likelihood changed the full-batch denominator")
            expected_loss = -(weights * log_probability).mean()
        result = native_delegate(batch, weights, collect_metrics=collect_metrics)
        require(torch.equal(result[0].detach(), expected_loss), "Actual native actor loss differs from likelihood mean")
        actor_weight_differences.append(float((weights - current["weights"]).abs().max()))
        if counts["actor"] == 0:
            first["dynamic_actor_weights"] = current["weights"].clone()
            first["actual_actor_weights"] = weights.detach().clone()
        counts["actor"] += 1
        return result

    def actor_entry(self, batch, weights, *, collect_metrics=True):
        require(batch is current["batch"] and torch.equal(weights, current["weights"]),
            "Actor-only intervention changed the original outer actor inputs")
        if fixed:
            return actor_original(batch, weights, collect_metrics=collect_metrics)
        return checked_native(self, batch, weights, collect_metrics=collect_metrics)

    replay.sample_mixed = MethodType(sample, replay)
    learner.augmentation.generate = MethodType(generate, learner.augmentation)
    learner._iql_value_terms = MethodType(value_terms, learner)
    learner._update_iql_actor = MethodType(actor_entry, learner)
    if fixed:
        learner._fixed_actor_native_update = MethodType(checked_native, learner)
    for _ in range(4):
        metrics = learner.update(online=True, collect_metrics=True)
        require(all(np.isfinite(value) for value in metrics.values()), "Nonfinite admission metrics")
        rng_steps.append(dict(torch=torch.get_rng_state().clone(), learner=copy.deepcopy(learner.rng.bit_generator.state),
            model=learner.augmentation.generator.get_state().clone(), trust=copy.deepcopy(learner.augmentation.rng.bit_generator.state)))
        paths.append(parameters(learner, ("critic", "target_critic", "value")))
        losses.append({key: metrics[key] for key in ("train/critic_loss", "train/offline_td_loss",
            "train/value_loss", "train/critic_grad_norm", "train/value_grad_norm")})
        require(not fixed or scorer_unchanged(learner), "Actual update changed the frozen actor scorer")
    require(initial == TEMPLATE["adaptation_updates"] and learner.total_updates - initial
        == learner.online_critic_updates - initial_online == 4 and replay.offline_size == TEMPLATE["target_fit_rows"]
        and steps == dict(actor=4, critic=4, value=4, alpha=4)
        and counts == dict(real=4, model=4, actor=4), "Actual update/optimizer/replay budget changed")
    require(not fixed or learner.fixed_actor_weight_updates == 4, "Frozen actor weight hook count changed")
    require(terminal_targets(learner, tails) == before, "Corrected terminal targets changed")
    report = dict(updates=4, config=config, loaded_learner_updates=initial, final_learner_updates=learner.total_updates,
        optimizer_steps=steps, online_critic_updates_added=4, model_td_coefficient=0., free_execution_actor_mask=False,
        real_batch_calls=4, model_generation_calls=4, actual_native_actor_calls=4,
        actor_weight_source="frozen55_target_critic_and_value" if fixed else "original_dynamic_target_critic_and_value",
        fixed_actor_weight_updates=learner.fixed_actor_weight_updates if fixed else 0,
        frozen_actor_scorer_unchanged=True if fixed else None,
        native_actor_weights_exact=True, native_actor_full_batch_mean_exact=True,
        actor_batch_denominator=len(first["actual_actor_weights"]), terminal_targets=before,
        actor_weight_max_absolute_differences_from_dynamic=actor_weight_differences, final_metrics=metrics)
    return report, dict(parameters=initial_parameters, optimizers=initial_optimizers, first=first,
        real_batches=real_batches, requests=requests, rng_steps=rng_steps, critic_value_paths=paths, losses=losses)


def run():
    dynamic, original = run_arm(False)
    fixed, modified = run_arm(True)
    require(dynamic["config"] == fixed["config"], "Frozen actor scoring changed learner configuration")
    for key, message in (("parameters", "Initial network parameters differ"),
            ("optimizers", "Initial complete optimizer states differ"), ("requests", "Real replay requests differ"),
            ("real_batches", "Actual real replay batches differ"), ("rng_steps", "Actual update RNG differs"),
            ("critic_value_paths", "Actual critic/value parameter update paths differ"),
            ("losses", "Actual critic/value losses or gradients differ")):
        require(equal(original[key], modified[key]), message)
    for key in ("model", "model_weights", "dynamic_actor_weights"):
        require(equal(original["first"][key], modified["first"][key]), "First " + key + " differs")
    require(dynamic["terminal_targets"] == fixed["terminal_targets"], "Terminal targets differ between arms")
    require(any(value > 0 for value in fixed["actor_weight_max_absolute_differences_from_dynamic"]),
        "The admitted actor-only weight intervention changed no actual weights")
    report = dict(complete=True, protocol=P["name"], seed=P["training_seeds"][0],
        arms=dict(dynamic_weights=dynamic, fixed55_weights=fixed), admission_training_updates=8,
        config_equal=True, initial_parameters_equal=True, initial_complete_optimizer_states_equal=True,
        actual_all_real_batches_equal=True, actual_first_model_batch_equal=True, first_model_weights_equal=True,
        first_original_actor_weights_equal=True, native_actor_weights_verified=True,
        native_actor_full_batch_mean_exact=True, fixed_actor_scorer_matches_initial55=True,
        fixed_actor_scorer_unchanged=True, actual_actor_weights_changed=True, replay_requests_equal=True,
        all_four_update_rng_equal=True, all_four_critic_value_paths_equal=True,
        all_four_critic_value_losses_equal=True, corrected_terminal_targets_exact=True,
        checkpoint_files_written=0, new_simulation_events=0, evaluation_jobs=0)
    write(HERE / PREFIX / "admission.json", report)
    print(json.dumps(dict(complete=True, protocol=P["name"], updates_per_arm=4, admission_training_updates=8)), flush=True)


if __name__ == "__main__":
    run()
