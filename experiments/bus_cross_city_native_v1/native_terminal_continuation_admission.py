"""Four real continuation updates per fixed arm, without simulation or checkpoints."""

import copy
from dataclasses import asdict
import json
from types import MethodType

import numpy as np
import torch

from native_terminal_continuation import P, build_learner, fit_terminal_rows
from train_native import HERE, write


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def terminal_targets(learner, batch):
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        actual = learner._target(batch, backup=learner.config.online_backup).reshape(-1)
        expected = learner.config.reward_scale*batch["rewards"].reshape(-1)
        require(torch.equal(actual, expected), "Actual continuation terminal targets differ from scaled rewards")
    return actual.tolist()


def clone_batch(batch):
    return {name: value.detach().clone() for name, value in batch.items()}


def run_arm(branch, seed):
    learner, replay, original = build_learner(branch, seed)
    require(original == {"training": P["training_settings"]}, "Continuation settings are not the frozen original")
    expected_config = dict(P["training_settings"]["trainer_config"], offline_objective="iql",
        online_fraction=0., use_ratio_correction=False, backup_entropy=False)
    require(asdict(learner.config) == expected_config, "Continuation config changed")
    require(learner.config.bootstrap_probability == 1 and learner.config.online_backup == "h2oplus",
        "Admission requires the actual frozen full-batch continuation TD kernel")
    require(learner.model_coefficient == P["branches"][branch]["model_td_coefficient"], "Model coefficient changed")
    expected_mode = P["branches"][branch]["model_weight_mode"]
    require(learner.augmentation.weight_mode == expected_mode, "Actual model weighting mode changed")
    tails = fit_terminal_rows(HERE / P["target_pool"])
    require(len(tails["rewards"]) == 6, "Corrected fit-terminal roster changed")
    before = terminal_targets(learner, tails)
    first, counts = {}, dict(real=0, model=0)
    original_sample, original_generate = replay.sample_mixed, learner.augmentation.generate
    native_owner = next(cls for cls in type(learner).__mro__
        if "_ensemble_td_loss" in cls.__dict__ and cls.__module__.startswith("SimpleSAC."))
    native_loss = native_owner.__dict__["_ensemble_td_loss"]

    def sample(self, *args, **kwargs):
        offline, online = original_sample(*args, **kwargs)
        require(offline is not None and online is None, "Continuation sampled online simulator records")
        if counts["real"] == 0:
            first["real"] = clone_batch(offline)
            first["request"] = dict(args=args, kwargs={key: value for key, value in kwargs.items() if key != "rng"})
            require(kwargs["rng"] is learner.rng, "Real batch used a different RNG")
        counts["real"] += 1
        return offline, online

    def generate(self, actor):
        batch, weights = original_generate(actor)
        require(bool(torch.isfinite(weights).all()), "Nonfinite model weights")
        if branch == "wm_uniform":
            require(torch.equal(weights, torch.ones_like(weights)), "Uniform model weights are not all one")
        if counts["model"] == 0:
            first["model"], first["weights"] = clone_batch(batch), weights.detach().clone()
        counts["model"] += 1
        return batch, weights

    def checked_native_loss(self, predictions, targets, weights=None):
        loss = native_loss(self, predictions, targets, weights)
        if "native_model_loss" not in first:
            require(counts["model"] == 1 and counts["real"] == 0 and
                torch.equal(weights, first["weights"]), "First actual model TD did not use generated weights")
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                repeated = native_loss(self, predictions, targets, first["weights"])
                unweighted = native_loss(self, predictions, targets, None)
            require(torch.equal(loss.detach(), repeated), "Native model TD did not preserve its supplied weights")
            if branch == "wm_uniform":
                require(torch.equal(loss.detach(), unweighted), "Uniform native TD differs from unweighted TD")
            first["native_model_loss"] = float(loss.detach())
            first["unweighted_model_loss"] = float(unweighted)
        return loss

    replay.sample_mixed = MethodType(sample, replay)
    learner.augmentation.generate = MethodType(generate, learner.augmentation)
    native_owner._ensemble_td_loss = checked_native_loss
    initial, initial_online = learner.total_updates, learner.online_critic_updates
    require(initial == P["adaptation_updates"] and replay.offline_size == P["target_fit_rows"],
        "Wrong repaired adapted checkpoint or target replay budget")
    try:
        for step in range(4):
            metrics = learner.update(online=True, collect_metrics=True)
            require(all(np.isfinite(value) for value in metrics.values()), "Nonfinite admission update metrics")
            if step == 0:
                require(metrics["model_td_loss"] == first["native_model_loss"], "Reported model TD differs from actual TD")
                first["torch_rng"] = torch.get_rng_state().clone()
                first["learner_rng"] = copy.deepcopy(learner.rng.bit_generator.state)
        require(learner.total_updates-initial == learner.online_critic_updates-initial_online == 4 and
            counts == dict(real=4, model=4), "Admission changed actual update or sampling budget")
        after = terminal_targets(learner, tails)
        require(before == after, "Corrected terminal targets changed during continuation")
    finally:
        native_owner._ensemble_td_loss = native_loss
        del replay.sample_mixed
        del learner.augmentation.generate
    weights = first["weights"]
    report = dict(updates=4, loaded_learner_updates=initial, final_learner_updates=learner.total_updates,
        online_critic_updates_added=4, real_batch_calls=counts["real"], model_generation_calls=counts["model"],
        config=asdict(learner.config), model_td_coefficient=learner.model_coefficient,
        weight_mode=expected_mode, model_weight_min=float(weights.min()), model_weight_max=float(weights.max()),
        model_weight_mean=float(weights.mean()), native_weighted_td_loss_checked=True,
        first_model_td_loss=first["native_model_loss"], first_unweighted_model_td_loss=first["unweighted_model_loss"],
        terminal_target_actual=True, terminal_targets=before, final_metrics=metrics)
    return report, first


def run():
    seed, arms, observations = 75101, {}, {}
    for branch in ("real", "wm_uniform", "wm_confidence"):
        arms[branch], observations[branch] = run_arm(branch, seed)
    reference = observations["real"]
    for branch in ("wm_uniform", "wm_confidence"):
        actual = observations[branch]
        require(actual["request"] == reference["request"], "First real replay requests differ across arms")
        for scope in ("real", "model"):
            require(actual[scope].keys() == reference[scope].keys() and all(torch.equal(value, actual[scope][name])
                for name, value in reference[scope].items()), f"First {scope} batches differ across arms")
        require(torch.equal(actual["torch_rng"], reference["torch_rng"]) and
            actual["learner_rng"] == reference["learner_rng"], "First-step training RNG differs across arms")
        require(arms[branch]["terminal_targets"] == arms["real"]["terminal_targets"], "Actual terminal targets differ across arms")
    require(torch.equal(observations["wm_confidence"]["weights"], reference["weights"]),
        "Confidence arm changed native first-batch weights")
    require(torch.equal(observations["wm_uniform"]["weights"], torch.ones_like(reference["weights"])),
        "Uniform arm changed more than the returned TD weights")
    report = dict(complete=True, protocol=P["name"], seed=seed, arms=arms,
        terminal_target_actual=True, model_batch_first_equal=True, real_first_batch_equal=True,
        first_step_torch_and_learner_rng_equal=True, confidence_weights_preserved=True,
        uniform_weights_all_one=True, uniform_native_td_equals_unweighted=True,
        checkpoint_files_written=0, new_simulation_events=0, evaluation_jobs=0,
        interpretation="Actual four-update interface admission; later critic and actor trajectories may differ.")
    write(HERE / "native_terminal_continuation/admission.json", report)
    print(json.dumps(dict(complete=True, protocol=P["name"], updates_per_arm=4)), flush=True)


if __name__ == "__main__":
    run()
