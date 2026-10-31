"""Paired actual updates for removing only the recorded-IQL free multiplier."""

import copy
from dataclasses import asdict
import json
from types import MethodType

import numpy as np
import torch

from native_terminal_actor_unmasked import P, PREFIX, build_learner
from native_terminal_v_only import build_learner as reference_builder
from native_terminal_continuation import count_optimizer_steps, fit_terminal_rows
from native_terminal_continuation_admission import clone_batch, require, terminal_targets
from train_native import HERE, write


def run_arm(unmasked):
    seed = P["training_seeds"][0]
    learner, replay, original = (build_learner if unmasked else reference_builder)(seed)
    require(original == {"training": P["training_settings"]}, "Frozen training settings changed")
    config = asdict(learner.config)
    require(config["mixed_backup_lambda"] == 1. and learner.model_coefficient == 0 and
        config["online_fraction"] == 0 and not config["backup_entropy"] and not config["use_ratio_correction"],
        "Common V-only real continuation changed")
    from canonical_execution_projection import canonical_execution_bounds
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    parameters = {module+"/"+name: value.detach().clone()
        for module in ("actor", "critic", "target_critic", "value")
        for name, value in getattr(learner, module).state_dict().items()}
    parameters["log_alpha"] = learner.log_alpha.detach().clone()
    tails = fit_terminal_rows(HERE / P["target_pool"])
    require(len(tails["rewards"]) == 6, "Corrected terminal roster changed")
    before = terminal_targets(learner, tails)
    initial, initial_online = learner.total_updates, learner.online_critic_updates
    first, current, requests, rng_steps = {}, {}, [], []
    counts = dict(real=0, model=0, actor=0)
    steps = count_optimizer_steps(learner)
    sample_original, generate_original = replay.sample_mixed, learner.augmentation.generate
    value_original, actor_original = learner._iql_value_terms, learner._update_iql_actor
    owner = next(cls for cls in type(learner).__mro__
        if "_update_iql_actor" in cls.__dict__ and cls.__module__.startswith("SimpleSAC."))
    native_actor = owner.__dict__["_update_iql_actor"]
    if unmasked:
        require(actor_original.__func__ is native_actor, "Unmasked entry does not bind the native actor method")
    native_delegate = actor_original if unmasked else MethodType(native_actor, learner)

    def sample(self, *args, **kwargs):
        offline, online = sample_original(*args, **kwargs)
        require(offline is not None and online is None and kwargs["rng"] is learner.rng, "Replay source or RNG changed")
        requests.append(dict(args=args, kwargs={k:v for k,v in kwargs.items() if k != "rng"}))
        if counts["real"] == 0:
            first["real"] = clone_batch(offline)
        counts["real"] += 1
        return offline, online

    def generate(self, actor):
        batch, weights = generate_original(actor)
        if counts["model"] == 0:
            first["model"], first["weights"] = clone_batch(batch), weights.detach().clone()
        counts["model"] += 1
        return batch, weights

    def value_terms(self, batch):
        result = value_original(batch)
        require(not result[2].requires_grad, "Original IQL actor weights are not detached")
        current.update(batch=batch, weights=result[2].detach().clone())
        return result

    def checked_native(self, batch, weights, *, collect_metrics=True):
        require(batch is current["batch"], "Native actor changed the recorded batch")
        lower, upper = canonical_execution_bounds(batch["observations"], DEFAULT_CONFIG_V7)
        free = upper[:, 0] > lower[:, 0]
        expected_weights = current["weights"] if unmasked else current["weights"]*free.to(weights.dtype)
        require(torch.equal(weights, expected_weights), "Actual native actor weights differ from the intervention")
        with torch.no_grad():
            log_probability = self.actor.action_log_prob(batch["observations"], batch["actions"])
            require(weights.ndim == log_probability.ndim == 1 and weights.numel() == len(batch["observations"]),
                "Native actor reduction no longer has the original batch denominator")
            expected_loss = -(weights*log_probability).mean()
        result = native_delegate(batch, weights, collect_metrics=collect_metrics)
        require(torch.equal(result[0].detach(), expected_loss), "Actual native actor loss differs from full-batch likelihood mean")
        if counts["actor"] == 0:
            original_weights = current["weights"]
            require(bool(free.any()) and bool((~free).any()) and bool((original_weights[~free] > 0).any()),
                "First actual actor batch does not exercise positive collapsed-row weights")
            first.update(actor_weights=original_weights.clone(), native_weights=weights.detach().clone(), free=free.clone())
        counts["actor"] += 1
        return result

    def actor_entry(self, batch, weights, *, collect_metrics=True):
        require(batch is current["batch"] and torch.equal(weights, current["weights"]), "Outer actor inputs changed")
        if unmasked:
            return checked_native(self, batch, weights, collect_metrics=collect_metrics)
        return actor_original(batch, weights, collect_metrics=collect_metrics)

    replay.sample_mixed = MethodType(sample, replay)
    learner.augmentation.generate = MethodType(generate, learner.augmentation)
    learner._iql_value_terms = MethodType(value_terms, learner)
    learner._update_iql_actor = MethodType(actor_entry, learner)
    owner._update_iql_actor = checked_native
    try:
        for _ in range(4):
            metrics = learner.update(online=True, collect_metrics=True)
            require(all(np.isfinite(value) for value in metrics.values()), "Nonfinite admission metrics")
            rng_steps.append(dict(torch=torch.get_rng_state().clone(), learner=copy.deepcopy(learner.rng.bit_generator.state),
                model=learner.augmentation.generator.get_state().clone(), trust=copy.deepcopy(learner.augmentation.rng.bit_generator.state)))
    finally:
        owner._update_iql_actor = native_actor
    require(initial == P["adaptation_updates"] and learner.total_updates-initial == learner.online_critic_updates-initial_online == 4
        and replay.offline_size == P["target_fit_rows"] and steps == dict(actor=4, critic=4, value=4, alpha=4)
        and counts == dict(real=4, model=4, actor=4), "Actual update/optimizer/replay budget changed")
    require(terminal_targets(learner, tails) == before, "Corrected terminal targets changed")
    collapsed_weights = first["actor_weights"][~first["free"]]
    report = dict(updates=4, config=config, loaded_learner_updates=initial, final_learner_updates=learner.total_updates,
        optimizer_steps=steps, online_critic_updates_added=4, model_td_coefficient=0., free_execution_actor_mask=not unmasked,
        real_batch_calls=4, model_generation_calls=4, actual_native_actor_calls=4, terminal_targets=before,
        native_actor_weights_exact=True, native_actor_full_batch_mean_exact=True, actor_batch_denominator=len(first["free"]),
        first_free_rows=int(first["free"].sum()), first_collapsed_rows=int((~first["free"]).sum()),
        first_collapsed_positive_weight_rows=int((collapsed_weights > 0).sum()), first_collapsed_weight_sum=float(collapsed_weights.sum()),
        final_metrics=metrics)
    return report, dict(first=first, parameters=parameters, requests=requests, rng_steps=rng_steps)


def run():
    old, original = run_arm(False)
    new, modified = run_arm(True)
    require(old["config"] == new["config"], "Actor unmasking changed learner configuration")
    require(original["parameters"].keys() == modified["parameters"].keys() and all(torch.equal(value,
        modified["parameters"][name]) for name,value in original["parameters"].items()), "Initial network parameters differ")
    require(original["requests"] == modified["requests"], "Actual replay requests differ")
    for scope in ("real", "model"):
        a,b = original["first"][scope], modified["first"][scope]
        require(a.keys() == b.keys() and all(torch.equal(value,b[name]) for name,value in a.items()), "First "+scope+" batches differ")
    for name in ("weights", "actor_weights", "free"):
        require(torch.equal(original["first"][name], modified["first"][name]), "First "+name+" differ")
    require(not torch.equal(original["first"]["native_weights"], modified["first"]["native_weights"]), "Unmasking changed no actual weights")
    for a,b in zip(original["rng_steps"], modified["rng_steps"]):
        require(torch.equal(a["torch"],b["torch"]) and torch.equal(a["model"],b["model"])
            and a["learner"] == b["learner"] and a["trust"] == b["trust"], "Actual update RNG differs")
    require(old["terminal_targets"] == new["terminal_targets"], "Terminal targets differ between arms")
    report = dict(complete=True, protocol=P["name"], seed=P["training_seeds"][0], arms=dict(real_v_only=old, real_v_unmasked=new),
        config_equal=True, initial_parameters_equal=True, actual_first_real_and_model_batches_equal=True,
        first_model_weights_equal=True, first_original_actor_weights_equal=True, actual_native_actor_weights_changed=True,
        native_actor_weights_verified=True, native_actor_full_batch_mean_exact=True, replay_requests_equal=True,
        all_four_update_rng_equal=True, corrected_terminal_targets_exact=True,
        checkpoint_files_written=0, new_simulation_events=0, evaluation_jobs=0)
    write(HERE / PREFIX / "admission.json", report)
    print(json.dumps(dict(complete=True, protocol=P["name"], updates_per_arm=4)), flush=True)


if __name__ == "__main__":
    run()
