"""Paired four-update admission for the isolated H2O+ V coefficient change."""

import copy
from dataclasses import asdict
import json
from types import MethodType

import numpy as np
import torch

from native_terminal_continuation import build_learner as original_builder, count_optimizer_steps, fit_terminal_rows
from native_terminal_continuation_admission import clone_batch, require, terminal_targets
from native_terminal_v_only import P, build_learner
from train_native import HERE, write


def run_arm(v_only):
    seed = P["training_seeds"][0]
    learner, replay, _ = build_learner(seed) if v_only else original_builder("real", seed)
    config = asdict(learner.config)
    require(config["mixed_backup_lambda"] == (1.0 if v_only else .1), "Wrong actual V coefficient")
    require(learner.model_coefficient == 0 and config["online_fraction"] == 0 and
        not config["backup_entropy"] and not config["use_ratio_correction"], "Common real continuation changed")
    initial_parameters = {module+"/"+name: value.detach().clone()
        for module in ("actor", "critic", "target_critic", "value")
        for name, value in getattr(learner, module).state_dict().items()}
    initial_parameters["log_alpha"] = learner.log_alpha.detach().clone()
    tails = fit_terminal_rows(HERE / P["target_pool"])
    before = terminal_targets(learner, tails)
    initial, initial_online = learner.total_updates, learner.online_critic_updates
    first, requests, rng_steps = {}, [], []
    counts = dict(real=0, model=0)
    optimizer_steps = count_optimizer_steps(learner)
    original_sample, original_generate, original_target = replay.sample_mixed, learner.augmentation.generate, learner._target

    def sample(self, *args, **kwargs):
        offline, online = original_sample(*args, **kwargs)
        require(offline is not None and online is None and kwargs["rng"] is learner.rng, "Unexpected replay source or RNG")
        requests.append(dict(args=args, kwargs={k:v for k,v in kwargs.items() if k != "rng"}))
        if counts["real"] == 0:
            first["real"], first["real_pointer"] = clone_batch(offline), offline["observations"].data_ptr()
        counts["real"] += 1
        return offline, online

    def generate(self, actor):
        batch, weights = original_generate(actor)
        if counts["model"] == 0:
            first["model"], first["weights"] = clone_batch(batch), weights.detach().clone()
        counts["model"] += 1
        return batch, weights

    def target(self, batch, *, backup):
        actual = original_target(batch, backup=backup)
        if batch["observations"].data_ptr() == first.get("real_pointer") and "actual_target" not in first:
            require(backup == "h2oplus", "Actual first real batch did not use the H2O+ backup")
            first["actual_target"] = actual.detach().clone().reshape(-1)
            with torch.random.fork_rng(devices=[]), torch.inference_mode():
                first["iql_target"] = original_target(batch, backup="iql").detach().clone().reshape(-1)
            if v_only:
                require(torch.equal(first["actual_target"], first["iql_target"]), "V-only actual H2O+ differs from IQL target")
        return actual

    replay.sample_mixed = MethodType(sample, replay)
    learner.augmentation.generate = MethodType(generate, learner.augmentation)
    learner._target = MethodType(target, learner)
    for step in range(4):
        metrics = learner.update(online=True, collect_metrics=True)
        require(all(np.isfinite(value) for value in metrics.values()), "Nonfinite paired admission metrics")
        rng_steps.append(dict(torch=torch.get_rng_state().clone(), learner=copy.deepcopy(learner.rng.bit_generator.state),
            model=learner.augmentation.generator.get_state().clone(), trust=copy.deepcopy(learner.augmentation.rng.bit_generator.state)))
    require(learner.total_updates-initial == learner.online_critic_updates-initial_online == 4 and
        initial == P["adaptation_updates"] and replay.offline_size == P["target_fit_rows"] and
        optimizer_steps == dict(actor=4, critic=4, value=4, alpha=4) and counts == dict(real=4, model=4),
        "Actual update/optimizer/replay budget changed")
    require("actual_target" in first, "Actual first real H2O+ target was not captured")
    require(terminal_targets(learner, tails) == before, "Corrected terminal targets changed")
    nonterminal = first["real"]["terminals"].reshape(-1) < .5
    require(bool(nonterminal.any()), "First real batch has no nonterminal transitions")
    difference = (first["actual_target"]-first["iql_target"]).abs()[nonterminal]
    if not v_only:
        require(bool((difference > 0).any()), "Original mixed target does not differ from V-only on nonterminal rows")
    report = dict(updates=4, config=config, loaded_learner_updates=initial, final_learner_updates=learner.total_updates,
        online_critic_updates_added=4, optimizer_steps=optimizer_steps, real_batch_calls=4, model_generation_calls=4,
        model_td_coefficient=0., terminal_target_actual=True, terminal_targets=before,
        first_nonterminal_rows=int(nonterminal.sum()), first_nonterminal_targets_different_from_iql=int((difference > 0).sum()),
        first_nonterminal_target_max_abs_difference_from_iql=float(difference.max()),
        first_nonterminal_target_mean_abs_difference_from_iql=float(difference.mean()), final_metrics=metrics)
    return report, dict(first=first, parameters=initial_parameters, requests=requests, rng_steps=rng_steps)


def run():
    old, original = run_arm(False)
    new, modified = run_arm(True)
    expected_config = dict(old["config"], mixed_backup_lambda=1.0)
    require(new["config"] == expected_config, "V-only changed configuration beyond the V coefficient")
    require(original["parameters"].keys() == modified["parameters"].keys() and all(torch.equal(value,
        modified["parameters"][name]) for name,value in original["parameters"].items()), "Initial full network parameters differ")
    require(original["requests"] == modified["requests"], "Actual replay requests differ")
    for scope in ("real", "model"):
        a,b = original["first"][scope], modified["first"][scope]
        require(a.keys() == b.keys() and all(torch.equal(value,b[name]) for name,value in a.items()), "First "+scope+" batches differ")
    require(torch.equal(original["first"]["weights"], modified["first"]["weights"]), "First model weights differ")
    require(torch.equal(original["first"]["iql_target"], modified["first"]["iql_target"]), "First same-batch IQL targets differ")
    for a,b in zip(original["rng_steps"],modified["rng_steps"]):
        require(torch.equal(a["torch"],b["torch"]) and torch.equal(a["model"],b["model"])
            and a["learner"] == b["learner"] and a["trust"] == b["trust"], "Actual update RNG differs")
    require(old["terminal_targets"] == new["terminal_targets"], "Terminal targets differ between arms")
    report = dict(complete=True, protocol=P["name"], seed=P["training_seeds"][0],
        arms=dict(real=old, real_v_only=new), config_only_lambda_changed=True, initial_parameters_equal=True,
        actual_first_real_and_model_batches_equal=True, first_model_weights_equal=True, replay_requests_equal=True,
        all_four_update_rng_equal=True, v_only_actual_h2oplus_equals_iql=True,
        original_nonterminal_targets_differ=True, corrected_terminal_targets_exact=True,
        checkpoint_files_written=0, new_simulation_events=0, evaluation_jobs=0)
    write(HERE / "native_terminal_v_only/admission.json", report)
    print(json.dumps(dict(complete=True, protocol=P["name"], updates_per_arm=4)), flush=True)


if __name__ == "__main__":
    run()
