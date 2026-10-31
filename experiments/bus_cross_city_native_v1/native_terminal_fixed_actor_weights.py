"""Retain the unmasked V-only continuation, fixing only actor IQL scores at55k."""

import argparse
import copy
from dataclasses import asdict
import json
from types import MethodType

import torch

from native_terminal_actor_unmasked import P as TEMPLATE, build_learner as build_unmasked
from native_terminal_continuation import count_optimizer_steps, fit_terminal_rows, terminal_diagnostic
from train_native import HERE, update_loop, write

PREFIX = "native_terminal_fixed_actor_weights"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
BRANCH = P["branch"]


@torch.no_grad()
def fixed_actor_weights(learner, batch):
    target_q = learner.fixed_actor_target_critic(batch["observations"], batch["actions"]).min(0).values
    value = learner.fixed_actor_value(batch["observations"])
    return torch.exp((target_q - value) * learner.fixed_actor_beta).clamp_max(
        learner.fixed_actor_max_weight).detach()


def fixed_actor_update(self, batch, weights, *, collect_metrics=True):
    self.fixed_actor_weight_updates += 1
    return self._fixed_actor_native_update(batch, fixed_actor_weights(self, batch),
        collect_metrics=collect_metrics)


def scorer_unchanged(learner):
    for name in ("target_critic", "value"):
        scorer = getattr(learner, "fixed_actor_" + name)
        initial, actual = learner.fixed_actor_scorer_initial[name], scorer.state_dict()
        if (scorer.training or initial.keys() != actual.keys()
                or any(not torch.equal(value, actual[key]) for key, value in initial.items())
                or any(parameter.requires_grad or parameter.grad is not None for parameter in scorer.parameters())):
            return False
    return True


def build_learner(seed):
    if (seed not in P["training_seeds"] or P["training_seeds"] != TEMPLATE["training_seeds"]
            or P["training_template_protocol"] != "native_terminal_actor_unmasked_protocol.json"
            or P["continuation_updates_per_seed"] != TEMPLATE["continuation_updates"]
            or P["historical_final_updates"] != TEMPLATE["initialization_updates"]
                + TEMPLATE["adaptation_updates"] + TEMPLATE["continuation_updates"]):
        raise ValueError("Fixed-weight continuation differs from the frozen training template")
    learner, replay, original = build_unmasked(seed)
    if (original != {"training": TEMPLATE["training_settings"]}
            or learner.config.iql_advantage_temperature != 3. or learner.config.iql_max_weight != 100.):
        raise RuntimeError("Original continuation settings or native IQL constants changed")
    learner.fixed_actor_beta = learner.config.iql_advantage_temperature
    learner.fixed_actor_max_weight = learner.config.iql_max_weight
    learner.fixed_actor_weight_updates = 0
    learner.fixed_actor_scorer_initial = {}
    for name in ("target_critic", "value"):
        scorer = copy.deepcopy(getattr(learner, name)).eval().requires_grad_(False)
        setattr(learner, "fixed_actor_" + name, scorer)
        learner.fixed_actor_scorer_initial[name] = {
            key: value.detach().clone() for key, value in scorer.state_dict().items()}
    learner._fixed_actor_native_update = learner._update_iql_actor
    learner._update_iql_actor = MethodType(fixed_actor_update, learner)
    return learner, replay, original


def train(seed):
    learner, replay, original = build_learner(seed)
    initial, initial_critic = learner.total_updates, learner.online_critic_updates
    initial_log_alpha = float(learner.log_alpha.detach())
    counts = count_optimizer_steps(learner)
    settings, count = original["training"], P["continuation_updates_per_seed"]
    folder = HERE / PREFIX / BRANCH / f"seed{seed}"
    print(json.dumps(dict(phase="initialized", protocol=P["name"], branch=BRANCH, seed=seed,
        stored_updates=initial, historical_updates=TEMPLATE["initialization_updates"] + initial,
        actor_weight_source="frozen55_target_critic_and_value", actor_weight_beta=3.,
        actor_weight_max=100., target_fit_rows=replay.offline_size, mixed_backup_lambda=1.,
        backup_entropy=False, free_execution_actor_mask=False, model_td_coefficient=0.)), flush=True)
    update_loop(learner, count, BRANCH, folder, online=True, log_every=settings["log_every"],
        offset=TEMPLATE["initialization_updates"] + TEMPLATE["adaptation_updates"])
    if (learner.total_updates != initial + count or learner.online_critic_updates - initial_critic != count
            or counts != {name: count for name in counts} or learner.fixed_actor_weight_updates != count
            or not scorer_unchanged(learner)):
        raise RuntimeError("Actual update budget or frozen actor scorer changed")
    initializer = str(HERE / TEMPLATE["initializer_folder"] / f"seed{seed}" / "adapted.pt")
    report = dict(complete=True, protocol=P["name"], branch=BRANCH, seed=seed,
        initializer=initializer, optimizer_transfer=True, target_pool=str(HERE / TEMPLATE["target_pool"]),
        target_fit_rows=replay.offline_size, target_union_rows=TEMPLATE["target_union_rows"],
        source_rows=TEMPLATE["source_rows"], initialization_updates=TEMPLATE["initialization_updates"],
        adaptation_updates=TEMPLATE["adaptation_updates"], continuation_updates=count,
        historical_policy_updates=TEMPLATE["initialization_updates"] + learner.total_updates,
        loaded_stored_updates=initial, stored_total_updates=learner.total_updates,
        optimizer_steps=counts, online_critic_updates_added=learner.online_critic_updates - initial_critic,
        config=asdict(learner.config), actor_objective=TEMPLATE["actor_objective"],
        actor_weight_source="frozen55_target_critic_and_value", actor_weight_scorer_initializer=initializer,
        actor_weight_beta=learner.fixed_actor_beta, actor_weight_max=learner.fixed_actor_max_weight,
        fixed_actor_weight_updates=learner.fixed_actor_weight_updates,
        actor_weight_formula=P["actor_weights"], frozen_actor_scorer_unchanged=True,
        free_execution_actor_mask=False, sampled_action_space=TEMPLATE["sampled_action_space"],
        entropy_space=TEMPLATE["entropy_space"], single_intervention="freeze_actor_advantage_scorer_at_initial55k",
        config_equal_to_unmasked=True, reference_mixed_backup_lambda=1.,
        temperature_algorithm="Original SAC latent-temperature update retained", backup_entropy=False,
        effective_critic_backup="h2oplus", mixed_backup_lambda=learner.config.mixed_backup_lambda,
        passenger_discount_gamma=1., terminal_discount=0., fit_terminal_before=learner.fit_terminal_before,
        fit_terminal_after=terminal_diagnostic(learner, fit_terminal_rows(HERE / TEMPLATE["target_pool"])),
        continuation_model_td_coefficient=learner.model_coefficient, continuation_real_td_coefficient=1.,
        model_weight_mode=learner.augmentation.weight_mode,
        model_td_weight_reduction="Mean of weight times elementwise Huber; no sum-of-weights division",
        confidence_definition=TEMPLATE["confidence_definition"], comparison_boundary=P["interpretation"],
        trust_discount_gamma=float(learner.augmentation.trust.gamma),
        model_generation_and_trust_enabled_in_all_arms=True, shadow_model_enabled=True,
        world_model_artifact=str(HERE / TEMPLATE["world_model_folder"] / f"seed{seed}" / "model.pt"),
        world_model_data_pool=str(HERE / TEMPLATE["world_model_data_pool"]), world_model_refit_updates=0,
        trust_update_every=learner.trust_update_every, model_seed=learner.model_seed,
        continuation_rng_seed=seed + settings["continuation_rng_offset"], target_online_training_events=0,
        log_alpha_before=initial_log_alpha, log_alpha_after=float(learner.log_alpha.detach()),
        actor_export_equal=None, checkpoint_selection="fixed final", interpretation=P["interpretation"])
    artifact = folder / "policy.pt"
    learner.save_policy_artifact(artifact, extra=report)
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    actor, _ = load_actor_checkpoint_v7(artifact, "cpu", require_policy_artifact=True)
    expected, actual = learner.actor.state_dict(), actor.state_dict()
    if expected.keys() != actual.keys() or any(not torch.equal(actual[key], value) for key, value in expected.items()):
        raise RuntimeError("Exported deployment actor differs from the final continuation actor")
    report["actor_export_equal"] = True
    learner.save(folder / "final_learner.pt", extra=report, include_online_replay=False)
    write(folder / "train_summary.json", report)
    del actor, learner, replay
    from iql_actor_continuation import evaluate
    evaluate(BRANCH, seed, protocol=P, prefix=PREFIX)
    write(folder / "summary.json", dict(report, evaluation_cases=len(P["evaluation"]["cases"])))
    print(json.dumps(dict(complete=True, branch=BRANCH, seed=seed, continuation_updates=count)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train",))
    parser.add_argument("--seed", type=int, required=True, choices=P["training_seeds"])
    train(parser.parse_args().seed)
