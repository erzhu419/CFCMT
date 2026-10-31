"""Keep the real continuation recipe and change only the V backup coefficient."""

import argparse
import copy
from dataclasses import asdict, replace
import json

import torch

from native_terminal_continuation import (P as REFERENCE, build_learner as build_real,
    count_optimizer_steps, fit_terminal_rows, terminal_diagnostic)
from train_native import HERE, update_loop, write

PREFIX, BRANCH = "native_terminal_v_only", "real_v_only"
P = json.loads((HERE / (PREFIX+"_protocol.json")).read_text())


def build_learner(seed):
    if seed not in P["training_seeds"]:
        raise ValueError("Seed is outside the fixed roster")
    expected = copy.deepcopy(REFERENCE["training_settings"])
    expected["trainer_config"]["mixed_backup_lambda"] = 1.0
    actual = copy.deepcopy(P["training_settings"])
    expected.pop("continuation_backup")
    actual.pop("continuation_backup")
    if actual != expected:
        raise RuntimeError("Frozen settings changed more than the V coefficient")
    for key in ("training_seeds", "initialization_updates", "adaptation_updates", "continuation_updates",
            "target_fit_rows", "target_union_rows", "source_rows", "target_pool", "initializer_folder",
            "world_model_folder", "world_model_data_pool", "world_model_refit_updates", "target_online_training_events",
            "actor_objective", "backup_entropy", "free_execution_actor_mask", "sampled_action_space", "entropy_space"):
        if P[key] != REFERENCE[key]:
            raise RuntimeError(f"Frozen continuation changed {key}")
    learner, replay, original = build_real("real", seed)
    if original != {"training": REFERENCE["training_settings"]}:
        raise RuntimeError("Reference build did not use its frozen settings")
    before = asdict(learner.config)
    if before["mixed_backup_lambda"] != 0.1 or learner.model_coefficient != 0.:
        raise RuntimeError("Reference is not the original real continuation")
    learner.config = replace(learner.config, mixed_backup_lambda=1.0)
    changed = {key for key, value in asdict(learner.config).items() if value != before[key]}
    if changed != {"mixed_backup_lambda"}:
        raise RuntimeError("Actual learner config changed more than the V coefficient")
    spec = P["branches"][BRANCH]
    if learner.model_coefficient != spec["model_td_coefficient"] or learner.augmentation.weight_mode != spec["model_weight_mode"]:
        raise RuntimeError("Shadow model coefficient or weight mode changed")
    return learner, replay, {"training": P["training_settings"]}


def train(seed):
    learner, replay, original = build_learner(seed)
    initial_updates, initial_critic = learner.total_updates, learner.online_critic_updates
    initial_log_alpha = float(learner.log_alpha.detach())
    counts = count_optimizer_steps(learner)
    settings, spec, count = original["training"], P["branches"][BRANCH], P["continuation_updates"]
    folder = HERE / spec["artifact_folder"] / f"seed{seed}"
    print(json.dumps(dict(phase="initialized", protocol=P["name"], branch=BRANCH, seed=seed,
        stored_updates=initial_updates, historical_updates=P["initialization_updates"]+initial_updates,
        target_fit_rows=replay.offline_size, mixed_backup_lambda=1., backup_entropy=False,
        model_td_coefficient=learner.model_coefficient, model_weight_mode=learner.augmentation.weight_mode)), flush=True)
    update_loop(learner, count, BRANCH, folder, online=True, log_every=settings["log_every"],
        offset=P["initialization_updates"]+P["adaptation_updates"])
    if learner.total_updates != initial_updates+count or learner.online_critic_updates-initial_critic != count or counts != {name: count for name in counts}:
        raise RuntimeError("Actual continuation optimizer/update counts differ from the frozen budget")
    report = dict(complete=True, protocol=P["name"], branch=BRANCH, seed=seed,
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
        original_mixed_backup_lambda=0.1, single_config_field_changed="mixed_backup_lambda",
        passenger_discount_gamma=1., terminal_discount=0.,
        fit_terminal_before=learner.fit_terminal_before,
        fit_terminal_after=terminal_diagnostic(learner, fit_terminal_rows(HERE / P["target_pool"])),
        continuation_model_td_coefficient=learner.model_coefficient, continuation_real_td_coefficient=1.,
        model_weight_mode=learner.augmentation.weight_mode,
        model_td_weight_reduction="Mean of weight times elementwise Huber; no sum-of-weights division",
        confidence_definition=P["confidence_definition"], comparison_boundary=P["comparison_boundary"],
        trust_discount_gamma=float(learner.augmentation.trust.gamma),
        model_generation_and_trust_enabled_in_all_arms=True, shadow_model_enabled=True,
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
    parser.add_argument("--branch", required=True, choices=(BRANCH,))
    parser.add_argument("--seed", type=int, required=True, choices=P["training_seeds"])
    train(parser.parse_args().seed)
