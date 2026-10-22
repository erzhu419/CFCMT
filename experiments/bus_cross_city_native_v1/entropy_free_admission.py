"""Paired real-learner admission for the single entropy-backup flag change."""

import copy
import json
from types import MethodType

import numpy as np
import torch

from iql_actor_admission import observe
from iql_actor_continuation import HERE, build_learner
from train_native import write

P = json.loads((HERE / "entropy_free_continuation_protocol.json").read_text())


def run_arm(branch, seed, entropy):
    learner, replay, _ = build_learner(branch, seed, iql_actor=True,
        protocol=P, backup_entropy=entropy)
    result, batches = observe(learner, True)
    targets, independent, generated = dict(real=0, model=0), {}, []
    original_generate = learner.augmentation.generate
    def generate(self, actor):
        batch, weights = original_generate(actor)
        generated.append(batch)
        return batch, weights
    learner.augmentation.generate = MethodType(generate, learner.augmentation)
    original_target = learner._target
    def target(self, batch, *, backup):
        assert self.config.backup_entropy is entropy and backup == "h2oplus"
        scope = "model" if generated and batch is generated[-1] else "real"
        before = torch.get_rng_state().clone()
        actual = original_target(batch, backup=backup)
        if not entropy and targets[scope] == 0:
            with torch.random.fork_rng(devices=[]), torch.no_grad():
                torch.set_rng_state(before)
                saved_alpha = self.log_alpha.detach().clone()
                self.log_alpha.add_(1)
                try:
                    perturbed = original_target(batch, backup=backup)
                finally:
                    self.log_alpha.copy_(saved_alpha)
            assert torch.equal(actual, perturbed), scope
            independent[scope] = True
        targets[scope] += 1
        return actual
    learner._target = MethodType(target, learner)
    initial, online = learner.total_updates, learner.online_critic_updates
    rng_steps, first = [], None
    for step in range(4):
        metrics = learner.update(online=True, collect_metrics=step in (0, 3))
        assert all(np.isfinite(value) for value in metrics.values())
        rng_steps.append(dict(torch=torch.get_rng_state().clone(),
            learner=copy.deepcopy(learner.rng.bit_generator.state)))
        if step == 0:
            first = {name:{key:value.detach().clone() for key,value in
                getattr(learner, name).state_dict().items()} for name in ("actor", "value")}
            first["log_alpha"] = learner.log_alpha.detach().clone()
    assert learner.total_updates-initial == learner.online_critic_updates-online == 4
    assert result["optimizer_steps"] == dict(actor=4, critic=4, value=4, alpha=4)
    assert result["replay_call_counts"]["sample_mixed"] == len(batches) == len(generated) == 4
    assert targets == dict(real=4, model=4)
    result.update(updates=4, backup_entropy=entropy, target_calls=targets,
        actual_model_generation_calls=len(generated),
        alpha_independent_off_targets=independent, model_td_coefficient=learner.model_coefficient)
    del learner, replay
    return result, batches, rng_steps, first


def run():
    seed, branches = 75101, {}
    for branch in ("td0", "td025"):
        on, on_batches, on_rng, on_first = run_arm(branch, seed, True)
        off, off_batches, off_rng, off_first = run_arm(branch, seed, False)
        assert on["replay_requests"] == off["replay_requests"]
        assert on["replay_call_counts"] == off["replay_call_counts"]
        for old, new in zip(on_batches, off_batches):
            assert old["request"] == new["request"]
            assert torch.equal(old["observations"], new["observations"])
            assert torch.equal(old["actions"], new["actions"])
        for old, new in zip(on_rng, off_rng):
            assert torch.equal(old["torch"], new["torch"])
            assert old["learner"] == new["learner"]
        for name in ("actor", "value"):
            assert all(torch.equal(value, off_first[name][key]) for key,value in on_first[name].items())
        assert torch.equal(on_first["log_alpha"], off_first["log_alpha"])
        assert on["actor_steps"][0]["actor_loss"] == off["actor_steps"][0]["actor_loss"]
        branches[branch] = dict(entropy_on=on, entropy_off=off,
            all_four_replay_batches_and_rng_equal=True,
            first_actor_value_and_alpha_updates_equal=True, additional_replay_samples=0)
        print(json.dumps(dict(branch=branch, complete=True)), flush=True)
    write(HERE / "entropy_free_continuation/admission.json", dict(complete=True,
        protocol=P["name"], seed=seed, branches=branches, evaluation_jobs=0,
        checkpoint_files_written=0, interpretation="Interface admission; critic updates may differ."))
    print(json.dumps(dict(complete=True, protocol=P["name"], seed=seed)), flush=True)


if __name__ == "__main__":
    run()
