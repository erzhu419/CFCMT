"""Paired four-update admission for masking only the collapsed actor weights."""

import copy
import json
from types import MethodType

import numpy as np
import torch

from canonical_execution_projection import canonical_execution_bounds
from iql_actor_admission import first_step_state, observe
from iql_actor_continuation import HERE, build_learner
from train_native import write

P = json.loads((HERE / "free_execution_actor_protocol.json").read_text())


def run_arm(branch, seed, masked):
    learner, replay, _ = build_learner(branch, seed, protocol=P, backup_entropy=False,
        free_execution_actor=masked)
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    # Observe the frozen likelihood implementation below the masking subclass.
    native_owner = next(cls for cls in type(learner).__mro__
        if "_update_iql_actor" in cls.__dict__ and cls.__module__.startswith("SimpleSAC."))
    original_native = native_owner.__dict__["_update_iql_actor"]
    actor_calls, target_rows, generated = [], dict(real=[], model=[]), []
    def native_actor(self, batch, weights, *, collect_metrics=True):
        recorded_batch, original_weights = self._recorded_actor_update
        assert batch is recorded_batch
        lower, upper = canonical_execution_bounds(batch["observations"], DEFAULT_CONFIG_V7)
        free = upper[:, 0] > lower[:, 0]
        expected = original_weights * free.to(weights.dtype) if masked else original_weights
        assert torch.equal(weights, expected)
        assert torch.equal(weights[free], original_weights[free])
        if masked:
            assert bool((weights[~free] == 0).all())
        before = original_weights.clone()
        loss, norm = original_native(self, batch, weights, collect_metrics=collect_metrics)
        assert torch.equal(original_weights, before)
        actor_calls.append(dict(rows=len(weights), free_rows=int(free.sum()),
            collapsed_rows=int((~free).sum()), actual_collapsed_weight_sum=float(weights[~free].sum()),
            original_collapsed_weight_sum=float(original_weights[~free].sum()),
            original_batch_mean_denominator=len(weights), actual_native_mask_exact=True))
        return loss, norm
    native_owner._update_iql_actor = native_actor
    result, batches = observe(learner, True, free_execution_actor=masked)
    original_generate = learner.augmentation.generate
    def generate(self, actor):
        batch, weights = original_generate(actor)
        generated.append(batch)
        return batch, weights
    learner.augmentation.generate = MethodType(generate, learner.augmentation)
    original_target = learner._target
    def target(self, batch, *, backup):
        assert not self.config.backup_entropy
        scope = "model" if generated and batch is generated[-1] else "real"
        target_rows[scope].append(len(batch["observations"]))
        return original_target(batch, backup=backup)
    learner._target = MethodType(target, learner)
    initial, online = learner.total_updates, learner.online_critic_updates
    first, rng_steps = None, []
    try:
        for step in range(4):
            metrics = learner.update(online=True, collect_metrics=step in (0, 3))
            assert all(np.isfinite(value) for value in metrics.values())
            if step == 0:
                first = first_step_state(learner)
            rng_steps.append(dict(torch_rng=torch.get_rng_state().clone(),
                learner_rng=copy.deepcopy(learner.rng.bit_generator.state)))
    finally:
        native_owner._update_iql_actor = original_native
    assert learner.total_updates-initial == learner.online_critic_updates-online == 4
    assert result["optimizer_steps"] == dict(actor=4, critic=4, value=4, alpha=4)
    assert result["replay_call_counts"]["sample_mixed"] == len(batches) == len(actor_calls) == len(generated) == 4
    assert len(target_rows["real"]) == len(target_rows["model"]) == 4
    assert target_rows["real"] == [len(batch["observations"]) for batch in batches]
    assert target_rows["model"] == [len(batch["observations"]) for batch in generated]
    assert [item["rows"] for item in result["batch_weights"]] == target_rows["real"]
    result.update(updates=4, free_execution_actor=masked, backup_entropy=False,
        actor_native_calls=actor_calls, target_rows=target_rows, full_replay_labels_and_iql_weights_preserved=True,
        model_td_coefficient=learner.model_coefficient, final_alpha=float(learner.alpha.detach()))
    del learner, replay
    return result, first, dict(batches=batches, rng_steps=rng_steps)


def run():
    seed, branches = 75101, {}
    for branch in ("td0", "td025"):
        old, old_first, old_observed = run_arm(branch, seed, False)
        new, new_first, new_observed = run_arm(branch, seed, True)
        assert old["replay_requests"] == new["replay_requests"]
        assert old["replay_call_counts"] == new["replay_call_counts"]
        assert old["target_rows"] == new["target_rows"]
        for left, right in zip(old_observed["batches"], new_observed["batches"]):
            assert left["request"] == right["request"]
            assert torch.equal(left["observations"], right["observations"])
            assert torch.equal(left["actions"], right["actions"])
        for left, right in zip(old_observed["rng_steps"], new_observed["rng_steps"]):
            assert torch.equal(left["torch_rng"], right["torch_rng"])
            assert left["learner_rng"] == right["learner_rng"]
        for key in ("pre_actor_td_loss", "pre_actor_model_td_loss", "alpha_loss"):
            assert old["actor_steps"][0][key] == new["actor_steps"][0][key]
        assert torch.equal(old_first["log_alpha"], new_first["log_alpha"])
        for name in ("critic", "value"):
            assert all(torch.equal(value, new_first[name][key]) for key,value in old_first[name].items())
        branches[branch] = dict(mask_off=old, mask_on=new, all_four_replay_and_rng_equal=True,
            first_critic_value_alpha_and_pre_actor_losses_equal=True, additional_replay_samples=0)
        print(json.dumps(dict(branch=branch, complete=True)), flush=True)
    write(HERE / "free_execution_actor/admission.json", dict(complete=True,
        protocol=P["name"], seed=seed, branches=branches, evaluation_jobs=0, checkpoint_files_written=0,
        interpretation="Actor-only interface admission; later learned values may differ."))
    print(json.dumps(dict(complete=True, protocol=P["name"], seed=seed)), flush=True)


if __name__ == "__main__":
    run()
