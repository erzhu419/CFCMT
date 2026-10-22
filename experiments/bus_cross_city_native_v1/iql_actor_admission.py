"""Paired real-learner admission for the actor-only IQL intervention."""

import copy
import json
from types import MethodType

import numpy as np
import torch

from iql_actor_continuation import HERE, P, build_learner
from train_native import write


def observe(learner, iql_actor, *, free_execution_actor=False):
    result = dict(optimizer_steps={}, actor_steps=[], batch_weights=[], td_losses=[])
    current, sampled = {}, []
    replay = learner.replay_buffer
    result["replay_call_counts"] = dict(sample_mixed=0, sample=0)
    result["replay_requests"] = []
    for name in ("sample_mixed", "sample"):
        original = getattr(replay, name)
        def sample(self, *args, _original=original, _name=name, **kwargs):
            request = dict(method=_name, args=args,
                kwargs={key:value for key,value in kwargs.items() if key != "rng"},
                rng_is_learner_rng=kwargs.get("rng") is learner.rng)
            result["replay_call_counts"][_name] += 1
            result["replay_requests"].append(request)
            batch = _original(*args, **kwargs)
            if _name == "sample_mixed":
                offline, online = batch
                assert offline is not None and online is None
                sampled.append(dict(request=request,
                    observations=offline["observations"].clone(),
                    actions=offline["actions"].clone()))
            return batch
        setattr(replay, name, MethodType(sample, replay))
    for name in ("actor", "critic", "value", "alpha"):
        optimizer = getattr(learner, name + "_optimizer")
        original = optimizer.step
        result["optimizer_steps"][name] = 0
        def step(*args, _original=original, _name=name, **kwargs):
            result["optimizer_steps"][_name] += 1
            return _original(*args, **kwargs)
        optimizer.step = step

    original_value_terms = learner._iql_value_terms
    def value_terms(self, batch):
        loss, advantage, weights = original_value_terms(batch)
        expected = torch.exp(advantage * self.config.iql_advantage_temperature).clamp_max(
            self.config.iql_max_weight)
        assert torch.equal(weights, expected) and not weights.requires_grad
        current.update(batch=batch, labels=batch["actions"].clone(), weights=weights.clone())
        result["batch_weights"].append(dict(mean=float(weights.mean()),
            max=float(weights.max()), rows=len(weights)))
        return loss, advantage, weights
    learner._iql_value_terms = MethodType(value_terms, learner)

    original_td_loss = learner._ensemble_td_loss
    def td_loss(self, predictions, targets, weights=None):
        loss = original_td_loss(predictions, targets, weights)
        assert bool(torch.isfinite(loss))
        result["td_losses"].append(float(loss.detach()))
        return loss
    learner._ensemble_td_loss = MethodType(td_loss, learner)

    if iql_actor:
        original_iql_actor = learner._update_iql_actor
        def iql_update(self, batch, weights, *, collect_metrics=True):
            assert batch is current["batch"]
            assert torch.equal(batch["actions"], current["labels"])
            assert torch.equal(weights, current["weights"])
            assert "action_log_prob" not in self.actor.__dict__
            with torch.no_grad():
                expected_weights = weights
                if free_execution_actor:
                    from canonical_execution_projection import canonical_execution_bounds
                    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
                    lower, upper = canonical_execution_bounds(batch["observations"], DEFAULT_CONFIG_V7)
                    expected_weights = weights * (upper[:, 0] > lower[:, 0]).to(weights.dtype)
                expected = -(expected_weights * self.actor.action_log_prob(
                    batch["observations"], batch["actions"])).mean()
            loss, norm = original_iql_actor(batch, weights, collect_metrics=collect_metrics)
            assert torch.equal(loss.detach(), expected)
            assert torch.equal(batch["actions"], current["labels"])
            current["iql_loss"] = float(loss.detach())
            return loss, norm
        learner._update_iql_actor = MethodType(iql_update, learner)

    original_actor = learner._update_sac_actor
    def actor_update(self, actor_states, *, temperature_states=None,
                     temperature_action_samples=1, collect_metrics=True):
        assert self.config.temperature_state_source == "actor"
        assert temperature_action_samples == 1
        assert actor_states.data_ptr() == current["batch"]["observations"].data_ptr()
        # Reproduce the pre-step temperature sample without consuming a draw
        # from the real update's RNG stream or requesting another replay batch.
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            components = self.actor.soft_action_components(actor_states)
            log_probability = (components["zero_probability"] * components["zero_log_prob"]
                + components["hold_probability"] * components["positive_log_prob"])
        expected_log_alpha = self.log_alpha.detach().clone().requires_grad_(True)
        expected_alpha_loss = -(expected_log_alpha * (
            log_probability + self.config.target_entropy)).mean()
        expected_alpha_grad = torch.autograd.grad(expected_alpha_loss, expected_log_alpha)[0]
        model_loss = float(self.model_loss.detach())
        real_td = result["td_losses"][-1]
        loss, alpha_loss, norm = original_actor(actor_states,
            temperature_states=temperature_states,
            temperature_action_samples=temperature_action_samples,
            collect_metrics=collect_metrics)
        assert torch.equal(alpha_loss.detach(), expected_alpha_loss.detach())
        assert torch.equal(self.log_alpha.grad, expected_alpha_grad)
        assert torch.equal(current["batch"]["actions"], current["labels"])
        result["actor_steps"].append(dict(actor_loss=float(loss.detach()),
            alpha_loss=float(alpha_loss.detach()), pre_actor_td_loss=real_td,
            pre_actor_model_td_loss=model_loss, alpha_pre_step_sample_exact=True,
            recorded_labels_unchanged=True, collect_metrics=collect_metrics,
            iql_weighted_likelihood_exact=bool(iql_actor)))
        return loss, alpha_loss, norm
    learner._update_sac_actor = MethodType(actor_update, learner)
    return result, sampled


def first_step_state(learner):
    return dict(torch_rng=torch.get_rng_state().clone(),
        learner_rng=copy.deepcopy(learner.rng.bit_generator.state),
        log_alpha=learner.log_alpha.detach().clone(),
        critic={key:value.detach().clone() for key,value in learner.critic.state_dict().items()},
        value={key:value.detach().clone() for key,value in learner.value.state_dict().items()})


def run_arm(branch, seed, *, iql_actor):
    learner, replay, _ = build_learner(branch, seed, iql_actor=iql_actor)
    initial_updates, initial_online = learner.total_updates, learner.online_critic_updates
    result, sampled = observe(learner, iql_actor)
    first, rng_steps = None, []
    for step in range(4):
        metrics = learner.update(online=True, collect_metrics=step in (0, 3))
        assert all(np.isfinite(value) for value in metrics.values())
        if step == 0:
            first = first_step_state(learner)
        rng_steps.append(dict(torch_rng=torch.get_rng_state().clone(),
            learner_rng=copy.deepcopy(learner.rng.bit_generator.state)))
    assert learner.total_updates - initial_updates == 4
    assert learner.online_critic_updates - initial_online == 4
    assert result["optimizer_steps"] == dict(actor=4, critic=4, value=4, alpha=4)
    assert len(result["batch_weights"]) == len(result["actor_steps"]) == 4
    assert result["replay_call_counts"]["sample_mixed"] == len(sampled) == 4
    result.update(updates=4, model_td_coefficient=learner.model_coefficient,
        actor_objective="recorded_iql" if iql_actor else "original_sac",
        final_alpha=float(learner.alpha.detach()), raw_recorded_action_likelihood=True)
    del learner, replay
    return result, first, dict(sampled_batches=sampled, rng_steps=rng_steps)


def run():
    seed, branches = 75101, {}
    for branch in ("td0", "td025"):
        original, original_first, original_observed = run_arm(branch, seed, iql_actor=False)
        modified, modified_first, modified_observed = run_arm(branch, seed, iql_actor=True)
        assert original["replay_call_counts"] == modified["replay_call_counts"]
        assert original["replay_requests"] == modified["replay_requests"]
        for old, new in zip(original_observed["sampled_batches"], modified_observed["sampled_batches"]):
            assert old["request"] == new["request"]
            assert torch.equal(old["observations"], new["observations"])
            assert torch.equal(old["actions"], new["actions"])
        for old, new in zip(original_observed["rng_steps"], modified_observed["rng_steps"]):
            assert torch.equal(old["torch_rng"], new["torch_rng"])
            assert old["learner_rng"] == new["learner_rng"]
        for key in ("pre_actor_td_loss", "pre_actor_model_td_loss", "alpha_loss"):
            assert original["actor_steps"][0][key] == modified["actor_steps"][0][key]
        assert torch.equal(original_first["log_alpha"], modified_first["log_alpha"])
        for name in ("critic", "value"):
            assert all(torch.equal(value, modified_first[name][key])
                       for key,value in original_first[name].items())
        branches[branch] = dict(original_sac=original, actor_only_iql=modified,
            first_pre_actor_td_and_model_loss_equal=True,
            first_alpha_equal=True, first_global_torch_rng_equal=True,
            first_learner_rng_equal=True, first_critic_and_value_equal=True,
            all_four_replay_batches_and_requests_equal=True,
            all_four_global_torch_and_learner_rng_equal=True,
            additional_replay_calls=0)
        print(json.dumps(dict(branch=branch, complete=True, paired_updates=4)), flush=True)
    report = dict(complete=True, protocol=P["name"], seed=seed, branches=branches,
        target_fit_rows=P["target_fit_rows"], evaluation_jobs=0, checkpoint_files_written=0,
        additional_replay_samples=0, extra_value_updates=0,
        temperature_frozen=False, existing_temperature_update_preserved=True,
        first_step_rng_delta=False,
        interpretation="Paired interface admission, not a policy performance comparison.")
    write(HERE / "iql_actor_continuation/admission.json", report)
    print(json.dumps(dict(complete=True, protocol=P["name"], seed=seed,
        branches=list(branches), updates_per_arm=4)), flush=True)


if __name__ == "__main__":
    run()
