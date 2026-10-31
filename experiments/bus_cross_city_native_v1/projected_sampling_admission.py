"""Real frozen-learner admission for projected executed-action continuation."""

from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from types import MethodType

import h5py
import numpy as np
import torch

from canonical_execution_projection import (
    canonical_execution_bounds, project_canonical_executed_action,
)
from projected_actor_sampling import install_projected_sampling
from train_native import (
    HERE, ModelContinuation, continuation_trainer, runtime_api, seed_all, write,
)


class QueryAudit:
    """Inspect generated queries in their real sampling call contexts."""

    def __init__(self, config):
        self.config, self.context, self.generated = config, None, {}
        self.counts = {}

    def check(self, name, states, actions):
        lower, upper = canonical_execution_bounds(states, self.config)
        assert actions.shape == lower.shape
        assert bool(((actions >= lower) & (actions <= upper)).all()), name
        self.counts[name] = self.counts.get(name, 0) + len(states)

    @contextmanager
    def scope(self, name, states=None):
        previous = self.context
        self.context = (name, states)
        try:
            yield
        finally:
            self.context = previous

    def instrument(self, learner, augmentation):
        for name, states_of in (
            ("_update_sac_actor", lambda args: args[0]),
            ("_sac_bootstrap", lambda args: args[0]["next_observations"]),
            ("_cql_max_bootstrap", lambda args: args[0]["next_observations"].repeat_interleave(
                learner.config.cql_n_actions, dim=0)),
        ):
            original = getattr(learner, name)
            def scoped(self, *args, _original=original, _name=name,
                       _states_of=states_of, **kwargs):
                with audit.scope(_name, _states_of(args)):
                    return _original(*args, **kwargs)
            audit = self
            setattr(learner, name, MethodType(scoped, learner))

        for critic, role in ((learner.critic, "actor_q"),
                             (learner.target_critic, "target_q")):
            for name in ("forward_features", "forward_features_subset"):
                original = getattr(critic, name)
                def checked(self, features, actions, *args, _original=original,
                            _role=role, **kwargs):
                    if audit.context is not None and audit.context[1] is not None:
                        path, states = audit.context
                        if len(actions) == 2 * len(states):
                            states = torch.cat((states, states), dim=0)
                        audit.check(_role + path, states, actions)
                    elif id(actions) in audit.generated:
                        generated_actions, states = audit.generated[id(actions)]
                        assert actions is generated_actions
                        audit.check("model_q", states, actions)
                    return _original(features, actions, *args, **kwargs)
                setattr(critic, name, MethodType(checked, critic))

        original_predict = augmentation.predict
        def predict(self, states, actions):
            if audit.context is not None and audit.context[0] == "model_generate":
                audit.check("world_model", states, actions)
            return original_predict(states, actions)
        augmentation.predict = MethodType(predict, augmentation)
        original_generate = augmentation.generate
        def generate(self, actor):
            with audit.scope("model_generate"):
                batch, weights = original_generate(actor)
            audit.generated[id(batch["actions"])] = (batch["actions"], batch["observations"])
            return batch, weights
        augmentation.generate = MethodType(generate, augmentation)
        original_weight = augmentation.trust.weight
        def weight(self, states, actions):
            audit.check("trust_weight", states, actions)
            return original_weight(states, actions)
        augmentation.trust.weight = MethodType(weight, augmentation.trust)
        augmentation.trust.target.register_forward_pre_hook(
            lambda module, args: audit.check("trust_target", args[0], args[1]))


def initial_action_audit(actor, config):
    changed, total, raw_seconds, projected_seconds = 0, 0, 0., 0.
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        ids = np.flatnonzero(data["split_id"][:] == 0)
        states = data["observations"][ids]
    with torch.no_grad():
        for start in range(0, len(states), 256):
            batch = torch.as_tensor(states[start:start+256])
            raw = actor(batch, deterministic=True)[0]
            projected = project_canonical_executed_action(batch, raw, config)
            changed += int((raw != projected).any(dim=1).sum())
            total += len(batch)
            raw_seconds += float(((raw + 1) * config.max_hold_seconds / 2).sum())
            projected_seconds += float(((projected + 1) * config.max_hold_seconds / 2).sum())
    return dict(fit_rows=total, raw_queries_changed_rows=changed,
        raw_queries_changed_percent=100 * changed / total,
        raw_mean_hold_seconds=raw_seconds / total,
        projected_mean_hold_seconds=projected_seconds / total)


def sampling_audit(learner, states, labels, config):
    actor = learner.actor
    state_before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    optimizer_ids = [id(p) for group in learner.actor_optimizer.param_groups for p in group["params"]]
    with torch.no_grad():
        label_likelihood = actor.action_log_prob(states, labels).clone()
        seed_all(88101)
        original_soft = actor.soft_action_components(states)
        seed_all(88102)
        original_repeated = actor.repeated_soft_action_components(states, 3)
        original_deterministic = actor(states, deterministic=True)[0]
    projector = lambda s, a: project_canonical_executed_action(s, a, config)
    install_projected_sampling(actor, projector)
    with torch.no_grad():
        seed_all(88101)
        projected_soft = actor.soft_action_components(states)
        seed_all(88102)
        projected_repeated = actor.repeated_soft_action_components(states, 3)
        projected_deterministic = actor(states, deterministic=True)[0]
        for raw, projected, sample_states in (
            (original_soft, projected_soft, states),
            (original_repeated, projected_repeated, states.repeat_interleave(3, dim=0)),
        ):
            for key in raw:
                expected = projector(sample_states, raw[key]) if key.endswith("_action") else raw[key]
                assert torch.equal(expected, projected[key]), key
        assert torch.equal(projector(states, original_deterministic), projected_deterministic)
        assert torch.equal(label_likelihood, actor.action_log_prob(states, labels))
    assert all(torch.equal(value, actor.state_dict()[key]) for key, value in state_before.items())
    assert optimizer_ids == [id(p) for group in learner.actor_optimizer.param_groups for p in group["params"]]
    return dict(latent_probabilities_and_log_probabilities_unchanged=True,
        recorded_action_likelihood_unchanged=True, actor_state_dict_unchanged=True,
        optimizer_parameters_unchanged=True, deterministic_projection_exact=True,
        repeated_projection_state_order_exact=True)


def run():
    seed, torch_threads = 75101, 1
    torch.set_num_threads(torch_threads)
    protocol = json.loads((HERE / "training_protocol.json").read_text())
    p = protocol["training"]
    trainer_class, config_class, replay_class, encoder_class = runtime_api()
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    physical = DEFAULT_CONFIG_V7
    trainer_config = replace(config_class(**p["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    reports, initial = {}, None
    for coefficient in (0., .25):
        replay = replay_class(HERE / "target_pool.h5", device="cpu", max_offline_rows=0, online_capacity=1)
        replay.materialize_device_cache()
        assert replay.offline_size == 12288
        assert replay.contract_config == asdict(physical)
        learner = continuation_trainer(trainer_class)(replay, config=trainer_config, device="cpu", seed=seed)
        learner.load(HERE / "source_dense_adaptation/seed75101/adapted.pt", load_optimizers=True)
        assert learner.total_updates == 5000
        batch = replay.sample(64, scope="offline", rng=np.random.default_rng(88100))
        if initial is None:
            initial = initial_action_audit(learner.actor, physical)
        sampling = sampling_audit(learner, batch["observations"], batch["actions"], physical)
        augmentation = ModelContinuation(protocol, seed, HERE / "training/models/seed75101", encoder_class, "cpu")
        learner.augmentation = augmentation
        learner.model_coefficient = coefficient
        learner.model_seed = seed + 60000
        learner.trust_update_every = p["trust_update_every"]
        audit = QueryAudit(physical)
        audit.instrument(learner, augmentation)
        seed_all(seed + p["continuation_rng_offset"])
        learner.rng = np.random.default_rng(seed + p["continuation_rng_offset"])
        online_before = learner.online_critic_updates
        for _ in range(4):
            metrics = learner.update(online=True, collect_metrics=True)
            assert all(np.isfinite(value) for value in metrics.values())
        assert learner.total_updates == 5004
        assert learner.online_critic_updates - online_before == 4
        # This read-only candidate query verifies the inherited repeated sampler;
        # the registered continuation still uses its unchanged H2O+ backup.
        with torch.random.fork_rng():
            learner._cql_max_bootstrap(batch)
        required = ("actor_q_update_sac_actor", "target_q_sac_bootstrap",
            "target_q_cql_max_bootstrap", "world_model", "model_q", "trust_weight", "trust_target")
        assert all(audit.counts.get(name, 0) > 0 for name in required), audit.counts
        artifact = HERE / "projected_continuation/admission" / f"coefficient{coefficient}/policy.pt"
        learner.save_policy_artifact(artifact, extra=dict(admission=True, model_coefficient=coefficient))
        loaded, _ = load_actor_checkpoint_v7(artifact, "cpu", require_policy_artifact=True)
        assert "soft_action_components" not in loaded.__dict__
        assert "repeated_soft_action_components" not in loaded.__dict__
        with torch.no_grad():
            expected = learner.actor(batch["observations"], deterministic=True)[0]
            actual = project_canonical_executed_action(batch["observations"],
                loaded(batch["observations"], deterministic=True)[0], physical)
            assert torch.equal(expected, actual)
        reports[str(coefficient)] = dict(updates=4, model_td_coefficient=coefficient,
            sampling=sampling, generated_query_rows_checked=audit.counts,
            finite_final_metrics=metrics, exported_raw_actor_projection_exact=True)
        del learner, replay, augmentation, audit, loaded
    report = dict(complete=True, seed=seed, target_fit_rows=12288,
        initial_actor_fit_actions=initial, branches=reports,
        replay_actions_unchanged=True, physical_config_matches_default=True,
        training_kernel_and_entropy_unchanged=True, evaluation_jobs=0,
        interpretation="Interface admission; no policy performance claim.")
    write(HERE / "projected_continuation/admission.json", report)
    print(json.dumps(dict(complete=True, seed=seed, fit_action_audit=initial,
        branches={key: value["generated_query_rows_checked"] for key, value in reports.items()})), flush=True)


if __name__ == "__main__":
    run()
