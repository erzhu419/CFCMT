"""Read-only entropy ON/OFF backup decomposition on the original fit states."""

import argparse
from dataclasses import replace
import json

import h5py
import numpy as np
import torch

from canonical_execution_projection import canonical_execution_bounds, project_canonical_executed_action
from diagnose_values import learner_from_payload
from projected_actor_sampling import install_projected_sampling
from train_native import HERE, runtime_api, write

P = json.loads((HERE / "iql_backup_diagnostic_protocol.json").read_text())


@torch.no_grad()
def target_parts(learner, batch, repeats):
    following = batch["next_observations"]
    actor = learner.actor
    part = (actor.soft_action_components(following) if repeats == 1 else
            actor.repeated_soft_action_components(following, repeats))
    features = learner.target_critic.encode_states(following).repeat_interleave(repeats, dim=0)
    combined_actions = torch.cat((part["zero_action"], part["positive_action"]), dim=0)
    chosen = learner._sample_target_subset()
    q = learner.target_critic.forward_features_subset(
        torch.cat((features, features), dim=0), combined_actions, chosen).min(0).values
    q0, qp = q.chunk(2)
    expected_q = (part["zero_probability"] * q0 + part["hold_probability"] * qp).reshape(-1, repeats)
    expected_log = (part["zero_probability"] * part["zero_log_prob"]
        + part["hold_probability"] * part["positive_log_prob"]).reshape(-1, repeats)
    discount = batch["discounts"].reshape(-1) * (1 - batch["terminals"].reshape(-1))
    mixing = learner.config.mixed_backup_lambda
    reward = learner.config.reward_scale * batch["rewards"].reshape(-1)
    value = learner.value(following)
    value_term = discount * mixing * value
    q_term = discount[:, None] * (1 - mixing) * expected_q
    entropy_term = -discount[:, None] * (1 - mixing) * learner.alpha.detach() * expected_log
    soft_q = (part["zero_probability"] * (q0-learner.alpha.detach()*part["zero_log_prob"])
        + part["hold_probability"] * (qp-learner.alpha.detach()*part["positive_log_prob"])).reshape(-1, repeats)
    off = reward[:, None] + discount[:, None] * (mixing*value[:, None] + (1-mixing)*expected_q)
    on = reward[:, None] + discount[:, None] * (mixing*value[:, None] + (1-mixing)*soft_q)
    return dict(on=on, off=off, reward=reward, value_term=value_term,
        q_term=q_term, entropy_term=entropy_term, entropy=-expected_log)


def summaries(values, mask):
    selected = np.asarray(values)[mask]
    return dict(mean=float(selected.mean()), median=float(np.median(selected)),
        p01=float(np.quantile(selected, .01)), p99=float(np.quantile(selected, .99)),
        negative_fraction=float((selected < 0).mean()))


@torch.no_grad()
def diagnose(seed, admission=False):
    torch.set_num_threads(1)
    base, config_class, _, _ = runtime_api()
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        ids = np.flatnonzero(data["split_id"][:] == P["fit_role"])
        assert len(ids) == P["target_fit_rows"]
        if admission:
            ids = ids[np.linspace(0, len(ids)-1, P["admission_rows"], dtype=int)]
        raw = {key:torch.as_tensor(data[key][ids], dtype=torch.float32) for key in
            ("observations", "actions", "next_observations", "rewards", "discounts", "terminals")}
        contract = json.loads(data.attrs["config"])
    lower, upper = canonical_execution_bounds(raw["observations"], DEFAULT_CONFIG_V7)
    next_lower, next_upper = canonical_execution_bounds(raw["next_observations"], DEFAULT_CONFIG_V7)
    groups = dict(all=np.ones(len(ids), dtype=bool),
        free=(upper[:, 0] > lower[:, 0]).numpy(),
        forced60=((lower[:, 0] == 1) & (upper[:, 0] == 1)).numpy(),
        next_free=(next_upper[:, 0] > next_lower[:, 0]).numpy(),
        next_forced60=((next_lower[:, 0] == 1) & (next_upper[:, 0] == 1)).numpy())
    groups = {name:mask for name,mask in groups.items() if mask.any()}
    report = dict(complete=True, protocol=P["name"], seed=seed, admission=admission,
        rows=len(ids), updates=0, evaluation_rows_read=0, stages={},
        group_observation_side=dict(all="all fit transitions", free="observations",
            forced60="observations", next_free="next_observations", next_forced60="next_observations"))
    for stage, spec in P["stages"].items():
        path = HERE / spec["folder"] / f"seed{seed}" / spec["filename"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["total_updates"] == spec["stored_updates"]
        learner = learner_from_payload(base, config_class, contract, payload, seed)
        del payload
        assert learner.config.online_backup == "h2oplus"
        assert learner.config.ensemble_size == learner.config.target_subset_size == 2
        install_projected_sampling(learner.actor, lambda states,actions:
            project_canonical_executed_action(states, actions, DEFAULT_CONFIG_V7))
        collected, verified = {}, 0
        def add(name, value):
            collected.setdefault(name, []).append(value.numpy().reshape(-1))
        for start in range(0, len(ids), P["state_batch_size"]):
            batch = {key:value[start:start+P["state_batch_size"]] for key,value in raw.items()}
            draw_seed = seed + P["common_draw_seed_offset"] + start
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(draw_seed)
                part = target_parts(learner, batch, 1 if admission else P["positive_samples_per_state"])
            if admission:
                saved_config = learner.config
                for flag, name in ((True, "on"), (False, "off")):
                    learner.config = replace(saved_config, backup_entropy=flag)
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(draw_seed)
                        actual = learner._target(batch, backup="h2oplus")
                    assert torch.allclose(actual, part[name][:, 0],
                        atol=P["admission_atol"], rtol=P["admission_rtol"]), (stage, name)
                learner.config = saved_config
                verified += len(batch["observations"])
            recorded_q = learner.critic(batch["observations"], batch["actions"])
            target_recorded_q = learner.target_critic(batch["observations"], batch["actions"]).min(0).values
            advantage = target_recorded_q - learner.value(batch["observations"])
            weights = torch.exp(advantage * learner.config.iql_advantage_temperature).clamp_max(
                learner.config.iql_max_weight)
            if admission:
                _, native_advantage, native_weights = learner._iql_value_terms(batch)
                assert torch.equal(advantage, native_advantage)
                assert torch.equal(weights, native_weights)
            residual_on = (recorded_q[:, :, None] - part["on"][None]).abs().mean((0, 2))
            residual_off = (recorded_q[:, :, None] - part["off"][None]).abs().mean((0, 2))
            fields = dict(reward_scaled=part["reward"], value_term=part["value_term"],
                target_q_term=part["q_term"].mean(1), entropy_target_delta=part["entropy_term"].mean(1),
                next_state_latent_entropy=part["entropy"].mean(1),
                target_on=part["on"].mean(1), target_off=part["off"].mean(1),
                recorded_q_mean=recorded_q.mean(0), abs_td_on=residual_on,
                abs_td_off=residual_off, abs_td_reduction=residual_on-residual_off,
                advantage=advantage, weights=weights)
            for name, value in fields.items():
                assert bool(torch.isfinite(value).all()), (stage, name)
                add(name, value)
        fields = {name:np.concatenate(value) for name,value in collected.items()}
        stage_report = dict(artifact=str(path), alpha=float(learner.alpha.detach()),
            admission_target_rows_verified=verified, groups={})
        for name, mask in groups.items():
            weights = fields["weights"][mask].astype(np.float64)
            squared_sum = np.square(weights).sum()
            stage_report["groups"][name] = dict(rows=int(mask.sum()),
                decomposition={key:summaries(value, mask) for key,value in fields.items()},
                entropy_off_reduces_abs_td_fraction=float((fields["abs_td_reduction"][mask] > 0).mean()),
                weight_zero_fraction=float((weights == 0).mean()),
                weight_near_zero_fraction=float((weights < P["near_zero_weight_threshold"]).mean()),
                weight_cap_fraction=float((weights == learner.config.iql_max_weight).mean()),
                weight_effective_sample_fraction=float(weights.sum()**2 / (len(weights)*squared_sum))
                    if squared_sum else 0.)
        report["stages"][stage] = stage_report
        print(json.dumps(dict(seed=seed, stage=stage, complete=True, rows=len(ids))), flush=True)
        del learner
    report["interpretation"] = ("Same frozen parameters; expected absolute TD over heads and common draws. "
        "Entropy OFF changes only the target, not the recorded-action IQL weights. "
        "Counterfactual residual decomposition, not a retraining or performance result.")
    folder = "admission" if admission else f"seed{seed}"
    write(HERE / "iql_backup_diagnostic" / folder / "summary.json", report)
    print(json.dumps(dict(complete=True, seed=seed, admission=admission, rows=len(ids))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=P["training_seeds"])
    parser.add_argument("--admission", action="store_true")
    args = parser.parse_args()
    diagnose(args.seed, args.admission)
