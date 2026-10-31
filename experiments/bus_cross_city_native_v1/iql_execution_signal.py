"""Read-only recorded-likelihood and executed-action signals at fixed checkpoints."""

import argparse
import json

import h5py
import numpy as np
import torch

from canonical_execution_projection import canonical_execution_bounds, project_canonical_executed_action
from diagnose_values import learner_from_payload
from train_native import HERE, runtime_api, write

P = json.loads((HERE / "iql_execution_signal_protocol.json").read_text())


def describe(value):
    value = np.asarray(value, dtype=np.float64)
    return dict(mean=float(value.mean()), median=float(np.median(value)),
        p01=float(np.quantile(value, .01)), p99=float(np.quantile(value, .99)),
        min=float(value.min()), max=float(value.max()))


def vector(gradients):
    result = torch.cat([gradient.detach().reshape(-1).double() for gradient in gradients])
    assert bool(torch.isfinite(result).all())
    return result


def check_close(left, right):
    assert torch.allclose(left, right, atol=P["admission_atol"], rtol=P["admission_rtol"])


def diagnose(seed, admission=False):
    torch.set_num_threads(1)
    base, config_class, _, _ = runtime_api()
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        ids = np.flatnonzero(data["split_id"][:] == P["fit_role"])
        assert len(ids) == P["target_fit_rows"]
        if admission:
            ids = ids[np.linspace(0, len(ids)-1, P["admission_rows"], dtype=int)]
        raw = {key:torch.as_tensor(data[key][ids], dtype=torch.float32)
            for key in ("observations", "actions")}
        contract = json.loads(data.attrs["config"])
    n = len(ids)
    lower, upper = canonical_execution_bounds(raw["observations"], DEFAULT_CONFIG_V7)
    groups = dict(all=np.ones(n, dtype=bool), free=(upper[:, 0] > lower[:, 0]).numpy(),
        forced60=((lower[:, 0] == 1) & (upper[:, 0] == 1)).numpy())
    assert np.all(groups["free"] | groups["forced60"])
    assert all(mask.any() for mask in groups.values())
    learners, fields, gradients, scores = {}, {}, {}, {}
    for stage, spec in P["stages"].items():
        path = HERE / spec["folder"] / f"seed{seed}" / spec["filename"]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        assert payload["total_updates"] == spec["stored_updates"]
        learners[stage] = learner_from_payload(base, config_class, contract, payload, seed)
        del payload
        learners[stage].actor.requires_grad_(True)
        fields[stage], scores[stage] = {}, {}
        size = sum(parameter.numel() for parameter in learners[stage].actor.parameters())
        gradients[stage] = {group:torch.zeros(size, dtype=torch.float64) for group in ("free", "forced60")}
    verified = dict(native_q_queries=0, actual_weight_rows=0, gradient_batches=0,
        forced_projection_rows=0, native_loss_batches=0)

    def add(store, name, value):
        assert bool(torch.isfinite(value).all()), name
        store.setdefault(name, []).append(value.detach().numpy().reshape(-1))

    for start in range(0, n, P["state_batch_size"]):
        batch = {key:value[start:start+P["state_batch_size"]] for key,value in raw.items()}
        states, labels = batch["observations"], batch["actions"]
        masks = {group:torch.as_tensor(mask[start:start+len(states)]) for group,mask in groups.items()}
        actions = {"recorded":labels, "zero":project_canonical_executed_action(
            states, torch.full_like(labels, -1), DEFAULT_CONFIG_V7)}
        for stage, learner in learners.items():
            with torch.no_grad():
                _, advantage, weights = learner._iql_value_terms(batch)
                desired = learner.actor(states, deterministic=True)[0]
                part = learner.actor.soft_action_components(states, deterministic_positive=True)
                executed = project_canonical_executed_action(states, desired, DEFAULT_CONFIG_V7)
                actions[stage] = executed
                for name, value in dict(advantage=advantage, weights=weights, desired=desired,
                    executed=executed, hold_probability=part["hold_probability"],
                    conditional_positive_median=part["positive_action"]).items():
                    add(fields[stage], name, value)
                if admission:
                    expected_advantage = learner.target_critic(states, labels).min(0).values - learner.value(states)
                    assert torch.equal(advantage, expected_advantage)
                    assert torch.equal(weights, torch.exp(advantage * learner.config.iql_advantage_temperature)
                        .clamp_max(learner.config.iql_max_weight))
                    assert not weights.requires_grad
                    assert bool((executed[masks["forced60"]] == 1).all())
                    assert bool((actions["zero"][masks["forced60"]] == 1).all())
                    verified["actual_weight_rows"] += len(states)
                    verified["forced_projection_rows"] += int(masks["forced60"].sum())
            logp = learner.actor.action_log_prob(states, labels)
            add(fields[stage], "recorded_log_prob", logp)
            terms = -weights * logp
            parameters = tuple(learner.actor.parameters())
            local_gradients, group_losses = {}, []
            for group in ("free", "forced60"):
                loss = terms[masks[group]].sum() / n
                group_losses.append(loss)
                gradient = (vector(torch.autograd.grad(loss, parameters, retain_graph=True))
                    if masks[group].any() else torch.zeros_like(gradients[stage][group]))
                local_gradients[group] = gradient
                gradients[stage][group] += gradient
            if admission:
                native_loss = -(weights * learner.actor.action_log_prob(states, labels)).mean()
                check_close(sum(group_losses) * (n / len(states)), native_loss)
                total_gradient = vector(torch.autograd.grad(terms.sum() / n, parameters))
                difference = (sum(local_gradients.values()) - total_gradient).norm()
                assert difference <= P["admission_atol"] + P["admission_rtol"] * total_gradient.norm()
                assert all(parameter.grad is None for parameter in parameters)
                verified["gradient_batches"] += 1
                verified["native_loss_batches"] += 1
            del logp, terms, group_losses
        with torch.no_grad():
            for stage, learner in learners.items():
                features = learner.critic.encode_states(states)
                for candidate, action in actions.items():
                    q = learner.critic.forward_features(features, action).min(0).values
                    add(scores[stage], candidate, q)
                    if admission:
                        check_close(q, learner.critic(states, action).min(0).values)
                        verified["native_q_queries"] += len(states)
    fields = {stage:{name:np.concatenate(value) for name,value in store.items()}
        for stage,store in fields.items()}
    scores = {stage:{name:np.concatenate(value) for name,value in store.items()}
        for stage,store in scores.items()}
    seconds = lambda action: (action + 1) * DEFAULT_CONFIG_V7.max_hold_seconds / 2
    labels = raw["actions"].numpy().reshape(-1)
    report = dict(complete=True, protocol=P["name"], seed=seed, admission=admission,
        rows=n, updates=0, evaluation_rows_read=0, admission_checks=verified,
        groups={}, stages={}, cross_weighted_likelihood={}, pairs={})
    for group, mask in groups.items():
        report["groups"][group] = dict(rows=int(mask.sum()), observation_side="observations",
            recorded_positive_holding_fraction=float((labels[mask] > -1 + P["action_equality_tolerance"]).mean()),
            recorded_holding_seconds=describe(seconds(labels[mask])))
        matrix = report["cross_weighted_likelihood"][group] = {}
        for stage, store in fields.items():
            weights = store["weights"][mask].astype(np.float64)
            total, squares = weights.sum(), np.square(weights).sum()
            matrix[stage] = {}
            for actor_stage, actor_store in fields.items():
                weighted_sum = -np.dot(weights, actor_store["recorded_log_prob"][mask].astype(np.float64))
                matrix[stage][actor_stage] = dict(normalized_weighted_nll=float(weighted_sum / total)
                    if total else None, raw_mean_weighted_loss=float(weighted_sum / len(weights)))
            stage_report = report["stages"].setdefault(stage, dict(groups={}, gradients={}))
            stage_report["groups"][group] = dict(advantage=describe(store["advantage"][mask]),
                weights=describe(weights), weight_zero_fraction=float((weights == 0).mean()),
                weight_near_zero_fraction=float((weights < P["near_zero_weight_threshold"]).mean()),
                weight_cap_fraction=float((weights == learners[stage].config.iql_max_weight).mean()),
                weight_effective_sample_fraction=float(total**2 / (len(weights)*squares)) if squares else 0.,
                desired_holding_seconds=describe(seconds(store["desired"][mask])),
                executed_holding_seconds=describe(seconds(store["executed"][mask])),
                desired_positive_holding_fraction=float((store["desired"][mask] > -1 + P["action_equality_tolerance"]).mean()),
                executed_positive_holding_fraction=float((store["executed"][mask] > -1 + P["action_equality_tolerance"]).mean()),
                gate_holding_fraction=float((store["hold_probability"][mask] >= .5).mean()),
                hold_probability=describe(store["hold_probability"][mask]),
                conditional_positive_median_holding_seconds=describe(seconds(store["conditional_positive_median"][mask])),
                mean_min_q={candidate:float(value[mask].mean()) for candidate,value in scores[stage].items()})
    for stage, vectors in gradients.items():
        total = sum(vectors.values())
        report["stages"][stage]["gradients"]["all"] = dict(l2_norm=float(total.norm()))
        for group, gradient in vectors.items():
            denominator = gradient.norm() * total.norm()
            report["stages"][stage]["gradients"][group] = dict(l2_norm=float(gradient.norm()),
                cosine_with_total=float(torch.dot(gradient, total) / denominator) if denominator else None)
    for name, (left, right) in P["pairs"].items():
        lhs, rhs = fields[left], fields[right]
        raw_changed = abs(lhs["desired"]-rhs["desired"]) > P["action_equality_tolerance"]
        changed = abs(lhs["executed"]-rhs["executed"]) > P["action_equality_tolerance"]
        gate_changed = (lhs["hold_probability"] >= .5) != (rhs["hold_probability"] >= .5)
        pair = report["pairs"][name] = dict(left=left, right=right, groups={}, changed_free_action_q={})
        for group, mask in groups.items():
            pair["groups"][group] = dict(raw_action_changed_rows=int((raw_changed & mask).sum()),
                executed_action_changed_rows=int((changed & mask).sum()), gate_flipped_rows=int((gate_changed & mask).sum()),
                desired_mean_holding_drift_seconds=float((seconds(lhs["desired"])-seconds(rhs["desired"]))[mask].mean()),
                executed_mean_holding_drift_seconds=float((seconds(lhs["executed"])-seconds(rhs["executed"]))[mask].mean()))
        selected = groups["free"] & changed
        for critic_stage, q in scores.items():
            gap = (q[left]-q[right])[selected]
            pair["changed_free_action_q"][critic_stage] = dict(rows=len(gap),
                mean_gap=float(gap.mean()) if len(gap) else None,
                positive_gap_fraction=float((gap > 0).mean()) if len(gap) else None)
    report["interpretation"] = ("Fixed-checkpoint fit signals, not integrated historical update attribution. "
        "Raw recorded likelihood is unchanged; forced60 execution is constant despite latent gradients. "
        "Group gradients use the overall selected N and sum before reporting norms; norms are not additive percentages. "
        "Conditional positive deterministic holding is a median, not a sampled expectation. "
        "Learned Q rankings are estimates, not actual counterfactual control outcomes.")
    folder = "admission" if admission else f"seed{seed}"
    write(HERE / "iql_execution_signal" / folder / "summary.json", report)
    print(json.dumps(dict(complete=True, seed=seed, admission=admission, rows=n, updates=0)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=P["training_seeds"])
    parser.add_argument("--admission", action="store_true")
    args = parser.parse_args()
    diagnose(args.seed, args.admission)
