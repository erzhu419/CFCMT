"""Frozen-learner IQL and actor drift on the corrected target fit records only."""

import argparse
from dataclasses import asdict
import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch.distributions import Normal
import torch.nn.functional as F

from canonical_execution_projection import (
    canonical_execution_bounds, project_canonical_executed_action,
)
from train_native import HERE, runtime_api, write

PREFIX = "native_terminal_iql_drift"


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite frozen diagnostic output")
    if not len(values):
        return dict(rows=0)
    return dict(rows=len(values), mean=float(values.mean()), std=float(values.std()),
        min=float(values.min()), p01=float(np.quantile(values, .01)),
        median=float(np.median(values)), p99=float(np.quantile(values, .99)),
        max=float(values.max()), mean_absolute=float(np.abs(values).mean()))


def strata(states, actions, config):
    lower, upper = canonical_execution_bounds(states, config)
    free = (upper[:, 0] > lower[:, 0]).numpy()
    scalar = actions[:, 0].numpy()
    zero, positive, endpoint = scalar == -1., scalar > -1., scalar == 1.
    return dict(all=np.ones(len(states), dtype=bool), free=free, collapsed=~free,
        recorded_zero=zero, recorded_positive=positive,
        recorded_positive_interior=positive & ~endpoint, recorded_upper_endpoint=endpoint,
        free_zero=free & zero, free_positive=free & positive,
        collapsed_positive=~free & positive)


def read_fit(protocol):
    path = HERE / protocol["target_pool"]
    with h5py.File(path, "r") as data:
        ids = np.flatnonzero(data["split_id"][:] == protocol["fit_role"])
        if len(ids) != protocol["target_fit_rows"] or len(data["actions"]) != protocol["target_union_rows"]:
            raise RuntimeError("Corrected fit/union row counts differ from the fixed protocol")
        if not json.loads(data.attrs["pool_summary"])["terminal_horizon_corrected"]:
            raise RuntimeError("The fixed target pool lacks the full-horizon terminal correction")
        contract = json.loads(data.attrs["config"])
        batch = {name: torch.as_tensor(data[name][ids], dtype=torch.float32)
            for name in ("observations", "actions")}
        terminals = int((data["terminals"][ids] > .5).sum())
    batch["actions"] = batch["actions"].reshape(-1, 1)
    if batch["observations"].shape != (len(ids), 229) or terminals != 6:
        raise RuntimeError("Fixed canonical states or six corrected fit terminal rows changed")
    return batch, contract


def load_frozen(stage, seed, trainer_class, config_class, contract):
    path = HERE / stage["artifact"].format(seed=seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = config_class(**payload["config"])
    learner = trainer_class(SimpleNamespace(contract_config=contract),
        config=config, seed=seed, device="cpu")
    learner.load(path, load_optimizers=False)
    if (asdict(learner.config) != payload["config"]
            or learner.total_updates != stage["stored_updates"]):
        raise RuntimeError("Learner configuration/counter differs from the fixed checkpoint")
    for name in ("actor", "critic", "target_critic", "value", "ratio"):
        getattr(learner, name).eval().requires_grad_(False)
    optimizers = {name: getattr(learner, name+"_optimizer")
        for name in ("actor", "critic", "value", "alpha")}
    optimizers["ratio"] = learner.ratio.optimizer
    if any(optimizer.state for optimizer in optimizers.values()):
        raise RuntimeError("Read-only learner unexpectedly contains initialized optimizer state")
    return learner, payload, optimizers, path


@torch.inference_mode()
def arrays(learner, batch, config, batch_size):
    parts = {}
    def add(name, value):
        parts.setdefault(name, []).append(value.detach().numpy().reshape(-1))
    for start in range(0, len(batch["actions"]), batch_size):
        chunk = {key: value[start:start+batch_size] for key, value in batch.items()}
        states, actions = chunk["observations"], chunk["actions"]
        _, advantage, weights = learner._iql_value_terms(chunk)
        target_q = learner.target_critic(states, actions).min(0).values
        value = learner.value(states)
        expected = target_q-value
        expected_weights = torch.exp(expected*learner.config.iql_advantage_temperature).clamp_max(
            learner.config.iql_max_weight)
        if not torch.equal(advantage, expected) or not torch.equal(weights, expected_weights):
            raise RuntimeError("Native IQL advantage/weights differ from the reported formula")
        log_probability = learner.actor.action_log_prob(states, actions)
        gate, mean, log_std = learner.actor.distribution_parameters(states)
        is_zero = actions[:, 0] == -1.
        gate_log_probability = torch.where(is_zero, F.logsigmoid(-gate), F.logsigmoid(gate))
        epsilon = torch.finfo(actions.dtype).eps*.5
        pre_sigmoid = torch.logit(((actions[:, 0]+1)*.5).clamp(epsilon, 1.-epsilon))
        positive_log_probability = (F.logsigmoid(gate)
            + Normal(mean, log_std.exp()).log_prob(pre_sigmoid)
            - learner.actor._positive_log_det(pre_sigmoid))
        reconstructed = torch.where(is_zero, gate_log_probability,
            positive_log_probability)
        if not torch.equal(log_probability, reconstructed):
            raise RuntimeError("Native recorded action likelihood decomposition differs")
        conditional_nll = -(log_probability-gate_log_probability)
        if not torch.allclose(-log_probability, -gate_log_probability+conditional_nll,
                atol=1e-6, rtol=1e-6):
            raise RuntimeError("Gate and positive density do not reconstruct native NLL")
        requested = learner.actor(states, deterministic=True)[0]
        executed = project_canonical_executed_action(states, requested, config)
        conditional_action = learner.actor.positive_action_and_log_prob(states, deterministic=True)[0]
        for name, values in dict(target_q=target_q, current_q=learner.critic(states, actions).min(0).values,
                value=value, advantage=advantage, weight=weights, full_nll=-log_probability,
                gate_nll=-gate_log_probability, positive_conditional_nll=conditional_nll,
                hold_probability=gate.sigmoid(), requested_hold_seconds=(requested+1)*config.max_hold_seconds/2,
                executed_hold_seconds=(executed+1)*config.max_hold_seconds/2,
                conditional_positive_hold_seconds=(conditional_action+1)*config.max_hold_seconds/2).items():
            add(name, values)
    return {name: np.concatenate(values) for name, values in parts.items()}


def summarize_stage(fields, masks, config):
    iql_names = ("target_q", "current_q", "value", "advantage", "weight")
    actor_names = ("full_nll", "gate_nll", "positive_conditional_nll", "hold_probability",
        "requested_hold_seconds", "executed_hold_seconds", "conditional_positive_hold_seconds")
    result = dict(iql={}, actor={}, weight_mass_fraction_by_stratum={})
    total_weight = float(fields["weight"].sum(dtype=np.float64))
    all_weights = fields["weight"].astype(np.float64)
    squared_mass = float(np.square(all_weights).sum())
    ess = total_weight**2/squared_mass if squared_mass > 0 else 0.
    top_rows = int(np.ceil(len(all_weights)*.01))
    result["weight_concentration"] = dict(effective_sample_size=ess,
        effective_sample_size_fraction=ess/len(all_weights), top_one_percent_rows=top_rows,
        top_one_percent_weight_mass_fraction=(
            float(np.sort(all_weights)[-top_rows:].sum()/total_weight) if total_weight > 0 else None))
    for name, mask in masks.items():
        result["iql"][name] = {key: describe(fields[key][mask]) for key in iql_names}
        result["actor"][name] = {key: describe(fields[key][mask]) for key in actor_names}
        result["actor"][name]["requested_positive_fraction"] = (
            float(np.mean(fields["requested_hold_seconds"][mask] > 0)) if mask.any() else None)
        result["actor"][name]["executed_positive_fraction"] = (
            float(np.mean(fields["executed_hold_seconds"][mask] > 0)) if mask.any() else None)
        result["weight_mass_fraction_by_stratum"][name] = (
            float(fields["weight"][mask].sum(dtype=np.float64)/total_weight) if total_weight > 0 else None)
    result["weight_clip_fraction"] = float(np.mean(fields["weight"] == config.iql_max_weight))
    result["weight_definition"] = "exp(beta*(min target-critic Q(s,recorded a)-V(s))).clamp_max(saved max_weight)"
    result["positive_conditional_nll_definition"] = "Recorded positive-action conditional density NLL; zero labels contribute zero. This differential NLL can be negative."
    return result


def cross_objectives(fields, masks):
    result = {}
    for weight_source, weighted_fields in fields.items():
        for actor_source, actor_fields in fields.items():
            key = weight_source+"_weights/"+actor_source+"_actor"
            row = dict(weight_source=weight_source, actor_source=actor_source, strata={})
            weights = weighted_fields["weight"].astype(np.float64)
            for name, mask in masks.items():
                terms = {component: weights*actor_fields[component].astype(np.float64)
                    for component in ("full_nll", "gate_nll", "positive_conditional_nll")}
                contributions = {component: float(term[mask].sum()/len(mask))
                    for component, term in terms.items()}
                if not np.isclose(contributions["full_nll"], contributions["gate_nll"]+
                        contributions["positive_conditional_nll"], atol=1e-6, rtol=1e-6):
                    raise RuntimeError("Cross-weight objective components do not reconstruct native loss")
                row["strata"][name] = dict(rows=int(mask.sum()),
                    full_record_denominator=len(mask), full_record_contributions=contributions,
                    within_stratum_means={component: float(term[mask].mean()) if mask.any() else None
                        for component, term in terms.items()})
            result[key] = row
    return result


def drift(fields, masks):
    before, after = fields["adapted55"], fields["unmasked75"]
    result = {}
    for name, mask in masks.items():
        row = {key+"_75_minus_55": describe(after[key][mask]-before[key][mask])
            for key in ("advantage", "weight", "full_nll", "gate_nll", "positive_conditional_nll",
                "hold_probability", "requested_hold_seconds", "executed_hold_seconds")}
        old_hold, new_hold = before["hold_probability"] >= .5, after["hold_probability"] >= .5
        row["gate_zero_to_positive_fraction"] = float(np.mean((~old_hold & new_hold)[mask])) if mask.any() else None
        row["gate_positive_to_zero_fraction"] = float(np.mean((old_hold & ~new_hold)[mask])) if mask.any() else None
        result[name] = row
    return result


def run(seed):
    protocol = json.loads((HERE / (PREFIX+"_protocol.json")).read_text())
    if seed not in protocol["training_seeds"]:
        raise ValueError("Seed is outside the fixed three-learner roster")
    torch.set_num_threads(1)
    trainer_class, config_class, _, _ = runtime_api()
    from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
    batch, contract = read_fit(protocol)
    for key in ("max_hold_seconds", "scheduled_headway_scale_seconds"):
        if contract[key] != getattr(DEFAULT_CONFIG_V7, key):
            raise RuntimeError("Pool contract differs from the frozen canonical projection")
    masks = strata(batch["observations"], batch["actions"], DEFAULT_CONFIG_V7)
    projected_labels = project_canonical_executed_action(
        batch["observations"], batch["actions"], DEFAULT_CONFIG_V7)
    residual = (projected_labels-batch["actions"]).abs()
    if float(residual.max()) > protocol["action_equality_tolerance"]:
        raise RuntimeError("Recorded execution labels lie outside the fixed canonical interval")
    stages, fields, artifacts = {}, {}, {}
    for name, stage in protocol["stages"].items():
        learner, payload, optimizers, path = load_frozen(stage, seed, trainer_class, config_class, contract)
        counters = learner.total_updates, learner.online_critic_updates
        fields[name] = arrays(learner, batch, DEFAULT_CONFIG_V7, protocol["state_batch_size"])
        if counters != (learner.total_updates, learner.online_critic_updates) or any(
                optimizer.state for optimizer in optimizers.values()):
            raise RuntimeError("Read-only inference performed an optimizer/counter update")
        for module_name in ("actor", "critic", "target_critic", "value", "ratio"):
            state = getattr(learner, module_name).state_dict()
            if state.keys() != payload[module_name].keys() or any(
                    not torch.equal(value, payload[module_name][key]) for key, value in state.items()):
                raise RuntimeError("Read-only inference changed a checkpoint tensor")
        if not torch.equal(learner.log_alpha.detach(), payload["log_alpha"]):
            raise RuntimeError("Read-only inference changed the saved temperature")
        artifacts[name] = str(path)
        stages[name] = dict(config=asdict(learner.config), stored_updates=counters[0],
            historical_updates=stage["historical_updates"], online_critic_updates=counters[1],
            optimizer_steps={key: 0 for key in optimizers},
            **summarize_stage(fields[name], masks, learner.config))
        del learner, payload, optimizers
        print(json.dumps(dict(seed=seed, completed_stage=name)), flush=True)
    result = dict(complete=True, protocol=protocol["name"], seed=seed, rows=len(batch["actions"]),
        updates=0, evaluation_rows_read=0, new_simulations=0, world_model_refit_updates=0,
        target_pool=str(HERE / protocol["target_pool"]), artifacts=artifacts,
        stored_updates={name: stage["stored_updates"] for name, stage in protocol["stages"].items()},
        historical_updates={name: stage["historical_updates"] for name, stage in protocol["stages"].items()},
        strata_counts={name: int(mask.sum()) for name, mask in masks.items()},
        validations=dict(native_weights_exact=True, native_likelihood_decomposition_exact=True,
            labels_projection_max_absolute_error=float(residual.max()),
            labels_projection_within_tolerance=True, configs_preserved=True,
            checkpoint_tensors_unchanged=True, optimizer_steps_zero=True),
        stages=stages, cross_weight_actor_objectives=cross_objectives(fields, masks),
        drift=drift(fields, masks), interpretation=protocol["interpretation"],
        action_semantics="Native deterministic requested holding, then canonical guard-image projection; seconds=(action+1)*max_hold_seconds/2.",
        diagnostic_role="Fixed training-fit records only; frozen weights are recomputed, not cached last-update weights. Likelihood/advantage drift is not passenger-cost or action-ranking validation.")
    write(HERE / PREFIX / f"seed{seed}" / "summary.json", result)
    print(json.dumps(dict(complete=True, seed=seed, rows=result["rows"], updates=0,
        new_simulations=0, evaluation_rows_read=0)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    run(parser.parse_args().seed)
