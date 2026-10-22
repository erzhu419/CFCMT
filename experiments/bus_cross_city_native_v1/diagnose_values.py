"""Read-only value/actor diagnosis on existing stop/calibration states.

No fitting, parameter selection, final-test reading, or simulator evaluation.
The final deployment artifacts contain actors only; their critics are unavailable.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch import nn

from train_native import (
    HERE, EventModel, ModelContinuation, Normalizer, TRANSFER_MODULES,
    runtime_api, transfer_parameters, write,
)


ARRAYS = ("observations", "actions", "rewards", "next_observations",
          "discounts", "terminals", "delta_t_seconds")


def describe(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if not finite.all():
        raise RuntimeError("Nonfinite diagnostic output")
    return dict(rows=len(values), mean=float(values.mean()), std=float(values.std()),
                min=float(values.min()), p01=float(np.quantile(values, .01)),
                median=float(np.median(values)), p99=float(np.quantile(values, .99)),
                max=float(values.max()), positive_fraction=float(np.mean(values > 0)))


def residual(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(describe(values), mae=float(np.abs(values).mean()),
                rmse=float(np.sqrt(np.square(values).mean())))


def read_roles(path):
    with h5py.File(path, "r") as data:
        codes = data["split_id"][:]
        result = {}
        for name, code in (("stop", 1), ("calibrate", 2)):
            ids = np.flatnonzero(codes == code)
            result[name] = {key: data[key][ids] for key in ARRAYS}
            result[name]["row_indices"] = ids
        config = json.loads(data.attrs["config"])
        stations = json.loads(data.attrs["station_max_by_line"])
    return result, config, stations


def read_source_samples(path):
    with h5py.File(path, "r") as data:
        fit = np.flatnonzero(data["split_id"][:] == 0)
        ids = fit[np.linspace(0, len(fit)-1, 256, dtype=np.int64)]
        anchors = fit[data["terminals"][fit] > .5]
        result = {}
        for name, rows in (("uniform256", ids), ("all_terminal_tails", anchors)):
            result[name] = {key: data[key][rows] for key in ARRAYS}
            result[name]["row_indices"] = rows
    return result


def tensors(raw):
    return {key: torch.as_tensor(raw[key], dtype=torch.float32) for key in ARRAYS}


def selected(batch, ids):
    return {key: value[ids] for key, value in batch.items()}


def model_decoder(protocol, seed, folder, encoder_class, stations):
    """Load only fitted predictor parameters; do not initialize or update trust."""
    payload = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    decoder = ModelContinuation.__new__(ModelContinuation)
    decoder.p, decoder.device = protocol["training"], torch.device("cpu")
    decoder.stations = torch.tensor(stations, dtype=torch.float32)
    decoder.norm = Normalizer.__new__(Normalizer)
    nn.Module.__init__(decoder.norm)
    for key, value in payload["normalizer"].items():
        decoder.norm.register_buffer(key, value)
    decoder.models = []
    for state in payload["models"]:
        model = EventModel(encoder_class, payload["station_classes"])
        model.load_state_dict(state)
        decoder.models.append(model.eval().requires_grad_(False))
    decoder.generator = torch.Generator().manual_seed(seed + 40000)
    return decoder


@torch.no_grad()
def model_rows(decoder, raw, *, seed):
    batch = tensors(raw)
    eligible = ((batch["delta_t_seconds"] > 0) & (batch["terminals"] == 0)
                & (batch["observations"][:, 0] < decoder.p["model_start_max_horizon_fraction"]))
    ids = torch.nonzero(eligible).flatten()
    actual = selected(batch, ids)
    decoder.generator.manual_seed(seed)
    outputs = []
    for start in range(0, len(ids), 256):
        states = actual["observations"][start:start+256]
        actions = actual["actions"][start:start+256]
        outputs.append(decoder.decode(states, decoder.predict(states, actions)))
    ns, reward, duration = [torch.cat([output[i] for output in outputs]) for i in range(3)]
    model = dict(actual, next_observations=ns, rewards=reward, delta_t_seconds=duration,
                 discounts=torch.ones(len(ids)), terminals=torch.zeros(len(ids)))
    scale = decoder.p["trainer_config"]["reward_scale"]
    report = dict(rows=len(ids), selection="positive dt, nonterminal, state horizon<0.75",
                  categorical_draw_seed=seed,
                  duration_bias_seconds=residual((duration-actual["delta_t_seconds"]).numpy()),
                  reward_bias_scaled=residual(((reward-actual["rewards"])*scale).numpy()),
                  actual_reward_scaled=describe((actual["rewards"]*scale).numpy()),
                  model_reward_scaled=describe((reward*scale).numpy()),
                  state_absolute_error_mean=float((ns-actual["next_observations"]).abs().mean()))
    return actual, model, report


@torch.no_grad()
def actor_arrays(actor, batch):
    parts = dict(action=[], probability=[], conditional_hold_seconds=[])
    for start in range(0, len(batch["observations"]), 256):
        states = batch["observations"][start:start+256]
        action = actor(states, deterministic=True)[0].reshape(-1)
        gate, _, _ = actor.distribution_parameters(states)
        positive = actor.positive_action_and_log_prob(states, deterministic=True)[0]
        parts["action"].append(action.numpy())
        parts["probability"].append(gate.sigmoid().numpy())
        parts["conditional_hold_seconds"].append(((positive+1)*30).numpy())
    return {key: np.concatenate(value) for key, value in parts.items()}


def actor_summary(actor, batch):
    parts = actor_arrays(actor, batch)
    proposed = (parts["action"]+1)*30
    recorded = (batch["actions"].reshape(-1).numpy()+1)*30
    return dict(rows=len(recorded), hold_probability=describe(parts["probability"]),
                proposed_hold_seconds=describe(proposed),
                conditional_positive_hold_seconds=describe(parts["conditional_hold_seconds"]),
                proposed_positive_fraction=float(np.mean(parts["action"] > -1)),
                recorded_action_hold_seconds=describe(recorded),
                proposal_minus_recorded_seconds=residual(proposed-recorded),
                semantics="Deterministic actor proposal before early60/completion guard; not executed hold")


@torch.no_grad()
def value_arrays(learner, batch, rng_seed):
    fields = {}
    def add(key, value):
        fields.setdefault(key, []).append(value.detach().numpy())
    for start in range(0, len(batch["observations"]), 256):
        chunk = selected(batch, slice(start, start+256))
        states, actions = chunk["observations"], chunk["actions"]
        proposal = learner.actor(states, deterministic=True)[0]
        recorded_q = learner.critic(states, actions)
        zero_q = learner.critic(states, torch.full_like(actions, -1))
        proposal_q = learner.critic(states, proposal)
        target_q = learner.target_critic(states, actions)
        value = learner.value(states)
        following_value = learner.value(chunk["next_observations"])
        next_proposal = learner.actor(chunk["next_observations"], deterministic=True)[0]
        next_q = learner.target_critic(chunk["next_observations"], next_proposal).min(0).values
        iql_target = learner._target(chunk, backup="iql")
        # Repeat the same stochastic positive-action and ensemble subset draws
        # for actual/model next states and for all checkpoint stages.
        with torch.random.fork_rng():
            torch.manual_seed(rng_seed+start)
            learner.rng = np.random.default_rng(rng_seed+start)
            mixed_target = learner._target(chunk, backup="h2oplus")
        for key, q in (("q_recorded", recorded_q), ("target_q_recorded", target_q),
                       ("q_zero", zero_q), ("q_proposed", proposal_q)):
            add(key+"_mean", q.mean(0))
            add(key+"_min", q.min(0).values)
        add("value", value)
        add("following_value", following_value)
        add("following_target_q_deterministic_proposal_min", next_q)
        add("iql_target", iql_target)
        add("mixed_entropy_target", mixed_target)
        add("iql_residual_mean", recorded_q.mean(0)-iql_target)
        add("iql_residual_min", recorded_q.min(0).values-iql_target)
        add("mixed_entropy_residual_min", recorded_q.min(0).values-mixed_target)
        add("proposal_advantage_vs_zero_min", proposal_q.min(0).values-zero_q.min(0).values)
        add("recorded_advantage_vs_zero_min", recorded_q.min(0).values-zero_q.min(0).values)
    return {key: np.concatenate(value) for key, value in fields.items()}


def value_summary(learner, batch, rng_seed):
    fields = value_arrays(learner, batch, rng_seed)
    result = {key: (residual(value) if "residual" in key else describe(value)) for key, value in fields.items()}
    terminal = batch["terminals"].numpy().reshape(-1) > .5
    for name, mask in (("terminal", terminal), ("nonterminal", ~terminal)):
        if mask.any():
            result[name] = {key: (residual(value[mask]) if "residual" in key else describe(value[mask]))
                           for key, value in fields.items() if key in (
                "q_recorded_mean", "q_recorded_min", "value", "iql_target", "iql_residual_min")}
    return result


def learner_from_payload(trainer_class, config_class, contract, payload, seed, *, reset_columns=False):
    config = config_class(**payload.get("config", payload.get("source_config")))
    config = replace(config, offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    learner = trainer_class(SimpleNamespace(contract_config=contract), config=config, seed=seed, device="cpu")
    transfer_parameters(learner, payload, reset_line_columns=reset_columns)
    if not reset_columns:
        learner.log_alpha.data.copy_(payload["log_alpha"])
    for name in TRANSFER_MODULES:
        getattr(learner, name).eval().requires_grad_(False)
    return learner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    protocol = json.loads((HERE / "training_protocol.json").read_text())
    if args.seed not in protocol["training_seeds"]:
        raise ValueError("Diagnostic seed must be one of the already trained seeds")
    torch.set_num_threads(1)
    trainer_class, config_class, _, encoder_class = runtime_api()
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    raw_roles, contract, stations = read_roles(HERE / "target_pool.h5")
    roles = {name: tensors(raw) for name, raw in raw_roles.items()}
    root = HERE / "training"
    decoder = model_decoder(protocol, args.seed, root / "models" / f"seed{args.seed}", encoder_class, stations)
    predictions, model_reports = {}, {}
    for index, (name, raw) in enumerate(raw_roles.items()):
        actual, model, report = model_rows(decoder, raw, seed=args.seed+40000+index)
        predictions[name], model_reports[name] = (actual, model), report
    stages = dict(source_initializer_native=("init/source", "initializer.pt", False),
                  source_initializer_transferred=("init/source", "initializer.pt", True),
                  target_initializer=("init/target", "initializer.pt", False),
                  source_adapted=("adapt/source", "adapted.pt", False),
                  target_adapted=("adapt/target", "adapted.pt", False))
    reports, source_report = {}, None
    for name, (folder, filename, reset) in stages.items():
        path = root / folder / f"seed{args.seed}" / filename
        payload = torch.load(path, map_location="cpu", weights_only=False)
        learner = learner_from_payload(trainer_class, config_class, contract, payload, args.seed, reset_columns=reset)
        report = dict(artifact=str(path.relative_to(HERE)), stored_updates=payload["total_updates"],
                      reset_city_and_arrival_columns=reset, temperature=float(learner.alpha.detach()), roles={})
        for role_index, (role, batch) in enumerate(roles.items()):
            actual, model = predictions[role]
            rng_seed = args.seed+80000+10000*role_index
            actual_fields = value_arrays(learner, actual, rng_seed)
            model_fields = value_arrays(learner, model, rng_seed)
            paired = {key+"_model_minus_actual": residual(model_fields[key]-actual_fields[key])
                      for key in ("following_value", "following_target_q_deterministic_proposal_min",
                                  "iql_target", "mixed_entropy_target")}
            report["roles"][role] = dict(actor=actor_summary(learner.actor, batch),
                values=value_summary(learner, batch, rng_seed),
                same_recorded_action_model_pairs=paired)
        if name == "source_initializer_native":
            source_report = dict(sampling="256 equally spaced source-fit row indices plus separately all source terminal tails; existing compact pool only", samples={})
            for sample, source_raw in read_source_samples(HERE / "source_pool.h5").items():
                source_batch = tensors(source_raw)
                source_report["samples"][sample] = dict(rows=len(source_batch["actions"]),
                    values=value_summary(learner, source_batch, args.seed+90000),
                    actor=actor_summary(learner.actor, source_batch))
        reports[name] = report
        del payload, learner
        print(json.dumps(dict(seed=args.seed, completed_stage=name)), flush=True)
    final_actors = {}
    for branch in protocol["branches"]:
        path = root / "branches" / branch / f"seed{args.seed}" / "policy.pt"
        actor, _ = load_actor_checkpoint_v7(path, "cpu", require_policy_artifact=True)
        final_actors[branch] = dict(artifact=str(path.relative_to(HERE)),
            critic_available=False, roles={name: actor_summary(actor, batch) for name, batch in roles.items()})
    result = dict(complete=True, seed=args.seed, target_test_rows_read=0, updates=0,
        role_rows={name: len(batch["actions"]) for name, batch in roles.items()},
        source_initializer_on_source_fit=source_report, stages=reports,
        model_predictions=model_reports, final_actors=final_actors,
        value_convention="scaled passenger-seconds (reward_scale=1e-4); IQL target=r*scale+discount*(1-terminal)*V(ns)",
        mixed_backup_convention="Actual frozen _target h2oplus, 0.1 V + 0.9 SAC with entropy; matched stochastic draws per role/chunk",
        interpretation="Same-state numerical diagnostics, not action counterfactual labels or a performance estimate. Source-native stages retain source line identities; transferred stage uses the exact frozen column reset. Final critics were not saved and cannot be diagnosed.")
    output = args.output or HERE / "diagnostics" / f"seed{args.seed}" / "summary.json"
    write(output, result)
    print(json.dumps(dict(complete=True, seed=args.seed, output=str(output))), flush=True)


if __name__ == "__main__":
    main()
