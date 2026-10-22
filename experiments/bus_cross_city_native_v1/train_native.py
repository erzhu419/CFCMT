"""Fixed-budget IQL initialization and native target world-model continuation.

The event predictor follows the completed Thesis MC-WM shared model, with
station output width inferred from this target's real topology. Runtime imports
come exclusively from CFCMT's physical frozen snapshots.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import h5py
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
TRANSFER_MODULES = ("actor", "critic", "target_critic", "value")
LINE_COLUMNS_PARAMETER = "encoder.slot_encoder.network.0.weight"
RESET_TRANSFER_COLUMNS = tuple(range(12)) + (19,)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def runtime_api():
    runtime = json.loads((HERE / "runtime.json").read_text())
    sys.path.insert(0, str(Path(runtime["frozen_snapshot"]) / "runtime/H2Oplus"))
    from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7, TransitTrainerConfig
    from SimpleSAC.transit_v7.replay_buffer import TransitReplayBufferV7
    from SimpleSAC.transit_v7.networks import TransitStateEncoderV7
    return TransitH2OTrainerV7, TransitTrainerConfig, TransitReplayBufferV7, TransitStateEncoderV7


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def transfer_parameters(learner, payload, *, reset_line_columns):
    """Transfer network tensors only; all destination optimizers remain fresh."""
    for name in TRANSFER_MODULES:
        destination = getattr(learner, name)
        state = {key: value.detach().clone() for key, value in payload[name].items()}
        if reset_line_columns:
            fresh = destination.state_dict()[LINE_COLUMNS_PARAMETER]
            state[LINE_COLUMNS_PARAMETER][:, RESET_TRANSFER_COLUMNS] = fresh[:, RESET_TRANSFER_COLUMNS]
        destination.load_state_dict(state)


def load_model_roles(path, seed, fit_cap):
    """World-model roles are explicit H5 roles, independent of actor replay."""
    result, selected = {}, {}
    with h5py.File(path, "r") as data:
        role = data["split_id"][:]
        eligible = (data["delta_t_seconds"][:] > 0) & (data["terminals"][:] == 0)
        rng = np.random.default_rng(seed)
        for name, code in (("fit", 0), ("stop", 1), ("calibrate", 2)):
            rows = np.flatnonzero((role == code) & eligible)
            if name == "fit":
                rows = np.sort(rng.choice(rows, min(len(rows), fit_cap), replace=False))
            selected[name] = rows
            result[name] = {key: data[array][rows] for key, array in (
                ("s", "observations"), ("a", "actions"), ("ns", "next_observations"),
                ("r", "rewards"), ("dt", "delta_t_seconds"))}
        stations = np.asarray(json.loads(data.attrs["station_max_by_line"]), np.float32)
    return result, selected, stations


def event_targets(raw, stations, device):
    s = torch.as_tensor(raw["s"], dtype=torch.float32, device=device)
    ns = torch.as_tensor(raw["ns"], dtype=torch.float32, device=device)
    slots = ns[:, 5:].reshape(-1, 16, 14)
    line = slots[:, :, 1].round().long()
    return dict(s=s, a=torch.as_tensor(raw["a"], device=device).float().reshape(-1, 1),
        ns=ns,
        g=torch.stack((torch.log1p(torch.as_tensor(raw["dt"], device=device).reshape(-1)),
            torch.log1p(-torch.as_tensor(raw["r"], device=device).reshape(-1)),
            (ns[:, 3] - s[:, 3]) * 20000, (ns[:, 4] - s[:, 4]) * 20000), 1).float(),
        count=(ns[:, 1] * 16).round().long() - 1, mask=slots[:, :, 0] > .5,
        line=line, station=(slots[:, :, 2] * stations[line]).round().long(),
        flags=slots[:, :, 11:13], local=slots[:, :, 3:11])


class Normalizer(nn.Module):
    def __init__(self, fit):
        super().__init__()
        for name, values in (("g", fit["g"]), ("local", fit["local"][fit["mask"]])):
            self.register_buffer(name + "_mean", values.mean(0))
            self.register_buffer(name + "_std", values.std(0).clamp_min(.05))


class EventModel(nn.Module):
    """Five shared-encoder heads, with sufficient station categories for Halifax."""
    def __init__(self, encoder_class, station_classes):
        super().__init__()
        self.encoder = encoder_class()
        self.station_classes = int(station_classes)
        self.slot_width = 12 + self.station_classes + 2 + 8
        self.width = 20 + 16 * self.slot_width
        base = lambda slot, column: 20 + self.slot_width * slot + column
        groups = dict(
            event=list(range(4, 20)) + [base(j, k) for j in range(16) for k in range(12+self.station_classes)],
            duration=[0], service=[2, 3] + [base(j, k) for j in range(16) for k in range(self.slot_width-4, self.slot_width)],
            headway=[base(j, k) for j in range(16) for k in range(12+self.station_classes, self.slot_width-4)],
            cost=[1])
        self.head = nn.ModuleDict({name: nn.Sequential(nn.Linear(257, 128), nn.SiLU(),
            nn.Linear(128, len(indices))) for name, indices in groups.items()})
        for name, indices in groups.items():
            self.register_buffer("outputs_" + name, torch.tensor(indices))

    def forward(self, states, actions):
        features = torch.cat((self.encoder(states), actions), -1)
        result = states.new_zeros((len(states), self.width))
        for name, head in self.head.items():
            result[:, getattr(self, "outputs_" + name)] = head(features)
        return result

    def unpack(self, output):
        slots = output[:, 20:].reshape(-1, 16, self.slot_width)
        end = 12 + self.station_classes
        return dict(g=output[:, :4], count=output[:, 4:20], line=slots[:, :, :12],
            station=slots[:, :, 12:end], flags=slots[:, :, end:end+2], local=slots[:, :, end+2:])


def prediction_losses(model, output, batch, norm):
    pred = model.unpack(output)
    mask = batch["mask"].float()
    average = lambda value: (value * mask).sum(1) / mask.sum(1).clamp_min(1)
    ce = lambda logits, labels: F.cross_entropy(logits.flatten(0, 1), labels.flatten(),
        reduction="none").reshape_as(mask)
    return torch.stack((
        ((pred["g"] - (batch["g"] - norm.g_mean) / norm.g_std) ** 2).mean(1),
        average(((pred["local"] - (batch["local"] - norm.local_mean) / norm.local_std) ** 2).mean(2)),
        F.cross_entropy(pred["count"], batch["count"], reduction="none"),
        average(ce(pred["line"], batch["line"])), average(ce(pred["station"], batch["station"])),
        average(F.binary_cross_entropy_with_logits(pred["flags"], batch["flags"], reduction="none").mean(2))), 1)


def fit_models(protocol, seed, folder, encoder_class, device, *, smoke=False):
    p = protocol["training"]
    raw, rows, station_values = load_model_roles(HERE / "target_pool.h5", p["model_split_seed"], p["world_model_fit_rows"])
    stations = torch.as_tensor(station_values, device=device)
    batches = {name: event_targets(value, stations, device) for name, value in raw.items()}
    norm = Normalizer(batches["fit"]).to(device)
    station_classes = int(stations.max().item()) + 1
    members = 1 if smoke else p["model_ensemble_members"]
    updates = 2 if smoke else p["model_updates_per_member"]
    models, reports = [], []
    for member in range(members):
        model_seed = seed + 50000 + member
        seed_all(model_seed)
        model = EventModel(encoder_class, station_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=p["model_learning_rate"])
        rng = np.random.default_rng(model_seed)
        best, selected, saved = float("inf"), 0, None
        for step in range(1, updates+1):
            ids = torch.as_tensor(rng.integers(len(batches["fit"]["s"]), size=p["model_batch_size"]), device=device)
            batch = {key: value[ids] for key, value in batches["fit"].items()}
            loss = prediction_losses(model, model(batch["s"], batch["a"]), batch, norm).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite native event-model fit")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            if step % p["model_validation_every"] == 0 or step == updates:
                with torch.no_grad():
                    stop = batches["stop"]
                    score = float(prediction_losses(model, model(stop["s"], stop["a"]), stop, norm).mean())
                if score < best:
                    best, selected, saved = score, step, copy.deepcopy(model.state_dict())
        model.load_state_dict(saved)
        model.eval().requires_grad_(False)
        models.append(model)
        reports.append(dict(member=member, updates=updates, selected_step=selected, stop_loss=best))
    with torch.no_grad():
        cal = batches["calibrate"]
        output = torch.stack([model(cal["s"], cal["a"]) for model in models]).mean(0)
        calibration = float(prediction_losses(models[0], output, cal, norm).mean())
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(dict(models=[model.state_dict() for model in models], normalizer=norm.state_dict(),
        station_classes=station_classes, selected_rows=rows), folder / "model.pt")
    report = dict(complete=True, seed=seed, models=reports, station_classes=station_classes,
        calibration_loss=calibration, rows={name: len(value) for name, value in rows.items()},
        normalization_role="fit", model_family="fixed direct shared five-head event model",
        calibration_usage="diagnostic only; no family, model coefficient or controller selection",
        target_test_rows_read=0, diagnostic=smoke)
    write(folder / "summary.json", report)
    print(json.dumps(report), flush=True)


class ModelContinuation:
    def __init__(self, protocol, seed, folder, encoder_class, device):
        self.p, self.device = protocol["training"], device
        raw, _, station_values = load_model_roles(HERE / "target_pool.h5", self.p["model_split_seed"], self.p["world_model_fit_rows"])
        self.stations = torch.as_tensor(station_values, device=device)
        self.fit = event_targets(raw["fit"], self.stations, device)
        self.norm = Normalizer(self.fit).to(device)
        payload = torch.load(folder / "model.pt", map_location=device, weights_only=False)
        self.norm.load_state_dict(payload["normalizer"])
        self.models = []
        for state in payload["models"]:
            model = EventModel(encoder_class, payload["station_classes"]).to(device)
            model.load_state_dict(state)
            self.models.append(model.eval().requires_grad_(False))
        frozen = HERE / "runtime_train_snapshot"
        trust_module = module("native_frozen_qdelta", frozen / "qdelta_bellman.py")
        mechanisms = module("native_frozen_trust", frozen / "mechanisms.py")
        self.normalized_update = mechanisms.update_normalized_trust
        seed_all(seed + 30000)
        self.trust = trust_module.QDeltaBellman(229, 1, gamma=self.p["qdelta_gamma"], device=device)
        self.rng = np.random.default_rng(seed + 30000)
        self.generator = torch.Generator(device=device).manual_seed(seed + 40000)
        with torch.no_grad():
            self.errors = torch.cat([prediction_losses(self.models[0], self.predict(self.fit["s"][i:i+256], self.fit["a"][i:i+256]),
                {key: value[i:i+256] for key, value in self.fit.items()}, self.norm).mean(1)
                for i in range(0, len(self.fit["s"]), 256)])

    @torch.no_grad()
    def predict(self, states, actions):
        return torch.stack([model(states, actions) for model in self.models]).mean(0)

    def update_trust(self, actor):
        ids = torch.as_tensor(self.rng.integers(len(self.fit["s"]), size=self.p["model_batch_size"]), device=self.device)
        return self.normalized_update(self.trust, self.fit["s"][ids], self.fit["a"][ids],
            self.fit["ns"][ids], torch.zeros(len(ids), device=self.device), self.errors[ids],
            lambda states: actor(states, deterministic=True)[0], self.p["qdelta_updates_per_step"])

    @torch.no_grad()
    def generate(self, actor):
        eligible = torch.nonzero(self.fit["s"][:, 0] < self.p["model_start_max_horizon_fraction"]).flatten()
        ids = eligible[torch.randint(len(eligible), (self.p["model_batch_size"],), generator=self.generator, device=self.device)]
        states = self.fit["s"][ids]
        actions = actor(states, deterministic=True)[0]
        following, reward, duration = self.decode(states, self.predict(states, actions))
        return dict(observations=states, actions=actions, next_observations=following,
            rewards=reward, delta_t_seconds=duration, discounts=torch.ones(len(ids), device=self.device),
            terminals=torch.zeros(len(ids), device=self.device)), self.trust.weight(states, actions)

    @torch.no_grad()
    def decode(self, states, output):
        pred = self.models[0].unpack(output)
        global_values = pred["g"] * self.norm.g_std + self.norm.g_mean
        horizon = float(self.p["horizon_seconds"])
        duration = torch.expm1(global_values[:, 0].clamp(0, 10)).clamp_min(1.)
        duration = torch.minimum(duration, (1-states[:, 0])*horizon-1)
        reward = -torch.expm1(global_values[:, 1].clamp(0, 20))
        draw = lambda logits: torch.multinomial(logits.softmax(-1).flatten(0, -2), 1,
            generator=self.generator).reshape(logits.shape[:-1])
        count = draw(pred["count"]) + 1
        line = draw(pred["line"])
        station_logits = pred["station"].clone()
        legal = torch.arange(station_logits.shape[-1], device=self.device)[None, None] <= self.stations[line, None]
        station_logits.masked_fill_(~legal, -torch.inf)
        station = draw(station_logits)
        mask = torch.arange(16, device=self.device)[None] < count[:, None]
        following = torch.zeros_like(states)
        following[:, 0] = states[:, 0] + duration / horizon
        following[:, 1] = count / 16
        following[:, 3:5] = (states[:, 3:5] + global_values[:, 2:4] / 20000).clamp(0, 1.25)
        slots = following[:, 5:].reshape(-1, 16, 14)
        slots[:, :, 0] = mask
        slots[:, :, 1] = line
        slots[:, :, 2] = station / self.stations[line]
        local = pred["local"] * self.norm.local_std + self.norm.local_mean
        lower = torch.tensor([0, 0, .001, .001, 0, 0, 0, 0], device=self.device)
        upper = torch.tensor([5, 5, 10, 10, 2, 1, 2, 2], device=self.device)
        slots[:, :, 3:11] = local.clamp(lower, upper)
        flags = torch.bernoulli(pred["flags"].sigmoid(), generator=self.generator)
        slots[:, :, 11:13] = flags
        slots[:, :, 3:5] = torch.where(flags > .5, slots[:, :, 3:5], 1.)
        slots[:, :, :13] *= mask[:, :, None]
        slots[:, :, 13] = -1.
        return following, reward, duration


def continuation_trainer(base_class):
    class NativeContinuationTrainer(base_class):
        augmentation = None
        model_loss = None

        def _ensemble_td_loss(self, predictions, targets, weights=None):
            real = super()._ensemble_td_loss(predictions, targets, weights)
            if self.model_loss is None:
                return real
            fraction = self.model_coefficient
            return (1-fraction)*real + fraction*self.model_loss

        def update(self, *, online, online_only=False, collect_metrics=True):
            self.model_loss = None
            metrics = {}
            if online and self.augmentation is not None:
                devices = [self.device.index or 0] if self.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(self.model_seed + self.total_updates)
                    if self.total_updates % self.trust_update_every == 0:
                        self.augmentation.update_trust(self.actor)
                    batch, weights = self.augmentation.generate(self.actor)
                    prediction = self.critic(batch["observations"], batch["actions"])
                    target = self._target(batch, backup=self.config.online_backup)
                    self.model_loss = super()._ensemble_td_loss(prediction, target, weights)
                    if collect_metrics:
                        metrics = dict(model_td_loss=float(self.model_loss.detach()), model_weight_mean=float(weights.mean()))
            try:
                result = super().update(online=online, online_only=online_only, collect_metrics=collect_metrics)
                result.update(metrics)
                return result
            finally:
                self.model_loss = None
    return NativeContinuationTrainer


def update_loop(learner, count, phase, output, *, online, log_every, offset=0):
    started = time.monotonic()
    for update in range(1, count+1):
        log = update % log_every == 0 or update == count
        metrics = learner.update(online=online, collect_metrics=log)
        if log:
            if not all(np.isfinite(value) for value in metrics.values()):
                raise RuntimeError(f"Nonfinite {phase} metrics")
            row = dict(phase=phase, phase_updates=update, completed_updates=offset+update,
                elapsed_seconds=time.monotonic()-started, metrics=metrics)
            write(output / "progress.json", row)
            print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", "initialize", "model", "adapt", "branch"), required=True)
    parser.add_argument("--domain", choices=("source", "target"))
    parser.add_argument("--branch", choices=("target_only", "target_model", "cross_city"))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    protocol = json.loads((HERE / "training_protocol.json").read_text())
    p = protocol["training"]
    torch.set_num_threads(1)
    trainer_class, config_class, replay_class, encoder_class = runtime_api()
    root = HERE / ("training_smoke" if args.smoke else "training")
    if args.phase == "all":
        phases = [("initialize", "--domain", domain) for domain in ("target", "source")]
        phases += [("model",)]
        phases += [("adapt", "--domain", domain) for domain in ("target", "source")]
        phases += [("branch", "--branch", branch) for branch in ("target_only", "target_model", "cross_city")]
        for phase in phases:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--phase", *phase,
                "--seed", str(args.seed), "--device", args.device]
            if args.smoke:
                command.append("--smoke")
            subprocess.run(command, check=True)
        if not args.smoke:
            runtime = json.loads((HERE / "runtime.json").read_text())
            for branch in protocol["branches"]:
                for case in protocol["evaluation"]["cases"]:
                    subprocess.run([sys.executable, "-u", str(HERE / "evaluate_native.py"),
                        "--method", branch, "--seed", str(args.seed), "--case", str(case["case"]),
                        "--assets", runtime["assets_root"] + f"/test{case['case']:02d}/asset_manifest.json",
                        "--checkpoint", str(root / "branches" / branch / f"seed{args.seed}" / "policy.pt"),
                        "--output", str(HERE / "evaluation" / branch / f"seed{args.seed}" / f"case{case['case']:02d}")],
                        check=True)
        write(root / f"seed{args.seed}_complete.json", dict(complete=True, seed=args.seed,
            branches=["target_only", "target_model", "cross_city"], diagnostic=args.smoke,
            evaluation_days_per_branch=0 if args.smoke else len(protocol["evaluation"]["cases"])))
        return
    model_folder = root / "models" / f"seed{args.seed}"
    if args.phase == "model":
        fit_models(protocol, args.seed, model_folder, encoder_class, args.device, smoke=args.smoke)
        return
    config = config_class(**p["trainer_config"])
    config = replace(config, offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    if args.phase == "initialize":
        if args.domain is None:
            parser.error("initialize requires --domain")
        folder = root / "init" / args.domain / f"seed{args.seed}"
        dataset = Path(p["source_dataset"]) if args.domain == "source" else HERE / "target_pool.h5"
        replay = replay_class(dataset, device=args.device, max_offline_rows=0, online_capacity=1)
        replay.materialize_device_cache()
        learner = trainer_class(replay, config=config, device=args.device, seed=args.seed)
        count = 5 if args.smoke else p["initialization_updates"]
        update_loop(learner, count, "initialization", folder, online=False, log_every=p["log_every"])
        learner.save_offline_initializer(folder / "initializer.pt", extra=dict(seed=args.seed,
            domain=args.domain, fresh_initialization=True, initialization_updates=count))
        write(folder / "summary.json", dict(complete=True, seed=args.seed, domain=args.domain,
            offline_rows=replay.offline_size, initialization_updates=count, target_test_rows_read=0,
            config=asdict(config), diagnostic=args.smoke))
        return
    if args.phase == "branch" and args.branch is None:
        parser.error("branch phase requires --branch")
    if args.phase == "adapt" and args.domain is None:
        parser.error("adapt phase requires --domain")
    domain = args.domain if args.phase == "adapt" else "source" if args.branch == "cross_city" else "target"
    adapted = root / "adapt" / domain / f"seed{args.seed}"
    folder = adapted if args.phase == "adapt" else root / "branches" / args.branch / f"seed{args.seed}"
    replay = replay_class(HERE / "target_pool.h5", device=args.device, max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    learner = continuation_trainer(trainer_class)(replay, config=config, device=args.device, seed=args.seed)
    expected = 5 if args.smoke else p["initialization_updates"]
    adapt = 3 if args.smoke else p["adaptation_updates"]
    if args.phase == "adapt":
        payload = torch.load(root / "init" / domain / f"seed{args.seed}" / "initializer.pt",
            map_location=args.device, weights_only=False)
        if payload["total_updates"] != expected:
            raise ValueError("Initializer update count differs from the frozen protocol")
        transfer_parameters(learner, payload, reset_line_columns=(domain == "source"))
        del payload
        seed_all(args.seed + p["adaptation_rng_offset"])
        learner.rng = np.random.default_rng(args.seed + p["adaptation_rng_offset"])
        update_loop(learner, adapt, "target_adaptation", folder, online=False, log_every=p["log_every"], offset=expected)
        report = dict(complete=True, domain=domain, seed=args.seed, initialization_updates=expected,
            adaptation_updates=adapt, optimizer_transfer=False, source_line_onehot_transfer=False,
            source_arrival_rate_transfer=False, diagnostic=args.smoke)
        learner.save(folder / "adapted.pt", extra=report, include_online_replay=False)
        write(folder / "summary.json", report)
        return
    learner.load(adapted / "adapted.pt", load_optimizers=True)
    if learner.total_updates != adapt:
        raise ValueError("Adapted checkpoint update count differs from the frozen protocol")
    if args.branch != "target_only":
        learner.augmentation = ModelContinuation(protocol, args.seed, model_folder, encoder_class, args.device)
    learner.model_coefficient = p["model_loss_coefficient"]
    learner.model_seed = args.seed + 60000
    learner.trust_update_every = p["trust_update_every"]
    seed_all(args.seed + p["continuation_rng_offset"])
    learner.rng = np.random.default_rng(args.seed + p["continuation_rng_offset"])
    count = 4 if args.smoke else p["continuation_updates"]
    update_loop(learner, count, "continuation", folder, online=True, log_every=p["log_every"], offset=expected+adapt)
    report = dict(complete=True, seed=args.seed, branch=args.branch, initialization_domain=domain,
        initialization_updates=expected, adaptation_updates=adapt, continuation_updates=count,
        total_policy_updates=expected+adapt+count, target_unique_rows=16384,
        actor_target_fit_rows=replay.offline_size, target_online_training_events=0,
        continuation_real_td_coefficient=1. if args.branch == "target_only" else 1-p["model_loss_coefficient"],
        continuation_model_td_coefficient=0. if args.branch == "target_only" else p["model_loss_coefficient"],
        optimizer_transfer=False, source_line_onehot_transfer=False,
        checkpoint_selection="fixed final", diagnostic=args.smoke)
    learner.save_policy_artifact(folder / "policy.pt", extra=report)
    write(folder / "train_summary.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
