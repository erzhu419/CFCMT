"""Fixed full-source continuation and paired original-case closed-loop evaluation."""

import argparse
from dataclasses import asdict, replace
import json
import subprocess
import sys

import numpy as np
import torch

from train_native import (HERE, ModelContinuation, continuation_trainer,
    runtime_api, seed_all, update_loop, write)


def campaign_protocol(variant):
    prefix = {"model-td": "source_dense_continuation", "real-td": "source_dense_real_only",
              "adapted": "source_dense_adapted_eval", "iql": "source_dense_iql_continuation",
              "target-adapted": "target_adapted_eval"}[variant]
    return prefix, json.loads((HERE / (prefix + "_protocol.json")).read_text())


def continue_and_evaluate(seed, *, admission=False, variant="model-td"):
    prefix, campaign = campaign_protocol(variant)
    original = json.loads((HERE / "training_protocol.json").read_text())
    runtime = json.loads((HERE / "runtime.json").read_text())
    p = original["training"]
    torch.set_num_threads(1)
    trainer_class, config_class, replay_class, encoder_class = runtime_api()
    config = replace(config_class(**p["trainer_config"]),
        offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    folder = HERE / prefix / ("admission" if admission else f"seed{seed}")
    replay = replay_class(HERE / "target_pool.h5", device="cpu", max_offline_rows=0, online_capacity=1)
    replay.materialize_device_cache()
    if replay.offline_size != campaign["target_fit_rows"]:
        raise ValueError("Continuation must use the original target-fit rows")
    learner = continuation_trainer(trainer_class)(replay, config=config, device="cpu", seed=seed)
    learner.load(HERE / "source_dense_adaptation" / f"seed{seed}" / "adapted.pt", load_optimizers=True)
    if learner.total_updates != campaign["adaptation_updates"]:
        raise ValueError("Adapted checkpoint update count differs from the frozen protocol")
    online_updates_before = learner.online_critic_updates
    log_alpha_before = float(learner.log_alpha.detach())
    model_folder = HERE / "training/models" / f"seed{seed}"
    online = variant != "iql"
    if online:
        learner.augmentation = ModelContinuation(original, seed, model_folder, encoder_class, "cpu")
    learner.model_coefficient = campaign.get("model_loss_coefficient", p["model_loss_coefficient"])
    learner.model_seed = seed+60000
    learner.trust_update_every = p["trust_update_every"]
    seed_all(seed+p["continuation_rng_offset"])
    learner.rng = np.random.default_rng(seed+p["continuation_rng_offset"])
    count = 4 if admission else campaign["continuation_updates"]
    phase = {"model-td": "full_source_model_continuation", "real-td": "full_source_real_td_continuation",
             "iql": "full_source_iql_continuation"}[variant]
    update_loop(learner, count, phase, folder,
        online=online, log_every=p["log_every"], offset=campaign["initialization_updates"]+campaign["adaptation_updates"])
    if learner.total_updates != campaign["adaptation_updates"]+count:
        raise ValueError("Learner continuation count differs from the fixed budget")
    if not online and (learner.online_critic_updates != online_updates_before
            or float(learner.log_alpha.detach()) != log_alpha_before):
        raise ValueError("IQL continuation changed SAC critic or temperature state")
    report = dict(complete=True, protocol=campaign["name"], seed=seed,
        branch="cross_city", variant={"model-td": "full_source", "real-td": "full_source_real_td", "iql": "full_source_iql"}[variant], initialization_updates=campaign["initialization_updates"],
        adaptation_updates=campaign["adaptation_updates"], continuation_updates=count,
        total_policy_updates=campaign["initialization_updates"]+campaign["adaptation_updates"]+count,
        target_unique_rows=16384, actor_target_fit_rows=replay.offline_size, source_rows=3284718,
        target_online_training_events=0, continuation_real_td_coefficient=1-learner.model_coefficient,
        continuation_model_td_coefficient=learner.model_coefficient,
        continuation_kernel="offline_iql" if not online else "online_sac",
        continuation_online_flag=online, effective_critic_backup=learner.config.online_backup if online else "iql",
        online_critic_updates_added=learner.online_critic_updates-online_updates_before,
        log_alpha_before=log_alpha_before, log_alpha_after=float(learner.log_alpha.detach()),
        world_model_artifact=str(model_folder / "model.pt") if online else None, world_model_refit_updates=0,
        config=asdict(learner.config), checkpoint_selection="fixed final", admission=admission)
    learner.save_policy_artifact(folder / "policy.pt", extra=report)
    # Preserve final Q/V for diagnosis; exact mid-continuation resume is not requested.
    learner.save(folder / "final_learner.pt", extra=report, include_online_replay=False)
    write(folder / "train_summary.json", report)
    del learner, replay
    cases = campaign["evaluation"]["cases"][:1] if admission else campaign["evaluation"]["cases"]
    for case in cases:
        subprocess.run([sys.executable, "-u", str(HERE / "evaluate_native.py"),
            "--method", "cross_city", "--seed", str(seed), "--case", str(case["case"]),
            "--assets", runtime["assets_root"] + f"/test{case['case']:02d}/asset_manifest.json",
            "--checkpoint", str(folder / "policy.pt"),
            "--output", str(folder / "evaluation" / f"case{case['case']:02d}")], check=True)
    write(folder / "summary.json", dict(report, evaluation_cases=len(cases), evaluation_seconds=24000))
    print(json.dumps(dict(complete=True, seed=seed, continuation_updates=count,
        evaluation_cases=len(cases), admission=admission)), flush=True)


def compact_result(seed, *, admission=False, variant="model-td"):
    prefix, _ = campaign_protocol(variant)
    folder = HERE / prefix / ("admission" if admission else f"seed{seed}")
    if not (folder / "summary.json").exists():
        return None
    evaluation = {}
    for path in (folder / "evaluation").glob("case*/summary.json"):
        row = json.loads(path.read_text())
        passengers = row["passengers"]
        evaluation[str(row["case"])] = dict(complete=row["complete"], case=row["case"],
            training_seed=row["training_seed"], demand_seed=row["demand_seed"], sumo_seed=row["sumo_seed"],
            horizon_seconds=row["horizon_seconds"], simulator_version=row["simulator_version"], line_ids=row["line_ids"],
            cost=row["generalized_seconds_per_departed_passenger"],
            departed=passengers["departed_passengers"], completed=passengers["completed_passengers"],
            completion=passengers["completion_rate"], ledger_balance_error=passengers["active_state_count_balance_error"],
            wait_seconds=passengers["accrued_wait_seconds_per_departed_passenger"],
            ride_seconds=passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
            decision_events=row["decision_events"], mean_executed_hold_seconds=row["mean_hold_seconds"],
            positive_holds=row["positive_holds"], service_departed=row["service"]["departed_trips"],
            service_completed=row["service"]["completed_trips"])
    return dict(training=json.loads((folder / "summary.json").read_text()), evaluation=evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--admission", action="store_true")
    parser.add_argument("--campaign", choices=("model-td", "real-td", "iql"), default="model-td")
    args = parser.parse_args()
    _, campaign = campaign_protocol(args.campaign)
    original = json.loads((HERE / "training_protocol.json").read_text())
    if args.seed not in campaign["training_seeds"]:
        parser.error("Seed must belong to the original fixed roster")
    if campaign["evaluation"] != original["evaluation"]:
        raise ValueError("Evaluation cases or protocol differ from the existing pilot")
    continue_and_evaluate(args.seed, admission=args.admission, variant=args.campaign)
