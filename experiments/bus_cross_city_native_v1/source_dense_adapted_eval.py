"""Export fixed adapted actors and evaluate them without any continuation."""

import argparse
import json
import subprocess
import sys

import h5py
import torch

from diagnose_values import learner_from_payload
from source_dense_continuation import campaign_protocol
from train_native import HERE, runtime_api, write


def evaluate(seed, *, variant="adapted"):
    prefix, protocol = campaign_protocol(variant)
    local = variant == "target-adapted"
    original = json.loads((HERE / "training_protocol.json").read_text())
    runtime = json.loads((HERE / "runtime.json").read_text())
    if seed not in protocol["training_seeds"] or protocol["evaluation"] != original["evaluation"]:
        raise ValueError("Adapted evaluation must retain the original seeds and cases")
    torch.set_num_threads(1)
    trainer_class, config_class, _, _ = runtime_api()
    from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7
    with h5py.File(HERE / "target_pool.h5", "r") as data:
        contract = json.loads(data.attrs["config"])
    source = HERE / ("training/adapt/target" if local else "source_dense_adaptation") / f"seed{seed}" / "adapted.pt"
    payload = torch.load(source, map_location="cpu", weights_only=False)
    saved = payload["extra"]
    expected = dict(seed=seed, initialization_updates=protocol["initialization_updates"],
        adaptation_updates=protocol["adaptation_updates"])
    if local:
        expected.update(domain="target", diagnostic=False)
        initial = json.loads((HERE / "training/init/target" / f"seed{seed}" / "summary.json").read_text())
        initial_expected = dict(complete=True, domain="target", seed=seed, diagnostic=False,
            initialization_updates=protocol["initialization_updates"], offline_rows=protocol["target_fit_rows"])
        if any(initial[key] != value for key, value in initial_expected.items()):
            raise ValueError("Wrong fixed target initializer or fit budget")
    else:
        expected.update(target_fit_rows=protocol["target_fit_rows"], target_union_rows=protocol["target_union_rows"])
    if payload["total_updates"] != protocol["adaptation_updates"] or any(saved[key] != value for key, value in expected.items()):
        raise ValueError("Wrong fixed adapted checkpoint")
    learner = learner_from_payload(trainer_class, config_class, contract, payload, seed)
    folder = HERE / prefix / f"seed{seed}"
    report = dict(complete=True, protocol=protocol["name"], seed=seed,
        branch="target_only" if local else "cross_city", variant="target_adapted" if local else "full_source_adapted",
        initialization_domain="target" if local else "source", admission=False,
        initialization_updates=protocol["initialization_updates"],
        adaptation_updates=protocol["adaptation_updates"], continuation_updates=0,
        total_policy_updates=protocol["initialization_updates"]+protocol["adaptation_updates"],
        checkpoint_stored_updates=payload["total_updates"], export_training_updates=0,
        target_unique_rows=protocol["target_union_rows"], actor_target_fit_rows=protocol["target_fit_rows"],
        source_rows=protocol["source_rows"], target_online_training_events=0,
        continuation_real_td_coefficient=None, continuation_model_td_coefficient=None,
        world_model_refit_updates=0, checkpoint_selection="fixed final adaptation", source_checkpoint=str(source))
    artifact = folder / "policy.pt"
    learner.save_policy_artifact(artifact, extra=report)
    actor, _ = load_actor_checkpoint_v7(artifact, "cpu", require_policy_artifact=True)
    actual = actor.state_dict()
    if actual.keys() != payload["actor"].keys() or any(
            not torch.equal(actual[key], value) for key, value in payload["actor"].items()):
        raise ValueError("Deployment actor differs from the adapted checkpoint")
    report["actor_export_equal"] = True
    write(folder / "export_summary.json", report)
    del actor, learner, payload
    for case in protocol["evaluation"]["cases"]:
        subprocess.run([sys.executable, "-u", str(HERE / "evaluate_native.py"),
            "--method", report["branch"], "--seed", str(seed), "--case", str(case["case"]),
            "--assets", runtime["assets_root"] + f"/test{case['case']:02d}/asset_manifest.json",
            "--checkpoint", str(artifact),
            "--output", str(folder / "evaluation" / f"case{case['case']:02d}")], check=True)
    write(folder / "summary.json", dict(report, evaluation_cases=len(protocol["evaluation"]["cases"]),
        evaluation_seconds=protocol["evaluation"]["episode_seconds"]))
    print(json.dumps(dict(complete=True, seed=seed, continuation_updates=0,
        actor_export_equal=True, evaluation_cases=3)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--campaign", choices=("adapted", "target-adapted"), default="adapted")
    args = parser.parse_args()
    evaluate(args.seed, variant=args.campaign)
