"""Confirm the three fixed repaired actors on three prospective demand days."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "runtime.json").read_text())
PREFIX = "native_terminal_confirmation"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


def stage():
    from launch import scheduler
    names = (PREFIX + "_launch.py", PREFIX + "_protocol.json")
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        *[str(HERE / name) for name in names],
        scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
        check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=list(names))), flush=True)


def prepare():
    from fresh_confirmation_launch import prepare as prepare_assets
    prepare_assets(P, PREFIX)


def evaluate():
    from training_launch import job, submit
    prior = json.loads((HERE / "native_terminal_adaptation_reporting.json").read_text())
    assets = json.loads((HERE / (PREFIX + "_assets_summary.json")).read_text())
    seeds = {str(seed) for seed in P["training_seeds"]}
    if not (prior["complete"] and prior["training_ready"] == len(seeds)
            and set(prior["training"]) == seeds and assets["complete"]):
        raise ValueError("Repaired fixed models or prospective assets are incomplete")
    choices = [(branch, seed) for branch in P["branches"] for seed in P["training_seeds"]]
    choices.append(("strong_rule", None))
    rows = []
    for branch, seed in choices:
        command = [R["python"], "-u", R["remote_root"] + "/" + PREFIX + "_launch.py",
            "run", "--branch", branch]
        if seed is not None:
            command += ["--seed", str(seed)]
        row = job(command, PREFIX + f"/{branch}/{seed}")
        row.update(signature=P["name"] + f"/{branch}/{seed}", resource_family=P["name"],
            description=f"Fixed repaired actor/rule confirmation {branch} {seed}; three prospective cases")
        rows.append(row)
    submit(PREFIX, rows)


def run(branch, seed):
    from iql_actor_continuation import evaluate as evaluate_policy
    evaluate_policy(branch, seed, protocol=P, prefix=PREFIX)


def collect():
    from launch import run_remote
    from fresh_confirmation_summary import summarize
    code = "import sys,json\nfrom pathlib import Path\nsys.path.insert(0," + repr(R["remote_root"]) + ")\n"
    code += "from fresh_confirmation_summary import compact_remote\n"
    code += "p=json.loads(Path(" + repr(R["remote_root"] + "/" + PREFIX + "_protocol.json") + ").read_text())\n"
    code += "print(json.dumps(compact_remote(p)))\n"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report = summarize(rows, P)
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "prepare", "evaluate", "run", "collect"))
    parser.add_argument("--branch", choices=(*P["branches"], "strong_rule"))
    parser.add_argument("--seed", type=int, choices=P["training_seeds"])
    args = parser.parse_args()
    if args.mode == "run":
        if args.branch is None or ((args.branch == "strong_rule") != (args.seed is None)):
            parser.error("A learned branch needs its fixed seed; the rule has no training seed")
        run(args.branch, args.seed)
    else:
        {"stage": stage, "prepare": prepare, "evaluate": evaluate, "collect": collect}[args.mode]()
