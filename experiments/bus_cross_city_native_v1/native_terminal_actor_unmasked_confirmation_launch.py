"""Confirm all frozen unmasked75k and repaired55k actors on fresh demand."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "runtime.json").read_text())
PREFIX = "native_terminal_actor_unmasked_confirmation"
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
    from launch import run_remote
    from training_launch import job, submit
    unmasked = json.loads((HERE / "native_terminal_actor_unmasked_reporting.json").read_text())
    repaired = json.loads((HERE / "native_terminal_adaptation_reporting.json").read_text())
    assets = json.loads((HERE / (PREFIX + "_assets_summary.json")).read_text())
    seeds = {str(seed) for seed in P["training_seeds"]}
    if not (unmasked["complete"] and repaired["complete"] and assets["complete"]
            and unmasked["training_ready"] == repaired["training_ready"] == len(seeds)
            and set(unmasked["training"]) == {"real_v_unmasked/seed" + seed for seed in seeds}
            and set(repaired["training"]) == seeds):
        raise ValueError("Frozen actors or fresh assets are incomplete")
    paths = [R["remote_root"] + f"/{spec['artifact_folder']}/seed{seed}/policy.pt"
        for spec in P["branches"].values() for seed in P["training_seeds"]]
    code = "from pathlib import Path\npaths=" + repr(paths) + "\n"
    code += "assert all(Path(p).is_file() and Path(p).stat().st_size > 0 for p in paths), 'Missing frozen actor'\n"
    code += "print('Six frozen policy artifacts present')\n"
    print(run_remote(shlex.join([R["python"], "-c", code])).strip(), flush=True)
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
            description=f"Frozen actor/rule confirmation {branch} {seed}; three fresh cases")
        rows.append(row)
    submit(PREFIX, rows)


def run(branch, seed):
    from iql_actor_continuation import evaluate as evaluate_policy
    evaluate_policy(branch, seed, protocol=P, prefix=PREFIX)


def collect():
    from launch import run_remote
    from fresh_confirmation_summary import key, summarize
    code = "import sys,json\nfrom pathlib import Path\nsys.path.insert(0," + repr(R["remote_root"]) + ")\n"
    code += "from fresh_confirmation_summary import compact_remote\n"
    code += "p=json.loads(Path(" + repr(R["remote_root"] + "/" + PREFIX + "_protocol.json") + ").read_text())\n"
    code += "print(json.dumps(compact_remote(p)))\n"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report = summarize(rows, P)
    if report["complete"]:
        for comparison in report["comparison"].values():
            for pair in comparison["pairs"]:
                seed, case = pair["training_seed"], pair["case"]
                left = rows[key(comparison["left"], seed, case)]
                right_seed = None if comparison["right"] == "strong_rule" else seed
                right = rows[key(comparison["right"], right_seed, case)]
                pair["completed_delta"] = left["completed"] - right["completed"]
                pair["departed_delta"] = left["departed"] - right["departed"]
            comparison["paired_completed_delta_sum"] = sum(pair["completed_delta"] for pair in comparison["pairs"])
            comparison["better_demand_case_means"] = sum(row["cost"] < 0 for row in comparison["by_case"].values())
        report["new_training_updates"] = P["new_training_updates"]
        report["world_model_refit_updates"] = P["world_model_refit_updates"]
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
