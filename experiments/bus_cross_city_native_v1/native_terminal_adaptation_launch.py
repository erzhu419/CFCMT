"""Launch fixed terminal-repaired adaptation and retrieve small JSON results."""

import argparse
import json
import shlex
import subprocess

from launch import HERE, R, run_remote, scheduler
from training_launch import job, submit as submit_jobs

PREFIX = "native_terminal_adaptation"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


def stage():
    names = (PREFIX + ".py", PREFIX + "_launch.py", PREFIX + "_protocol.json")
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        *[str(HERE / name) for name in names],
        scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
        check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=list(names))), flush=True)


def submit(admission=False):
    if not admission:
        report = json.loads((HERE / (PREFIX + "_admission_summary.json")).read_text())
        if not report["complete"]:
            raise ValueError("Actual terminal-repaired adaptation admission has not completed")
    mode = "admission" if admission else "train"
    seeds = P["training_seeds"][:1] if admission else P["training_seeds"]
    rows = []
    for seed in seeds:
        command = [R["python"], "-u", R["remote_root"] + "/" + PREFIX + ".py", "--seed", str(seed)]
        if admission:
            command.append("--admission")
        row = job(command, PREFIX + f"/{mode}/seed{seed}", training=True)
        row.update(signature=P["name"] + f"/{mode}/seed{seed}", resource_family=P["name"],
            description=f"Fixed terminal-repaired target adaptation {mode} seed{seed}",
            allow_cpu_training=True,
            cpu_training_justification="Retain the frozen CPU target IQL adaptation and source initializers; only repaired native terminal labels change.")
        rows.append(row)
    submit_jobs(PREFIX + "_" + mode, rows)


def collect_admission():
    path = R["remote_root"] + "/" + PREFIX + "/admission/summary.json"
    code = "from pathlib import Path\nimport json\np=Path(" + repr(path) + ")\n"
    code += "print(p.read_text() if p.exists() else json.dumps(dict(complete=False)))\n"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if report["complete"] and (report["protocol"] != P["name"] or report["seed"] != P["training_seeds"][0]):
        raise ValueError("Wrong terminal-repaired adaptation admission result")
    (HERE / (PREFIX + "_admission_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=report["complete"])), flush=True)


def collect():
    seeds = P["training_seeds"]
    expected = [f"seed{seed}/evaluation/case{case['case']:02d}/summary.json"
        for seed in seeds for case in P["evaluation"]["cases"]]
    code = "from pathlib import Path\nimport json\nroot=Path(" + repr(R["remote_root"] + "/" + PREFIX) + ")\n"
    code += "training={}\n"
    code += "for seed in " + repr(seeds) + ":\n"
    code += "    folder=root/f'seed{seed}'\n    row={}\n"
    code += "    for name in ('progress.json','train_summary.json'):\n"
    code += "        path=folder/name\n        if path.exists(): row[name]=json.loads(path.read_text())\n"
    code += "    training[str(seed)]=row\n"
    code += "evaluation={}\nfor name in " + repr(expected) + ":\n"
    code += "    path=root/name\n    if path.exists(): evaluation[name]=json.loads(path.read_text())\n"
    code += "print(json.dumps(dict(training=training,evaluation=evaluation)))\n"
    remote = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    training_ready = sum(row.get("train_summary.json", {}).get("complete", False)
        for row in remote["training"].values())
    evaluation_ready = sum(remote["evaluation"].get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=P["name"], complete=training_ready == len(seeds) and evaluation_ready == len(expected),
        training_ready=training_ready, expected_training=len(seeds), ready=evaluation_ready,
        expected=len(expected), training=remote["training"], evaluation=remote["evaluation"])
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "training_ready", "expected_training", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "admission", "collect-admission", "train", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage()
    elif mode in ("admission", "train"):
        submit(admission=mode == "admission")
    elif mode == "collect-admission":
        collect_admission()
    else:
        collect()
