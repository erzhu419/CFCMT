"""Launch the corrected-pool local initialization-route comparison."""

import argparse
import json
import shlex
import subprocess

from launch import HERE, R, run_remote, scheduler
from training_launch import job, submit as submit_jobs

PREFIX = "native_terminal_target_only"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


def stage():
    names = (PREFIX + ".py", PREFIX + "_launch.py", PREFIX + "_protocol.json", PREFIX + "_report.py")
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        *[str(HERE / name) for name in names],
        scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
        check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=list(names))), flush=True)


def submit(admission=False):
    if not admission:
        report = json.loads((HERE / (PREFIX + "_admission_summary.json")).read_text())
        if not (report["complete"] and report["protocol"] == P["name"]
                and report["seed"] == P["training_seeds"][0]
                and report["initialization_updates"] == report["adaptation_updates"] == 3):
            raise ValueError("Original two-stage target IQL admission is incomplete")
    mode = "admission" if admission else "train"
    seeds = P["training_seeds"][:1] if admission else P["training_seeds"]
    rows = []
    for seed in seeds:
        command = [R["python"], "-u", R["remote_root"] + "/" + PREFIX + ".py", "--seed", str(seed)]
        if admission:
            command.append("--admission")
        row = job(command, PREFIX + f"/{mode}/seed{seed}", training=True)
        row.update(signature=P["name"] + f"/{mode}/seed{seed}", resource_family=P["name"],
            description=f"Corrected-pool target-only IQL 50000+5000 {mode} seed{seed}",
            allow_cpu_training=True,
            cpu_training_justification="Retain original CPU IQL initialization/adaptation settings and total update budget; only target terminal labels change.")
        rows.append(row)
    submit_jobs(PREFIX + "_" + mode, rows)


def collect_admission():
    path = R["remote_root"] + "/" + PREFIX + "/admission/summary.json"
    code = "from pathlib import Path\nimport json\np=Path(" + repr(path) + ")\n"
    code += "print(p.read_text() if p.exists() else json.dumps(dict(complete=False)))\n"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if report["complete"] and (report["protocol"] != P["name"] or report["seed"] != P["training_seeds"][0]):
        raise ValueError("Wrong target-only admission result")
    (HERE / (PREFIX + "_admission_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=report["complete"])), flush=True)


def collect():
    expected = [f"seed{seed}/evaluation/case{case['case']:02d}/summary.json"
        for seed in P["training_seeds"] for case in P["evaluation"]["cases"]]
    code = "from pathlib import Path\nimport json\nroot=Path(" + repr(R["remote_root"] + "/" + PREFIX) + ")\ntraining={}\n"
    code += "for seed in " + repr(P["training_seeds"]) + ":\n    folder=root/f'seed{seed}'\n    row={}\n"
    code += "    for name,path in [('initialization_progress.json',folder/'initialization/progress.json'),('progress.json',folder/'progress.json'),('train_summary.json',folder/'train_summary.json')]:\n"
    code += "        if path.exists(): row[name]=json.loads(path.read_text())\n    training[str(seed)]=row\n"
    code += "evaluation={}\nfor name in " + repr(expected) + ":\n    path=root/name\n    if path.exists(): evaluation[name]=json.loads(path.read_text())\n"
    code += "print(json.dumps(dict(training=training,evaluation=evaluation)))\n"
    remote = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    trained = sum(row.get("train_summary.json", {}).get("complete", False) for row in remote["training"].values())
    evaluated = sum(remote["evaluation"].get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=P["name"], complete=trained == len(P["training_seeds"]) and evaluated == len(expected),
        training_ready=trained, expected_training=len(P["training_seeds"]), ready=evaluated, expected=len(expected), **remote)
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key:report[key] for key in ("complete", "training_ready", "ready", "expected")}), flush=True)


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
