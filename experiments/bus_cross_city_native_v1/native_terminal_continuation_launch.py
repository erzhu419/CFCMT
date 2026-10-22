"""Launch the fixed three-arm repaired-label continuation comparison."""

import argparse
import json
import shlex
import subprocess

from launch import HERE, R, run_remote, scheduler
from training_launch import job, submit as submit_jobs

PREFIX = "native_terminal_continuation"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


def stage():
    names = (PREFIX + ".py", PREFIX + "_admission.py", PREFIX + "_launch.py", PREFIX + "_protocol.json")
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        *[str(HERE / name) for name in names],
        scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
        check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=list(names))), flush=True)


def submit(admission=False):
    if not json.loads((HERE / P["references"]["summary"]).read_text())["complete"]:
        raise ValueError("Fixed repaired model references are incomplete")
    if not admission:
        report = json.loads((HERE / (PREFIX + "_admission_summary.json")).read_text())
        if not (report["complete"] and report["protocol"] == P["name"]
                and report["seed"] == P["training_seeds"][0]
                and set(report["arms"]) == set(P["branches"])):
            raise ValueError("Actual three-arm continuation admission is incomplete")
    choices = [(None, P["training_seeds"][0])] if admission else [
        (branch, seed) for branch in P["branches"] for seed in P["training_seeds"]]
    mode, rows = ("admission" if admission else "train"), []
    for branch, seed in choices:
        if admission:
            command = [R["python"], "-u", R["remote_root"] + "/" + PREFIX + "_admission.py"]
        else:
            command = [R["python"], "-u", R["remote_root"] + "/" + PREFIX + ".py",
                "train", "--branch", branch, "--seed", str(seed)]
        row = job(command, PREFIX + f"/{mode}/{branch}/seed{seed}", training=True)
        row.update(signature=P["name"] + f"/{mode}/{branch}/seed{seed}", resource_family=P["name"],
            description=f"Repaired-label continuation {mode} {branch} seed{seed}; fixed final20k",
            allow_cpu_training=True,
            cpu_training_justification="Keep original frozen CPU IQL actor/entropy-OFF critic continuation; compare real TD, unit-weight WM and native normalized confidence at fixed budgets.")
        rows.append(row)
    submit_jobs(PREFIX + "_" + mode, rows)


def collect_admission():
    path = R["remote_root"] + "/" + PREFIX + "/admission.json"
    code = "from pathlib import Path\nimport json\np=Path(" + repr(path) + ")\n"
    code += "print(p.read_text() if p.exists() else json.dumps(dict(complete=False)))\n"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if report["complete"] and (report["protocol"] != P["name"] or report["seed"] != P["training_seeds"][0]):
        raise ValueError("Wrong continuation admission result")
    (HERE / (PREFIX + "_admission_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=report["complete"])), flush=True)


def collect():
    folders = [f"{branch}/seed{seed}" for branch in P["branches"] for seed in P["training_seeds"]]
    expected = [folder + f"/case{case['case']:02d}/summary.json"
        for folder in folders for case in P["evaluation"]["cases"]]
    code = "from pathlib import Path\nimport json\nroot=Path(" + repr(R["remote_root"] + "/" + PREFIX) + ")\ntraining={}\n"
    code += "for folder in " + repr(folders) + ":\n    row={}\n"
    code += "    for name in ('progress.json','train_summary.json'):\n        path=root/folder/name\n"
    code += "        if path.exists(): row[name]=json.loads(path.read_text())\n    training[folder]=row\n"
    code += "evaluation={}\nfor name in " + repr(expected) + ":\n    path=root/name\n"
    code += "    if path.exists(): evaluation[name]=json.loads(path.read_text())\n"
    code += "print(json.dumps(dict(training=training,evaluation=evaluation)))\n"
    remote = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    trained = sum(row.get("train_summary.json", {}).get("complete", False) for row in remote["training"].values())
    evaluated = sum(remote["evaluation"].get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=P["name"], complete=trained == len(folders) and evaluated == len(expected),
        training_ready=trained, expected_training=len(folders), ready=evaluated, expected=len(expected), **remote)
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
