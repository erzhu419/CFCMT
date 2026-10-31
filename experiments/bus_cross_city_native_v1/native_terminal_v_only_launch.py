"""Dispatch the fixed V-only intervention and collect small result JSON only."""

import argparse
import json
import shlex
import subprocess

from launch import HERE, R, run_remote, scheduler
from training_launch import job, submit as submit_jobs

PREFIX = "native_terminal_v_only"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


def stage():
    names = [PREFIX + suffix for suffix in (".py", "_admission.py", "_protocol.json")]
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        *[str(HERE / name) for name in names],
        scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
        check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=names)), flush=True)


def submit(admission=False):
    for reference in (P["references"], P["matched_reference"]):
        if not json.loads((HERE / reference["summary"]).read_text())["complete"]:
            raise ValueError("Fixed reference results are incomplete")
    if not admission:
        report = json.loads((HERE / (PREFIX + "_admission_summary.json")).read_text())
        if not (report["complete"] and report["protocol"] == P["name"]
                and report["seed"] == P["training_seeds"][0]):
            raise ValueError("Actual V-only continuation admission is incomplete")
    mode = "admission" if admission else "train"
    rows = []
    for seed in P["training_seeds"][:1] if admission else P["training_seeds"]:
        script = PREFIX + ("_admission.py" if admission else ".py")
        command = [R["python"], "-u", R["remote_root"] + "/" + script]
        if not admission:
            command += ["train", "--branch", "real_v_only", "--seed", str(seed)]
        row = job(command, PREFIX + f"/{mode}/seed{seed}", training=True)
        row.update(signature=P["name"] + f"/{mode}/seed{seed}", resource_family=P["name"],
            description=f"V-only backup {mode} seed{seed}; only lambda0.1->1.0, fixed final20k",
            allow_cpu_training=True,
            cpu_training_justification="Fixed original CPU real continuation; change only the V backup coefficient while retaining20k updates and all sampling/actor settings.")
        rows.append(row)
    submit_jobs(PREFIX + "_" + mode, rows)


def collect_admission():
    path = R["remote_root"] + "/" + PREFIX + "/admission.json"
    code = "from pathlib import Path\nimport json\np=Path(" + repr(path) + ")\n"
    code += "print(p.read_text() if p.exists() else json.dumps(dict(complete=False)))\n"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if report["complete"] and (report["protocol"] != P["name"] or report["seed"] != P["training_seeds"][0]):
        raise ValueError("Wrong V-only admission result")
    (HERE / (PREFIX + "_admission_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=report["complete"])), flush=True)


def collect():
    folders = [f"real_v_only/seed{seed}" for seed in P["training_seeds"]]
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
