"""Run fixed full-source target adaptation; retrieve summaries only."""

import argparse
import json
import shlex
import subprocess

from launch import HERE, R, run_remote, stage_code
from training_launch import job

P = json.loads((HERE / "source_dense_adaptation_protocol.json").read_text())


def submit(admission=False):
    name = "source_dense_adaptation_" + ("admission" if admission else "train")
    record = HERE / (name + "_submission.json")
    if record.exists():
        raise ValueError("This fixed adaptation batch was already submitted")
    prerequisite = HERE / ("dense_source_summary.json" if admission else "source_dense_adaptation_admission_summary.json")
    if not json.loads(prerequisite.read_text())["complete"]:
        raise ValueError("Previous full-source stage has not completed")
    rows = []
    for seed in P["training_seeds"][:1] if admission else P["training_seeds"]:
        command = [R["python"], "-u", R["remote_root"] + "/source_dense_adaptation.py", "--seed", str(seed)]
        if admission:
            command.append("--admission")
        row = job(command, name + f"/seed{seed}", training=True)
        row.update(signature=P["name"] + ("/admission" if admission else "/train") + f"/seed{seed}",
            description="Full-source target IQL adaptation " + str(seed), resource_family=P["name"],
            allow_cpu_training=True,
            cpu_training_justification="Keep CPU and original5000-update target IQL settings identical to the completed sparse-source reference; only the fixed source initializer changes.")
        rows.append(row)
    path = HERE / (name + "_jobs.json")
    path.write_text(json.dumps(rows, indent=2) + "\n")
    result = json.loads(subprocess.check_output(["python3", R["scheduler"], "submit-jsonl",
        "--file", str(path), "--trusted", "--json"], text=True))
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def collect(admission=False):
    code = "import sys,json; from pathlib import Path; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    code += "root=Path(" + repr(R["remote_root"]) + ")/'source_dense_adaptation'; "
    if admission:
        code += "path=root/'admission/summary.json'; print(path.read_text() if path.exists() else json.dumps(dict(complete=False)))"
        report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
        filename = "source_dense_adaptation_admission_summary.json"
    else:
        code += "from source_dense_adaptation import compact_result; "
        code += "print(json.dumps({str(seed):r for seed in " + repr(P["training_seeds"]) + " for r in [compact_result(seed)] if r is not None}))"
        rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
        report = dict(protocol=P["name"], complete=len(rows)==len(P["training_seeds"]),
            ready=len(rows), expected=len(P["training_seeds"]), results=rows, interpretation=P["interpretation"])
        filename = "source_dense_adaptation_summary.json"
    (HERE / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected") if key in report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "admission", "collect-admission", "train", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage_code()
    elif mode in ("admission", "train"):
        submit(admission=mode == "admission")
    else:
        collect(admission=mode == "collect-admission")
