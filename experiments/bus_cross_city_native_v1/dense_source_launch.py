"""Launch source-only compression ablation; collect small summaries only."""

import argparse
import json
import shlex
import subprocess

from launch import HERE, R, run_remote, stage_code
from training_launch import job

P = json.loads((HERE / "dense_source_protocol.json").read_text())


def submit(name, seeds):
    record = HERE / ("dense_source_" + name + "_submission.json")
    if record.exists():
        raise ValueError("This dense-source batch was already submitted")
    if name == "train":
        if not json.loads((HERE / "dense_source_admission_summary.json").read_text())["complete"]:
            raise ValueError("Full-data replay admission has not completed")
    rows = []
    for seed in seeds:
        command = [R["python"], "-u", R["remote_root"] + "/dense_source_ablation.py",
                   name, "--seed", str(seed)]
        row = job(command, "dense-source/" + name + f"/seed{seed}", training=True)
        row.update(signature=P["name"] + "/" + name + f"/seed{seed}",
            description="Source compression ablation " + name + f" seed{seed}",
            ram_mb=12000, vram=0, resource_family=P["name"],
            allow_cpu_training=True,
            cpu_training_justification="Keep CPU execution identical to the completed sparse-source reference; full-source replay needs about 6.2GB plus loader/model overhead. Same seeds and 50000 IQL updates.")
        rows.append(row)
    path = HERE / ("dense_source_" + name + "_jobs.json")
    path.write_text(json.dumps(rows, indent=2) + "\n")
    result = json.loads(subprocess.check_output(["python3", R["scheduler"], "submit-jsonl",
        "--file", str(path), "--trusted", "--json"], text=True))
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def collect(admission=False):
    base = "import sys,json; from pathlib import Path; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    base += "root=Path(" + repr(R["remote_root"]) + ")/'source_dense_ablation'; "
    if admission:
        base += "path=root/'admission/summary.json'; "
        base += "row=json.loads(path.read_text()) if path.exists() else dict(complete=False); "
        base += "print(json.dumps(dict(admission=row,snapshot=json.loads((root/'snapshot_summary.json').read_text()) if (root/'snapshot_summary.json').exists() else None)))"
        rows = json.loads(run_remote(shlex.join([R["python"], "-c", base])))
        report = dict(protocol=P["name"], complete=rows["admission"]["complete"], **rows)
        filename = "dense_source_admission_summary.json"
    else:
        base += "from dense_source_ablation import compact_result; "
        base += "print(json.dumps({str(seed):row for seed in " + repr(P["training_seeds"]) + " for row in [compact_result(seed)] if row is not None}))"
        rows = json.loads(run_remote(shlex.join([R["python"], "-c", base])))
        report = dict(protocol=P["name"], complete=len(rows)==len(P["training_seeds"]),
            ready=len(rows), expected=len(P["training_seeds"]), results=rows,
            interpretation=P["interpretation"])
        filename = "dense_source_summary.json"
    (HERE / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected") if key in report}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "prepare", "collect-admission", "train", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage_code()
    elif mode in ("prepare", "train"):
        submit(mode, P["training_seeds"][:1] if mode == "prepare" else P["training_seeds"])
    else:
        collect(admission=mode == "collect-admission")
