"""Launch the frozen gate55/amount75 component test and collect small JSON."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_actor_splice_reverse"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def stage():
    from launch import scheduler
    names = [PREFIX + suffix for suffix in (".py", "_launch.py", "_protocol.json")]
    names += ["native_terminal_actor_splice.py", "evaluate_native.py"]
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        *[str(HERE / name) for name in names],
        scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
        check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=names)), flush=True)


def submit(admission=False):
    from training_launch import job, submit as submit_jobs
    ref = json.loads((HERE / P["reference_summary"]).read_text())
    rp = json.loads((HERE / P["reference_protocol"]).read_text())
    if not (ref["complete"] and ref["protocol"] == rp["name"]
            and rp["training_seeds"] == P["training_seeds"]
            and rp["evaluation"]["cases"] == P["evaluation"]["cases"]
            and ref["actual_evaluations"] == len(ref["results"]) == 30
            and P["reference_evaluations_reused"] == 30
            and P["actual_evaluations_planned"] == 39):
        raise ValueError("Fixed component references changed or are incomplete")
    if not admission:
        admitted = json.loads((HERE / (PREFIX + "_admission_summary.json")).read_text())
        if not (admitted["complete"] and admitted["protocol"] == P["name"]
                and admitted["training_seeds"] == P["training_seeds"]):
            raise ValueError("Native component functions have not passed admission")
    rows = []
    for seed in [None] if admission else P["training_seeds"]:
        mode = "admission" if admission else "run"
        command = [R["python"], "-u", R["remote_root"] + "/" + PREFIX + ".py", mode]
        if seed is not None:
            command += ["--seed", str(seed)]
        row = job(command, PREFIX + f"/{mode}/{seed}")
        row.update(signature=P["name"] + f"/{mode}/{seed}", resource_family=P["name"], ram_mb=4000,
            description="Read-only gate55/amount75 function check, all three seeds" if admission
                else f"Frozen gate55/amount75 evaluation seed{seed}, three fixed cases")
        rows.append(row)
    submit_jobs(PREFIX + ("_admission" if admission else ""), rows)


def collect_admission():
    from launch import run_remote
    code = "from pathlib import Path\nimport json\np=Path(" + repr(R["remote_root"] + "/" + PREFIX + "_admission_summary.json") + ")\n"
    code += "print(p.read_text() if p.exists() else json.dumps(dict(complete=False)))\n"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if report["complete"]:
        if not (report["protocol"] == P["name"] and report["variant"] == P["variant"]
                and report["training_seeds"] == P["training_seeds"]
                and set(report["results"]) == {str(seed) for seed in P["training_seeds"]}
                and all(row["complete"] and row["rows"] == 12288 for row in report["results"].values())):
            raise ValueError("Wrong native component admission identity")
    (HERE / (PREFIX + "_admission_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=report["complete"])), flush=True)


def collect():
    from launch import run_remote
    names = [f"seed{seed}/case{case['case']:02d}" for seed in P["training_seeds"] for case in P["evaluation"]["cases"]]
    code = "import json\nfrom pathlib import Path\nroot=Path(" + repr(R["remote_root"] + "/" + PREFIX + "/" + P["variant"]) + ")\nrows={}\n"
    code += "for name in " + repr(names) + ":\n    path=root/name/'summary.json'\n"
    code += "    if path.exists(): rows[name]=json.loads(path.read_text())\nprint(json.dumps(rows))\n"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    ready = sum(row.get("complete", False) for row in rows.values())
    report = dict(protocol=P["name"], complete=ready == len(names), ready=ready, expected=len(names), results=rows)
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "admission", "collect-admission", "evaluate", "collect"))
    mode = parser.parse_args().mode
    if mode in ("admission", "evaluate"):
        submit(mode == "admission")
    else:
        {"stage": stage, "collect-admission": collect_admission, "collect": collect}[mode]()
