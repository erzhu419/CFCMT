"""Run the frozen one-event native counterfactual; retrieve small JSON only."""

import argparse
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
PREFIX = "native_action_replay"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def submit():
    from training_launch import job, submit as submit_jobs
    if not json.loads((HERE / (PREFIX + "_assets_summary.json")).read_text())["complete"]:
        raise ValueError("Frozen native replay assets are not ready")
    row = job([R["python"], "-u", R["remote_root"] + "/native_action_replay.py"], PREFIX)
    row.update(signature=P["name"], resource_family=P["name"], ram_mb=4000,
        description="One fresh native prefix: adapted baseline, single masked action, continuous masked continuation")
    submit_jobs(PREFIX, [row])


def collect():
    from launch import run_remote
    code = "from pathlib import Path; print(Path(" + repr(R["remote_root"] + "/native_action_replay/summary.json") + ").read_text())"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if not (report["complete"] and report["protocol"] == P["name"]
            and report["training_seed"] == P["training_seed"] and report["case"] == P["case"]
            and report["updates"] == 0):
        raise ValueError("Native replay result does not match frozen protocol")
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=True, protocol=report["protocol"], updates=report["updates"])), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "prepare", "submit", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        from launch import stage_code
        stage_code()
    elif mode == "prepare":
        from fresh_confirmation_launch import prepare
        prepare(P, PREFIX)
    elif mode == "submit":
        submit()
    else:
        collect()
