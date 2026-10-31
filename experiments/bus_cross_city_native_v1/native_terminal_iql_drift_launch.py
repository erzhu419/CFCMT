"""Run fixed-fit IQL drift diagnostics and retrieve statistics only."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_iql_drift"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def stage():
    from launch import scheduler
    names = [PREFIX + suffix for suffix in (".py", "_launch.py", "_protocol.json")]
    subprocess.run(["rsync", "-a", "-e", scheduler._ssh_rsync_shell_for_node("node007"),
        *[str(HERE / name) for name in names],
        scheduler._ssh_target_for_node("node007") + ":" + R["remote_root"] + "/"],
        check=True, capture_output=True, text=True, timeout=60)
    print(json.dumps(dict(staged=names)), flush=True)


def diagnose():
    from training_launch import job, submit
    for name in ("native_terminal_adaptation_reporting.json", "native_terminal_actor_unmasked_reporting.json"):
        reference = json.loads((HERE / name).read_text())
        if not (reference["complete"] and reference["training_ready"] == len(P["training_seeds"])):
            raise ValueError("Fixed learner reference is incomplete: " + name)
    rows = []
    for seed in P["training_seeds"]:
        command = [R["python"], "-u", R["remote_root"] + "/" + PREFIX + ".py", "--seed", str(seed)]
        row = job(command, PREFIX + f"/seed{seed}")
        row.update(signature=P["name"] + f"/seed{seed}", resource_family=P["name"], ram_mb=4000,
            description=f"Read-only fixed-fit IQL weight/actor decomposition seed{seed}")
        rows.append(row)
    submit(PREFIX, rows)


def collect():
    from launch import run_remote
    folders = [f"seed{seed}" for seed in P["training_seeds"]]
    code = "import json\nfrom pathlib import Path\nroot=Path(" + repr(R["remote_root"] + "/" + PREFIX) + ")\n"
    code += "rows={}\nfor name in " + repr(folders) + ":\n    path=root/name/'summary.json'\n"
    code += "    if path.exists(): rows[name]=json.loads(path.read_text())\nprint(json.dumps(rows))\n"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    for folder, row in rows.items():
        seed = int(folder[4:])
        if not (row["complete"] and row["protocol"] == P["name"] and row["seed"] == seed
                and row["rows"] == P["target_fit_rows"] and row["updates"] == 0
                and row["evaluation_rows_read"] == 0 and row["new_simulations"] == 0):
            raise ValueError("Wrong fixed-fit diagnostic identity: " + folder)
        expected_paths = {name: R["remote_root"] + "/" + spec["artifact"].format(seed=seed)
            for name, spec in P["stages"].items()}
        if row["artifacts"] != expected_paths or row["stored_updates"] != {
                name: spec["stored_updates"] for name, spec in P["stages"].items()}:
            raise ValueError("Wrong frozen learners: " + folder)
    report = dict(complete=len(rows) == len(folders), ready=len(rows), expected=len(folders),
        protocol=P["name"], new_training_updates=0, world_model_refit_updates=0,
        new_simulations=0, interpretation=P["interpretation"], results=rows)
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: report[name] for name in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "diagnose", "collect"))
    mode = parser.parse_args().mode
    {"stage": stage, "diagnose": diagnose, "collect": collect}[mode]()
