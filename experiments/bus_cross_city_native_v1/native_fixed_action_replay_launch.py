"""Run the frozen twelve-event native action comparison, pulling small JSON only."""

import argparse
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
PREFIX = "native_fixed_action_replay"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def submit():
    from training_launch import job, submit as submit_jobs
    prior = json.loads((HERE / "native_actor_preference_summary.json").read_text())
    if not prior["complete"]:
        raise ValueError("The existing native preference trajectories are incomplete")
    rows = []
    for seed in P["training_seeds"]:
        row = job([R["python"], "-u", R["remote_root"] + "/native_fixed_action_replay.py",
                   "--seed", str(seed)], PREFIX + f"/seed{seed}")
        row.update(signature=P["name"] + f"/seed{seed}", resource_family=P["name"],
                   ram_mb=4000, description=f"Four fixed native events and single-action outcomes, seed{seed}")
        rows.append(row)
    submit_jobs(PREFIX, rows)


def collect():
    from launch import run_remote
    code = "from pathlib import Path; import json; root=Path(" + repr(R["remote_root"]) + "); "
    code += "seeds=" + repr(P["training_seeds"]) + "; "
    code += "rows={str(s):json.loads((root/'native_fixed_action_replay'/f'seed{s}'/'summary.json').read_text()) "
    code += "for s in seeds if (root/'native_fixed_action_replay'/f'seed{s}'/'summary.json').exists()}; print(json.dumps(rows))"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    for seed, row in rows.items():
        if not (row["complete"] and row["protocol"] == P["name"] and row["training_seed"] == int(seed)
                and row["case"] == P["case"] and row["updates"] == 0):
            raise ValueError("Native action outcomes differ from frozen protocol")
    report = dict(protocol=P["name"], complete=len(rows) == len(P["training_seeds"]),
                  ready=len(rows), expected=len(P["training_seeds"]), summaries=rows)
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k:v for k,v in report.items() if k != "summaries"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "submit", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        from launch import stage_code
        stage_code()
    elif mode == "submit":
        submit()
    else:
        collect()
