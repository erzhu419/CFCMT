"""Launch the frozen actor-only intervention and collect small JSON summaries."""

import argparse
import json
from pathlib import Path
import shlex

HERE = Path(__file__).resolve().parent
PREFIX = "iql_actor_continuation"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())
R = json.loads((HERE / "runtime.json").read_text())


def submit(mode):
    from training_launch import job, submit as submit_jobs
    if mode == "admission":
        rows = [job([R["python"], "-u", R["remote_root"] + "/iql_actor_admission.py"],
                    PREFIX + "/admission", training=True)]
        rows[0].update(signature=P["name"] + "/admission", resource_family=P["name"],
                       description="Four-step paired SAC/IQL actor admission; no evaluations")
    else:
        if not json.loads((HERE / (PREFIX + "_assets_summary.json")).read_text())["complete"]:
            raise ValueError("New fixed assets are not ready")
        if not json.loads((HERE / PREFIX / "admission.json").read_text())["complete"]:
            raise ValueError("Actual actor-only continuation has not passed admission")
        training = mode == "train"
        choices = [(b,s) for b in ("td0", "td025") for s in P["training_seeds"]] if training else [
            (b,s) for b in ("sac0", "sac025", "adapted") for s in P["training_seeds"]] + [("strong_rule", None)]
        rows = []
        for branch, seed in choices:
            command = [R["python"], "-u", R["remote_root"] + "/iql_actor_continuation.py",
                "train" if training else "evaluate", "--branch", branch]
            if seed is not None:
                command += ["--seed", str(seed)]
            row = job(command, PREFIX + f"/{branch}/{seed}", training=training)
            row.update(signature=P["name"] + f"/{branch}/{seed}", resource_family=P["name"],
                description=f"Fixed actor-only comparison {branch} {seed}; three prospective demand cases")
            rows.append(row)
    submit_jobs(PREFIX + "_" + mode, rows)


def collect():
    from launch import run_remote
    from fresh_confirmation_summary import summarize
    code = "import sys,json; from pathlib import Path; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    code += "from fresh_confirmation_summary import compact_remote; root=Path(" + repr(R["remote_root"]) + "); "
    code += "p=json.loads((root/'iql_actor_continuation_protocol.json').read_text()); "
    code += "training={str(q.parent.relative_to(root/'iql_actor_continuation')):json.loads(q.read_text()) for pattern in ['*/seed*/progress.json','*/seed*/train_summary.json'] for q in (root/'iql_actor_continuation').glob(pattern)}; "
    code += "print(json.dumps(dict(results=compact_remote(p),training=training)))"
    remote = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report = summarize(remote["results"], P)
    report["training"] = remote["training"]
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=report["complete"], ready=report["ready"], expected=report["expected"],
        training={key:row.get("phase_updates",row.get("continuation_updates")) for key,row in report["training"].items()})), flush=True)


def collect_admission():
    from launch import run_remote
    code = "from pathlib import Path; print(Path(" + repr(R["remote_root"] + "/iql_actor_continuation/admission.json") + ").read_text())"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    if not (report["complete"] and report["protocol"] == P["name"] and report["seed"] == P["training_seeds"][0]):
        raise ValueError("Wrong actor-only admission result")
    folder = HERE / PREFIX
    folder.mkdir(exist_ok=True)
    (folder / "admission.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "prepare", "admission", "collect-admission", "train", "references", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        from launch import stage_code
        stage_code()
    elif mode == "prepare":
        from fresh_confirmation_launch import prepare
        prepare(P, PREFIX)
    elif mode == "collect-admission":
        collect_admission()
    elif mode in ("admission", "train", "references"):
        submit(mode)
    else:
        collect()
