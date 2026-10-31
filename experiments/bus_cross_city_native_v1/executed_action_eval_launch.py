"""Fixed adapted checkpoints, execution-space projection, small JSON collection."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "runtime.json").read_text())
PREFIX = "executed_action_eval"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


def run(branch, seed):
    spec = P["branches"][branch]
    checkpoint = HERE / spec["artifact_folder"] / f"seed{seed}" / "policy.pt"
    for case in P["evaluation"]["cases"]:
        subprocess.run([sys.executable, "-u", str(HERE / "evaluate_native.py"),
            "--method", spec["method"], "--seed", str(seed), "--case", str(case["case"]),
            "--actor-action-space", "executed", "--checkpoint", str(checkpoint),
            "--assets", R["assets_root"] + f"/test{case['case']:02d}/asset_manifest.json",
            "--output", str(HERE / PREFIX / branch / f"seed{seed}" / f"case{case['case']:02d}")], check=True)


def evaluate():
    from training_launch import job, submit
    original = json.loads((HERE / "training_protocol.json").read_text())
    if P["evaluation"] != original["evaluation"] or P["training_seeds"] != original["training_seeds"]:
        raise ValueError("The original evaluation cases and training seeds must remain fixed")
    gate = json.loads((HERE / "action_semantics_audit.json").read_text())["execution_projection"]
    if gate["fit_labels_max_action_error"] or gate["idempotence_max_action_error"] or gate["formerly_shrunk_rows_preserved"] != 2010:
        raise ValueError("Execution projection does not preserve the real fit labels")
    rows = []
    for branch, spec in P["branches"].items():
        if not json.loads((HERE / (spec["reference"] + "_summary.json")).read_text())["complete"]:
            raise ValueError("Original fixed-checkpoint reference is incomplete")
        for seed in P["training_seeds"]:
            command = [R["python"], "-u", R["remote_root"] + "/executed_action_eval_launch.py",
                "run", "--branch", branch, "--seed", str(seed)]
            row = job(command, PREFIX + f"/{branch}/seed{seed}")
            row.update(signature=P["name"] + f"/{branch}/seed{seed}", resource_family=P["name"],
                description="Fixed55k adapted actor, execution-space projection " + branch + " " + str(seed))
            rows.append(row)
    submit(PREFIX, rows)


def compact_remote():
    rows = {}
    for branch in P["branches"]:
        for seed in P["training_seeds"]:
            for case in P["evaluation"]["cases"]:
                key = f"{branch}/{seed}/{case['case']}"
                path = HERE / PREFIX / branch / f"seed{seed}" / f"case{case['case']:02d}" / "summary.json"
                if not path.exists():
                    continue
                row = json.loads(path.read_text())
                passengers = row["passengers"]
                values = {name:row[name] for name in ("complete", "method", "training_seed", "case", "demand_seed",
                    "sumo_seed", "horizon_seconds", "simulator_version", "line_ids", "checkpoint", "actor_action_space",
                    "decision_events", "positive_holds", "mean_requested_hold_seconds", "execution_layer", "completion_guard",
                    "guard_increased_holds", "guard_decreased_holds", "same_state_legacy_mean_hold_seconds", "same_state_legacy_different_events")}
                values.update(cost=row["generalized_seconds_per_departed_passenger"],
                    completion=passengers["completion_rate"], departed=passengers["departed_passengers"],
                    completed=passengers["completed_passengers"], ledger_balance_error=passengers["active_state_count_balance_error"],
                    wait_seconds=passengers["accrued_wait_seconds_per_departed_passenger"],
                    ride_seconds=passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
                    mean_executed_hold_seconds=row["mean_hold_seconds"])
                rows[key] = values
    return rows


def collect():
    from launch import run_remote
    from zero_hold_eval_launch import metrics
    code = "import sys,json; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    code += "from executed_action_eval_launch import compact_remote; print(json.dumps(compact_remote()))"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report = dict(protocol=P["name"], complete=len(rows)==18, ready=len(rows), expected=18, results=rows)
    references = {branch:json.loads((HERE / (spec["reference"] + "_summary.json")).read_text())["results"]
        for branch,spec in P["branches"].items()}
    rule = json.loads((HERE / "pilot_summary.json").read_text())["results"]
    zero = json.loads((HERE / "zero_hold_eval_summary.json").read_text())["results"]
    mean = lambda items: {k:sum(x[k] for x in items)/len(items) for k in items[0]}
    comparisons = {}
    for branch, spec in P["branches"].items():
        selected = []
        for seed in P["training_seeds"]:
            for case in P["evaluation"]["cases"]:
                row = rows.get(f"{branch}/{seed}/{case['case']}")
                if row is None:
                    continue
                old = references[branch][str(seed)]["evaluation"][str(case["case"])]
                artifact = R["remote_root"] + f"/{spec['artifact_folder']}/seed{seed}/policy.pt"
                if (not row["complete"] or row["actor_action_space"] != "executed" or row["method"] != spec["method"]
                        or row["checkpoint"] != artifact or row["ledger_balance_error"] or row["departed"] != old["departed"]
                        or row["execution_layer"] != "early60" or row["completion_guard"] != "quarter-load-sixty"):
                    raise ValueError("Wrong checkpoint, action mapping or passenger ledger")
                for field in ("case", "training_seed", "demand_seed", "sumo_seed", "horizon_seconds", "simulator_version", "line_ids"):
                    if row[field] != old[field]:
                        raise ValueError("Paired evaluation changed: " + field)
                selected.append(dict(seed=seed, case=case["case"], actual=metrics(row),
                    original=metrics(old), rule=metrics(rule[f"forward_proportional/case{case['case']:02d}"]),
                    zero=metrics(zero[str(case["case"])])))
        if len(selected) == 9:
            actual = mean([x["actual"] for x in selected])
            refs = {name:mean([x[name] for x in selected]) for name in ("original", "rule", "zero")}
            comparisons[branch] = dict(actual=actual, references=refs,
                differences={name:{k:actual[k]-v[k] for k in actual} for name,v in refs.items()},
                better_cases={name:sum(x["actual"]["cost"]<x[name]["cost"] for x in selected) for name in refs},
                by_seed={str(seed):dict(cost=mean([x["actual"] for x in selected if x["seed"]==seed])["cost"],
                    cost_minus_original=sum(x["actual"]["cost"]-x["original"]["cost"] for x in selected if x["seed"]==seed)/3)
                    for seed in P["training_seeds"]})
    report["comparison"] = comparisons
    if report["complete"]:
        source = comparisons["source_adapted"]["actual"]
        target = comparisons["target_adapted"]["actual"]
        report["source_minus_target"] = {key:source[key]-target[key] for key in source}
        report["training_updates"] = 0
        report["actor_checkpoints_changed"] = False
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({key:report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "evaluate", "collect", "run"))
    parser.add_argument("--branch", choices=tuple(P["branches"]))
    parser.add_argument("--seed", type=int, choices=P["training_seeds"])
    args = parser.parse_args()
    if args.mode == "run":
        if args.branch is None or args.seed is None:
            parser.error("run requires a frozen branch and seed")
        run(args.branch, args.seed)
    elif args.mode == "stage":
        from launch import stage_code
        stage_code()
    else:
        {"evaluate":evaluate, "collect":collect}[args.mode]()
