"""Launch three fixed continuation jobs with nine evaluations; collect small results."""

import argparse
import json
import shlex
import subprocess

from launch import HERE, R, run_remote, stage_code
from training_launch import job


def campaign_protocol(variant):
    prefix = {"model-td": "source_dense_continuation", "real-td": "source_dense_real_only",
              "adapted": "source_dense_adapted_eval", "iql": "source_dense_iql_continuation",
              "target-adapted": "target_adapted_eval"}[variant]
    return prefix, json.loads((HERE / (prefix + "_protocol.json")).read_text())


def paired_comparison(rows, *, admission, variant="model-td"):
    _, P = campaign_protocol(variant)
    baseline = json.loads((HERE / "pilot_summary.json").read_text())["results"]
    continued = {}
    if variant in ("real-td", "adapted", "iql"):
        continued["full_source_model"] = json.loads((HERE / "source_dense_continuation_summary.json").read_text())["results"]
    if variant in ("adapted", "iql"):
        continued["full_source_real"] = json.loads((HERE / "source_dense_real_only_summary.json").read_text())["results"]
    if variant in ("iql", "target-adapted"):
        continued["full_source_adapted"] = json.loads((HERE / "source_dense_adapted_eval_summary.json").read_text())["results"]
    pairs = []
    for seed, result in rows.items():
        training = result["training"]
        updates = 4 if admission else P["continuation_updates"]
        if (training["continuation_updates"] != updates or training["admission"] != admission
                or training["protocol"] != P["name"]
                or training["continuation_model_td_coefficient"] != P.get("model_loss_coefficient", .25)):
            raise ValueError("Wrong fixed-final continuation artifact")
        if variant == "iql" and (training["continuation_online_flag"] or training["effective_critic_backup"] != "iql"
                or training["world_model_artifact"] is not None or training["online_critic_updates_added"]
                or training["log_alpha_before"] != training["log_alpha_after"]):
            raise ValueError("IQL continuation used the SAC or model path")
        cases = P["evaluation"]["cases"][:1] if admission else P["evaluation"]["cases"]
        if set(result["evaluation"]) != {str(c["case"]) for c in cases}:
            raise ValueError("Incomplete fixed evaluation cases")
        for case in cases:
            actual = result["evaluation"][str(case["case"])]
            if not actual["complete"] or actual["training_seed"] != int(seed) or actual["ledger_balance_error"]:
                raise ValueError("Evaluation identity or passenger ledger mismatch")
            references = {}
            for name in P["comparisons"]:
                if name in continued:
                    old = continued[name][seed]["evaluation"][str(case["case"])]
                    if old["training_seed"] != int(seed):
                        raise ValueError("Full-source reference seed mismatch")
                else:
                    method = "forward_proportional" if name == "strong_rule" else name
                    key = f"{method}/case{case['case']:02d}" if name == "strong_rule" else f"{method}/seed{seed}/case{case['case']:02d}"
                    raw = baseline[key]
                    passengers = raw["passengers"]
                    old = dict(raw, cost=raw["generalized_seconds_per_departed_passenger"],
                        departed=passengers["departed_passengers"], completion=passengers["completion_rate"],
                        ledger_balance_error=passengers["active_state_count_balance_error"],
                        wait_seconds=passengers["accrued_wait_seconds_per_departed_passenger"],
                        ride_seconds=passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
                        mean_executed_hold_seconds=raw["mean_hold_seconds"])
                for field in ("case", "demand_seed", "sumo_seed", "horizon_seconds", "simulator_version", "line_ids"):
                    if actual[field] != old[field]:
                        raise ValueError("Paired reference changed: " + field)
                if (actual["horizon_seconds"] != 24000 or old["departed"] != actual["departed"]
                        or not old["complete"] or old["ledger_balance_error"]):
                    raise ValueError("Paired horizon or demand passenger count mismatch")
                references[name] = dict(cost=old["cost"], completion=old["completion"],
                    wait_seconds=old["wait_seconds"], ride_seconds=old["ride_seconds"],
                    mean_executed_hold_seconds=old["mean_executed_hold_seconds"],
                    positive_hold_fraction=old["positive_holds"]/old["decision_events"])
            metrics = {key: actual[key] for key in ("cost", "completion", "wait_seconds", "ride_seconds", "mean_executed_hold_seconds")}
            metrics["positive_hold_fraction"] = actual["positive_holds"]/actual["decision_events"]
            pairs.append(dict(seed=int(seed), case=case["case"], actual=metrics, references=references))
    def summarize(selected):
        average = lambda values: sum(values)/len(values)
        actual = {key: average([row["actual"][key] for row in selected]) for key in selected[0]["actual"]}
        references = {name: {key: average([row["references"][name][key] for row in selected])
            for key in actual} for name in P["comparisons"]}
        differences = {name: dict(cost=actual["cost"]-ref["cost"],
            cost_percent=100*(actual["cost"]-ref["cost"])/ref["cost"],
            completion_pp=100*(actual["completion"]-ref["completion"]),
            wait_seconds=actual["wait_seconds"]-ref["wait_seconds"],
            ride_seconds=actual["ride_seconds"]-ref["ride_seconds"]) for name,ref in references.items()}
        return dict(actual=actual,references=references,differences=differences)
    return dict(paired_evaluations=len(pairs), all_seed_case_mean=summarize(pairs),
        by_seed={seed:summarize([row for row in pairs if row["seed"]==int(seed)]) for seed in rows},
        pairs=pairs, interpretation="Executed holding and equal-weight case/seed costs; previously analyzed fixed demand cases. Admission is interface-only.")


def submit(admission=False, *, variant="model-td"):
    prefix, P = campaign_protocol(variant)
    name = prefix + "_" + ("admission" if admission else "train")
    record = HERE / (name + "_submission.json")
    if record.exists():
        raise ValueError("This fixed continuation batch was already submitted")
    admission_reference = {"model-td": "source_dense_adaptation_summary.json", "real-td": "source_dense_continuation_summary.json",
                           "iql": "source_dense_adapted_eval_summary.json"}[variant]
    prerequisite = admission_reference if admission else prefix + "_admission_summary.json"
    if not json.loads((HERE / prerequisite).read_text())["complete"]:
        raise ValueError("Continuation prerequisite has not completed")
    rows = []
    for seed in P["training_seeds"][:1] if admission else P["training_seeds"]:
        command = [R["python"], "-u", R["remote_root"] + "/source_dense_continuation.py", "--seed", str(seed)]
        if variant != "model-td":
            command += ["--campaign", variant]
        if admission:
            command.append("--admission")
        row = job(command, name + f"/seed{seed}", training=True)
        row.update(signature=P["name"] + ("/admission" if admission else "/train") + f"/seed{seed}",
            description="Full-source " + variant + " continuation and fixed evaluation " + str(seed),
            resource_family=P["name"], allow_cpu_training=True,
            cpu_training_justification=("Keep original CPU and fixed budget; replace SAC continuation with the frozen pure offline IQL kernel."
                if variant == "iql" else "Keep original CPU, continuation kernel and fixed budget; this campaign freezes the specified model TD coefficient."))
        rows.append(row)
    path = HERE / (name + "_jobs.json")
    path.write_text(json.dumps(rows, indent=2) + "\n")
    result = json.loads(subprocess.check_output(["python3", R["scheduler"], "submit-jsonl",
        "--file", str(path), "--trusted", "--json"], text=True))
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def collect(admission=False, *, variant="model-td"):
    prefix, P = campaign_protocol(variant)
    seeds = P["training_seeds"][:1] if admission else P["training_seeds"]
    code = "import sys,json; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    code += "from source_dense_continuation import compact_result; "
    code += "print(json.dumps({str(seed):r for seed in " + repr(seeds) + " for r in [compact_result(seed,admission=" + repr(admission) + ",variant=" + repr(variant) + ")] if r is not None}))"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report = dict(protocol=P["name"], complete=len(rows)==len(seeds), ready=len(rows),
        expected=len(seeds), expected_evaluations=1 if admission else 9,
        admission=admission, results=rows, interpretation=P["interpretation"])
    if report["complete"]:
        report["comparison"] = paired_comparison(rows, admission=admission, variant=variant)
    filename = prefix + "_" + ("admission_summary.json" if admission else "summary.json")
    (HERE / filename).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "admission", "collect-admission", "train", "collect"))
    parser.add_argument("--campaign", choices=("model-td", "real-td", "iql"), default="model-td")
    args = parser.parse_args()
    mode = args.mode
    if mode == "stage":
        stage_code()
    elif mode in ("admission", "train"):
        submit(admission=mode == "admission", variant=args.campaign)
    else:
        collect(admission=mode == "collect-admission", variant=args.campaign)
