"""Evaluate zero autonomous holding under the original executor; collect small JSON."""

import argparse
import json
import shlex

from launch import HERE, R, run_remote, stage_code
from training_launch import job, submit

PREFIX = "zero_hold_eval"
P = json.loads((HERE / (PREFIX + "_protocol.json")).read_text())


def evaluate():
    original = json.loads((HERE / "training_protocol.json").read_text())
    if P["evaluation"] != original["evaluation"]:
        raise ValueError("Zero holding must retain the original evaluation cases")
    for prefix in ("source_dense_adapted_eval", "target_adapted_eval"):
        if not json.loads((HERE / (prefix + "_summary.json")).read_text())["complete"]:
            raise ValueError("Fixed adaptation-stage reference has not completed")
    rows = []
    for case in P["evaluation"]["cases"]:
        command = [R["python"], "-u", R["remote_root"] + "/evaluate_native.py",
            "--method", "zero", "--case", str(case["case"]),
            "--assets", R["assets_root"] + f"/test{case['case']:02d}/asset_manifest.json",
            "--output", R["remote_root"] + f"/{PREFIX}/case{case['case']:02d}"]
        row = job(command, PREFIX + f"/case{case['case']:02d}")
        row.update(signature=P["name"] + f"/case{case['case']:02d}",
            description="Zero autonomous holding, original executor, case " + str(case["case"]),
            resource_family=P["name"])
        rows.append(row)
    submit(PREFIX, rows)


def metrics(row):
    if "cost" in row:
        out = {key: row[key] for key in ("cost", "completion", "wait_seconds", "ride_seconds", "mean_executed_hold_seconds")}
    else:
        passengers = row["passengers"]
        out = dict(cost=row["generalized_seconds_per_departed_passenger"],
            completion=passengers["completion_rate"],
            wait_seconds=passengers["accrued_wait_seconds_per_departed_passenger"],
            ride_seconds=passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
            mean_executed_hold_seconds=row["mean_hold_seconds"])
    out["positive_hold_fraction"] = row["positive_holds"]/row["decision_events"]
    return out


def compare(rows):
    baseline = json.loads((HERE / "pilot_summary.json").read_text())["results"]
    models = {name: json.loads((HERE / (prefix + "_summary.json")).read_text())["results"]
        for name, prefix in (("source_adapted", "source_dense_adapted_eval"), ("target_adapted", "target_adapted_eval"))}
    pairs = []
    for case in P["evaluation"]["cases"]:
        zero = rows[str(case["case"])]
        if (not zero["complete"] or zero["method"] != "zero" or zero["checkpoint"] is not None
                or zero["training_seed"] is not None or zero["mean_proposed_hold_seconds"]
                or zero["max_proposed_hold_seconds"] or zero["positive_proposed_holds"]
                or zero["execution_layer"] != P["execution_layer"] or zero["completion_guard"] != P["completion_guard"]):
            raise ValueError("Wrong zero-autonomous-holding artifact or executor")
        references = [("strong_rule", None, baseline[f"forward_proportional/case{case['case']:02d}"])]
        references += [(name, seed, models[name][str(seed)]["evaluation"][str(case["case"])])
            for name in models for seed in P["reference_training_seeds"]]
        for name, seed, old in references:
            for field in ("case", "demand_seed", "sumo_seed", "horizon_seconds", "simulator_version", "line_ids"):
                if zero[field] != old[field]:
                    raise ValueError("Paired reference changed: " + field)
            if seed is not None and old["training_seed"] != seed:
                raise ValueError("Reference training seed mismatch")
            departed = old["departed"] if "departed" in old else old["passengers"]["departed_passengers"]
            balance = old["ledger_balance_error"] if "ledger_balance_error" in old else old["passengers"]["active_state_count_balance_error"]
            if (zero["horizon_seconds"] != 24000 or not old["complete"] or balance
                    or zero["passengers"]["active_state_count_balance_error"]
                    or departed != zero["passengers"]["departed_passengers"]):
                raise ValueError("Paired horizon or passenger ledger mismatch")
            pairs.append(dict(method=name, training_seed=seed, case=case["case"],
                zero=metrics(zero), policy=metrics(old)))
    def mean(items):
        return {key: sum(row[key] for row in items)/len(items) for key in items[0]}
    def summarize(selected):
        zero, policy = mean([x["zero"] for x in selected]), mean([x["policy"] for x in selected])
        return dict(policy=policy, zero=zero, policy_minus_zero={key: policy[key]-zero[key] for key in policy},
            cost_percent_of_zero=100*(policy["cost"]-zero["cost"])/zero["cost"],
            better_pairs=sum(x["policy"]["cost"]<x["zero"]["cost"] for x in selected))
    methods = {}
    for name in ("strong_rule", "source_adapted", "target_adapted"):
        selected = [x for x in pairs if x["method"] == name]
        methods[name] = summarize(selected)
        if name != "strong_rule":
            methods[name]["by_seed"] = {str(seed): summarize([x for x in selected if x["training_seed"] == seed])
                for seed in P["reference_training_seeds"]}
    return dict(zero_evaluations=len(rows), paired_policy_comparisons=len(pairs),
        zero_mean=mean([metrics(row) for row in rows.values()]), by_method=methods, pairs=pairs)


def collect():
    code = "from pathlib import Path; import json; root=Path(" + repr(R["remote_root"] + "/" + PREFIX) + "); "
    code += "print(json.dumps({str(case):json.loads(p.read_text()) for case in " + repr([c["case"] for c in P["evaluation"]["cases"]])
    code += " for p in [root/f'case{case:02d}'/'summary.json'] if p.exists()}))"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report = dict(protocol=P["name"], complete=len(rows)==3, ready=len(rows), expected=3,
        results=rows, interpretation=P["interpretation"])
    if report["complete"]:
        report["comparison"] = compare(rows)
    (HERE / (PREFIX + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "evaluate", "collect"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage_code()
    elif mode == "evaluate":
        evaluate()
    else:
        collect()
