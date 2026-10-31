"""Launch the fixed adaptation-stage evaluations and collect small summaries."""

import argparse
import json
import shlex

from launch import HERE, R, run_remote, stage_code
from source_dense_continuation_launch import campaign_protocol, paired_comparison
from training_launch import job, submit

def launch(variant="adapted"):
    prefix, P = campaign_protocol(variant)
    prerequisites = ("source_dense_adapted_eval",) if variant == "target-adapted" else (
        "source_dense_adaptation", "source_dense_continuation", "source_dense_real_only")
    for name in prerequisites:
        if not json.loads((HERE / (name + "_summary.json")).read_text())["complete"]:
            raise ValueError("Fixed phase reference has not completed")
    rows = []
    for seed in P["training_seeds"]:
        command = [R["python"], "-u", R["remote_root"] + "/source_dense_adapted_eval.py", "--seed", str(seed)]
        if variant == "target-adapted":
            command += ["--campaign", variant]
        row = job(command, prefix + f"/seed{seed}")
        row.update(signature=P["name"] + f"/seed{seed}",
            description=variant + " fixed actor, zero continuation, three paired cases " + str(seed),
            resource_family=P["name"])
        rows.append(row)
    submit(prefix, rows)


def collect(variant="adapted"):
    prefix, P = campaign_protocol(variant)
    code = "import sys,json; sys.path.insert(0," + repr(R["remote_root"]) + "); "
    code += "from source_dense_continuation import compact_result; "
    code += "print(json.dumps({str(seed):r for seed in " + repr(P["training_seeds"]) + " for r in [compact_result(seed,variant=" + repr(variant) + ")] if r is not None}))"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    report = dict(protocol=P["name"], complete=len(rows)==len(P["training_seeds"]),
        ready=len(rows), expected=len(P["training_seeds"]), expected_evaluations=9,
        results=rows, interpretation=P["interpretation"])
    if report["complete"]:
        for result in rows.values():
            training = result["training"]
            if not training["actor_export_equal"] or training["export_training_updates"] or training["total_policy_updates"] != 55000:
                raise ValueError("Adapted evaluation changed the actor or update history")
            if variant == "target-adapted" and (training["initialization_domain"] != "target"
                    or training["source_rows"] or training["checkpoint_stored_updates"] != P["adaptation_updates"]):
                raise ValueError("Local stage evaluated a source or different-stage checkpoint")
        report["comparison"] = paired_comparison(rows, admission=False, variant=variant)
    (HERE / (prefix + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "evaluate", "collect"))
    parser.add_argument("--campaign", choices=("adapted", "target-adapted"), default="adapted")
    args = parser.parse_args()
    mode = args.mode
    if mode == "stage":
        stage_code()
    elif mode == "evaluate":
        launch(args.campaign)
    else:
        collect(args.campaign)
