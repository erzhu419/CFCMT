"""Stage and launch the fixed offline pilot; retrieve summaries, never datasets."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess

from launch import HERE, P, R, run_remote, stage_code


T = json.loads((HERE / "training_protocol.json").read_text())


def job(command, signature, *, training=False):
    return dict(project="CFCMT", signature=T["name"] + "/" + signature,
        description="Fixed Halifax pilot " + signature, cwd=str(HERE),
        cmd=shlex.join(command), allowed_nodes=R["nodes"],
        cpu=2 if training else 1, ram_mb=6000 if training else 2500,
        vram=2000 if training and R["training_device"] == "cuda" else 0, env=R["env"],
        resource_family=T["name"] + ("/training" if training else "/evaluation"),
        allow_cpu_training=training and R["training_device"] == "cpu",
        cpu_training_justification="All allowed server nodes currently expose no CUDA devices; this fixed small-pool pilot intentionally uses CPU with unchanged model and update counts." if training else "",
        skip_launch_staging=True, allow_no_ckpt=True, allow_no_resume=True)


def submit(name, rows):
    record = HERE / (name + "_submission.json")
    if record.exists():
        raise ValueError("This batch was already submitted: " + str(record))
    path = HERE / (name + "_jobs.json")
    path.write_text(json.dumps(rows, indent=2) + "\n")
    result = json.loads(subprocess.check_output(["python3", R["scheduler"], "submit-jsonl",
        "--file", str(path), "--trusted", "--json"], text=True))
    record.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def prepare_source():
    command = [R["python"], R["remote_root"] + "/source_pool.py", "--source",
        "/home/zhengliang01/scheduleurm_work/Thesis/experiments/mcwm_bus/runtime/dataset.h5",
        "--output", R["remote_root"]]
    submit("source_pool", [job(command, "source-pool")])


def prepare_tests():
    selection = P["asset_selection"]
    for case in T["evaluation"]["cases"]:
        folder = R["assets_root"] + f"/test{case['case']:02d}"
        command = [R["python"], R["remote_root"] + "/asset_adapter.py", "--city-manifest",
            R["city_input_root"] + "/gtfs_city_manifest.json", "--out", folder,
            "--start-hour", str(selection["start_hour"]), "--service-hours", str(selection["service_hours"]),
            "--episode-seconds", str(T["evaluation"]["episode_seconds"]), "--max-services",
            str(selection["max_directional_services"]), "--demand-seed", str(case["demand_seed"]),
            "--netconvert", R["netconvert"], "--weekday-timetables",
            R["city_input_root"] + "/weekday_timetables.json"]
        print(run_remote(shlex.join(command), timeout=60), flush=True)


def collect_training(smoke=False):
    folder = "training_smoke" if smoke else "training"
    code = "from pathlib import Path; import json; root=Path(" + repr(R["remote_root"]) + ")/" + repr(folder) + "; "
    code += "rows={str(p.relative_to(root)):json.loads(p.read_text()) for p in root.rglob('*summary.json')}; "
    code += "print(json.dumps(dict(summaries=rows)))"
    report = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    seeds = T["training_seeds"][:1] if smoke else T["training_seeds"]
    expected = [f"branches/{branch}/seed{seed}/train_summary.json" for branch in T["branches"] for seed in seeds]
    report.update(protocol=T["name"], diagnostic=smoke,
        complete=all(report["summaries"].get(path, {}).get("complete") for path in expected),
        expected_branch_policies=len(expected))
    (HERE / (folder + "_summary.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(complete=report["complete"], ready=sum(path in report["summaries"] for path in expected),
        expected=len(expected), diagnostic=smoke)), flush=True)


def train(smoke=False):
    if not smoke:
        admission = json.loads((HERE / "training_smoke_summary.json").read_text())
        if not admission["complete"]:
            raise ValueError("The actual frozen training interface has not passed admission")
    seeds = T["training_seeds"][:1] if smoke else T["training_seeds"]
    rows = []
    for seed in seeds:
        command = [R["python"], "-u", R["remote_root"] + "/train_native.py",
            "--phase", "all", "--seed", str(seed), "--device", R["training_device"]]
        if smoke:
            command += ["--smoke"]
        rows.append(job(command, ("training-smoke" if smoke else "training") + f"/seed{seed}", training=True))
    submit("training_smoke" if smoke else "training", rows)


def collect_evaluation():
    code = "from pathlib import Path; import json; root=Path(" + repr(R["remote_root"]) + ")/'evaluation'; "
    code += "rows={str(p.parent.relative_to(root)):json.loads(p.read_text()) for p in root.rglob('summary.json')}; "
    code += "print(json.dumps(rows))"
    rows = json.loads(run_remote(shlex.join([R["python"], "-c", code])))
    cases, seeds = T["evaluation"]["cases"], T["training_seeds"]
    expected = [f"forward_proportional/case{c['case']:02d}" for c in cases]
    expected += [f"{branch}/seed{seed}/case{c['case']:02d}" for branch in T["branches"] for seed in seeds for c in cases]
    missing = [path for path in expected if not rows.get(path, {}).get("complete")]
    report = dict(protocol=T["name"], complete=not missing, ready=len(expected)-len(missing),
        expected=len(expected), missing=missing, results=rows)
    if not missing:
        mean = lambda values: sum(values) / len(values)
        branches = {}
        for branch in T["branches"]:
            by_seed = []
            for seed in seeds:
                policy = [rows[f"{branch}/seed{seed}/case{c['case']:02d}"] for c in cases]
                rules = [rows[f"forward_proportional/case{c['case']:02d}"] for c in cases]
                by_seed.append(dict(seed=seed,
                    cost=mean([p["generalized_seconds_per_departed_passenger"] for p in policy]),
                    cost_delta_rule=mean([p["generalized_seconds_per_departed_passenger"]-r["generalized_seconds_per_departed_passenger"] for p, r in zip(policy, rules)]),
                    completion_delta_rule_pp=100*mean([p["passengers"]["completion_rate"]-r["passengers"]["completion_rate"] for p, r in zip(policy, rules)])))
            branches[branch] = dict(cost=mean([r["cost"] for r in by_seed]),
                cost_delta_rule=mean([r["cost_delta_rule"] for r in by_seed]),
                completion_delta_rule_pp=mean([r["completion_delta_rule_pp"] for r in by_seed]), by_seed=by_seed)
        report["branches"] = branches
        report["cross_city_minus_target_model"] = dict(
            cost=branches["cross_city"]["cost"]-branches["target_model"]["cost"],
            completion_pp=branches["cross_city"]["completion_delta_rule_pp"]-branches["target_model"]["completion_delta_rule_pp"])
        report["interpretation"] = "Fixed-final three-seed offline pilot on three generated demand days; not a physical cross-fidelity result or population-level confidence interval."
    (HERE / "pilot_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("complete", "ready", "expected")}), flush=True)


def evaluation(admission=False, rules_only=False):
    seeds = T["training_seeds"][:1] if admission else T["training_seeds"]
    cases = T["evaluation"]["cases"][:1] if admission else T["evaluation"]["cases"]
    folder = "training_smoke" if admission else "training"
    family = "evaluation-admission" if admission else "evaluation"
    rows = []
    for case in cases:
        choices = [("forward_proportional", None)] if rules_only else [
            (branch, seed) for branch in T["branches"] for seed in seeds]
        for method, seed in choices:
            suffix = f"{method}/seed{seed}/case{case['case']:02d}" if seed is not None else f"{method}/case{case['case']:02d}"
            command = [R["python"], "-u", R["remote_root"] + "/evaluate_native.py",
                "--method", method, "--case", str(case["case"]), "--assets",
                R["assets_root"] + f"/test{case['case']:02d}/asset_manifest.json",
                "--output", R["remote_root"] + "/" + family + "/" + suffix]
            if seed is not None:
                command += ["--seed", str(seed), "--checkpoint", R["remote_root"] +
                    f"/{folder}/branches/{method}/seed{seed}/policy.pt"]
            rows.append(job(command, family + "/" + suffix))
    submit("evaluation_admission" if admission else "rule_evaluation" if rules_only else "policy_evaluation", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("stage", "prepare-source", "prepare-tests", "smoke", "collect-smoke",
        "train", "collect", "evaluate-admission", "evaluate-rules", "evaluate"))
    mode = parser.parse_args().mode
    if mode == "stage":
        stage_code()
    elif mode == "prepare-source":
        prepare_source()
    elif mode == "prepare-tests":
        prepare_tests()
    elif mode == "smoke":
        train(True)
    elif mode == "collect-smoke":
        collect_training(True)
    elif mode == "train":
        train()
    elif mode == "collect":
        collect_training()
        collect_evaluation()
    else:
        evaluation(admission=mode == "evaluate-admission", rules_only=mode == "evaluate-rules")
