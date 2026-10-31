"""Report fixed corrected-pool target-only training against unchanged local references."""

import argparse
import json
import math
from pathlib import Path

from fresh_confirmation_summary import difference, mean, metrics
from native_terminal_adaptation_report import compact

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_target_only"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return json.loads((HERE / name).read_text())


def average(rows):
    return dict(mean(rows), completed=sum(row["completed"] for row in rows)/len(rows),
        departed=sum(row["departed"] for row in rows)/len(rows))


def delta(left, right):
    return dict(difference(left, right), completed=left["completed"]-right["completed"],
        departed=left["departed"]-right["departed"])


def summarize(p, collected):
    seeds, cases = p["training_seeds"], p["evaluation"]["cases"]
    expected = {f"seed{seed}/evaluation/case{case['case']:02d}/summary.json" for seed in seeds for case in cases}
    training, evaluation = collected.get("training", {}), collected.get("evaluation", {})
    require(collected.get("protocol", p["name"]) == p["name"] and not (set(evaluation)-expected) and
        not (set(training)-set(map(str, seeds))), "Collected protocol or roster changed")
    training_ready = sum(training.get(str(seed), {}).get("train_summary.json", {}).get("complete", False) for seed in seeds)
    ready = sum(evaluation.get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=p["name"], complete=False, training_ready=training_ready,
        expected_training=len(seeds), ready=ready, expected=len(expected))
    if training_ready != len(seeds) or ready != len(expected):
        return report
    ref_p, ref = read("native_terminal_confirmation_protocol.json"), read(p["source_reference"]["summary"])
    require(ref["complete"] and ref["protocol"] == ref_p["name"] == p["source_reference"]["protocol"] and
        ref_p["name"] != p["name"] and ref_p["training_seeds"] == seeds and ref_p["evaluation"]["cases"] == cases,
        "Source references must retain their complete original protocol and roster")
    require(p["initialization_updates"] == 50000 and p["adaptation_updates"] == 5000 and
        p["source_rows"] == 0 and p["target_fit_rows"] == 12288 and p["target_union_rows"] == 16384 and
        p["evaluation"]["episode_seconds"] == ref_p["evaluation"]["episode_seconds"] == 24000,
        "Frozen training budget or evaluation horizon changed")
    runtime, native = read("runtime.json"), read("protocol.json")
    root, config = Path(runtime["remote_root"]), dict(p["training_settings"]["trainer_config"], **p["config_overrides"])
    train_report = {}
    for seed in seeds:
        entry = training[str(seed)]
        t = entry["train_summary.json"]
        fields = dict(complete=True, protocol=p["name"], seed=seed, admission=False, initialization_updates=50000,
            adaptation_updates=5000, historical_policy_updates=55000, target_fit_rows=12288, target_union_rows=16384,
            target_pool=str(root / p["target_pool"]), initializer=str(root / p["artifact_folder"] /
                f"seed{seed}" / "initialization/initializer.pt"), config=config, fresh_initialization=True,
            optimizer_transfer=False, encoder_columns_reset=False, source_rows=0, actor_export_equal=True,
            world_model_updates=0, continuation_updates=0, target_online_training_events=0)
        require(all(t[key] == value for key, value in fields.items()), f"Target-only seed{seed} changed training settings")
        for name, count, historical in (("initialization_progress.json", 50000, 50000), ("progress.json", 5000, 55000)):
            require(entry[name]["phase_updates"] == count and entry[name]["completed_updates"] == historical,
                f"Target-only seed{seed} did not finish {name}")
        stages = ["fit_terminal_initial", "fit_terminal_initialized", "fit_terminal_adaptation_before", "fit_terminal_adapted"]
        require(all(t[stage]["rows"] == 6 and t[stage]["terminal_target_exact"] and
            t[stage]["targets"] == t[stages[0]]["targets"] for stage in stages), f"Target-only seed{seed} terminal targets differ")
        require(t["fit_terminal_initialized"]["recorded_min_q"] ==
            t["fit_terminal_adaptation_before"]["recorded_min_q"], f"Target-only seed{seed} handoff changed recorded Q")
        train_report[str(seed)] = dict(initialization_updates=50000, adaptation_updates=5000, historical_policy_updates=55000,
            target_pool=t["target_pool"], source_rows=0, settings_exact=True,
            fit_terminal_mae={stage: t[stage]["mae"] for stage in stages})
    rows = dict(ref["results"])
    old_expected = {f"repaired/{seed}/{case['case']}" for seed in seeds for case in cases} | {f"strong_rule/{case['case']}" for case in cases}
    require(set(rows) == old_expected, "Source/rule references contain another cohort")
    for seed in seeds:
        for case in cases:
            name = f"seed{seed}/evaluation/case{case['case']:02d}/summary.json"
            raw = evaluation[name]
            require(raw["evaluation_protocol"] == str(root / (PREFIX+"_protocol.json")) and raw["asset_manifest"] ==
                str(Path(runtime["assets_root"]) / p["asset_folder"] / f"case{case['case']:02d}/asset_manifest.json"),
                f"Target-only {name} changed protocol or assets")
            rows[f"target_only/{seed}/{case['case']}"] = compact(raw)
    for name, row in rows.items():
        branch = name.split("/")[0]
        seed = None if branch == "strong_rule" else int(name.split("/")[1])
        case = next(case for case in cases if case["case"] == int(name.split("/")[-1]))
        rule = rows[f"strong_rule/{case['case']}"]
        folder = p["artifact_folder"] if branch == "target_only" else p["source_reference"]["artifact_folder"]
        fields = dict(complete=True, training_seed=seed, **case, horizon_seconds=24000,
            method="forward_proportional" if seed is None else ("target_only" if branch == "target_only" else "cross_city"),
            checkpoint=None if seed is None else str(root / folder / f"seed{seed}" / "policy.pt"),
            actor_action_space="proposal" if seed is None else "executed", protocol=p["name"] if branch == "target_only" else ref_p["name"],
            execution_layer="early60", completion_guard="quarter-load-sixty", simulator_version=rule["simulator_version"],
            line_ids=rule["line_ids"], departed=rule["departed"])
        require(all(row[key] == value for key, value in fields.items()), f"Paired evaluation identity changed: {name}")
        require("SUMO "+native["smoke"]["sumo_version"] in row["simulator_version"][1] and row["ledger_balance_error"] == 0 and
            math.isclose(row["cost"], 2*row["wait_seconds"]+row["ride_seconds"], rel_tol=1e-12, abs_tol=1e-8) and
            math.isclose(row["completion"], row["completed"]/row["departed"], rel_tol=1e-12), f"Passenger ledger or cost changed: {name}")
    values = {name: dict(metrics(row), completed=row["completed"], departed=row["departed"]) for name, row in rows.items()}
    by_method = {}
    for branch in ("repaired", "target_only", "strong_rule"):
        selected = [value for name, value in values.items() if name.split("/")[0] == branch]
        by_method[branch] = dict(mean=average(selected), actual_evaluations=len(selected))
        if branch != "strong_rule":
            by_method[branch]["by_seed"] = {str(seed): average([values[f"{branch}/{seed}/{case['case']}"] for case in cases]) for seed in seeds}
    comparisons = {}
    for name, left, right in (("source_vs_target", "repaired", "target_only"),
            ("target_vs_rule", "target_only", "strong_rule"), ("source_vs_rule", "repaired", "strong_rule")):
        pairs = [dict(training_seed=seed, case=case["case"], differences=delta(values[f"{left}/{seed}/{case['case']}"],
            values[f"strong_rule/{case['case']}" if right == "strong_rule" else f"{right}/{seed}/{case['case']}"])) for seed in seeds for case in cases]
        per_seed = {str(seed): dict(differences=delta(by_method[left]["by_seed"][str(seed)], by_method[right]["mean"] if right ==
            "strong_rule" else by_method[right]["by_seed"][str(seed)]), better_pairs=sum(pair["differences"]["cost"] < 0
                for pair in pairs if pair["training_seed"] == seed)) for seed in seeds}
        difference_mean = delta(by_method[left]["mean"], by_method[right]["mean"])
        comparisons[name] = dict(left=left, right=right, differences=difference_mean,
            cost_percent_of_right=100*difference_mean["cost"]/by_method[right]["mean"]["cost"], compared_pairs=len(pairs),
            better_pairs=sum(pair["differences"]["cost"] < 0 for pair in pairs),
            better_training_seed_means=sum(value["differences"]["cost"] < 0 for value in per_seed.values()),
            completion_count_difference_sum=sum(pair["differences"]["completed"] for pair in pairs),
            by_case={str(case["case"]): delta(average([values[f"{left}/{seed}/{case['case']}"] for seed in seeds]),
                values[f"strong_rule/{case['case']}"] if right == "strong_rule" else
                average([values[f"{right}/{seed}/{case['case']}"] for seed in seeds])) for case in cases},
            by_seed=per_seed, pairs=pairs)
    report.update(complete=True, results=rows, training=train_report, by_method=by_method, comparison=comparisons,
        actual_new_evaluations=len(expected), old_evaluations_reused=len(old_expected), actual_evaluations=len(rows),
        actual_rule_evaluations=len(cases), training_seeds=seeds, reference_protocol=ref_p["name"],
        budget_boundary=p["budget_boundary"], interpretation=p["interpretation"],
        weighting="Equal case/seed weights; three actual rule runs reused across nine pairs; all seeds retained, no population CI.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / (PREFIX+"_protocol.json"))
    parser.add_argument("--summary", type=Path, default=HERE / (PREFIX+"_summary.json"))
    parser.add_argument("--output", type=Path, default=HERE / (PREFIX+"_reporting.json"))
    args = parser.parse_args()
    p = json.loads(args.protocol.read_text())
    report = summarize(p, json.loads(args.summary.read_text()) if args.summary.exists() else {})
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("complete", "training_ready", "ready", "expected")}))


if __name__ == "__main__":
    main()
