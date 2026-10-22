"""Report the three fixed continuations using local small JSON summaries."""

import argparse
import json
import math
from pathlib import Path

from fresh_confirmation_summary import metrics
from native_terminal_adaptation_report import compact
from native_terminal_target_only_report import average, delta, read, require

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_continuation"


def summarize(p, collected):
    seeds, cases, branches = p["training_seeds"], p["evaluation"]["cases"], p["branches"]
    jobs = {f"{branch}/seed{seed}" for branch in branches for seed in seeds}
    expected = {f"{job}/case{case['case']:02d}/summary.json" for job in jobs for case in cases}
    training, evaluation = collected.get("training", {}), collected.get("evaluation", {})
    require(collected.get("protocol", p["name"]) == p["name"] and not (set(training)-jobs)
        and not (set(evaluation)-expected), "Collected continuation protocol or roster changed")
    training_ready = sum(training.get(job, {}).get("train_summary.json", {}).get("complete", False) for job in jobs)
    ready = sum(evaluation.get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=p["name"], complete=False, training_ready=training_ready,
        expected_training=len(jobs), ready=ready, expected=len(expected))
    if training_ready != len(jobs) or ready != len(expected):
        return report
    ref_p, ref = read("native_terminal_confirmation_protocol.json"), read(p["references"]["summary"])
    require(ref["complete"] and ref["protocol"] == ref_p["name"] == p["references"]["protocol"]
        and ref_p["name"] != p["name"] and ref_p["training_seeds"] == seeds
        and ref_p["evaluation"]["cases"] == cases, "References changed their original protocol or cohort")
    require(p["initialization_updates"] == 50000 and p["adaptation_updates"] == 5000
        and p["continuation_updates"] == 20000 and p["new_training_updates_per_policy"] == 20000
        and p["target_fit_rows"] == 12288 and p["target_union_rows"] == 16384
        and p["world_model_refit_updates"] == p["target_online_training_events"] == 0
        and p["evaluation"]["episode_seconds"] == ref_p["evaluation"]["episode_seconds"] == 24000,
        "Frozen continuation budget or horizon changed")
    runtime, native = read("runtime.json"), read("protocol.json")
    root, settings = Path(runtime["remote_root"]), p["training_settings"]
    config = dict(settings["trainer_config"], offline_objective="iql", online_fraction=0.,
        use_ratio_correction=False, backup_entropy=False)
    train_report = {}
    for branch, spec in branches.items():
        for seed in seeds:
            job = f"{branch}/seed{seed}"
            entry = training[job]
            t, progress = entry["train_summary.json"], entry["progress.json"]
            fields = dict(complete=True, protocol=p["name"], branch=branch, seed=seed,
                initializer=str(root / p["initializer_folder"] / f"seed{seed}/adapted.pt"),
                optimizer_transfer=True, target_pool=str(root / p["target_pool"]),
                target_fit_rows=12288, target_union_rows=16384, source_rows=p["source_rows"],
                initialization_updates=50000, adaptation_updates=5000, continuation_updates=20000,
                historical_policy_updates=75000, loaded_stored_updates=5000, stored_total_updates=25000,
                optimizer_steps={name: 20000 for name in ("actor", "critic", "value", "alpha")},
                online_critic_updates_added=20000, config=config, actor_objective=p["actor_objective"],
                free_execution_actor_mask=True, sampled_action_space=p["sampled_action_space"],
                entropy_space=p["entropy_space"], backup_entropy=False, effective_critic_backup="h2oplus",
                mixed_backup_lambda=config["mixed_backup_lambda"], passenger_discount_gamma=1., terminal_discount=0.,
                continuation_model_td_coefficient=spec["model_td_coefficient"],
                continuation_real_td_coefficient=1-spec["model_td_coefficient"], model_weight_mode=spec["model_weight_mode"],
                trust_discount_gamma=settings["qdelta_gamma"], model_generation_and_trust_enabled_in_all_arms=True,
                shadow_model_enabled=branch == "real", world_model_refit_updates=0,
                world_model_artifact=str(root / p["world_model_folder"] / f"seed{seed}/model.pt"),
                world_model_data_pool=str(root / p["world_model_data_pool"]),
                trust_update_every=settings["trust_update_every"], model_seed=seed+60000,
                continuation_rng_seed=seed+settings["continuation_rng_offset"], target_online_training_events=0,
                actor_export_equal=True, checkpoint_selection="fixed final")
            require(all(t[key] == value for key, value in fields.items()), f"Frozen continuation settings changed: {job}")
            require(progress["phase"] == branch and progress["phase_updates"] == 20000
                and progress["completed_updates"] == 75000, f"Continuation updates are incomplete: {job}")
            before, after = t["fit_terminal_before"], t["fit_terminal_after"]
            require(before["rows"] == after["rows"] == 6 and before["backup"] == after["backup"] == "h2oplus"
                and before["terminal_target_exact"] and after["terminal_target_exact"]
                and before["targets"] == after["targets"], f"Actual terminal targets changed: {job}")
            train_report[job] = dict(settings_exact=True, continuation_updates=20000,
                stored_updates=[5000, 25000], historical_updates=[55000, 75000], optimizer_steps=t["optimizer_steps"],
                fit_terminal_before=before, fit_terminal_after=after,
                model_td_coefficient=t["continuation_model_td_coefficient"], model_weight_mode=t["model_weight_mode"],
                final_model_weight_mean=progress["metrics"]["model_weight_mean"],
                final_model_td_loss=progress["metrics"]["model_td_loss"],
                model_td_weight_reduction=t["model_td_weight_reduction"], passenger_discount_gamma=t["passenger_discount_gamma"],
                trust_discount_gamma=t["trust_discount_gamma"], confidence_definition=t["confidence_definition"],
                log_alpha_before=t["log_alpha_before"], log_alpha_after=t["log_alpha_after"], final_training_metrics=progress["metrics"])
    rows = dict(ref["results"])
    old_expected = {f"repaired/{seed}/{case['case']}" for seed in seeds for case in cases} | {f"strong_rule/{case['case']}" for case in cases}
    require(set(rows) == old_expected, "References contain another cohort")
    for branch in branches:
        for seed in seeds:
            for case in cases:
                name = f"{branch}/seed{seed}/case{case['case']:02d}/summary.json"
                raw = evaluation[name]
                require(raw["evaluation_protocol"] == str(root / (PREFIX+"_protocol.json")) and raw["asset_manifest"] ==
                    str(Path(runtime["assets_root"]) / p["asset_folder"] / f"case{case['case']:02d}/asset_manifest.json"),
                    f"New evaluation protocol or assets changed: {name}")
                rows[f"{branch}/{seed}/{case['case']}"] = compact(raw)
    for name, row in rows.items():
        branch, ci = name.split("/")[0], int(name.split("/")[-1])
        seed = None if branch == "strong_rule" else int(name.split("/")[1])
        case, rule = next(case for case in cases if case["case"] == ci), rows[f"strong_rule/{ci}"]
        folder = branches[branch]["artifact_folder"] if branch in branches else p["references"]["artifact_folder"]
        fields = dict(complete=True, training_seed=seed, **case, horizon_seconds=24000,
            method="forward_proportional" if seed is None else "cross_city",
            checkpoint=None if seed is None else str(root / folder / f"seed{seed}/policy.pt"),
            actor_action_space="proposal" if seed is None else "executed", protocol=p["name"] if branch in branches else ref_p["name"],
            execution_layer="early60", completion_guard="quarter-load-sixty", simulator_version=rule["simulator_version"],
            line_ids=rule["line_ids"], departed=rule["departed"])
        require(all(row[key] == value for key, value in fields.items()), f"Paired evaluation identity changed: {name}")
        require("SUMO "+native["smoke"]["sumo_version"] in row["simulator_version"][1] and row["ledger_balance_error"] == 0
            and math.isclose(row["cost"], 2*row["wait_seconds"]+row["ride_seconds"], rel_tol=1e-12, abs_tol=1e-8)
            and math.isclose(row["completion"], row["completed"]/row["departed"], rel_tol=1e-12), f"Passenger ledger changed: {name}")
    values = {name: dict(metrics(row), completed=row["completed"], departed=row["departed"]) for name, row in rows.items()}
    by_method = {}
    for branch in (*branches, "repaired", "strong_rule"):
        selected = [value for name, value in values.items() if name.split("/")[0] == branch]
        by_method[branch] = dict(mean=average(selected), actual_evaluations=len(selected))
        if branch != "strong_rule":
            by_method[branch]["by_seed"] = {str(seed): average([values[f"{branch}/{seed}/{case['case']}"] for case in cases]) for seed in seeds}
    comparisons = {}
    for name, (left, right) in p["paired_comparisons"].items():
        pairs = [dict(training_seed=seed, case=case["case"], differences=delta(values[f"{left}/{seed}/{case['case']}"],
            values[f"strong_rule/{case['case']}" if right == "strong_rule" else f"{right}/{seed}/{case['case']}"])) for seed in seeds for case in cases]
        per_seed = {str(seed): dict(differences=delta(by_method[left]["by_seed"][str(seed)], by_method[right]["mean"] if right ==
            "strong_rule" else by_method[right]["by_seed"][str(seed)]), better_pairs=sum(pair["differences"]["cost"] < 0
                for pair in pairs if pair["training_seed"] == seed)) for seed in seeds}
        per_case = {str(case["case"]): delta(average([values[f"{left}/{seed}/{case['case']}"] for seed in seeds]),
            values[f"strong_rule/{case['case']}"] if right == "strong_rule" else average([values[f"{right}/{seed}/{case['case']}"] for seed in seeds])) for case in cases}
        diff = delta(by_method[left]["mean"], by_method[right]["mean"])
        comparisons[name] = dict(left=left, right=right, differences=diff, cost_percent_of_right=100*diff["cost"]/by_method[right]["mean"]["cost"],
            compared_pairs=len(pairs), better_pairs=sum(pair["differences"]["cost"] < 0 for pair in pairs),
            better_training_seed_means=sum(row["differences"]["cost"] < 0 for row in per_seed.values()),
            better_case_means=sum(row["cost"] < 0 for row in per_case.values()),
            completion_count_difference_sum=sum(pair["differences"]["completed"] for pair in pairs), by_seed=per_seed, by_case=per_case, pairs=pairs)
    report.update(complete=True, results=rows, training=train_report, by_method=by_method, comparison=comparisons,
        actual_new_evaluations=len(expected), old_evaluations_reused=len(old_expected), actual_evaluations=len(rows),
        actual_new_policy_updates=len(jobs)*20000, new_training_updates_per_policy=20000,
        actual_rule_evaluations=len(cases), training_seeds=seeds, reference_protocol=ref_p["name"],
        comparison_boundary=p["comparison_boundary"], model_loss_formula=p["model_loss_formula"], interpretation=p["interpretation"],
        weighting="Equal case/seed weights; three actual rule runs reused across nine pairs; all seeds retained, no population CI.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / (PREFIX+"_protocol.json"))
    parser.add_argument("--summary", type=Path, default=HERE / (PREFIX+"_summary.json"))
    parser.add_argument("--output", type=Path, default=HERE / (PREFIX+"_reporting.json"))
    args = parser.parse_args()
    report = summarize(json.loads(args.protocol.read_text()), json.loads(args.summary.read_text()) if args.summary.exists() else {})
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("complete", "training_ready", "ready", "expected")}))


if __name__ == "__main__":
    main()
