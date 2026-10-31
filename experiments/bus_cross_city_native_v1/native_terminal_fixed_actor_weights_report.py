"""Report fixed55k IQL actor weights against the original fresh-day references."""

import argparse
import json
import math
from pathlib import Path

from fresh_confirmation_summary import metrics, summarize as validate_reference
from native_terminal_adaptation_report import compact
from native_terminal_target_only_report import average, delta, read, require

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_fixed_actor_weights"


def summarize(p, collected):
    seeds, cases, branch = p["training_seeds"], p["evaluation"]["cases"], p["branch"]
    jobs = {f"{branch}/seed{seed}" for seed in seeds}
    expected = {f"{job}/case{case['case']:02d}/summary.json" for job in jobs for case in cases}
    training, evaluation = collected.get("training", {}), collected.get("evaluation", {})
    require(collected.get("protocol", p["name"]) == p["name"] and not (set(training)-jobs) and not (set(evaluation)-expected),
        "Collected fixed-weight protocol or training/evaluation roster changed")
    tr = sum(training.get(job, {}).get("train_summary.json", {}).get("complete", False) for job in jobs)
    er = sum(evaluation.get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=p["name"], complete=False, training_ready=tr, expected_training=len(jobs), ready=er, expected=len(expected))
    if tr != len(jobs) or er != len(expected):
        return report
    ref_p, ref = read(p["reference_protocol"]), read(p["reference_summary"])
    checked = validate_reference(ref["results"], ref_p)
    require(collected["complete"] and ref["complete"] and checked["complete"] and ref["protocol"] == checked["protocol"] ==
        ref_p["name"] != p["name"] and ref_p["training_seeds"] == seeds and ref_p["evaluation"]["cases"] == cases and
        ref_p["asset_folder"] == p["asset_folder"] and p["evaluation"]["episode_seconds"] == ref_p["evaluation"]["episode_seconds"] == 24000,
        "Original fresh references, identities, cases or horizon changed")
    require(branch == "real_v_fixed55_weights" and p["continuation_updates_per_seed"] == 20000 and
        p["new_training_updates"] == len(jobs)*20000 == 60000 and p["historical_final_updates"] == 75000 and
        p["world_model_refit_updates"] == p["target_online_training_events"] == 0 and p["admission_training_updates"] == 8 and
        p["new_evaluations"] == len(expected) == 9 and p["reference_evaluations_reused"] == len(ref["results"]) == 21 and
        p["actual_evaluations_planned"] == 30, "Fixed actor-weight intervention budget changed")
    runtime, native, template = read("runtime.json"), read("protocol.json"), read(p["training_template_protocol"])
    root, baseline = Path(runtime["remote_root"]), read("native_terminal_actor_unmasked_summary.json")
    require(baseline["complete"] and baseline["protocol"] == template["name"] and template["training_seeds"] == seeds,
        "Original dynamic-weight training reference changed")
    admission = read(PREFIX+"_admission_summary.json")
    checks = ("config_equal initial_parameters_equal initial_complete_optimizer_states_equal actual_all_real_batches_equal "
        "actual_first_model_batch_equal first_model_weights_equal first_original_actor_weights_equal native_actor_weights_verified "
        "native_actor_full_batch_mean_exact fixed_actor_scorer_matches_initial55 fixed_actor_scorer_unchanged actual_actor_weights_changed "
        "replay_requests_equal all_four_update_rng_equal all_four_critic_value_paths_equal all_four_critic_value_losses_equal corrected_terminal_targets_exact").split()
    require(admission["complete"] and admission["protocol"] == p["name"] and admission["seed"] == seeds[0] and
        admission["admission_training_updates"] == 8 and all(admission[key] is True for key in checks) and
        admission["checkpoint_files_written"] == admission["new_simulation_events"] == admission["evaluation_jobs"] == 0 and
        set(admission["arms"]) == {"dynamic_weights", "fixed55_weights"}, "Fixed-weight eight-update admission changed or is incomplete")
    initial_config = baseline["training"][f"real_v_unmasked/seed{seeds[0]}"]["train_summary.json"]["config"]
    for arm, hooks in (("dynamic_weights", 0), ("fixed55_weights", 4)):
        a = admission["arms"][arm]
        require(a["updates"] == a["actual_native_actor_calls"] == 4 and a["optimizer_steps"] ==
            {name: 4 for name in ("actor", "critic", "value", "alpha")} and a["fixed_actor_weight_updates"] == hooks and
            a["native_actor_weights_exact"] and a["native_actor_full_batch_mean_exact"] and a["config"] == initial_config and
            a["actor_batch_denominator"] == initial_config["batch_size"], f"Fixed-weight admission arm {arm} changed the native actor path or budget")
    unchanged = ("initializer optimizer_transfer target_pool target_fit_rows target_union_rows source_rows initialization_updates "
        "adaptation_updates continuation_updates historical_policy_updates loaded_stored_updates stored_total_updates optimizer_steps "
        "online_critic_updates_added config actor_objective free_execution_actor_mask sampled_action_space entropy_space backup_entropy effective_critic_backup "
        "mixed_backup_lambda passenger_discount_gamma terminal_discount continuation_model_td_coefficient continuation_real_td_coefficient "
        "model_weight_mode trust_discount_gamma model_generation_and_trust_enabled_in_all_arms shadow_model_enabled world_model_artifact "
        "world_model_data_pool world_model_refit_updates trust_update_every model_seed continuation_rng_seed target_online_training_events "
        "actor_export_equal checkpoint_selection").split()
    train_reports = {}
    for seed in seeds:
        job = f"{branch}/seed{seed}"
        t, progress = training[job]["train_summary.json"], training[job]["progress.json"]
        old = baseline["training"][f"real_v_unmasked/seed{seed}"]["train_summary.json"]
        require(t["protocol"] == p["name"] and t["branch"] == branch and t["seed"] == seed and all(t[key] == old[key] for key in unchanged)
            and t["optimizer_steps"] == {name: 20000 for name in ("actor", "critic", "value", "alpha")} and
            t["config"]["iql_advantage_temperature"] == 3. and t["config"]["iql_max_weight"] == 100. and
            progress["phase"] == branch and progress["phase_updates"] == 20000 and progress["completed_updates"] == 75000,
            f"Fixed-weight seed{seed} changed matched training configuration or completed budget")
        scorer_fields = dict(actor_weight_source="frozen55_target_critic_and_value", actor_weight_scorer_initializer=old["initializer"],
            actor_weight_beta=3., actor_weight_max=100., frozen_actor_scorer_unchanged=True, fixed_actor_weight_updates=20000,
            actor_weight_formula=p["actor_weights"],
            single_intervention="freeze_actor_advantage_scorer_at_initial55k", config_equal_to_unmasked=True)
        require(all(t[key] == value for key, value in scorer_fields.items()), f"Fixed-weight seed{seed} frozen scorer identity or formula changed")
        before, after = t["fit_terminal_before"], t["fit_terminal_after"]
        require(before == old["fit_terminal_before"] and before["rows"] == after["rows"] == 6 and before["terminal_target_exact"] and
            after["terminal_target_exact"] and before["targets"] == after["targets"], f"Fixed-weight seed{seed} initial or terminal targets changed")
        train_reports[job] = dict(settings_exact=True, historical_updates=[55000, 75000], stored_updates=[5000, 25000],
            optimizer_steps=t["optimizer_steps"], fit_terminal_before=before, fit_terminal_after=after, final_training_metrics=progress["metrics"],
            **scorer_fields)
    rows = dict(ref["results"])
    for seed in seeds:
        for case in cases:
            name = f"{branch}/seed{seed}/case{case['case']:02d}/summary.json"
            raw, rule = evaluation[name], rows[f"strong_rule/{case['case']}"]
            fields = dict(complete=True, method="cross_city", training_seed=seed, **case, horizon_seconds=24000,
                protocol=p["name"], checkpoint=str(root / PREFIX / branch / f"seed{seed}/policy.pt"), actor_action_space="executed",
                execution_layer="early60", completion_guard="quarter-load-sixty", simulator_version=rule["simulator_version"], line_ids=rule["line_ids"],
                evaluation_protocol=str(root / (PREFIX+"_protocol.json")), asset_manifest=str(Path(runtime["assets_root"]) / p["asset_folder"] /
                    f"case{case['case']:02d}/asset_manifest.json"))
            require(all(raw[key] == value for key, value in fields.items()), f"Fixed-weight evaluation identity changed: {name}")
            row, passengers = compact(raw), raw["passengers"]
            require(row["departed"] == rule["departed"] and row["ledger_balance_error"] == 0 and
                "SUMO "+native["smoke"]["sumo_version"] in row["simulator_version"][1] and all(math.isclose(row[key],
                    passengers[total]/row["departed"], rel_tol=1e-12, abs_tol=1e-8) for key, total in (("wait_seconds",
                    "accrued_wait_seconds_total"), ("ride_seconds", "accrued_in_vehicle_seconds_total"))) and
                math.isclose(row["completion"], row["completed"]/row["departed"], rel_tol=1e-12),
                f"Fixed-weight passenger ledger, paired population or denominator changed: {name}")
            row["mean_requested_hold_seconds"] = raw["mean_requested_hold_seconds"]
            rows[f"{branch}/{seed}/{case['case']}"] = row
    values = {name: dict(metrics(row), completed=row["completed"], departed=row["departed"],
        mean_requested_hold_seconds=row["mean_requested_hold_seconds"]) for name, row in rows.items()}
    def mean(chosen):
        return dict(average(chosen), mean_requested_hold_seconds=sum(v["mean_requested_hold_seconds"] for v in chosen)/len(chosen))
    def difference(left, right):
        return dict(delta(left, right), mean_requested_hold_seconds=left["mean_requested_hold_seconds"]-right["mean_requested_hold_seconds"])
    by_method = {}
    for method in (branch, "unmasked", "repaired", "strong_rule"):
        chosen_seeds = [None] if method == "strong_rule" else seeds
        def key(seed, case):
            return f"{method}/{case}" if seed is None else f"{method}/{seed}/{case}"
        selected = [values[key(seed, c["case"])] for seed in chosen_seeds for c in cases]
        by_method[method] = dict(mean=mean(selected), actual_evaluations=len(selected),
            by_case={str(c["case"]): mean([values[key(seed, c["case"])] for seed in chosen_seeds]) for c in cases})
        if method != "strong_rule":
            by_method[method]["by_seed"] = {str(seed): mean([values[key(seed, c["case"])] for c in cases]) for seed in seeds}
    comparison = {}
    for right in ("unmasked", "repaired", "strong_rule"):
        pairs = [dict(training_seed=seed, case=c["case"], differences=difference(values[f"{branch}/{seed}/{c['case']}"],
            values[f"strong_rule/{c['case']}" if right == "strong_rule" else f"{right}/{seed}/{c['case']}"])) for seed in seeds for c in cases]
        per_seed = {str(seed): dict(differences=difference(by_method[branch]["by_seed"][str(seed)], by_method[right]["mean"]
            if right == "strong_rule" else by_method[right]["by_seed"][str(seed)]), better_pairs=sum(v["differences"]["cost"] < 0
                for v in pairs if v["training_seed"] == seed)) for seed in seeds}
        per_case = {str(c["case"]): difference(by_method[branch]["by_case"][str(c["case"])], by_method[right]["by_case"][str(c["case"])]) for c in cases}
        diff = difference(by_method[branch]["mean"], by_method[right]["mean"])
        comparison[f"fixed_weights_vs_{right}"] = dict(left=branch, right=right, differences=diff,
            cost_percent_of_right=100*diff["cost"]/by_method[right]["mean"]["cost"], compared_pairs=len(pairs),
            better_pairs=sum(v["differences"]["cost"] < 0 for v in pairs), better_training_seed_means=sum(v["differences"]["cost"] < 0
                for v in per_seed.values()), better_case_means=sum(v["cost"] < 0 for v in per_case.values()),
            completion_count_difference_sum=sum(v["differences"]["completed"] for v in pairs), by_seed=per_seed, by_case=per_case, pairs=pairs)
    report.update(complete=True, training=train_reports, results=rows, by_method=by_method, comparison=comparison,
        actual_new_evaluations=len(expected), old_evaluations_reused=len(ref["results"]), actual_evaluations=len(rows), actual_rule_evaluations=len(cases),
        actual_new_policy_updates=len(jobs)*20000, admission_training_updates=8, admission_validated=True, world_model_refit_updates=0, training_seeds=seeds,
        training_template_protocol=template["name"], reference_protocol=ref_p["name"], interpretation=p["interpretation"], decision=p["decision"],
        weighting="Equal seed/case weights; three actual rule runs reused across nine pairs; all30 actual evaluations retained, no population CI.")
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
