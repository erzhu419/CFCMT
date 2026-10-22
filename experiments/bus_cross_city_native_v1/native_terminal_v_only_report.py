"""Report the single mixed-backup change using local small JSON results."""

import argparse
import json
import math
from pathlib import Path

from fresh_confirmation_summary import metrics
from native_terminal_adaptation_report import compact
from native_terminal_target_only_report import average, delta, read, require

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_v_only"


def summarize(p, collected):
    seeds, cases = p["training_seeds"], p["evaluation"]["cases"]
    jobs = {f"real_v_only/seed{seed}" for seed in seeds}
    expected = {f"{job}/case{case['case']:02d}/summary.json" for job in jobs for case in cases}
    training, evaluation = collected.get("training", {}), collected.get("evaluation", {})
    require(collected.get("protocol", p["name"]) == p["name"] and not (set(training)-jobs)
        and not (set(evaluation)-expected), "Collected V-only protocol or roster changed")
    tr = sum(training.get(job, {}).get("train_summary.json", {}).get("complete", False) for job in jobs)
    er = sum(evaluation.get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=p["name"], complete=False, training_ready=tr, expected_training=len(jobs), ready=er, expected=len(expected))
    if tr != len(jobs) or er != len(expected):
        return report
    matched = p["matched_reference"]
    old_p, old, old_report = read(matched["protocol_file"]), read(matched["summary"]), read(matched["reporting"])
    ref_p, ref = read("native_terminal_confirmation_protocol.json"), read(p["references"]["summary"])
    require(old["complete"] and old_report["complete"] and old["protocol"] == old_report["protocol"] == old_p["name"] == matched["protocol"]
        and ref["complete"] and ref["protocol"] == ref_p["name"] == p["references"]["protocol"], "Original reference protocols are incomplete or changed")
    require(list(p["branches"]) == ["real_v_only"] and all(q["training_seeds"] == seeds and q["evaluation"]["cases"] == cases
        and q["evaluation"]["episode_seconds"] == 24000 for q in (p, old_p, ref_p)), "Reference roster or horizon changed")
    settings = dict(p["training_settings"], trainer_config=dict(p["training_settings"]["trainer_config"], mixed_backup_lambda=.1),
        continuation_backup=old_p["training_settings"]["continuation_backup"])
    require(settings == old_p["training_settings"] and p["training_settings"]["trainer_config"]["mixed_backup_lambda"] == 1.
        and p["continuation_updates"] == p["new_training_updates_per_policy"] == 20000, "The frozen lambda-only settings changed")
    runtime, native = read("runtime.json"), read("protocol.json")
    root = Path(runtime["remote_root"])
    train_report = {}
    unchanged = ("complete initializer optimizer_transfer target_pool target_fit_rows target_union_rows source_rows initialization_updates "
        "adaptation_updates continuation_updates historical_policy_updates loaded_stored_updates stored_total_updates optimizer_steps "
        "online_critic_updates_added actor_objective free_execution_actor_mask sampled_action_space entropy_space backup_entropy "
        "effective_critic_backup passenger_discount_gamma terminal_discount continuation_model_td_coefficient continuation_real_td_coefficient "
        "model_weight_mode trust_discount_gamma model_generation_and_trust_enabled_in_all_arms shadow_model_enabled world_model_artifact "
        "world_model_data_pool world_model_refit_updates trust_update_every model_seed continuation_rng_seed target_online_training_events "
        "actor_export_equal checkpoint_selection").split()
    for seed in seeds:
        job, old_job = f"real_v_only/seed{seed}", f"real/seed{seed}"
        t, progress = training[job]["train_summary.json"], training[job]["progress.json"]
        baseline = old["training"][old_job]["train_summary.json"]
        require(old_report["training"][old_job]["settings_exact"] and all(t[key] == baseline[key] for key in unchanged)
            and t["protocol"] == p["name"] and t["branch"] == "real_v_only" and t["seed"] == seed,
            f"V-only seed{seed} changed the matched training identity or budget")
        require(t["config"] == dict(baseline["config"], mixed_backup_lambda=1.) and t["mixed_backup_lambda"] == 1.
            and t["original_mixed_backup_lambda"] == baseline["mixed_backup_lambda"] == .1
            and t["single_config_field_changed"] == "mixed_backup_lambda"
            and progress["phase"] == "real_v_only" and progress["phase_updates"] == 20000 and progress["completed_updates"] == 75000,
            f"V-only seed{seed} changed more than the mixed-backup lambda")
        before, after = t["fit_terminal_before"], t["fit_terminal_after"]
        require(before["rows"] == after["rows"] == 6 and before["backup"] == after["backup"] == "h2oplus"
            and before["terminal_target_exact"] and after["terminal_target_exact"] and before["targets"] == after["targets"]
            == baseline["fit_terminal_before"]["targets"] and before["recorded_min_q"] == baseline["fit_terminal_before"]["recorded_min_q"],
            f"V-only seed{seed} changed its initial parameters or corrected terminal targets")
        train_report[job] = dict(settings_exact=True, mixed_backup_lambda=1., original_mixed_backup_lambda=.1,
            stored_updates=[5000, 25000], historical_updates=[55000, 75000], optimizer_steps=t["optimizer_steps"],
            fit_terminal_before=before, fit_terminal_after=after, final_training_metrics=progress["metrics"])
    rows = dict(ref["results"])
    old_keys = {f"repaired/{seed}/{case['case']}" for seed in seeds for case in cases} | {f"strong_rule/{case['case']}" for case in cases}
    require(set(rows) == old_keys, "Repaired/rule reference includes another cohort")
    real_keys = {f"real/{seed}/{case['case']}" for seed in seeds for case in cases}
    rows.update({name: old_report["results"][name] for name in real_keys})
    for seed in seeds:
        for case in cases:
            name = f"real_v_only/seed{seed}/case{case['case']:02d}/summary.json"
            raw = evaluation[name]
            require(raw["evaluation_protocol"] == str(root / (PREFIX+"_protocol.json")) and raw["asset_manifest"] ==
                str(Path(runtime["assets_root"]) / p["asset_folder"] / f"case{case['case']:02d}/asset_manifest.json"), f"New protocol or assets changed: {name}")
            rows[f"real_v_only/{seed}/{case['case']}"] = compact(raw)
    specs = {"real_v_only": (p["name"], p["branches"]["real_v_only"]["artifact_folder"]),
        "real": (old_p["name"], matched["artifact_folder"]), "repaired": (ref_p["name"], p["references"]["artifact_folder"]),
        "strong_rule": (ref_p["name"], None)}
    for name, row in rows.items():
        branch, ci = name.split("/")[0], int(name.split("/")[-1])
        seed = None if branch == "strong_rule" else int(name.split("/")[1])
        case, rule = next(case for case in cases if case["case"] == ci), rows[f"strong_rule/{ci}"]
        protocol_name, folder = specs[branch]
        fields = dict(complete=True, training_seed=seed, **case, horizon_seconds=24000, protocol=protocol_name,
            method="forward_proportional" if seed is None else "cross_city", checkpoint=None if seed is None else str(root / folder / f"seed{seed}/policy.pt"),
            actor_action_space="proposal" if seed is None else "executed", execution_layer="early60", completion_guard="quarter-load-sixty",
            simulator_version=rule["simulator_version"], line_ids=rule["line_ids"], departed=rule["departed"])
        require(all(row[key] == value for key, value in fields.items()) and "SUMO "+native["smoke"]["sumo_version"] in row["simulator_version"][1]
            and row["ledger_balance_error"] == 0 and math.isclose(row["cost"], 2*row["wait_seconds"]+row["ride_seconds"], rel_tol=1e-12, abs_tol=1e-8)
            and math.isclose(row["completion"], row["completed"]/row["departed"], rel_tol=1e-12), f"Paired evaluation identity or ledger changed: {name}")
    values = {name: dict(metrics(row), completed=row["completed"], departed=row["departed"]) for name, row in rows.items()}
    by_method = {b: dict(mean=average([v for k, v in values.items() if k.split("/")[0] == b]),
        actual_evaluations=sum(k.split("/")[0] == b for k in values)) for b in specs}
    for b in ("real_v_only", "real", "repaired"):
        by_method[b]["by_seed"] = {str(seed): average([values[f"{b}/{seed}/{c['case']}"] for c in cases]) for seed in seeds}
    comparisons = {}
    for name, (left, right) in p["paired_comparisons"].items():
        pairs = [dict(training_seed=seed, case=c["case"], differences=delta(values[f"{left}/{seed}/{c['case']}"],
            values[f"strong_rule/{c['case']}" if right == "strong_rule" else f"{right}/{seed}/{c['case']}"])) for seed in seeds for c in cases]
        per_seed = {str(seed): dict(differences=delta(by_method[left]["by_seed"][str(seed)], by_method[right]["mean"] if right == "strong_rule"
            else by_method[right]["by_seed"][str(seed)]), better_pairs=sum(v["differences"]["cost"] < 0 for v in pairs if v["training_seed"] == seed)) for seed in seeds}
        per_case = {str(c["case"]): delta(average([values[f"{left}/{seed}/{c['case']}"] for seed in seeds]), values[f"strong_rule/{c['case']}"] if right ==
            "strong_rule" else average([values[f"{right}/{seed}/{c['case']}"] for seed in seeds])) for c in cases}
        diff = delta(by_method[left]["mean"], by_method[right]["mean"])
        comparisons[name] = dict(left=left, right=right, differences=diff, cost_percent_of_right=100*diff["cost"]/by_method[right]["mean"]["cost"],
            compared_pairs=len(pairs), better_pairs=sum(v["differences"]["cost"] < 0 for v in pairs),
            better_training_seed_means=sum(v["differences"]["cost"] < 0 for v in per_seed.values()), better_case_means=sum(v["cost"] < 0 for v in per_case.values()),
            completion_count_difference_sum=sum(v["differences"]["completed"] for v in pairs), by_seed=per_seed, by_case=per_case, pairs=pairs)
    report.update(complete=True, training=train_report, results=rows, by_method=by_method, comparison=comparisons,
        actual_new_evaluations=len(expected), old_evaluations_reused=len(old_keys | real_keys), actual_evaluations=len(rows),
        actual_new_policy_updates=len(jobs)*20000, actual_rule_evaluations=len(cases), training_seeds=seeds,
        reference_protocols=[old_p["name"], ref_p["name"]], comparison_boundary=p["comparison_boundary"], interpretation=p["interpretation"],
        weighting="Equal seed/case weights; three actual rule runs reused across nine pairs; every fixed result retained, no population CI.")
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
