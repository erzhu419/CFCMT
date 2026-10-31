"""Report frozen75k-gate/55k-amount evaluations against unchanged small references."""

import argparse
import json
import math
from pathlib import Path

from fresh_confirmation_summary import metrics, summarize as validate_reference
from native_terminal_adaptation_report import compact
from native_terminal_target_only_report import average, delta, read, require

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_actor_splice"


def summarize(p, collected):
    seeds, cases, branch = p["training_seeds"], p["evaluation"]["cases"], p["variant"]
    expected = {f"seed{seed}/case{case['case']:02d}" for seed in seeds for case in cases}
    evaluation = collected.get("results", {})
    require(collected.get("protocol", p["name"]) == p["name"] and not (set(evaluation)-expected),
        "Collected splice protocol or seed/case roster changed")
    ready = sum(evaluation.get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=p["name"], complete=False, ready=ready, expected=len(expected))
    if ready != len(expected):
        return report
    ref_p, ref = read(p["reference_protocol"]), read(p["reference_summary"])
    reference = validate_reference(ref["results"], ref_p)
    require(collected["complete"] and ref["complete"] and reference["complete"] and
        ref["protocol"] == reference["protocol"] == ref_p["name"] != p["name"] and
        ref_p["training_seeds"] == seeds and ref_p["evaluation"]["cases"] == cases and
        p["asset_folder"] == ref_p["asset_folder"] and p["evaluation"]["episode_seconds"] ==
        ref_p["evaluation"]["episode_seconds"] == 24000, "Original reference identity, roster or horizon changed")
    require(branch == "gate75_amount55" and p["new_training_updates"] == p["world_model_refit_updates"] == 0 and
        p["new_evaluations"] == len(expected) == 9 and p["reference_evaluations_reused"] ==
        len(ref["results"]) == 21 and p["actual_evaluations_planned"] == 30, "Fixed splice budget changed")
    runtime, native = read("runtime.json"), read("protocol.json")
    root, admitted = Path(runtime["remote_root"]), read(PREFIX+"_admission_summary.json")
    require(admitted["complete"] and admitted["protocol"] == p["name"] and admitted["variant"] == branch and
        admitted["training_seeds"] == seeds and set(admitted["results"]) == set(map(str, seeds)) and admitted["new_training_updates"] ==
        admitted["world_model_refit_updates"] == admitted["new_simulations"] == 0,
        "All fixed actor components must retain their zero-training admission")
    rows = dict(ref["results"])
    semantics = ("75k native hold probability>=0.5 selects55k native deterministic positive action; otherwise exact-1. "
        "Existing evaluation execution projection once; no hold division.")
    for seed in seeds:
        admission = admitted["results"][str(seed)]
        paths = {"gate75": str(root / p["gate_actor"].format(seed=seed)),
            "amount55": str(root / p["positive_amount_actor"].format(seed=seed))}
        checks = "native75_deterministic_reconstruction_exact native55_positive_action_exact native75_gate_threshold_half closed_gate_action_exact_minus_one requested_composition_exact single_canonical_projection_exact separate_native_encoders".split()
        require(admission["complete"] and admission["protocol"] == p["name"] and admission["seed"] == seed and
            admission["component_checkpoints"] == paths and admission["rows"] == 12288 and
            all(admission["validations"][key] is True for key in checks), f"Splice seed{seed} admission or native composition changed")
        for role, updates, protocol in (("gate75", 75000, "cfcmt-bus-native-terminal-actor-unmasked-v1"),
                ("amount55", 55000, "cfcmt-bus-native-terminal-adaptation-v1")):
            meta = admission["component_metadata"][role]
            require(meta["seed"] == seed and meta["protocol"] == protocol and meta["historical_policy_updates"] == updates
                and meta["actor_type"].endswith(".HurdleTransitActorV7") and meta["policy_artifact_schema"],
                f"Splice seed{seed} {role} native header changed")
        for case in cases:
            name = f"seed{seed}/case{case['case']:02d}"
            raw, rule = evaluation[name], rows[f"strong_rule/{case['case']}"]
            fields = dict(complete=True, method=branch, variant=branch, training_seed=seed, **case,
                horizon_seconds=24000, checkpoint=None, actor_action_space="executed", protocol=p["name"],
                execution_layer="early60", completion_guard="quarter-load-sixty", simulator_version=rule["simulator_version"],
                line_ids=rule["line_ids"], component_checkpoints=paths, component_metadata=admission["component_metadata"],
                new_training_updates=0, world_model_refit_updates=0, admission_protocol=p["name"], admission_fit_rows=12288,
                admission_all_training_seeds_complete=True,
                intervention=p["intervention"], actor_composition="native75_gate_native55_positive_action_each_with_own_encoder",
                requested_action_semantics=semantics, evaluation_protocol=str(root / (PREFIX+"_protocol.json")),
                asset_manifest=str(Path(runtime["assets_root"]) / p["asset_folder"] / f"case{case['case']:02d}/asset_manifest.json"))
            require(all(raw[key] == value for key, value in fields.items()), f"Splice evaluation identity changed: {name}")
            row = compact(raw)
            passengers = raw["passengers"]
            require(row["departed"] == rule["departed"] and row["ledger_balance_error"] == 0 and
                "SUMO "+native["smoke"]["sumo_version"] in row["simulator_version"][1] and
                all(math.isclose(row[key], passengers[total]/row["departed"], rel_tol=1e-12, abs_tol=1e-8)
                    for key, total in (("wait_seconds", "accrued_wait_seconds_total"),
                    ("ride_seconds", "accrued_in_vehicle_seconds_total"))) and
                math.isclose(row["completion"], row["completed"]/row["departed"], rel_tol=1e-12),
                f"Splice paired population, passenger ledger or denominator changed: {name}")
            row.update({key: raw[key] for key in ("variant", "component_checkpoints", "component_metadata",
                "admission_protocol", "new_training_updates", "world_model_refit_updates", "actor_composition",
                "requested_action_semantics", "mean_requested_hold_seconds")})
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
        per_case = {str(c["case"]): difference(by_method[branch]["by_case"][str(c["case"])],
            by_method[right]["by_case"][str(c["case"])]) for c in cases}
        diff = difference(by_method[branch]["mean"], by_method[right]["mean"])
        comparison[f"splice_vs_{right}"] = dict(left=branch, right=right, differences=diff,
            cost_percent_of_right=100*diff["cost"]/by_method[right]["mean"]["cost"], compared_pairs=len(pairs),
            better_pairs=sum(v["differences"]["cost"] < 0 for v in pairs), better_training_seed_means=sum(
                v["differences"]["cost"] < 0 for v in per_seed.values()), better_case_means=sum(v["cost"] < 0 for v in per_case.values()),
            completion_count_difference_sum=sum(v["differences"]["completed"] for v in pairs), by_seed=per_seed, by_case=per_case, pairs=pairs)
    report.update(complete=True, results=rows, by_method=by_method, comparison=comparison, actual_new_evaluations=len(expected),
        old_evaluations_reused=len(ref["results"]), actual_evaluations=len(rows), actual_rule_evaluations=len(cases),
        new_training_updates=0, world_model_refit_updates=0, training_seeds=seeds, reference_protocol=ref_p["name"],
        interpretation=p["interpretation"], weighting="Equal seed/case weights; three actual rule runs reused across nine pairs; all fixed results retained, no population CI.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / (PREFIX+"_protocol.json"))
    parser.add_argument("--summary", type=Path, default=HERE / (PREFIX+"_summary.json"))
    parser.add_argument("--output", type=Path, default=HERE / (PREFIX+"_reporting.json"))
    args = parser.parse_args()
    report = summarize(json.loads(args.protocol.read_text()), json.loads(args.summary.read_text()) if args.summary.exists() else {})
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("complete", "ready", "expected")}))


if __name__ == "__main__":
    main()
