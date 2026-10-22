"""Report the fixed terminal-label repair using local small JSON summaries only."""

import argparse
import json
import math
from pathlib import Path

from fresh_confirmation_summary import difference, mean, metrics

HERE = Path(__file__).resolve().parent
PREFIX = "native_terminal_adaptation"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(name):
    return json.loads((HERE / name).read_text())


def compact(row):
    passengers = row["passengers"]
    require(row["generalized_passenger_seconds"] ==
        2*passengers["accrued_wait_seconds_total"]+passengers["accrued_in_vehicle_seconds_total"],
        "New evaluation passenger-time total does not match its cost")
    require(math.isclose(row["generalized_seconds_per_departed_passenger"],
        row["generalized_passenger_seconds"]/passengers["departed_passengers"], rel_tol=1e-12),
        "New evaluation cost uses a different passenger denominator")
    result = {name: row[name] for name in ("complete", "method", "training_seed", "case", "demand_seed",
        "sumo_seed", "horizon_seconds", "simulator_version", "line_ids", "checkpoint", "actor_action_space",
        "protocol", "decision_events", "positive_holds", "execution_layer", "completion_guard")}
    result.update(cost=row["generalized_seconds_per_departed_passenger"],
        wait_seconds=passengers["accrued_wait_seconds_per_departed_passenger"],
        ride_seconds=passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
        completion=passengers["completion_rate"], departed=passengers["departed_passengers"],
        completed=passengers["completed_passengers"], ledger_balance_error=passengers["active_state_count_balance_error"],
        mean_executed_hold_seconds=row["mean_hold_seconds"])
    return result


def summarize(protocol, collected):
    seeds, cases = protocol["training_seeds"], protocol["evaluation"]["cases"]
    expected = {f"seed{seed}/evaluation/case{case['case']:02d}/summary.json"
        for seed in seeds for case in cases}
    training, evaluation = collected.get("training", {}), collected.get("evaluation", {})
    require(collected.get("protocol", protocol["name"]) == protocol["name"], "Wrong collected repair protocol")
    require(not (set(evaluation)-expected), "New evaluation includes a seed or case outside the protocol")
    require(not (set(training)-set(map(str, seeds))), "New training includes a seed outside the protocol")
    ready_training = sum(training.get(str(seed), {}).get("train_summary.json", {}).get("complete", False) for seed in seeds)
    ready = sum(evaluation.get(name, {}).get("complete", False) for name in expected)
    report = dict(protocol=protocol["name"], complete=False, training_ready=ready_training,
        expected_training=len(seeds), ready=ready, expected=len(expected))
    if ready_training != len(seeds) or ready != len(expected):
        return report
    runtime, native = read("runtime.json"), read("protocol.json")
    root = Path(runtime["remote_root"])
    reference_protocol = read(protocol["references"]["protocol"])
    references = read(protocol["references"]["summary"])
    original_training = read("source_dense_adaptation_summary.json")["results"]
    expected_config = read("training_protocol.json")["training"]["trainer_config"]
    expected_config.update(offline_objective="iql", online_fraction=0., use_ratio_correction=False)
    require(reference_protocol["name"] != protocol["name"] and references["protocol"] == reference_protocol["name"],
        "Old references must retain their original protocol")
    require(reference_protocol["evaluation"]["cases"] == cases and protocol["evaluation"]["episode_seconds"] ==
        reference_protocol["evaluation"]["episode_seconds"] == 24000, "Reference cases or horizon changed")
    require(protocol["source_initialization_updates"] == 50000 and protocol["adaptation_updates"] == 5000,
        "Frozen source or adaptation update budget changed")
    train_reports = {}
    for seed in seeds:
        entry = training[str(seed)]
        trained, progress = entry["train_summary.json"], entry["progress.json"]
        expected_training = dict(complete=True, protocol=protocol["name"], seed=seed, admission=False,
            initialization_updates=50000, adaptation_updates=5000, historical_policy_updates=55000,
            target_fit_rows=protocol["target_fit_rows"], target_union_rows=protocol["target_union_rows"],
            target_pool=str(root / protocol["target_pool"]),
            source_initializer=str(root / "source_dense_ablation" / f"seed{seed}" / "initializer.pt"),
            config=expected_config, optimizer_transfer=False, source_line_onehot_transfer=False,
            source_arrival_rate_transfer=False, world_model_updates=0, continuation_updates=0,
            target_online_training_events=0, actor_export_equal=True)
        require(all(trained[key] == value for key, value in expected_training.items()),
            f"Repaired training seed{seed} changed its frozen adaptation settings")
        require(trained["config"] == original_training[str(seed)]["training"]["config"],
            f"Repaired training seed{seed} config differs from original source adaptation")
        require(progress["phase"] == "corrected_terminal_target_adaptation" and
            progress["phase_updates"] == 5000 and progress["completed_updates"] == 55000,
            f"Repaired training seed{seed} did not finish the fixed updates")
        before, after = trained["fit_terminal_before"], trained["fit_terminal_after"]
        require(before["rows"] == after["rows"] == 6 and before["terminal_target_exact"] and
            after["terminal_target_exact"] and before["targets"] == after["targets"],
            f"Repaired training seed{seed} changed the six fit terminal targets")
        train_reports[str(seed)] = dict(adaptation_updates=5000, historical_policy_updates=55000,
            fit_terminal_before_mae=before["mae"], fit_terminal_after_mae=after["mae"],
            fit_terminal_rows=6, terminal_targets_exact=True, settings_match_original=True,
            target_pool=trained["target_pool"], source_initializer=trained["source_initializer"],
            final_training_metrics=progress["metrics"])
    rows = {}
    for case in cases:
        ci = case["case"]
        rows[f"strong_rule/{ci}"] = references["results"][f"strong_rule/{ci}"]
        for seed in seeds:
            path = f"seed{seed}/evaluation/case{ci:02d}/summary.json"
            raw = evaluation[path]
            require(raw["evaluation_protocol"] == str(root / (PREFIX+"_protocol.json")) and
                raw["asset_manifest"] == str(Path(runtime["assets_root"]) / protocol["asset_folder"] /
                    f"case{ci:02d}/asset_manifest.json"), f"New evaluation {path} changed protocol or assets")
            rows[f"repaired/{seed}/{ci}"] = compact(raw)
            rows[f"adapted/{seed}/{ci}"] = references["results"][f"adapted/{seed}/{ci}"]
    for name, row in rows.items():
        branch = name.split("/")[0]
        seed = None if branch == "strong_rule" else int(name.split("/")[1])
        case = next(case for case in cases if case["case"] == int(name.split("/")[-1]))
        rule = rows[f"strong_rule/{case['case']}"]
        folder = protocol["artifact_folder"] if branch == "repaired" else reference_protocol["branches"]["adapted"]["artifact_folder"]
        expected_fields = dict(complete=True, training_seed=seed, **case, horizon_seconds=24000,
            method="forward_proportional" if seed is None else "cross_city",
            checkpoint=None if seed is None else str(root / folder / f"seed{seed}" / "policy.pt"),
            actor_action_space="proposal" if seed is None else "executed",
            protocol=protocol["name"] if branch == "repaired" else reference_protocol["name"],
            execution_layer="early60", completion_guard="quarter-load-sixty",
            simulator_version=rule["simulator_version"], line_ids=rule["line_ids"], departed=rule["departed"])
        require(all(row[key] == value for key, value in expected_fields.items()), f"Paired identity changed: {name}")
        require("SUMO "+native["smoke"]["sumo_version"] in row["simulator_version"][1] and
            row["ledger_balance_error"] == 0, f"Simulator version or passenger ledger changed: {name}")
        require(math.isclose(row["cost"], 2*row["wait_seconds"]+row["ride_seconds"], rel_tol=1e-12, abs_tol=1e-8)
            and math.isclose(row["completion"], row["completed"]/row["departed"], rel_tol=1e-12),
            f"Passenger cost components or completion denominator changed: {name}")
    values = {key: metrics(row) for key, row in rows.items()}
    by_method = {}
    for branch in ("repaired", "adapted", "strong_rule"):
        chosen_seeds = [None] if branch == "strong_rule" else seeds
        keys = [f"{branch}/{case['case']}" if seed is None else f"{branch}/{seed}/{case['case']}"
            for seed in chosen_seeds for case in cases]
        by_method[branch] = dict(mean=mean([values[key] for key in keys]), actual_evaluations=len(keys))
        if branch != "strong_rule":
            by_method[branch]["by_seed"] = {str(seed): mean([values[f"{branch}/{seed}/{case['case']}"] for case in cases])
                for seed in seeds}
    comparison = {}
    for reference in ("adapted", "strong_rule"):
        pairs = [dict(training_seed=seed, case=case["case"], differences=difference(
            values[f"repaired/{seed}/{case['case']}"], values[f"{reference}/{case['case']}" if reference ==
                "strong_rule" else f"adapted/{seed}/{case['case']}"])) for seed in seeds for case in cases]
        per_seed = {str(seed): dict(differences=difference(by_method["repaired"]["by_seed"][str(seed)],
            by_method[reference]["mean"] if reference == "strong_rule" else by_method[reference]["by_seed"][str(seed)]),
            better_pairs=sum(pair["differences"]["cost"] < 0 for pair in pairs if pair["training_seed"] == seed)) for seed in seeds}
        delta = difference(by_method["repaired"]["mean"], by_method[reference]["mean"])
        comparison[f"repaired_vs_{reference}"] = dict(differences=delta,
            cost_percent_of_reference=100*delta["cost"]/by_method[reference]["mean"]["cost"],
            compared_pairs=len(pairs), better_pairs=sum(pair["differences"]["cost"] < 0 for pair in pairs),
            better_training_seed_means=sum(row["differences"]["cost"] < 0 for row in per_seed.values()),
            by_seed=per_seed, pairs=pairs)
    report.update(complete=True, results=rows, training=train_reports, by_method=by_method, comparison=comparison,
        actual_new_evaluations=len(expected), old_evaluations_reused=len(expected)+len(cases),
        actual_rule_evaluations=len(cases), training_seeds=seeds, generated_demand_cases=len(cases),
        reference_protocol=reference_protocol["name"], interpretation=protocol["interpretation"],
        weighting="Equal case/seed weights; three actual rule runs reused across nine pairs. No population CI or seed filtering.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / (PREFIX+"_protocol.json"))
    parser.add_argument("--summary", type=Path, default=HERE / (PREFIX+"_summary.json"))
    parser.add_argument("--output", type=Path, default=HERE / (PREFIX+"_reporting.json"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    collected = json.loads(args.summary.read_text()) if args.summary.exists() else {}
    report = summarize(protocol, collected)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({key: report[key] for key in ("complete", "training_ready", "ready", "expected")}))


if __name__ == "__main__":
    main()
