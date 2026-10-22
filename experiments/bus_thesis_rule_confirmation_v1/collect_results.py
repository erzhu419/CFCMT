"""Pair 600 resident v13 test records with ten new rule days, on the server."""

import argparse
import json
import math
from statistics import mean, stdev

from rule_common import HERE, P, case_spec, result_path, write


def validate_day(row, case, *, rule=False, scene=None, seed=None, smoke=False):
    expected = case_spec(case)
    assert row["horizon_seconds"] == (1200 if smoke else P["evaluation_horizon_seconds"])
    assert row["sumo_version"][1] == P["sumo_version"]
    assert row["deployment_hold_scale"] == P["deployment_hold_scale"]
    assert row["split"] == "test"
    for key in ("demand_scale", "demand_seed", "sumo_seed"):
        assert row[key] == expected[key]
    if rule:
        assert row["comparison_protocol"] == P["name"] and row["case"] == case
        assert row["method"] == P["rule"] and row["execution_layer"] == P["execution_layer"]
        assert row["completion_guard"] == P["completion_guard"] and row["smoke"] == smoke
        assert row["rule_gain"] == P["rule_gain"] and row["rule_max_hold_seconds"] == P["rule_max_hold_seconds"]
    else:
        assert (row["arm"], row["scenario"], row["training_seed"], row["case"]) == (P["primary_arm"], scene, seed, case+10)
        assert "execution_actions" in row
    assert row["passengers"]["active_state_count_balance_error"] == 0
    assert math.isfinite(row["generalized_seconds_per_departed_passenger"])


def statistics(values):
    average, sd = mean(values), stdev(values)
    radius = P["paired_ci"]["critical_value"] * sd / math.sqrt(len(values))
    return dict(n=len(values), mean=average, sd=sd, minimum=min(values), maximum=max(values),
        ci95=[average-radius, average+radius])


def compare(policies, rules):
    cells, seeds = [], []
    by_seed = {seed: [] for seed in P["training_seeds"]}
    for scene in P["scenarios"]:
        for scale_index, scale in enumerate(P["evaluation_demand_scales"]):
            cases = range(5*scale_index, 5*scale_index+5)
            rows = []
            for seed in P["training_seeds"]:
                cost, completion = [], []
                for case in cases:
                    row, control = policies[scene, seed, case], rules[case]
                    assert row["passengers"]["departed_passengers"] == control["passengers"]["departed_passengers"]
                    cost.append(row["generalized_seconds_per_departed_passenger"] - control["generalized_seconds_per_departed_passenger"])
                    completion.append(100*(row["passengers"]["completion_rate"] - control["passengers"]["completion_rate"]))
                item = dict(seed=seed, cost_delta_seconds=mean(cost), completion_delta_pp=mean(completion))
                by_seed[seed].append(item)
                rows.append(item)
            cells.append(dict(scenario=scene, demand_scale=scale,
                cost_delta_seconds=statistics([r["cost_delta_seconds"] for r in rows]),
                completion_delta_pp=statistics([r["completion_delta_pp"] for r in rows]),
                better_seeds=sum(r["cost_delta_seconds"] < 0 for r in rows), seed_deltas=rows))
    for seed, rows in by_seed.items():
        seeds.append(dict(seed=seed, cost_delta_seconds=mean(r["cost_delta_seconds"] for r in rows),
            completion_delta_pp=mean(r["completion_delta_pp"] for r in rows)))
    macro = statistics([r["cost_delta_seconds"] for r in seeds])
    conditions = dict(mean_cost_improvement=macro["mean"] <= -P["minimum_mean_cost_improvement_seconds"],
        paired_ci_upper_below_zero=macro["ci95"][1] < 0,
        no_cell_mean_cost_regression=all(r["cost_delta_seconds"]["mean"] <= 0 for r in cells),
        cell_completion_preserved=all(r["completion_delta_pp"]["mean"] >= -P["maximum_cell_completion_decrease_percentage_points"] for r in cells))
    return dict(primary_arm=P["primary_arm"], reference=P["rule"], macro_cost_delta_seconds=macro,
        macro_completion_delta_pp=statistics([r["completion_delta_pp"] for r in seeds]),
        better_seeds=sum(r["cost_delta_seconds"] < 0 for r in seeds),
        seed_deltas=seeds, cells=cells, conditions=conditions,
        decision="PASS" if all(conditions.values()) else "STOP")


def load_reference(root):
    reference = root / P["reference_outputs"]
    missing, policies, descriptive = [], {}, []
    analysis_path = reference / "completed_analysis.json"
    if analysis_path.exists():
        previous = json.loads(analysis_path.read_text())
        assert previous["complete"] and previous["initialization_count"] == 20
        assert previous["training_count"] == 240 and previous["evaluation_count"] == 4800
        assert previous["protocol"]["source_event_budget"] == 25000
        assert previous["protocol"]["offline_updates"] == 50000
        assert previous["protocol"]["continuation_updates"] == 20000
        descriptive = [r for r in previous["comparisons"]
            if (r["arm"] == P["primary_arm"] and r["reference"] in
                ("h2o_matched", "fixed0300", "shared_gated", "daganzo_duet"))
            or (r["arm"] == "shared_gated" and r["reference"] == P["primary_arm"])]
    else:
        missing.append(str(analysis_path))
    for scene in P["scenarios"]:
        for seed in P["training_seeds"]:
            folder = reference / "runs" / scene / f"{P['primary_arm']}_seed{seed}" / "results"
            for case in range(10):
                path = folder / f"eval_{case+10:02d}.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                row = json.loads(path.read_text())
                validate_day(row, case, scene=scene, seed=seed)
                policies[scene, seed, case] = row
    return policies, descriptive, missing


def inventory(root=HERE):
    policies, _, missing = load_reference(root)
    return dict(experiment=P["name"], analysis_location="server", complete=not missing,
        primary_evaluation_ready=len(policies), primary_evaluation_expected=600,
        missing_count=len(missing), missing_examples=missing[:5], decision="REFERENCE_ONLY")


def summarize(smoke=False, root=HERE):
    cases = [0, 5] if smoke else range(10)
    missing, rules, policies = [], {}, {}
    for case in cases:
        path = result_path(case, smoke, root)
        if not path.exists():
            missing.append(str(path))
            continue
        row = json.loads(path.read_text())
        validate_day(row, case, rule=True, smoke=smoke)
        rules[case] = row
    report = dict(experiment=P["name"], smoke=smoke, analysis_location="server",
        rule_ready=len(rules), rule_expected=len(cases), interpretation=P["interpretation"],
        gate=P["gate"], paired_ci=P["paired_ci"])
    report["runtime_validation"] = dict(decision="SMOKE_ONLY" if smoke else "PENDING")
    if not smoke:
        path = root / "runtime_validation/results/validation.json"
        if path.exists():
            validation = json.loads(path.read_text())
            assert validation["experiment"] == P["name"] and validation["case"] == 0 and validation["complete"]
            assert validation["valid"] == all(validation["checks"].values())
            report["runtime_validation"] = validation
            if not validation["valid"]:
                return dict(report, complete=True, missing_count=len(missing), missing_examples=missing[:5], decision="INVALID_RUNTIME")
        else:
            missing.append(str(path))
    if missing:
        return dict(report, complete=False, missing_count=len(missing), missing_examples=missing[:5], decision="PENDING")
    if not smoke:
        policies, descriptive, missing = load_reference(root)
        report["existing_descriptive_comparisons"] = descriptive
        report["primary_executor_provenance"] = dict(
            reference_outputs=P["reference_outputs"], original_implementation="mcwm_bus/evaluate_control.py:98",
            variant=P["completion_guard"], current_rule_executor=P["execution_code_snapshot"]+"/attribution_v1/run.py",
            execution_record="Frozen v13 test metadata retains execution_actions on every primary evaluation")
        report.update(primary_evaluation_ready=len(policies), primary_evaluation_expected=600)
    report.update(complete=not missing, missing_count=len(missing), missing_examples=missing[:5],
        decision="SMOKE_ONLY" if smoke and not missing else "PENDING")
    if not missing and not smoke:
        report["primary"] = compare(policies, rules)
        report["decision"] = report["primary"]["decision"]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--inventory", action="store_true")
    args = parser.parse_args()
    report = inventory() if args.inventory else summarize(args.smoke)
    write(HERE / ("reference_inventory.json" if args.inventory else
        ("smoke_" if args.smoke else "") + "analysis_summary.json"), report)
    fields = ("complete", "primary_evaluation_ready", "primary_evaluation_expected", "decision") if args.inventory else ("complete", "rule_ready", "rule_expected", "decision")
    print(json.dumps({key: report[key] for key in fields}))
