"""Read small frozen-case summaries and calculate paired results."""

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_BRANCHES = ("source_adapted", "target_adapted")
METRICS = ("cost", "wait_seconds", "ride_seconds", "completion",
           "mean_executed_hold_seconds", "positive_hold_fraction")


def protocol(p=None):
    return p if p is not None else json.loads((HERE / "fresh_confirmation_protocol.json").read_text())


def identities(p):
    return [(branch, seed, case) for branch in p["branches"]
            for seed in p["training_seeds"] for case in p["evaluation"]["cases"]] + [
        ("strong_rule", None, case) for case in p["evaluation"]["cases"]]


def key(branch, seed, case):
    return f"{branch}/{seed}/{case}" if seed is not None else f"{branch}/{case}"


def compact_remote(p=None):
    """Server-side reader: no scheduler imports, replay, checkpoint or simulation reads."""
    p = protocol(p)
    rows = {}
    for branch, seed, case in identities(p):
        directory = HERE / p["asset_folder"] / branch
        if seed is not None:
            directory = directory / f"seed{seed}"
        path = directory / f"case{case['case']:02d}" / "summary.json"
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        passengers = row["passengers"]
        compact = {name: row[name] for name in (
            "complete", "method", "training_seed", "case", "demand_seed", "sumo_seed",
            "horizon_seconds", "simulator_version", "line_ids", "checkpoint", "actor_action_space",
            "protocol", "decision_events", "positive_holds", "mean_requested_hold_seconds",
            "execution_layer", "completion_guard")}
        compact.update(cost=row["generalized_seconds_per_departed_passenger"],
            wait_seconds=passengers["accrued_wait_seconds_per_departed_passenger"],
            ride_seconds=passengers["accrued_in_vehicle_seconds_per_departed_passenger"],
            completion=passengers["completion_rate"], departed=passengers["departed_passengers"],
            completed=passengers["completed_passengers"],
            ledger_balance_error=passengers["active_state_count_balance_error"],
            mean_executed_hold_seconds=row["mean_hold_seconds"])
        rows[key(branch, seed, case["case"])] = compact
    return rows


def metrics(row):
    return {name: row[name] for name in METRICS if name != "positive_hold_fraction"} | {
        "positive_hold_fraction": row["positive_holds"] / row["decision_events"]}


def mean(rows):
    return {name: sum(row[name] for row in rows) / len(rows) for name in METRICS}


def difference(left, right):
    return {name: left[name] - right[name] for name in METRICS}


def summarize(rows, p=None):
    """Calculate effects only after every frozen branch, seed and fresh case is complete."""
    p = protocol(p)
    expected = identities(p)
    expected_keys = {key(branch, seed, case["case"]) for branch, seed, case in expected}
    if set(rows) - expected_keys:
        raise ValueError("Results include a branch, seed or case outside the fresh protocol")
    report = dict(protocol=p["name"], complete=False, ready=len(rows), expected=len(expected), results=rows)
    if set(rows) != expected_keys or not all(row["complete"] for row in rows.values()):
        return report
    remote_root = json.loads((HERE / "runtime.json").read_text())["remote_root"]
    version = rows[key("strong_rule", None, p["evaluation"]["cases"][0]["case"])]["simulator_version"]
    for branch, seed, case in expected:
        row = rows[key(branch, seed, case["case"])]
        rule = rows[key("strong_rule", None, case["case"])]
        if seed is None:
            method, checkpoint, action_space = "forward_proportional", None, "proposal"
        else:
            spec = p["branches"][branch]
            method = spec["method"]
            checkpoint = remote_root + f"/{spec['artifact_folder']}/seed{seed}/policy.pt"
            action_space = "executed"
        expected_fields = dict(protocol=p["name"], method=method, checkpoint=checkpoint,
            actor_action_space=action_space, training_seed=seed, **case,
            horizon_seconds=p["evaluation"]["episode_seconds"], simulator_version=version,
            execution_layer="early60", completion_guard="quarter-load-sixty")
        for name, value in expected_fields.items():
            if row[name] != value:
                raise ValueError(f"Fresh result {branch}/{seed}/{case['case']} changed {name}")
        if row["line_ids"] != rule["line_ids"] or row["departed"] != rule["departed"] or row["ledger_balance_error"]:
            raise ValueError(f"Fresh paired routes, departed population or ledger differ: {branch}/{seed}/{case['case']}")
        if not math.isclose(row["cost"], 2 * row["wait_seconds"] + row["ride_seconds"], rel_tol=1e-12, abs_tol=1e-8):
            raise ValueError(f"Fresh cost does not match its passenger-time components: {branch}/{seed}/{case['case']}")
    values = {name: metrics(row) for name, row in rows.items()}
    by_method = {}
    for branch in (*p["branches"], "strong_rule"):
        seeds = p["training_seeds"] if branch in p["branches"] else [None]
        selected = [values[key(branch, seed, case["case"])] for seed in seeds for case in p["evaluation"]["cases"]]
        by_method[branch] = dict(mean=mean(selected), evaluations=len(selected),
            by_case={str(case["case"]): mean([values[key(branch, seed, case["case"])] for seed in seeds])
                     for case in p["evaluation"]["cases"]})
        if branch in p["branches"]:
            by_method[branch]["by_seed"] = {str(seed): mean([
                values[key(branch, seed, case["case"])] for case in p["evaluation"]["cases"]])
                for seed in seeds}

    def comparison(left, right):
        pairs = []
        for seed in p["training_seeds"]:
            for case in p["evaluation"]["cases"]:
                a = values[key(left, seed, case["case"])]
                b = values[key(right, None if right == "strong_rule" else seed, case["case"])]
                pairs.append(dict(training_seed=seed, case=case["case"],
                    left_cost=a["cost"], right_cost=b["cost"], differences=difference(a, b)))
        a, b = by_method[left]["mean"], by_method[right]["mean"]
        per_seed = {}
        for seed in p["training_seeds"]:
            x = by_method[left]["by_seed"][str(seed)]
            y = by_method[right]["mean"] if right == "strong_rule" else by_method[right]["by_seed"][str(seed)]
            per_seed[str(seed)] = dict(left=x, right=y, differences=difference(x, y),
                better_pairs=sum(pair["differences"]["cost"] < 0 for pair in pairs if pair["training_seed"] == seed))
        return dict(left=left, right=right, left_mean=a, right_mean=b,
            differences=difference(a, b), cost_percent_of_right=100 * (a["cost"] - b["cost"]) / b["cost"],
            compared_pairs=len(pairs), better_pairs=sum(pair["differences"]["cost"] < 0 for pair in pairs),
            better_training_seed_means=sum(row["differences"]["cost"] < 0 for row in per_seed.values()),
            by_seed=per_seed, by_case={str(case["case"]): difference(
                by_method[left]["by_case"][str(case["case"])], by_method[right]["by_case"][str(case["case"])])
                for case in p["evaluation"]["cases"]}, pairs=pairs)

    pairs = p.get("paired_comparisons", {
        "source_vs_rule": ["source_adapted", "strong_rule"],
        "source_vs_local": ["source_adapted", "target_adapted"],
        "local_vs_rule": ["target_adapted", "strong_rule"]})
    report.update(complete=True, by_method=by_method,
        comparison={name:comparison(*branches) for name,branches in pairs.items()},
        actual_evaluations=len(expected), generated_demand_cases=len(p["evaluation"]["cases"]),
        rule_evaluations=len(p["evaluation"]["cases"]), training_seeds=len(p["training_seeds"]),
        new_training_updates=p.get("new_training_updates_per_policy", 0),
        interpretation=p.get("reporting_interpretation", "Fresh three-case confirmation only; each rule run is reused across the three model seeds. Equal case/seed weights, all fixed models retained, no old-case pooling or population confidence interval."))
    return report
