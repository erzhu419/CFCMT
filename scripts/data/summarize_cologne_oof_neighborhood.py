"""Describe the frozen raw-model OOF errors and their three training neighbors."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean, median

PROTOCOL = "tsc-v155e-cologne-oof-neighborhood-v1"
SEEDS = (2027, 3037, 4047)
QUERY_COUNTS = {2027: 17, 3037: 17, 4047: 14}
TOLERANCE = 1e-12
CHECKS = ("original_a_effective_records_exact", "native_bank_labels_and_candidate_order_exact",
          "original_a_selection_cost_and_counts_exact", "model_feature_order_exact", "accepted_query_roster_exact")
AUDIT_CHECKS = ("fold_exclusions_exact", "neighbor_groups_unique", "neighbors_exclude_query_seed",
                "candidate_order_and_raw_labels_exact")


def gain_sign(gain):
    return "positive" if gain > TOLERANCE else "negative" if gain < -TOLERANCE else "zero"


def neighbor_pattern(gains):
    signs = {gain_sign(gain) for gain in gains}
    return "all_" + next(iter(signs)) if len(signs) == 1 else "mixed"


def distribution(values):
    return {"count": len(values), "minimum": min(values) if values else None,
            "median": median(values) if values else None, "maximum": max(values) if values else None}


def summarize_queries(rows):
    patterns = ("all_positive", "all_negative", "all_zero", "mixed")
    harmful = [row for row in rows if row["outcome"] == "harmful"]
    feature_counts = Counter(feature["feature"] for row in rows for feature in row["outside_training_range"])
    constant_counts = Counter(feature["feature"] for row in rows for feature in row["outside_training_range"]
                              if feature["training_constant"])
    return {"queries": len(rows),
            "query_outcomes": {outcome: sum(row["outcome"] == outcome for row in rows) for outcome in ("beneficial", "harmful", "tie")},
            "queries_outside_any_training_range": sum(bool(row["outside_training_range"]) for row in rows),
            "queries_violating_training_constant": sum(any(item["training_constant"] for item in row["outside_training_range"]) for row in rows),
            "range_violations_by_query_outcome": {
                outcome: {"any_range": sum(bool(row["outside_training_range"]) for row in rows if row["outcome"] == outcome),
                          "training_constant": sum(any(item["training_constant"] for item in row["outside_training_range"])
                                                   for row in rows if row["outcome"] == outcome)}
                for outcome in ("beneficial", "harmful", "tie")},
            "neighbor_mean_gain_sign": {sign: sum(gain_sign(row["neighbor_mean_raw_gain"]) == sign for row in rows)
                                        for sign in ("positive", "negative", "zero")},
            "neighbor_sign_pattern": {pattern: sum(row["neighbor_sign_pattern"] == pattern for row in rows) for pattern in patterns},
            "harmful_queries_neighbor_mean_gain_sign": {sign: sum(gain_sign(row["neighbor_mean_raw_gain"]) == sign for row in harmful)
                                                        for sign in ("positive", "negative", "zero")},
            "harmful_queries_neighbor_sign_pattern": {pattern: sum(row["neighbor_sign_pattern"] == pattern for row in harmful) for pattern in patterns},
            "nearest_distance_by_query_outcome": {
                outcome: distribution([row["nearest_distance"] for row in rows if row["outcome"] == outcome])
                for outcome in ("beneficial", "harmful", "tie")},
            "feature_range_violation_counts": dict(sorted(feature_counts.items(), key=lambda item: (-item[1], item[0]))),
            "constant_feature_violation_counts": dict(sorted(constant_counts.items(), key=lambda item: (-item[1], item[0])))}


def aggregate(protocol, result):
    bound = ("source_root", "sumocfg", "model_path", "frozen_threshold", "training_group_ids", "feature_names", "folds")
    if (protocol["protocol"] != PROTOCOL or result["protocol"] != PROTOCOL or result["status"] != "COMPLETE"
            or any(result[key] != protocol[key] for key in bound)
            or len(result["training_group_ids"]) != 100 or len(set(result["training_group_ids"])) != 100
            or len(result["feature_names"]) != 29
            or result["selected_policy"] != protocol["expected_selected_policy"]
            or result["proposal_accounting"] != protocol["expected_proposal_accounting"]
            or any(result[key] != 0 for key in ("new_model_fits", "new_labels", "new_simulations"))
            or result["threshold_retuned"] is not False):
        raise ValueError("Require the frozen A model, all100 OOF groups, original features/folds and unchanged selection")
    audit = result["audit"]
    if (any(result.get("checks", {}).get(key) is not True for key in CHECKS)
            or any(audit.get("checks", {}).get(key) is not True for key in AUDIT_CHECKS)
            or audit["feature_names"] != protocol["feature_names"]):
        raise ValueError("Original labels, candidate order, fold exclusions and query roster must be verified")
    features = protocol["feature_names"]
    group_seeds = {group: int(group.split(":")[1][4:]) for group in protocol["training_group_ids"]}
    folds = {fold["heldout_seed"]: fold for fold in audit["folds"]}
    if len(audit["folds"]) != 3 or set(folds) != set(SEEDS):
        raise ValueError("All three original heldout folds are required")
    for spec in protocol["folds"]:
        seed, count = spec["heldout_seed"], spec["fit_group_count"]
        fold = folds[seed]
        arrays = [fold[key] for key in ("feature_mean", "feature_std", "feature_min", "feature_max")]
        if (fold["fit_group_count"] != count or fold["fit_row_count"] != 7 * count
                or fold["query_count"] != QUERY_COUNTS[seed]
                or fold["training_seeds"] != [other for other in SEEDS if other != seed]
                or fold["training_reference_rows"] != 0
                or any(len(values) != len(features) or not all(math.isfinite(v) for v in values) for values in arrays)
                or fold["constant_features"] != [name for name, lo, hi in zip(features, fold["feature_min"], fold["feature_max"]) if lo == hi]):
            raise ValueError("Neighborhood scaling must use the original fit-fold non-reference rows")
    expected = {row["group_id"]: row for row in protocol["expected_queries"]}
    queries = audit["queries"]
    if (audit["query_count"] != 48 or len(queries) != 48 or len(expected) != 48
            or {row["group_id"] for row in queries} != set(expected)
            or Counter(row["seed"] for row in queries) != QUERY_COUNTS):
        raise ValueError("The diagnosis requires exactly the original48 accepted queries")
    rows = []
    for query in queries:
        row, frozen = dict(query), expected[query["group_id"]]
        outcome = {"positive": "beneficial", "negative": "harmful", "zero": "tie"}[gain_sign(row["actual_gain"])]
        if (any(row[key] != frozen[key] for key in ("group_id", "seed", "selected_index", "reference_index"))
                or any(not math.isfinite(row[key]) or abs(row[key] - frozen[key]) > TOLERANCE for key in ("predicted_gain", "actual_gain"))
                or row["outcome"] != outcome or row["selected_index"] == row["reference_index"]):
            raise ValueError("Each query must retain the frozen A action, score and raw-label outcome")
        neighbors = row["neighbors"]
        if (len(neighbors) != 3 or len({neighbor["group_id"] for neighbor in neighbors}) != 3
                or any(neighbor["group_id"] not in group_seeds
                       or neighbor["seed"] != group_seeds[neighbor["group_id"]] or neighbor["seed"] == row["seed"]
                       or neighbor["selected_index"] not in range(8) or neighbor["reference_index"] not in range(8)
                       or neighbor["selected_index"] == neighbor["reference_index"]
                       or not math.isfinite(neighbor["raw_gain"]) or not math.isfinite(neighbor["distance"])
                       or neighbor["distance"] < 0 for neighbor in neighbors)
                or [neighbor["distance"] for neighbor in neighbors] != sorted(neighbor["distance"] for neighbor in neighbors)
                or row["nearest_distance"] != neighbors[0]["distance"]):
            raise ValueError("Three distinct training groups with non-reference candidates and ordered distances are required")
        fold = folds[row["seed"]]
        violations = row["outside_training_range"]
        if len({item["feature"] for item in violations}) != len(violations):
            raise ValueError("Each violated feature must be reported once")
        for item in violations:
            index = features.index(item["feature"])
            if (item["minimum"] != fold["feature_min"][index] or item["maximum"] != fold["feature_max"][index]
                    or item["training_constant"] != (item["feature"] in fold["constant_features"])
                    or not math.isfinite(item["value"])
                    or not (item["value"] < item["minimum"] or item["value"] > item["maximum"])):
                raise ValueError("Range violations must use their own training-fold bounds and constant-feature flags")
        gains = [neighbor["raw_gain"] for neighbor in neighbors]
        row.update(neighbor_mean_raw_gain=mean(gains), neighbor_mean_gain_sign=gain_sign(mean(gains)),
                   neighbor_sign_pattern=neighbor_pattern(gains))
        rows.append(row)
    per_seed = {str(seed): summarize_queries([row for row in rows if row["seed"] == seed]) for seed in SEEDS}
    for seed, counts in per_seed.items():
        policy = result["selected_policy"]["per_seed"][seed]
        if (counts["queries"] != policy["accepted_overrides"]
                or any(counts["query_outcomes"][outcome] != policy[key] for outcome, key in
                       (("beneficial", "beneficial"), ("harmful", "harmful"), ("tie", "equal_cost")))
                or abs(sum(row["actual_gain"] for row in rows if row["seed"] == int(seed)) / policy["groups"]
                       + policy["mean_cost_minus_phase_pressure"]) > TOLERANCE):
            raise ValueError("Accepted outcomes and raw gains must reproduce the original100-group policy denominator")
    return {"protocol": PROTOCOL, "status": "COMPLETE", **{key: result[key] for key in bound},
            "selected_policy": result["selected_policy"], "proposal_accounting": result["proposal_accounting"],
            "label_units": protocol["label_units"], "distance_definition": protocol["distance"],
            "neighbor_rule": protocol["neighbor_rule"], "overall": summarize_queries(rows), "per_seed": per_seed,
            "queries": rows, "fold_diagnostics": audit["folds"], "checks": result["checks"], "audit_checks": audit["checks"],
            "accounting": {key: result[key] for key in ("new_model_fits", "new_labels", "new_simulations", "threshold_retuned", "elapsed_sec")},
            "limitations": protocol["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "result", "launch-record", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(json.loads(args.protocol.read_text()), json.loads(args.result.read_text()))
    launch = json.loads(args.launch_record.read_text())
    submitted = json.loads(launch["scheduler_stdout"])["submitted"]
    task = next(row["id"] for row in submitted if row["signature"] == "CFCMT/v155e/oof-neighborhood/v1")
    report["accounting"].update(task_id=task, retrieved_result_bytes=args.result.stat().st_size)
    report["refs"] = {"protocol": str(args.protocol), "result": str(args.result), "launch_record": str(args.launch_record)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "overall": report["overall"], "accounting": report["accounting"]}))


if __name__ == "__main__":
    main()
