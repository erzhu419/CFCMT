"""Frozen OOF-roster and descriptive neighbor accounting for V155E."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_oof_neighborhood as summary

RESULTS = Path(__file__).resolve().parents[1] / "cf_h2o/results"


def fixture():
    protocol = json.loads((RESULTS / "cluster/tsc_v155e_cologne_oof_neighborhood_20260910/protocol_v1.json").read_text())
    result = {key: deepcopy(protocol[key]) for key in
              ("protocol", "source_root", "sumocfg", "model_path", "frozen_threshold", "training_group_ids", "feature_names", "folds")}
    result.update(status="COMPLETE", selected_policy=deepcopy(protocol["expected_selected_policy"]),
                  proposal_accounting=deepcopy(protocol["expected_proposal_accounting"]),
                  checks=dict.fromkeys(summary.CHECKS, True), new_model_fits=0, new_labels=0,
                  new_simulations=0, threshold_retuned=False, elapsed_sec=1.)
    folds = []
    for spec in protocol["folds"]:
        folds.append({"heldout_seed": spec["heldout_seed"], "fit_group_count": spec["fit_group_count"],
                      "fit_row_count": 7 * spec["fit_group_count"], "query_count": summary.QUERY_COUNTS[spec["heldout_seed"]],
                      "feature_mean": [0.] + [.5] * 28, "feature_std": [0.] + [1.] * 28,
                      "feature_min": [0.] * 29, "feature_max": [0.] + [1.] * 28,
                      "constant_features": [protocol["feature_names"][0]],
                      "training_seeds": [seed for seed in summary.SEEDS if seed != spec["heldout_seed"]],
                      "training_reference_rows": 0})
    queries = []
    for frozen in protocol["expected_queries"]:
        row = dict(frozen)
        row["outcome"] = {"positive": "beneficial", "negative": "harmful", "zero": "tie"}[summary.gain_sign(row["actual_gain"])]
        candidates = [group for group in protocol["training_group_ids"] if int(group.split(":")[1][4:]) != row["seed"]]
        row.update(nearest_distance=1., outside_training_range=[], neighbors=[
            {"group_id": group, "seed": int(group.split(":")[1][4:]), "selected_index": 1, "reference_index": 0,
             "raw_gain": .01, "distance": float(index + 1)} for index, group in enumerate(candidates[:3])])
        queries.append(row)
    result["audit"] = {"queries": queries, "query_count": 48, "folds": folds,
                       "feature_names": protocol["feature_names"], "checks": dict.fromkeys(summary.AUDIT_CHECKS, True)}
    return protocol, result


def test_complete_roster_preserves_original_policy_and_counts():
    protocol, result = fixture()
    report = summary.aggregate(protocol, result)
    assert report["status"] == "COMPLETE" and len(report["queries"]) == 48
    assert report["overall"]["query_outcomes"] == {"beneficial": 31, "harmful": 15, "tie": 2}
    assert [report["per_seed"][str(seed)]["queries"] for seed in summary.SEEDS] == [17, 17, 14]
    assert report["selected_policy"]["groups"] == 100
    assert report["selected_policy"]["mean_cost"] == protocol["expected_selected_policy"]["mean_cost"]
    assert report["overall"]["harmful_queries_neighbor_sign_pattern"]["all_positive"] == 15
    assert report["accounting"]["new_model_fits"] == report["accounting"]["new_simulations"] == 0
    assert "admission" not in report and "threshold_candidates" not in report


def test_neighbor_mean_sign_and_mixed_pattern_are_distinct_and_ties_retained():
    protocol, result = fixture()
    harmful = [row for row in result["audit"]["queries"] if row["outcome"] == "harmful"]
    for row, gains in zip(harmful, ((-.4, .1, .1), (-.1, -.2, -.3), (0., 0., 0.))):
        for neighbor, gain in zip(row["neighbors"], gains):
            neighbor["raw_gain"] = gain
    harmful[0]["outside_training_range"] = [{"feature": protocol["feature_names"][0],
        "value": 1., "minimum": 0., "maximum": 0., "training_constant": True}]
    beneficial = next(row for row in result["audit"]["queries"] if row["outcome"] == "beneficial")
    beneficial["outside_training_range"] = [{"feature": protocol["feature_names"][1],
        "value": 2., "minimum": 0., "maximum": 1., "training_constant": False}]
    report = summary.aggregate(protocol, result)
    overall = report["overall"]
    assert overall["harmful_queries_neighbor_sign_pattern"] == {"all_positive": 12, "all_negative": 1, "all_zero": 1, "mixed": 1}
    assert overall["harmful_queries_neighbor_mean_gain_sign"] == {"positive": 12, "negative": 2, "zero": 1}
    assert overall["queries_outside_any_training_range"] == 2 and overall["queries_violating_training_constant"] == 1
    assert overall["range_violations_by_query_outcome"]["harmful"] == {"any_range": 1, "training_constant": 1}
    assert overall["nearest_distance_by_query_outcome"]["tie"] == {"count": 2, "minimum": 1., "median": 1., "maximum": 1.}


@pytest.mark.parametrize("change", ["missing_query", "model", "feature_order", "fold_scaling", "neighbor_leak", "duplicate_neighbor", "failed_labels"])
def test_changed_roster_features_or_training_neighborhood_is_rejected(change):
    protocol, result = fixture()
    if change == "missing_query":
        result["audit"]["queries"].pop()
    elif change == "model":
        result["model_path"] = "different-model.pkl"
    elif change == "feature_order":
        result["feature_names"] = list(reversed(result["feature_names"]))
    elif change == "fold_scaling":
        result["audit"]["folds"][0]["fit_row_count"] += 66  # Reference rows must not enter fit scaling.
    elif change == "neighbor_leak":
        row = result["audit"]["queries"][0]
        row["neighbors"][0].update(group_id=row["group_id"], seed=row["seed"])
    elif change == "duplicate_neighbor":
        row = result["audit"]["queries"][0]
        row["neighbors"][1] = deepcopy(row["neighbors"][0])
    else:
        result["audit"]["checks"]["candidate_order_and_raw_labels_exact"] = False
    with pytest.raises(ValueError):
        summary.aggregate(protocol, result)


def test_wrong_constant_flag_and_changed_frozen_raw_gain_are_rejected():
    protocol, result = fixture()
    row = result["audit"]["queries"][0]
    row["outside_training_range"] = [{"feature": protocol["feature_names"][0],
        "value": 1., "minimum": 0., "maximum": 0., "training_constant": False}]
    with pytest.raises(ValueError, match="training-fold bounds"):
        summary.aggregate(protocol, result)
    row["outside_training_range"] = []
    row["actual_gain"] += .01
    with pytest.raises(ValueError, match="frozen A action"):
        summary.aggregate(protocol, result)
