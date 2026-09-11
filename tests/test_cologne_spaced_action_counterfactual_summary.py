"""Synthetic paired costs verify the Y frozen-roster estimand and missing-data rules."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_spaced_action_counterfactual as summary

PROTOCOL_PATH = Path(__file__).resolve().parents[1] / (
    "cf_h2o/results/cluster/tsc_v154y_cologne_spaced_action_counterfactual_20260910/protocol_v1.json")


def inputs():
    protocol = json.loads(PROTOCOL_PATH.read_text())
    results = []
    for seed in summary.SEEDS:
        result = {key: deepcopy(protocol[key]) for key in
                  ("protocol", "source_root", "sumocfg", "frozen_threshold", "cooldown_intervals", "label_contract")}
        result.update(seed=seed, status="PASS", groups=[], new_simulation_count=14, elapsed_sec=1,
                      task_id=f"fixture-{seed}", source_result=f"fixture-{seed}.json")
        for index, window in enumerate(w for w in protocol["windows"] if w["seed"] == seed):
            gain = (0.2, -0.1, 0)[index % 3]
            runs = {}
            for arm in summary.ARMS:
                runs[arm] = {"status": "PASS", "checks": dict.fromkeys(summary.BRANCH_CHECKS, True),
                             "sample_count": 450, "mean_halted_per_lane": 1 - gain if arm == summary.ARMS[0] else 1,
                             "prefix_safety": {"raw_collision_events": 0}, "safety": {"raw_collision_events": 0}}
            result["groups"].append({"seed": seed, "target_time_sec": window["time_sec"],
                "end_time_sec": window["window_end_sec"], "selected_trace": deepcopy(window["selected_trace"]),
                "status": "PASS", "checks": dict.fromkeys(summary.PAIR_CHECKS, True), "runs": runs})
        results.append(result)
    return protocol, results


def test_all_21_equal_seed_costs_sign_counts_and_descriptive_extrema():
    protocol, results = inputs()
    report = summary.aggregate(protocol, list(reversed(results)))
    metrics = report["summary"]
    assert report["status"] == "COMPLETE" and report["valid_pair_count"] == 21
    assert len(report["pairs"]) == 21 and metrics["pair_count"] == 21
    assert metrics["weighting"] == "equal_seed"
    assert metrics["model_mean_cost"] == pytest.approx(6.6 / 7)
    assert metrics["phase_pressure_mean_cost"] == 1
    assert metrics["mean_actual_advantage"] == pytest.approx(0.4 / 7)
    assert [metrics[key] for key in ("beneficial", "harmful", "tie")] == [9, 6, 6]
    assert metrics["score_sign_agreement_count"] == 9
    assert metrics["score_sign_disagreement_count"] == 12
    assert all([row[key] for key in ("beneficial", "harmful", "tie")] == [3, 2, 2] for row in metrics["per_seed"])
    assert report["descriptive_extrema"]["best_measured_window"]["actual_advantage"] == pytest.approx(0.2)
    assert report["descriptive_extrema"]["worst_measured_window"]["actual_advantage"] == pytest.approx(-0.1)
    assert report["accounting"]["new_simulations_reported"] == 42
    assert report["score_units"] == "group-normalized predicted score advantage"
    assert report["gain_units"] == "native 450-second mean halted vehicles per controlled lane"
    encoded = json.dumps(report)
    assert all(key not in encoded for key in ("prediction_bias", "prediction_mean_absolute_error", "prediction_error"))


def test_colliding_valid_window_stays_in_costs_and_prefix_is_separate():
    protocol, results = inputs()
    original = summary.aggregate(protocol, results)["summary"]
    branch = results[0]["groups"][0]["runs"][summary.ARMS[0]]
    branch["prefix_safety"]["raw_collision_events"] = 13
    branch["safety"]["raw_collision_events"] = 2
    results[0]["safety_status"] = results[0]["groups"][0]["safety_status"] = "FAIL"
    report = summary.aggregate(protocol, results)
    assert report["status"] == "COMPLETE" and report["summary"] == original
    diagnostic = report["pairs"][0]["branch_diagnostics"][summary.ARMS[0]]
    assert diagnostic["prefix"]["raw_collision_events"] == 13
    assert diagnostic["window"]["raw_collision_events"] == 2


@pytest.mark.parametrize("change", ["missing_pair", "missing_seed", "short_horizon", "native_mismatch", "source", "action"])
def test_incomplete_records_disable_all_21_summary(change):
    protocol, results = inputs()
    if change == "missing_pair":
        results[0]["groups"].pop()
    elif change == "missing_seed":
        results.pop()
    elif change == "short_horizon":
        results[0]["groups"][0]["runs"][summary.ARMS[0]]["sample_count"] = 449
    elif change == "native_mismatch":
        results[0]["groups"][0]["checks"]["physical_start_exact"] = False
    elif change == "source":
        results[0]["frozen_threshold"]["model_path"] = "different.pkl"
    else:
        results[0]["groups"][0]["selected_trace"]["priority"] += 1
    report = summary.aggregate(protocol, results)
    assert report["status"] == "INCOMPLETE" and len(report["pairs"]) == 21
    assert report["valid_pair_count"] < 21
    assert report["summary"] is report["descriptive_extrema"] is None
    assert any(row["validity_errors"] for row in report["pairs"])


def test_missing_pair_checks_cannot_pass_vacuously():
    protocol, results = inputs()
    results[0]["groups"][0]["checks"] = {}
    report = summary.aggregate(protocol, results)
    assert report["status"] == "INCOMPLETE" and report["valid_pair_count"] == 20


def test_frozen_roster_cannot_shrink_to_favorable_subset():
    protocol, results = inputs()
    protocol["windows"].pop()
    with pytest.raises(ValueError):
        summary.aggregate(protocol, results)


def test_different_units_use_signs_and_descriptive_association_only():
    rows = [{"predicted_advantage": 0.4, "actual_advantage": 0.1},
            {"predicted_advantage": 0.2, "actual_advantage": 0.0}]
    metrics = summary.association(rows)
    assert metrics["pearson"] == pytest.approx(1)
    assert summary.sign_agreement(rows) == {"score_sign_agreement_count": 1, "score_sign_disagreement_count": 1}
    # Changing the score scale does not create a raw native-cost error measure.
    scaled = [{**row, "predicted_advantage": row["predicted_advantage"] * 100} for row in rows]
    assert summary.sign_agreement(scaled) == summary.sign_agreement(rows)
    assert summary.association(scaled)["pearson"] == pytest.approx(metrics["pearson"])
    assert "prediction_bias" not in metrics and "prediction_mean_absolute_error" not in metrics
    rows[1]["actual_advantage"] = 0.1
    assert summary.association(rows)["pearson"] is None


def raw_inputs():
    protocol, results = inputs()
    protocol.update(protocol=summary.RAW_PROTOCOL, score_units=summary.RAW_SCORE_UNITS)
    protocol["frozen_threshold"].update(calibration_protocol="tsc-v155a-cologne-raw-target-refit-v1",
                                       threshold=0, score_units=summary.RAW_SCORE_UNITS)
    for window in protocol["windows"]:
        window["selected_trace"]["priority"] = .1
    for result in results:
        result.update(protocol=summary.RAW_PROTOCOL, score_units=summary.RAW_SCORE_UNITS,
                      frozen_threshold=deepcopy(protocol["frozen_threshold"]))
        for group in result["groups"]:
            group["selected_trace"]["priority"] = .1
            gain = group["runs"]["pp_only"]["mean_halted_per_lane"] - group["runs"]["model_once_then_pp"]["mean_halted_per_lane"]
            gains = [.2, 3 * gain - .1, -.1]
            for arm, branch in group["runs"].items():
                branch["per_150_seconds"] = [
                    {"start_sec": group["target_time_sec"] + 150 * index,
                     "end_sec": group["target_time_sec"] + 150 * (index + 1),
                     "mean_halted_per_lane": 1 - gain if arm == "model_once_then_pp" else 1}
                    for index, gain in enumerate(gains)]
    return protocol, results


def test_raw_units_allow_calibration_error_and_early_late_gain():
    protocol, results = raw_inputs()
    results[0]["groups"][0]["runs"][summary.ARMS[0]]["safety"]["raw_collision_events"] = 2
    report = summary.aggregate(protocol, results)
    assert report["protocol"] == summary.RAW_PROTOCOL and report["status"] == "COMPLETE"
    metrics = report["summary"]
    errors = metrics["raw_prediction_diagnostics"]
    assert metrics["mean_predicted_advantage"] == pytest.approx(.1)
    assert errors["prediction_bias"] == pytest.approx(.1 - .4 / 7)
    assert errors["prediction_mean_absolute_error"] == pytest.approx(.9 / 7)
    assert (errors["overprediction_count"], errors["underprediction_count"], errors["equal_prediction_count"]) == (12, 9, 0)
    assert all(row["raw_prediction_diagnostics"]["underprediction_count"] == 3 for row in metrics["per_seed"])
    assert report["pairs"][0]["prediction_error"] == pytest.approx(-.1)
    assert metrics["per_150_seconds"][0]["mean_actual_advantage"] == pytest.approx(.2)
    assert metrics["per_150_seconds"][2]["mean_actual_advantage"] == pytest.approx(-.1)


def test_raw_error_requires_a_model_calibration_binding():
    protocol, results = raw_inputs()
    protocol["frozen_threshold"]["calibration_protocol"] = "tsc-v154t-cologne-no-stay-oof-threshold-v1"
    with pytest.raises(ValueError):
        summary.aggregate(protocol, results)


def test_raw_result_score_unit_mismatch_disables_complete_statistics():
    protocol, results = raw_inputs()
    results[0]["score_units"] = summary.SCORE_UNITS
    report = summary.aggregate(protocol, results)
    assert report["status"] == "INCOMPLETE" and report["summary"] is None
    assert report["valid_pair_count"] == 14


def test_inconsistent_segment_mean_rejected():
    protocol, results = raw_inputs()
    results[0]["groups"][0]["runs"][summary.ARMS[0]]["per_150_seconds"][0]["mean_halted_per_lane"] += .1
    with pytest.raises(ValueError):
        summary.aggregate(protocol, results)
