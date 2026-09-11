"""Frozen OOF denominator and score-unit comparisons for the raw-target refit."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.data import summarize_cologne_raw_target_refit as summary
from scripts.data.audit_cologne_local_rigid_ranking import group_record
from scripts.data.calibrate_cologne_rigid_threshold import threshold_metrics

RESULTS = Path(__file__).resolve().parents[1] / "cf_h2o/results"


def calibrations():
    old = json.loads((RESULTS / "paper_artifacts/tsc_v154t_cologne_no_stay_oof_threshold_v1.json").read_text())["calibration"]
    new = deepcopy(old)
    new["selected"]["threshold"] = 0.01  # A raw-unit fixture, not a proposed fitted threshold.
    for row in new["selected"]["per_seed"].values():
        row["mean_cost"] -= 0.01
        row["mean_cost_minus_phase_pressure"] -= 0.01
    new["selected"]["mean_cost"] -= 0.01
    new["selected"]["mean_cost_minus_phase_pressure"] -= 0.01
    return new, old


def test_selected_policy_comparison_preserves_all_groups_and_separates_threshold_units():
    new, old = calibrations()
    result = summary.comparison(new, old)
    assert result["groups"] == 100 and result["weighting"] == "equal_seed"
    assert [row["groups"] for row in result["per_seed"]] == [34, 33, 33]
    assert result["new_minus_old_policy_cost"] == pytest.approx(-0.01)
    assert result["old_selected_policy"]["accepted_overrides"] == 44
    assert result["old_selected_policy"]["beneficial"] == 23
    assert result["old_score_units"] != result["new_score_units"]
    assert "threshold_difference" not in result


@pytest.mark.parametrize("change", ["dropped_group", "pp_labels", "accepted_count"])
def test_changed_denominator_labels_or_totals_rejected(change):
    new, old = calibrations()
    if change == "dropped_group":
        new["selected"]["per_seed"]["2027"]["groups"] -= 1
    elif change == "pp_labels":
        new["selected"]["per_seed"]["2027"]["mean_phase_pressure_cost"] += 0.1
    else:
        new["selected"]["accepted_overrides"] += 1
    with pytest.raises(ValueError):
        summary.comparison(new, old)


def fitted_fixture():
    """Synthetic costs fit the real roster; this is not a V155A result."""
    protocol = json.loads((RESULTS / "cluster/tsc_v155a_cologne_raw_target_refit_20260910/protocol_v1.json").read_text())
    actions, records = [], {"old": [], "new": []}
    for group in protocol["training_group_ids"]:
        seed = int(group.split(":")[1][4:])
        costs = [1., .9, .8, 1.1, 1.2, 1.3, 1.4, 1.5]
        actions.append({"group_id": group, "seed": seed, "reference_index": 0,
            "old_effective_index": 1, "new_effective_index": 2, "old_deployed_index": 1,
            "new_deployed_index": 2, "label_costs": costs,
            "old_normalized_predicted_advantage": .3, "new_raw_predicted_advantage": .02})
        for arm, selected, score in (("old", 1, .3), ("new", 2, .02)):
            scores = [0.] * 8
            scores[selected] = -score
            records[arm].append(group_record(group, [f"phase{i}" for i in range(8)], costs, scores, 0, selected))
    calibration = {"selected": threshold_metrics(records["new"], .01), "phase_pressure": threshold_metrics(records["new"], None)}
    oldcalib = {"selected": threshold_metrics(records["old"], protocol["old_frozen_threshold"]["threshold"])}
    protocol["expected_original_selected"] = oldcalib["selected"]
    frozen = deepcopy(protocol["old_frozen_threshold"])
    frozen.update(model_path="fixture-model.pkl", calibration_protocol=summary.PROTOCOL, threshold=.01)
    result = {key: protocol[key] for key in ("protocol", "source_root", "sumocfg", "old_model_path", "folds")}
    result.update(status="COMPLETE", model_path="fixture-model.pkl", frozen_threshold=frozen,
        score_units="raw_native_halted_per_lane_cost_difference", selected_groups=protocol["training_group_ids"],
        anchor_feature_names=protocol["expected_anchor_feature_names"], checks=dict.fromkeys(summary.CHECKS, True),
        fit_diagnostics={"full": {"fixture": True}, "oof": [{"heldout_seed": seed, "diagnostics": {}} for seed in summary.SEEDS]},
        calibration=calibration, old_calibration=oldcalib, matched_actions=actions, proposal_accounting={},
        new_model_fits=4, full_model_fits=1, oof_fold_fits=3, inactive_correction_fits=0,
        new_labels=0, new_simulations=0, elapsed_sec=1)
    return protocol, result


def test_matched_costs_and_ranks_validate_selected_policies():
    protocol, fitted = fitted_fixture()
    result = summary.aggregate(protocol, fitted)
    action = result["action_comparison"]
    assert result["status"] == "COMPLETE" and len(action["rows"]) == 100
    assert action["totals"]["deployed_action_changed"] == 100
    assert action["totals"]["new_lower_cost"] == action["totals"]["both_accept"] == 100
    assert action["means"]["old_cost_rank"] == 2 and action["means"]["new_cost_rank"] == 1
    assert result["policy_comparison"]["new_minus_old_policy_cost"] == pytest.approx(-.1)
    assert result["accounting"]["new_model_fits"] == 4


@pytest.mark.parametrize("change", ["features", "fold", "weight_check", "serialized_check", "matched_roster", "wrong_units_gate"])
def test_fit_and_matched_action_provenance_rejected(change):
    protocol, fitted = fitted_fixture()
    if change == "features":
        fitted["anchor_feature_names"] = list(reversed(fitted["anchor_feature_names"]))
    elif change == "fold":
        fitted["folds"] = []
    elif change == "weight_check":
        fitted["checks"]["sample_weight_source_unchanged"] = False
    elif change == "serialized_check":
        fitted["checks"]["serialized_scores_exact"] = False
    elif change == "matched_roster":
        fitted["matched_actions"][0]["group_id"] = "different"
    else:
        # Applying the old normalized threshold to .02 would wrongly revert this raw-score action.
        fitted["matched_actions"][0]["new_deployed_index"] = 0
    with pytest.raises(ValueError):
        summary.aggregate(protocol, fitted)


def test_phase_pressure_only_selection_is_preserved():
    new, old = calibrations()
    new["selected"] = deepcopy(new["phase_pressure"])
    compared = summary.comparison(new, old)
    assert compared["new_selected_policy"]["threshold"] is None
    assert compared["new_selected_policy"]["accepted_overrides"] == 0
    assert compared["new_minus_phase_pressure_cost"] == 0


def balanced_fixture():
    protocol, result = fitted_fixture()
    protocol["protocol"] = result["protocol"] = summary.BALANCED_PROTOCOL
    protocol["old_frozen_threshold"].update(calibration_protocol=summary.PROTOCOL,
                                            score_units="raw_native_halted_per_lane_cost_difference")
    result["frozen_threshold"]["calibration_protocol"] = summary.BALANCED_PROTOCOL
    result["checks"] = dict.fromkeys(summary.BALANCED_CHECKS, True)
    for diagnostic in [result["fit_diagnostics"]["full"], *(row["diagnostics"] for row in result["fit_diagnostics"]["oof"])]:
        diagnostic.update(sample_weight_source_protocol="domain_group_balanced_v1",
                          sample_weight_min=1., sample_weight_max=1., sample_weight_mean=1.,
                          sign_balance={"enabled": False}, weight_group_scale_summary=None)
    for row in result["matched_actions"]:
        row["old_raw_predicted_advantage"] = row.pop("old_normalized_predicted_advantage")
    return protocol, result


def test_group_balanced_refit_compares_two_raw_score_models():
    protocol, result = balanced_fixture()
    report = summary.aggregate(protocol, result)
    assert report["protocol"] == summary.BALANCED_PROTOCOL and report["status"] == "COMPLETE"
    compared = report["policy_comparison"]
    assert compared["old_score_units"] == compared["new_score_units"] == summary.NEW_SCORE_UNITS
    assert compared["new_minus_old_policy_cost"] == pytest.approx(-.1)
    assert report["action_comparison"]["totals"]["new_lower_cost"] == 100
    assert "old_normalized_predicted_advantage" not in report["action_comparison"]["rows"][0]


@pytest.mark.parametrize("change", ["weight", "sign_balance", "baseline"])
def test_group_balanced_contract_rejects_old_shaping_or_wrong_baseline(change):
    protocol, result = balanced_fixture()
    if change == "weight":
        result["fit_diagnostics"]["oof"][0]["diagnostics"]["sample_weight_min"] = .8
    elif change == "sign_balance":
        result["fit_diagnostics"]["full"]["sign_balance"]["enabled"] = True
    else:
        protocol["old_frozen_threshold"]["calibration_protocol"] = "tsc-v154t-cologne-no-stay-oof-threshold-v1"
    with pytest.raises(ValueError):
        summary.aggregate(protocol, result)
