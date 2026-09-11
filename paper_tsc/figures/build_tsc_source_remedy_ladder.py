#!/usr/bin/env python3
"""Build the post-v148 source-remedy evidence table and source data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / "paper_tsc"
RESULT_ROOT = REPO / "cf_h2o" / "results" / "paper_artifacts"
CSV_OUT = PAPER / "source_data" / "source_remedy_ladder.csv"
TABLE_OUT = PAPER / "tables" / "source_remedy_ladder.tex"
V157B_MANIFEST = RESULT_ROOT / "tsc_v157b_feature_aligned_b100_arm_manifest_v2.json"
V158_RESULT = (
    RESULT_ROOT / "tsc_v158_native_prefix_ranking_calibration_v1.json"
)
EXPECTED_CITIES = {
    "atlanta",
    "cologne",
    "hangzhou",
    "ingolstadt",
    "new_york",
    "resco_synthetic",
    "salt_lake_city",
}
RESULTS = {
    "v150a": (
        "tsc_v150a_source_benefit_repeatability.json",
        "tsc-v150a-source-benefit-repeatability-diagnostic-v1",
    ),
    "v150c": (
        "tsc_v150c_mechanism_parameter_prior.json",
        "tsc-v150c-mechanism-parameter-prior-aggregate-v1",
    ),
    "v150i": (
        "tsc_v150i_sample_coherent_source_prior.json",
        "tsc-v150i-sample-coherent-source-prior-aggregate-v1",
    ),
    "v150j": (
        "tsc_v150j_conditional_mechanism_prior.json",
        "tsc-v150j-conditional-mechanism-prior-aggregate-v1",
    ),
    "v150k": (
        "tsc_v150k_state_conditioned_source_utility.json",
        "tsc-v150k-state-conditioned-source-utility-aggregate-v1",
    ),
    "v150l": (
        "tsc_v150l_state_conditioned_source_closed_loop_smoke_v3_paired_safety.json",
        "tsc-v150l-state-conditioned-source-closed-loop-aggregate-v3",
    ),
    "v156a": (
        "tsc_v156a_feature_aligned_dense_source_utility.json",
        "tsc-v156a-feature-aligned-dense-source-utility-aggregate-v1",
    ),
    "v151a": (
        "tsc_v151a_cross_city_meta_source_utility.json",
        "tsc-v151a-cross-city-meta-source-utility-aggregate-v1",
    ),
    "v152a": (
        "tsc_v152a_hierarchical_mechanism_prior.json",
        "tsc-v152a-hierarchical-mechanism-prior-aggregate-v1",
    ),
    "v153a": (
        "tsc_v153a_target_calibrated_hierarchical_prior_v3.json",
        "tsc-v153a-target-calibrated-hierarchical-prior-aggregate-v3",
    ),
    "v157a": (
        "tsc_v157a_feature_aligned_target_budget_source_value_curve.json",
        "tsc-v157a-feature-aligned-target-budget-source-value-curve-aggregate-v1",
    ),
    "v157b": (
        "tsc_v157b_feature_aligned_b100_runtime_freeze_result_v2.json",
        "tsc-v157b-feature-aligned-b100-runtime-refit-freeze-result-v2",
    ),
    "v157c": (
        "tsc_v157c_jinan_native_action_branches_incomplete_v1.json",
        "tsc-v157c-jinan-native-one-action-incomplete-aggregate-v1",
    ),
}


def _read_results() -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for version, (filename, protocol) in RESULTS.items():
        path = RESULT_ROOT / filename
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("protocol") != protocol:
            raise ValueError(f"{version} source-remedy evidence changed")
        cities = result.get("target_city_groups")
        if cities is not None and set(str(value) for value in cities) != EXPECTED_CITIES:
            raise ValueError(f"{version} city cohort changed")
        admitted = result.get("admitted_cities")
        if admitted is not None and set(str(value) for value in admitted) != EXPECTED_CITIES:
            raise ValueError(f"{version} admission cohort changed")
        results[version] = result
    return results


def _city_effect(result: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = result[field]
    if "city_unit" in value:
        return value["city_unit"]
    return value


def _read_v157b_manifest(result: Mapping[str, Any]) -> dict[str, Any]:
    raw = V157B_MANIFEST.read_bytes()
    manifest = json.loads(raw)
    reference = result["artifacts"]["arm_manifest"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("protocol") != reference["protocol"]
        or hashlib.sha256(raw).hexdigest() != reference["sha256"]
        or len(raw) != int(reference["size_bytes"])
    ):
        raise ValueError("v157b arm manifest binding changed")
    runtime = manifest["runtime_bundle"]
    expected_runtime = result["artifacts"]["runtime_models"]
    if any(
        runtime[field] != expected_runtime[field]
        for field in ("protocol", "sha256", "size_bytes")
    ):
        raise ValueError("v157b runtime bundle binding changed")
    return manifest


def _read_v158_result() -> dict[str, Any]:
    result = json.loads(V158_RESULT.read_text(encoding="utf-8"))
    if (
        not isinstance(result, dict)
        or result.get("protocol")
        != "tsc-v158-native-prefix-ranking-calibration-result-v1"
    ):
        raise ValueError("v158 native-prefix result evidence changed")
    return result


def build_rows() -> list[dict[str, str]]:
    result = _read_results()
    v158 = _read_v158_result()
    repeat = result["v150a"]["repeatability_summary"][
        "forced_source_leave_one_seed_out"
    ]
    repeat_effect = repeat["effect_vs_target_only"]
    if int(repeat_effect["count"]) != 21:
        raise ValueError("v150a repeatability cohort changed")

    v150c = result["v150c"]
    c_effect = _city_effect(v150c, "selected_effect_vs_target_only")
    c_placebo = _city_effect(v150c, "selected_effect_vs_matched_placebo")
    c_improved = sum(float(value) < 0.0 for value in c_effect["city_means"].values())
    c_admitted = sum(
        bool(value["admitted"]) for value in v150c["selected_candidates"].values()
    )

    v150i = result["v150i"]["budget_results"]["100"]
    i_effect = _city_effect(v150i, "selected_effect_vs_rigid_target_only")
    v150l = result["v150l"]
    l_cities = v150l["city_summaries"]
    v156a = result["v156a"]
    o_effect = _city_effect(v156a, "selected_effect_vs_rigid_target_only")
    o_placebo = _city_effect(v156a, "selected_effect_vs_matched_placebo")
    o_blind = _city_effect(v156a, "selected_effect_vs_source_blind")
    v152a = result["v152a"]
    p152 = _city_effect(v152a, "forced_effect_vs_rigid_target_only")
    v153a = result["v153a"]
    p153 = _city_effect(v153a, "forced_effect_vs_rigid_target_only")
    v157a = result["v157a"]
    v157b = result["v157b"]
    v157b_manifest = _read_v157b_manifest(v157b)
    v157b_identity = v157b["v157a_exact_refit_identity_audit"]
    v157b_domain_diagnostic = v157b["runtime_domain_alignment_diagnostic"]
    v157b_gate = v157b["domain_aligned_source_gate"]
    v157b_effect = v157b_gate["paired_bootstrap"]
    v157b_seed_differences = v157b_gate["source_minus_target_by_seed"]
    v157b_improved = sum(float(value) < 0.0 for value in v157b_seed_differences)
    v157c = result["v157c"]
    v157c_completion = v157c["protocol_completion"]
    v157c_descriptive = v157c["descriptive_two_valid_seeds"]
    v157c_comparisons = v157c_descriptive["comparisons"]
    v157c_source_target = v157c_comparisons[
        "uniform_source_minus_target_only"
    ]
    v157c_source_placebo = v157c_comparisons[
        "uniform_source_minus_source_label_placebo"
    ]
    v157c_target_pp = v157c_comparisons[
        "target_only_minus_phase_pressure"
    ]
    v157c_source_pp = v157c_comparisons[
        "uniform_source_minus_phase_pressure"
    ]
    v158_development = v158["development"]
    v158_decision = v158["decision"]
    v158_oof = v158["oof"]
    v158_gate = v158_oof["gate"]
    v158_comparisons = v158_oof["comparisons"]
    v158_source_target = v158_comparisons["source_minus_target"]
    v158_source_placebo = v158_comparisons["source_minus_placebo"]
    v158_source_pp = v158_comparisons["source_minus_phase_pressure"]
    k_effect = _city_effect(
        result["v150k"], "selected_effect_vs_rigid_target_only"
    )

    rows = [
        {
            "stage": "Repeatability diagnostic",
            "information": "Other development seeds; forced source",
            "source_admission": "Oracle only",
            "quantitative_result": (
                f"{round(float(repeat_effect['improving_fraction']) * 21)}/21 "
                f"seed units and {repeat['improving_city_count_vs_target_only']}/7 "
                "city means improved"
            ),
            "adjudication": "Headroom exists; not deployable",
            "evidence": "v150a",
        },
        {
            "stage": "Mechanism parameter prior",
            "information": "Target folds plus source mechanism block",
            "source_admission": f"{c_admitted}/7 prototype",
            "quantitative_result": (
                f"{c_improved}/7 improved; mean {float(c_effect['mean']):+.5f}; "
                f"placebo upper 95% {float(c_placebo['upper_95']):+.5f}"
            ),
            "adjudication": "Offline pass; source identity unresolved",
            "evidence": "v150c",
        },
        {
            "stage": "Budget-aware shrinkage",
            "information": "B100 target folds; inverse-budget prior",
            "source_admission": f"{int(v150i['source_admission_count'])}/7",
            "quantitative_result": (
                f"mean {float(i_effect['mean']):+.5f}; no city regression"
            ),
            "adjudication": "Reject multicity generality",
            "evidence": "v150i",
        },
        {
            "stage": "Local mechanism context",
            "information": "B100 plus local right-of-way context",
            "source_admission": f"{int(result['v150j']['source_admission_count'])}/7",
            "quantitative_result": "Exact rigid fallback in all cities",
            "adjudication": "Reject",
            "evidence": "v150j",
        },
        {
            "stage": "State-conditioned utility",
            "information": "B100 nested state-level utility labels",
            "source_admission": f"{int(result['v150k']['source_admission_count'])}/7",
            "quantitative_result": (
                f"pre-correction historical: offline mean {float(k_effect['mean']):+.5f}; "
                "smoke: Cologne "
                f"{100 * float(l_cities['cologne']['source_relative_change_vs_rigid']):+.1f}%, "
                f"New York {100 * float(l_cities['new_york']['source_relative_change_vs_rigid']):+.1f}%, "
                f"RESCO {100 * float(l_cities['resco_synthetic']['source_relative_change_vs_rigid']):+.1f}%"
            ),
            "adjudication": "Historical; reject after closed-loop smoke",
            "evidence": "v150k--v150l",
        },
        {
            "stage": "Feature-aligned dense utility",
            "information": "V150O replay; stored-name rigid binding",
            "source_admission": (
                f"{int(v156a['source_admission_count'])}/7 inner; 0 follow-up"
            ),
            "quantitative_result": (
                f"macro {float(o_effect['mean']):+.5f} vs rigid; Hangzhou "
                f"{float(o_effect['city_means']['hangzhou']):+.5f}/"
                f"{float(o_placebo['city_means']['hangzhou']):+.5f}/"
                f"{float(o_blind['city_means']['hangzhou']):+.5f} vs rigid/placebo/blind"
            ),
            "adjudication": "Reject globally and city-specifically",
            "evidence": "v150o historical; v156a corrected",
        },
        {
            "stage": "Leave-city-out meta utility",
            "information": "Six pseudo-target cities; target B100",
            "source_admission": f"{int(result['v151a']['source_admission_count'])}/7",
            "quantitative_result": "Exact rigid fallback in all cities",
            "adjudication": "Reject",
            "evidence": "v151a",
        },
        {
            "stage": "Source-only hierarchical prior",
            "information": "Equal-city source prior; no target gate labels",
            "source_admission": f"{int(v152a['source_admission_count'])}/7",
            "quantitative_result": (
                f"forced mean {float(p152['mean']):+.5f}; "
                f"{sum(float(value) < 0 for value in p152['city_means'].values())}/7 improved"
            ),
            "adjudication": "Reject",
            "evidence": "v152a",
        },
        {
            "stage": "Target-calibrated hierarchical prior",
            "information": "B100 target selector; same prior grid",
            "source_admission": f"{int(v153a['source_admission_count'])}/7",
            "quantitative_result": (
                f"forced mean {float(p153['mean']):+.5f}, 95% "
                f"[{float(p153['lower_95']):+.5f}, {float(p153['upper_95']):+.5f}]"
            ),
            "adjudication": "Reject; exact rigid fallback",
            "evidence": "v153a",
        },
        {
            "stage": "Domain-aligned B100 source gate",
            "information": "V123/V157A confounded; reused 22-seed Jinan selector",
            "source_admission": "One-action branch only",
            "quantitative_result": (
                "source-target "
                f"{float(v157b_effect['mean']):+.5f}, "
                "95% "
                f"[{float(v157b_effect['lower_95']):+.5f}, "
                f"{float(v157b_effect['upper_95']):+.5f}]; "
                f"{v157b_improved}/22 improve; "
                f"{float(v157b_gate['uniform_source_mean_normalized_delta_vs_phase_pressure']):+.5f} vs PP"
            ),
            "adjudication": "B100 gate pass; V157C incomplete",
            "evidence": "v123/v157a historical; v157b corrected",
        },
        {
            "stage": "Native one-action branch",
            "information": "Fixed Jinan roster; seed 27178 t=480 invalid",
            "source_admission": "2/3 valid seeds",
            "quantitative_result": (
                "descriptive source-target "
                f"{float(v157c_source_target['mean_of_available_seed_means']):+.5f} "
                f"({int(v157c_source_target['negative_available_seed_count'])}/2 improve); "
                "source-placebo "
                f"{float(v157c_source_placebo['mean_of_available_seed_means']):+.5f}; "
                "target-PP "
                f"{float(v157c_target_pp['mean_of_available_seed_means']):+.5f}; "
                "source-PP "
                f"{float(v157c_source_pp['mean_of_available_seed_means']):+.5f} "
                f"({int(v157c_source_pp['negative_available_seed_count'])}/2 improve); "
                f"collisions {int(v157c_descriptive['valid_seed_collision_events'])}; "
                "PP=PhasePressure heuristic"
            ),
            "adjudication": "Incomplete; no three-seed inference",
            "evidence": "v157c",
        },
        {
            "stage": "Native-prefix ranking calibration",
            "information": (
                "Frozen V157B predictors; native labels; five seed-held-out folds"
            ),
            "source_admission": "OOF 0/3; no reserve",
            "quantitative_result": (
                "source-target "
                f"{float(v158_source_target['mean_difference']):+.5f}, 95% "
                f"[{float(v158_source_target['paired_bootstrap_95'][0]):+.5f}, "
                f"{float(v158_source_target['paired_bootstrap_95'][1]):+.5f}] "
                f"({int(v158_source_target['improved_seeds'])}/10 improve); "
                "source-placebo "
                f"{float(v158_source_placebo['mean_difference']):+.5f}, 95% "
                f"[{float(v158_source_placebo['paired_bootstrap_95'][0]):+.5f}, "
                f"{float(v158_source_placebo['paired_bootstrap_95'][1]):+.5f}] "
                f"({int(v158_source_placebo['improved_seeds'])}/10 improve); "
                "source-PP "
                f"{float(v158_source_pp['mean_difference']):+.5f}, 95% "
                f"[{float(v158_source_pp['paired_bootstrap_95'][0]):+.5f}, "
                f"{float(v158_source_pp['paired_bootstrap_95'][1]):+.5f}] "
                f"({int(v158_source_pp['improved_seeds'])}/10 improve); "
                f"{int(v158_development['action_groups'])} groups/"
                f"{int(v158_development['candidate_rows'])} rows/"
                f"{int(v158_development['native_nonreference_branches'])} branches; "
                f"{int(v158_development['snapshot_restores'])} restores"
            ),
            "adjudication": "Reject; OOF gate failed; reserve withheld",
            "evidence": "v158",
        },
    ]
    expected_v157c_means = {
        "uniform_source_minus_target_only": 0.002363683127571995,
        "uniform_source_minus_source_label_placebo": -0.002317386831275766,
        "target_only_minus_phase_pressure": 0.0012525720164610807,
        "uniform_source_minus_phase_pressure": 0.003616255144033076,
        "source_label_placebo_minus_phase_pressure": 0.005933641975308841,
    }
    invalid_window = v157c_completion["invalid_fixed_window"]
    expected_v158_seeds = [
        180314,
        188625,
        127178,
        177524,
        163775,
        150945,
        173412,
        161481,
        150744,
        134310,
    ]
    expected_v158_development = {
        "seeds": expected_v158_seeds,
        "action_groups": 100,
        "candidate_rows": 800,
        "native_nonreference_branches": 700,
        "baseline_trajectories": 30,
        "snapshot_restores": 0,
    }
    expected_v158_comparisons = {
        "source_minus_target": {
            "mean": -0.0011257716049382527,
            "interval": [-0.0023958487654320943, 0.00019170138888892246],
            "improved": 6,
        },
        "source_minus_placebo": {
            "mean": -0.0007228395061728232,
            "interval": [-0.002185034722222258, 0.0009064814814814914],
            "improved": 7,
        },
        "source_minus_phase_pressure": {
            "mean": 0.0001354938271605155,
            "interval": [-0.0023895949074074067, 0.0026061072530864163],
            "improved": 5,
        },
    }
    if not (
        v150c["development_gate"]["passed"] is True
        and all(
            result[key]["development_gate"]["passed"] is False
            for key in ("v150i", "v150j", "v156a", "v151a", "v152a", "v153a")
        )
        and v156a["city_specific_followup_gate"]["passed"] is False
        and v157a["correction_only_contract"][
            "all_input_split_and_fit_invariants_passed"
        ]
        is True
        and v157b_identity["passed"] is True
        and v157b_identity["failed_sections"] == []
        and v157b_identity["target_score_uncertainty_trust_equal"] is True
        and all(v157b_identity["per_source_score_uncertainty_trust_equal"].values())
        and v157b_identity["uniform_component_mean_score_uncertainty_trust_equal"]
        is True
        and v157b_identity["uniform_runtime_score_uncertainty_trust_equal"] is True
        and v157b_domain_diagnostic["passed"] is False
        and v157b_domain_diagnostic["failed_sections"] == ["target_only"]
        and v157b_domain_diagnostic["target_score_uncertainty_trust_equal"] is False
        and all(
            v157b_domain_diagnostic[
                "per_source_score_uncertainty_trust_equal"
            ].values()
        )
        and v157b_gate["passed"] is True
        and v157b_gate["comparison"]
        == "uniform_source_minus_domain_aligned_target_only"
        and len(v157b_gate["selector_seed_order"]) == 22
        and len(v157b_seed_differences) == 22
        and v157b_improved == 20
        and float(v157b_effect["mean"])
        <= float(v157b_gate["thresholds"]["mean_maximum"])
        and float(v157b_effect["upper_95"])
        < float(v157b_gate["thresholds"]["upper_95_strict_maximum"])
        and float(v157b_gate["uniform_source_mean_normalized_delta_vs_phase_pressure"])
        > 0.0
        and v157b["runtime_artifacts_written"] is True
        and v157b["v157c_authorized"] is True
        and v157b["scientific_status"] == "runtime_arms_frozen_branch_not_run"
        and v157b["information_budget"]["target_action_groups"] == 100
        and v157b["information_budget"]["b100_fit_roster"]["total_fit_tasks"] == 16
        and v157b_manifest["scope"]
        == "one_frozen_focal_tls_action_then_phase_pressure_continuation"
        and v157b_manifest["arms"]["target_only"]["training_domain_handling"]
        == "relabel_all_b100_rows_to_jinan"
        and v157b_manifest["arms"]["target_only"][
            "v157a_exact_target_refit_used_at_runtime"
        ]
        is False
        and v157c["status"] == "INCOMPLETE"
        and v157c["scientific_status"] == "three_seed_protocol_incomplete"
        and v157c["decision"]
        == "do_not_treat_v157c_as_completed_or_as_transfer_success"
        and v157c["parent_protocol"]
        == "tsc-v157c-jinan-native-one-action-branches-v1"
        and v157c["scenario"] == "jinan_3x4_real"
        and v157c_completion["expected_seed_count"] == 3
        and v157c_completion["expected_seeds"] == [80314, 88625, 27178]
        and v157c_completion["valid_seed_count"] == 2
        and v157c_completion["valid_seed_results"] == [80314, 88625]
        and v157c_completion["invalid_seed"] == 27178
        and invalid_window
        == {
            "seed": 27178,
            "checkpoint_sec": 480,
            "reason": "no_scorable_tls",
            "status": "INVALID_RETAINED",
            "checkpoint_shifted": False,
            "window_dropped": False,
        }
        and v157c_completion["expected_window_count"] == 9
        and v157c_completion["valid_window_count"] == 6
        and v157c_completion["three_seed_aggregate_computed"] is False
        and v157c_completion["primary_gate_evaluated"] is False
        and v157c_descriptive["status"] == "DESCRIPTIVE_ONLY"
        and v157c_descriptive["confidence_interval_computed"] is False
        and v157c_descriptive["gate_result"] is None
        and v157c_descriptive["valid_seed_collision_events"] == 0
        and set(v157c_comparisons) == set(expected_v157c_means)
        and all(
            comparison["available_seed_count"] == 2
            and set(comparison["available_seed_means"]) == {"80314", "88625"}
            and comparison["inferential_status"] == "DESCRIPTIVE_ONLY"
            and float(comparison["mean_of_available_seed_means"])
            == expected_v157c_means[name]
            for name, comparison in v157c_comparisons.items()
        )
        and v157c_source_target["negative_available_seed_count"] == 1
        and v157c_source_pp["negative_available_seed_count"] == 0
        and v157c["identical_action_windows_retained_as_zero"] is True
        and v157c["outcomes_used_for_checkpoint_or_focal_selection"] is False
        and v158["parent_protocol"]
        == "tsc-v158-jinan-native-prefix-ranking-calibration-v1"
        and v158["status"] == "OOF_GATE_FAIL"
        and v158["scientific_status"] == "remediation_closed_at_oof"
        and v158_decision
        == {
            "adopt_source_native_controller": False,
            "native_prefix_ranking_remediation": (
                "closed_after_preregistered_oof_gate_failure"
            ),
            "reserve_authorized": False,
            "reserve_cell_count": 0,
            "reserve_executed": False,
            "reserve_seed_count": 0,
        }
        and {
            key: v158_development.get(key)
            for key in expected_v158_development
        }
        == expected_v158_development
        and v158_development["valid_seed_banks"] == 10
        and v158_development["invalid_or_missing_seed_banks"] == 0
        and v158_development["validity_contract_passed_all_seed_banks"] is True
        and v158_development["teleport_check_passed_all_seed_banks"] is True
        and v158_oof["fold_count"] == 5
        and v158_oof["threshold"] == 0.0
        and v158_oof["unit"] == "whole_seed"
        and set(v158_comparisons) == set(expected_v158_comparisons)
        and all(
            comparison["seed_count"] == 10
            and set(comparison["seeds"]) == set(expected_v158_seeds)
            and comparison["mean_difference"] == expected["mean"]
            and comparison["paired_bootstrap_95"] == expected["interval"]
            and comparison["improved_seeds"] == expected["improved"]
            for name, expected in expected_v158_comparisons.items()
            for comparison in [v158_comparisons[name]]
        )
        and v158_gate["passed"] is False
        and v158_gate["specification"]
        == {
            "unit": "equal_seed",
            "source_minus_target_mean_maximum": -0.0005,
            "source_minus_placebo_mean_maximum": -0.0005,
            "source_minus_phase_pressure_mean_maximum": 0.0,
            "paired_bootstrap_upper_95_maximum": 0.0,
            "minimum_improved_seeds": 7,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20260911,
        }
        and v158_gate["checks"]
        == {
            "source_minus_target": {
                "mean_passed": True,
                "bootstrap_upper_passed": False,
                "seed_wins_passed": False,
                "passed": False,
            },
            "source_minus_placebo": {
                "mean_passed": True,
                "bootstrap_upper_passed": False,
                "seed_wins_passed": True,
                "passed": False,
            },
            "source_minus_phase_pressure": {
                "mean_passed": False,
                "bootstrap_upper_passed": False,
                "seed_wins_passed": False,
                "passed": False,
            },
        }
    ):
        raise ValueError("source-remedy adjudication changed")
    return rows


def render_table(rows: Sequence[Mapping[str, str]]) -> str:
    lines = [
        (
            r"\begin{tabular}{@{}"
            r">{\raggedright\arraybackslash}p{0.14\textwidth}"
            r">{\raggedright\arraybackslash}p{0.18\textwidth}"
            r">{\raggedright\arraybackslash}p{0.09\textwidth}"
            r">{\raggedright\arraybackslash}p{0.29\textwidth}"
            r">{\raggedright\arraybackslash}p{0.16\textwidth}@{}}"
        ),
        r"\toprule",
        r"Stage & Deployment information & Source & Held-out result & Adjudication \\",
        r"\midrule",
    ]
    for row in rows:
        escaped_result = row["quantitative_result"].replace("%", r"\%")
        lines.append(
            f"{row['stage']} & {row['information']} & {row['source_admission']} & "
            f"{escaped_result} & {row['adjudication']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build() -> list[dict[str, str]]:
    rows = build_rows()
    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    TABLE_OUT.write_text(render_table(rows), encoding="utf-8")
    return rows


if __name__ == "__main__":
    print(
        json.dumps(
            {"rows": build(), "table": str(TABLE_OUT), "csv": str(CSV_OUT)},
            sort_keys=True,
        )
    )
