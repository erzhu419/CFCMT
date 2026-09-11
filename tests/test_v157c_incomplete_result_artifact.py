import hashlib
import json
import statistics
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = (
    ROOT
    / "cf_h2o/results/cluster/tsc_v157c_jinan_native_action_branches_20260911/native_v4"
)
AGGREGATE = RUN_DIR / "incomplete_aggregate.json"
PAPER_ARTIFACT = (
    ROOT
    / "cf_h2o/results/paper_artifacts/tsc_v157c_jinan_native_action_branches_incomplete_v1.json"
)
FAILURE = RUN_DIR / "failed_seed_27178_evidence.json"
REPORT = ROOT / "paper/tsc_v157c_jinan_native_action_branches_result.md"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_incomplete_artifact_preserves_the_failed_fixed_window():
    result = _json(AGGREGATE)
    completion = result["protocol_completion"]

    assert result["status"] == "INCOMPLETE"
    assert result["scientific_status"] == "three_seed_protocol_incomplete"
    assert completion["expected_seeds"] == [80314, 88625, 27178]
    assert completion["valid_seed_results"] == [80314, 88625]
    assert completion["invalid_seed"] == 27178
    assert completion["primary_gate_evaluated"] is False
    assert completion["three_seed_aggregate_computed"] is False
    assert completion["invalid_fixed_window"] == {
        "checkpoint_sec": 480,
        "checkpoint_shifted": False,
        "reason": "no_scorable_tls",
        "seed": 27178,
        "status": "INVALID_RETAINED",
        "window_dropped": False,
    }
    assert result["descriptive_two_valid_seeds"]["gate_result"] is None
    assert result["descriptive_two_valid_seeds"]["confidence_interval_computed"] is False


def test_two_seed_numbers_are_recomputed_without_dropping_zero_windows():
    aggregate = _json(AGGREGATE)
    descriptive = aggregate["descriptive_two_valid_seeds"]
    checkpoints = (300, 480, 660)
    seed_results = {
        seed: _json(RUN_DIR / f"seed_{seed}/result.json")
        for seed in (80314, 88625)
    }
    contrast_names = tuple(descriptive["comparisons"])

    primary_windows = []
    disagreement_primary_windows = []
    disagreement_order_matches = 0
    source_placebo_same_action = 0
    effective_outcomes = {
        "target_only": [],
        "uniform_source": [],
        "source_label_placebo": [],
    }
    for seed, result in seed_results.items():
        for name in contrast_names:
            expected_seed_mean = statistics.fmean(
                result["windows"][str(checkpoint)]["contrasts"][name]
                for checkpoint in checkpoints
            )
            stored = descriptive["comparisons"][name]["available_seed_means"][str(seed)]
            assert stored == pytest.approx(expected_seed_mean)
        for checkpoint in checkpoints:
            window = result["windows"][str(checkpoint)]
            primary = window["contrasts"]["uniform_source_minus_target_only"]
            primary_windows.append(primary)
            if window["focal"]["source_target_selected_actions_differ"]:
                disagreement_primary_windows.append(primary)
                predicted_source_benefit = (
                    window["learned_arms"]["uniform_source"][
                        "predicted_advantage_vs_phase_pressure"
                    ]
                    - window["learned_arms"]["target_only"][
                        "predicted_advantage_vs_phase_pressure"
                    ]
                )
                native_source_benefit = -primary
                disagreement_order_matches += (
                    (predicted_source_benefit > 0.0)
                    == (native_source_benefit > 0.0)
                )
            if (
                window["learned_arms"]["uniform_source"]["selected_action"]
                == window["learned_arms"]["source_label_placebo"]["selected_action"]
            ):
                source_placebo_same_action += 1
            for arm in effective_outcomes:
                arm_result = window["learned_arms"][arm]
                if arm_result["selected_action"] != window["focal"]["reference_action"]:
                    assert arm_result["predicted_advantage_vs_phase_pressure"] > 0.0
                    effective_outcomes[arm].append(
                        window["costs"][arm] < window["costs"]["phase_pressure"]
                    )

    for name, comparison in descriptive["comparisons"].items():
        assert comparison["mean_of_available_seed_means"] == pytest.approx(
            statistics.fmean(comparison["available_seed_means"].values())
        )
        assert comparison["inferential_status"] == "DESCRIPTIVE_ONLY"

    assert sum(value < 0 for value in primary_windows) == 2
    assert sum(value > 0 for value in primary_windows) == 2
    assert sum(value == 0 for value in primary_windows) == 2
    assert sum(value < 0 for value in disagreement_primary_windows) == 2
    assert sum(value > 0 for value in disagreement_primary_windows) == 2
    assert disagreement_order_matches == 1
    assert source_placebo_same_action == 5
    assert {arm: len(values) for arm, values in effective_outcomes.items()} == {
        "target_only": 5,
        "uniform_source": 5,
        "source_label_placebo": 5,
    }
    assert {arm: sum(values) for arm, values in effective_outcomes.items()} == {
        "target_only": 1,
        "uniform_source": 2,
        "source_label_placebo": 1,
    }


def test_failure_evidence_and_provenance_are_bound_to_inputs():
    aggregate = _json(AGGREGATE)
    failure = _json(FAILURE)

    assert failure["evidence_kind"] == "scheduler_failure_not_native_seed_result"
    assert failure["seed_result_json_written"] is False
    assert failure["full_trace_written"] is False
    assert failure["native_simulations_completed_across_attempts"] == 1
    assert [attempt["task_id"] for attempt in failure["attempts"]] == [
        "t92564",
        "t92565",
        "t92566",
    ]
    assert [attempt["status"] for attempt in failure["attempts"]] == [
        "failed",
        "failed",
        "cancelled",
    ]
    assert aggregate["failed_seed_evidence"]["sha256"] == _sha256(FAILURE)

    for record in aggregate["provenance"]["valid_seed_results"].values():
        assert _sha256(ROOT / record["path"]) == record["sha256"]
    for key in ("protocol_config", "derived_snapshot_v7", "launch_v7"):
        record = aggregate["provenance"][key]
        assert _sha256(ROOT / record["path"]) == record["sha256"]

    assert AGGREGATE.read_bytes() == PAPER_ARTIFACT.read_bytes()
    assert _sha256(AGGREGATE) == "cc2595f5a938309281bcb2aa32e262b1d7235f4f809b182c867cec7246e7a6fa"
    report = REPORT.read_text(encoding="utf-8")
    compact_report = " ".join(report.split())
    assert "**Status: INCOMPLETE.**" in report
    assert "was not moved, dropped, replaced, or counted as zero" in compact_report
    assert "rather than as an outcome" in compact_report
    assert _sha256(AGGREGATE) in report
