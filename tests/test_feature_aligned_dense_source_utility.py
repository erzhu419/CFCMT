from __future__ import annotations

from copy import deepcopy

import pytest

from cf_h2o.eval import traffic_signal_feature_aligned_dense_source_utility as subject
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)


def _target(city: str, protocol: str, *, admitted: bool = False) -> dict:
    arms = {
        "rigid_target_only": {"mean": 1.0},
        "selected_dense_source_pairwise": {"mean": 0.9 if admitted else 1.0},
        "selected_dense_matched_placebo": {"mean": 0.95 if admitted else 1.0},
        "selected_dense_source_blind": {"mean": 0.96 if admitted else 1.0},
    }
    return {
        "protocol": protocol,
        "target_city": city,
        "target_scenarios": [f"{city}_scenario"],
        "source_city_groups": ["source"],
        "target_budget": 100,
        "estimand": "estimand",
        "selection_audit": {"selected_group_ids": [f"{city}:a"]},
        "evaluation_reserve_audit": {"selected_group_ids": [f"{city}:b"]},
        "information_budget": {"target_adaptation_group_count": 100},
        "inputs": {
            "cache": "fixed",
            "source_manifest": (
                "/old/snapshot/"
                "traffic_signal_cross_city_v2_saltlake18.json"
            ),
        },
        "input_audits": {
            "source_bank": {
                "cache_root": subject.EXPECTED_SOURCE_CACHE_ROOT,
                "used_file_count": 960,
                "seeds": [2027, 3037, 4047],
            }
        },
        "arm_summaries": arms,
        "method": {"selector": {"admitted": admitted}},
    }


def _provenance() -> dict:
    return {
        "protocol": subject.LAUNCH_PROTOCOL,
        "submitted": True,
        "target_cities": list(EXPECTED_CITY_GROUPS),
        "snapshot_sha256": subject.DERIVED_SNAPSHOT_SHA256,
        "snapshot_root": subject.DERIVED_SNAPSHOT_ROOT,
        "source_tree_sha256": subject.DERIVED_SOURCE_TREE_SHA256,
        "snapshot_marker_verified": True,
        "derived_from_snapshot_sha256": subject.PARENT_SNAPSHOT_SHA256,
        "snapshot_patch": {
            "path": "cf_h2o/traffic_signal/action_ranker.py",
            "parent_sha256": subject.ACTION_RANKER_PARENT_SHA256,
            "patched_sha256": subject.ACTION_RANKER_PATCHED_SHA256,
        },
        "input_sha256": deepcopy(subject.EXPECTED_INPUT_SHA256),
        "snapshot_file_sha256": {
            "source_manifest": subject.SOURCE_MANIFEST_SHA256,
            "fit_protocol": subject.FIT_PROTOCOL_SHA256,
            "action_ranker": subject.ACTION_RANKER_PATCHED_SHA256,
            "snapshot_marker": subject.DERIVED_MARKER_SHA256,
        },
        "execution_module": (
            "cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility"
        ),
        "source_cache_root": subject.EXPECTED_SOURCE_CACHE_ROOT,
        "conversion_root": subject.EXPECTED_CONVERSION_ROOT,
        "task_specs": [
            {
                "local_result_dir": f"/tmp/v156a/{city}",
                "signature": (
                    "CFCMT/v156a/feature-aligned-dense-source-utility-replay-v1/"
                    f"{city}"
                ),
            }
            for city in EXPECTED_CITY_GROUPS
        ],
        "submitted_tasks": [
            {
                "id": f"task-{city}",
                "signature": (
                    "CFCMT/v156a/feature-aligned-dense-source-utility-replay-v1/"
                    f"{city}"
                ),
            }
            for city in EXPECTED_CITY_GROUPS
        ],
    }


def _legacy_results() -> list[dict]:
    return [
        _target(city, subject.V150O_RESULT_PROTOCOL, admitted=city == "new_york")
        for city in EXPECTED_CITY_GROUPS
    ]


def _corrected_results(legacy: list[dict]) -> list[dict]:
    corrected = [subject.annotate_target_result(row) for row in legacy]
    for row in corrected:
        row["method"]["selector"]["admitted"] = False
        for arm in row["arm_summaries"].values():
            arm["mean"] = 1.0
    return corrected


def _stub_parent_aggregate(monkeypatch, *, passed: bool = False) -> None:
    monkeypatch.setattr(
        subject,
        "_aggregate_v150o_results",
        lambda rows: {
            "protocol": subject.V150O_AGGREGATE_PROTOCOL,
            "development_gate": {
                "passed": passed,
                "decision": (
                    "authorize_v150o_dense_pairwise_closed_loop_development"
                    if passed
                    else "retain_rigid_target_adaptation_and_reject_v150o_source_claim"
                ),
            },
        },
    )


def test_annotation_records_correction_only_contract() -> None:
    parent = _target("new_york", subject.V150O_RESULT_PROTOCOL, admitted=True)

    result = subject.annotate_target_result(parent)

    assert result["protocol"] == subject.RESULT_PROTOCOL
    assert result["correction_only_contract"] == {
        "parent_protocol": subject.V150O_RESULT_PROTOCOL,
        "feature_binding": "stored_feature_names",
        "parent_snapshot_sha256": subject.PARENT_SNAPSHOT_SHA256,
        "derived_snapshot_sha256": subject.DERIVED_SNAPSHOT_SHA256,
        "derived_source_tree_sha256": subject.DERIVED_SOURCE_TREE_SHA256,
        "action_ranker_parent_sha256": subject.ACTION_RANKER_PARENT_SHA256,
        "action_ranker_patched_sha256": subject.ACTION_RANKER_PATCHED_SHA256,
        "source_manifest_sha256": subject.SOURCE_MANIFEST_SHA256,
        "fit_protocol_sha256": subject.FIT_PROTOCOL_SHA256,
        "target_counterfactual_costs_reused": True,
        "source_counterfactual_costs_reused": True,
        "target_budget_unchanged": True,
        "folds_and_thresholds_unchanged": True,
        "utility_records_recomputed": True,
        "new_simulations": 0,
    }


def test_annotation_requires_raw_v150o_parent_protocol() -> None:
    row = _target("new_york", subject.RESULT_PROTOCOL)
    with pytest.raises(ValueError, match="parent result protocol changed"):
        subject.annotate_target_result(row)


@pytest.mark.parametrize(
    ("passed", "expected_decision"),
    [
        (True, "authorize_v156a_global_development_followup"),
        (False, "reject_v156a_global_source_selector"),
    ],
)
def test_aggregate_renames_parent_decision(
    monkeypatch, passed: bool, expected_decision: str
) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    _stub_parent_aggregate(monkeypatch, passed=passed)

    result = subject.aggregate_results(corrected, legacy, _provenance())

    assert result["protocol"] == subject.AGGREGATE_PROTOCOL
    assert result["development_gate"]["decision"] == expected_decision
    assert "v150o" not in result["development_gate"]["decision"]


def test_aggregate_accepts_raw_replay_and_records_fixed_invariants(monkeypatch) -> None:
    legacy = _legacy_results()
    corrected = [deepcopy(row) for row in legacy]
    for row in corrected:
        row["method"]["selector"]["admitted"] = False
        for arm in row["arm_summaries"].values():
            arm["mean"] = 1.0
    _stub_parent_aggregate(monkeypatch)

    result = subject.aggregate_results(corrected, legacy, _provenance())

    assert result["correction_only_contract"][
        "all_city_input_and_split_invariants_passed"
    ]
    assert result["legacy_v150o_comparison"]["new_york"]["legacy_admitted"]
    assert not result["legacy_v150o_comparison"]["new_york"][
        "corrected_admitted"
    ]


def test_aggregate_allows_snapshot_path_change_with_same_manifest_identity(
    monkeypatch,
) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    for row in corrected:
        row["inputs"]["source_manifest"] = (
            "/new/snapshot/traffic_signal_cross_city_v2_saltlake18.json"
        )
    _stub_parent_aggregate(monkeypatch)

    result = subject.aggregate_results(corrected, legacy, _provenance())

    assert result["correction_only_contract"][
        "all_city_input_and_split_invariants_passed"
    ]


@pytest.mark.parametrize("field", ["source_manifest", "fit_protocol"])
def test_aggregate_rejects_snapshot_file_identity_drift(
    monkeypatch, field: str
) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    _stub_parent_aggregate(monkeypatch)
    provenance = _provenance()
    provenance["snapshot_file_sha256"][field] = "0" * 64

    with pytest.raises(ValueError, match="replay provenance changed"):
        subject.aggregate_results(corrected, legacy, provenance)


def test_aggregate_rejects_fit_input_identity_drift(monkeypatch) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    _stub_parent_aggregate(monkeypatch)
    provenance = _provenance()
    provenance["input_sha256"]["fit_result"] = "0" * 64

    with pytest.raises(ValueError, match="replay provenance changed"):
        subject.aggregate_results(corrected, legacy, provenance)


def test_aggregate_rejects_data_split_drift(monkeypatch) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    corrected[0]["selection_audit"] = {"selected_group_ids": ["changed"]}
    _stub_parent_aggregate(monkeypatch)

    with pytest.raises(ValueError, match="changed more than feature binding"):
        subject.aggregate_results(corrected, legacy, _provenance())


def test_aggregate_rejects_source_bank_drift(monkeypatch) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    corrected[0]["input_audits"]["source_bank"]["used_file_count"] = 959
    _stub_parent_aggregate(monkeypatch)

    with pytest.raises(ValueError, match="changed more than feature binding"):
        subject.aggregate_results(corrected, legacy, _provenance())


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("feature_binding", "positional_indices"),
        ("derived_snapshot_sha256", "0" * 64),
        ("new_simulations", 1),
    ],
)
def test_aggregate_rejects_invalid_correction_contract(
    monkeypatch, field: str, changed_value: object
) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    corrected[0]["correction_only_contract"][field] = changed_value
    _stub_parent_aggregate(monkeypatch)

    with pytest.raises(ValueError, match="correction-only contract"):
        subject.aggregate_results(corrected, legacy, _provenance())


def test_aggregate_rejects_missing_correction_contract(monkeypatch) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    corrected[0].pop("correction_only_contract")
    _stub_parent_aggregate(monkeypatch)

    with pytest.raises(ValueError, match="correction-only contract"):
        subject.aggregate_results(corrected, legacy, _provenance())


def test_city_specific_candidate_does_not_override_failed_global_gate(
    monkeypatch,
) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    new_york = next(row for row in corrected if row["target_city"] == "new_york")
    new_york["method"]["selector"]["admitted"] = True
    new_york["arm_summaries"]["selected_dense_source_pairwise"]["mean"] = 0.90
    new_york["arm_summaries"]["rigid_target_only"]["mean"] = 1.00
    new_york["arm_summaries"]["selected_dense_matched_placebo"]["mean"] = 0.95
    new_york["arm_summaries"]["selected_dense_source_blind"]["mean"] = 0.96
    _stub_parent_aggregate(monkeypatch, passed=False)

    result = subject.aggregate_results(corrected, legacy, _provenance())

    assert result["development_gate"]["passed"] is False
    assert result["city_specific_followup_gate"] == {
        "requirements": {
            "crossfitted_selector_admission": True,
            "improves_rigid_target_only": True,
            "beats_matched_placebo": True,
            "beats_source_blind": True,
        },
        "candidates": ["new_york"],
        "passed": True,
        "decision": "authorize_untouched_seed_city_specific_closed_loop",
        "does_not_override_global_gate": True,
    }


@pytest.mark.parametrize(
    "failed_requirement",
    ["admission", "rigid", "matched_placebo", "source_blind"],
)
def test_city_specific_candidate_requires_every_condition(
    monkeypatch, failed_requirement: str
) -> None:
    legacy = _legacy_results()
    corrected = _corrected_results(legacy)
    new_york = next(row for row in corrected if row["target_city"] == "new_york")
    new_york["method"]["selector"]["admitted"] = failed_requirement != "admission"
    new_york["arm_summaries"]["selected_dense_source_pairwise"]["mean"] = 0.90
    new_york["arm_summaries"]["rigid_target_only"]["mean"] = (
        0.90 if failed_requirement == "rigid" else 1.00
    )
    new_york["arm_summaries"]["selected_dense_matched_placebo"]["mean"] = (
        0.90 if failed_requirement == "matched_placebo" else 0.95
    )
    new_york["arm_summaries"]["selected_dense_source_blind"]["mean"] = (
        0.90 if failed_requirement == "source_blind" else 0.96
    )
    _stub_parent_aggregate(monkeypatch, passed=False)

    result = subject.aggregate_results(corrected, legacy, _provenance())

    assert result["city_specific_followup_gate"]["candidates"] == []
    assert result["city_specific_followup_gate"]["passed"] is False
    assert (
        result["city_specific_followup_gate"]["decision"]
        == "stop_dense_source_selector_family"
    )


def test_cli_rejects_result_path_outside_launch_task_bindings(
    monkeypatch, tmp_path
) -> None:
    launch_path = tmp_path / "launch.json"
    result_paths = [
        tmp_path / "target_v2" / city / "result.json"
        for city in EXPECTED_CITY_GROUPS
    ]
    legacy_paths = [
        tmp_path / "legacy" / city / "result.json"
        for city in EXPECTED_CITY_GROUPS
    ]
    provenance = _provenance()
    provenance["task_specs"] = [
        {
            "local_result_dir": str(path.parent),
            "signature": (
                "CFCMT/v156a/feature-aligned-dense-source-utility-replay-v1/"
                f"{city}"
            ),
        }
        for city, path in zip(EXPECTED_CITY_GROUPS, result_paths, strict=True)
    ]
    wrong_paths = list(result_paths)
    wrong_paths[0] = tmp_path / "unbound" / "atlanta" / "result.json"

    monkeypatch.setattr(
        subject,
        "_read_json",
        lambda path: provenance if path == launch_path else {},
    )
    monkeypatch.setattr(
        subject,
        "aggregate_results",
        lambda *args, **kwargs: pytest.fail(
            "aggregate must not run with an unbound result path"
        ),
    )

    argv = [
        "--results",
        *(str(path) for path in wrong_paths),
        "--legacy-results",
        *(str(path) for path in legacy_paths),
        "--launch-manifest",
        str(launch_path),
        "--out",
        str(tmp_path / "aggregate.json"),
    ]
    with pytest.raises(ValueError, match="result paths"):
        subject.main(argv)
