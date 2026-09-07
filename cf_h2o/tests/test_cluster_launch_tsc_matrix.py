from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_PATH = PROJECT_ROOT / "scripts/cluster/launch_tsc_v9_matrix.py"
SPEC = importlib.util.spec_from_file_location("launch_tsc_v9_matrix", LAUNCHER_PATH)
assert SPEC is not None and SPEC.loader is not None
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


def test_default_r16_spec_matches_executable_protocols() -> None:
    spec = json.loads(LAUNCHER.DEFAULT_SPEC.read_text(encoding="utf-8"))
    LAUNCHER._validate_spec(spec)
    assert spec["version"] == 5
    assert spec["development"]["primary_policy"] == "cfcmt_fused_contrast_guard"
    assert len(spec["development"]["policies"]) == 14


def test_r17_exact_ablation_spec_matches_executable_protocols() -> None:
    path = (
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v17_exact_mechanism_ablation_development.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))

    LAUNCHER._validate_spec(spec)

    assert spec["version"] == 6
    assert LAUNCHER._matrix_protocol_tag(spec) == "v21"
    assert (
        spec["method_acceptance"]["exact_mechanism_ablation"]
        == "cfcmt_fused_rigid_contrast_guard"
    )


def test_version_six_spec_rejects_missing_exact_ablation() -> None:
    path = (
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v17_exact_mechanism_ablation_development.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["development"]["policies"].remove(
        "cfcmt_fused_rigid_contrast_guard"
    )

    with pytest.raises(ValueError, match="method_acceptance policies are absent"):
        LAUNCHER._validate_spec(spec)


def test_r18_joint_gate_spec_separates_model_and_deployment_ablations() -> None:
    path = (
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v18_joint_mechanism_gate_development.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))

    LAUNCHER._validate_spec(spec)

    assert spec["version"] == 7
    assert LAUNCHER._matrix_protocol_tag(spec) == "v22"
    acceptance = spec["method_acceptance"]
    assert acceptance["mechanism_primary_policy"] == "cfcmt_fused_contrast_mpc"
    assert acceptance["exact_mechanism_model_ablation"] == (
        "cfcmt_fused_rigid_contrast_mpc"
    )
    assert acceptance["exact_mechanism_ablation"] == (
        "cfcmt_fused_rigid_contrast_guard"
    )


def test_r19_policy_consistent_spec_and_confirmatory_manifest_are_valid() -> None:
    spec_path = (
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v19_policy_consistent_estimand_development.json"
    )
    spec = json.loads(spec_path.read_text(encoding="utf-8"))

    LAUNCHER._validate_spec(spec)

    assert LAUNCHER._matrix_protocol_tag(spec) == "v25"
    development = spec["estimand_development"]
    assert development["target_group_budget"] == 0
    assert len(development["candidate_families"]) == 10
    assert development["stage_a_gate"] == {
        "minimum_macro_relative_improvement": 0.10,
        "minimum_improved_city_count": 4,
        "maximum_absolute_city_regression": 0.05,
    }
    manifest = load_traffic_signal_manifest(
        PROJECT_ROOT / development["confirmatory_manifest"]
    )
    assert manifest.version == 2
    assert len(manifest.scenarios) == 18
    assert len(set(manifest.city_groups.values())) == 7
    assert {
        name
        for name, group in manifest.city_groups.items()
        if group == "salt_lake_city"
    } == {
        "saltlake_400s_200w_q1_weekday_peak",
        "saltlake_state_university_q1_weekday_peak",
    }


def test_version_seven_spec_rejects_missing_model_ablation() -> None:
    path = (
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v18_joint_mechanism_gate_development.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))
    spec["development"]["policies"].remove(
        "cfcmt_fused_rigid_contrast_mpc"
    )

    with pytest.raises(ValueError, match="method_acceptance policies are absent"):
        LAUNCHER._validate_spec(spec)


def test_spec_rejects_primary_policy_outside_evaluation_matrix() -> None:
    spec = json.loads(LAUNCHER.DEFAULT_SPEC.read_text(encoding="utf-8"))
    spec["development"]["primary_policy"] = "not_a_policy"

    with pytest.raises(ValueError, match="primary_policy is absent"):
        LAUNCHER._validate_spec(spec)


def test_version_four_spec_requires_explicit_primary_policy() -> None:
    spec = json.loads(LAUNCHER.DEFAULT_SPEC.read_text(encoding="utf-8"))
    spec["version"] = 4
    spec["development"].pop("primary_policy")

    with pytest.raises(ValueError, match="require an explicit"):
        LAUNCHER._validate_spec(spec)


def test_development_command_passes_explicit_primary_policy(tmp_path: Path) -> None:
    spec = json.loads(LAUNCHER.DEFAULT_SPEC.read_text(encoding="utf-8"))
    primary = "cfcmt_mechanism_contrast_guard"
    spec["development"]["primary_policy"] = primary
    commands = LAUNCHER._development_specs(
        spec=spec,
        snapshot={"snapshot_root": "/snapshot"},
        scheduler=Path("/scheduler.py"),
        run_id="formal-test",
        remote_results=Path("/remote/results"),
        local_results=tmp_path / "results",
        cache_root=Path("/cache/counterfactual"),
        source_rule_cache_root=Path("/cache/source-rules"),
        selected_budgets=[0],
    )

    command = commands[0][commands[0].index("--cmd") + 1]
    assert f"--primary-reference-policy {primary}" in command


def test_r17_development_command_uses_distinct_protocol_signature(
    tmp_path: Path,
) -> None:
    path = (
        PROJECT_ROOT
        / "cf_h2o/config/traffic_signal_tsc_v17_exact_mechanism_ablation_development.json"
    )
    spec = json.loads(path.read_text(encoding="utf-8"))
    commands = LAUNCHER._development_specs(
        spec=spec,
        snapshot={"snapshot_root": "/snapshot"},
        scheduler=Path("/scheduler.py"),
        run_id="exact-ablation-test",
        remote_results=Path("/remote/results"),
        local_results=tmp_path / "results",
        cache_root=Path("/cache/counterfactual"),
        source_rule_cache_root=Path("/cache/source-rules"),
        selected_budgets=[0],
    )

    command = commands[0]
    assert command[command.index("--project") + 1] == "CFCMT-TSC-v21"
    assert (
        command[command.index("--signature") + 1]
        == "CFCMT/tsc-v21/exact-ablation-test/development/budget-000"
    )


def test_closed_loop_selection_contract_is_explicit_and_seed_disjoint(
    tmp_path: Path,
) -> None:
    spec = json.loads(LAUNCHER.DEFAULT_SPEC.read_text(encoding="utf-8"))
    development = spec["development"]
    development.update(
        {
            "target_closed_loop_selection_seeds": [5057, 6067],
            "target_closed_loop_selection_duration_sec": 600.0,
            "target_closed_loop_min_improvement": 0.001,
            "target_closed_loop_refinement_min_improvement": 0.001,
            "target_closed_loop_worst_tolerance": 0.005,
        }
    )
    LAUNCHER._validate_spec(spec)
    commands = LAUNCHER._development_specs(
        spec=spec,
        snapshot={"snapshot_root": "/snapshot"},
        scheduler=Path("/scheduler.py"),
        run_id="closed-loop-test",
        remote_results=Path("/remote/results"),
        local_results=tmp_path / "results",
        cache_root=Path("/cache/counterfactual"),
        source_rule_cache_root=Path("/cache/source-rules"),
        selected_budgets=[8],
    )
    command = commands[0][commands[0].index("--cmd") + 1]
    assert "--target-closed-loop-selection-seeds 5057 6067" in command
    assert "--target-closed-loop-selection-duration-sec 600.0" in command
    assert "--target-closed-loop-min-improvement 0.001" in command
    assert "--target-closed-loop-worst-tolerance 0.005" in command

    development["target_closed_loop_selection_seeds"] = [2027]
    with pytest.raises(ValueError, match="source/calibration seeds"):
        LAUNCHER._validate_spec(spec)


def test_safety_survey_is_partitioned_across_all_six_nodes(tmp_path: Path) -> None:
    spec = json.loads(LAUNCHER.DEFAULT_SPEC.read_text(encoding="utf-8"))
    commands = LAUNCHER._safety_survey_specs(
        spec=spec,
        snapshot={"snapshot_root": "/snapshot"},
        scheduler=Path("/scheduler.py"),
        run_id="safety-test",
        remote_results=Path("/remote/results"),
        local_results=tmp_path / "results",
    )

    assert len(commands) == 6
    first = commands[0]
    command = first[first.index("--cmd") + 1]
    assert "cf_h2o.eval.traffic_signal_safety_survey" in command
    assert "--policies fixed_program max_pressure phase_pressure spillback_pressure" in command
    assert first[first.index("--require-node") + 1] == "node001"


def test_parallel_resources_are_bounded_by_work_and_phase_limit() -> None:
    cluster = {
        "workers_per_job": 96,
        "cache_workers_per_job": 60,
        "cpu_cores_per_job": 100,
        "ram_mb_per_job": 65536,
        "min_ram_mb_per_job": 16384,
    }

    assert LAUNCHER._parallel_job_resources(
        cluster,
        worker_key="cache_workers_per_job",
        parallel_items=144,
    ) == (60, 64, 40960)
    assert LAUNCHER._parallel_job_resources(
        cluster,
        worker_key="cache_workers_per_job",
        parallel_items=48,
    ) == (48, 52, 32768)


def test_cache_resources_honor_per_node_worker_and_cpu_overrides() -> None:
    cluster = {
        "workers_per_job": 128,
        "cache_workers_per_job": 128,
        "cpu_cores_per_job": 160,
        "ram_mb_per_job": 131072,
        "min_ram_mb_per_job": 32768,
        "cache_resource_overrides": {
            "node003": {"workers_per_job": 112, "cpu_overhead": 8},
            "node006": {"workers_per_job": 44, "cpu_overhead": 8},
        },
    }

    assert LAUNCHER._cache_job_resources(
        cluster, node="node003", parallel_items=144
    ) == (112, 120, 114688)
    assert LAUNCHER._cache_job_resources(
        cluster, node="node006", parallel_items=192
    ) == (44, 52, 45056)
    assert LAUNCHER._cache_job_resources(
        cluster, node="node001", parallel_items=48
    ) == (48, 80, 49152)


def test_assignment_subset_is_strict() -> None:
    assignments = {"node001": ["a"], "node002": ["b"]}

    assert LAUNCHER._selected_assignment_names(assignments, ["node002"]) == {
        "node002"
    }
    with pytest.raises(ValueError, match="unknown cache assignments"):
        LAUNCHER._selected_assignment_names(assignments, ["node003"])


def test_development_requires_both_explicit_cache_roots() -> None:
    counterfactual = Path("/cache/counterfactual")
    source_rules = Path("/cache/source_rules")

    assert LAUNCHER._development_cache_roots(
        counterfactual_cache_root=counterfactual,
        source_rule_cache_root=source_rules,
    ) == (counterfactual, source_rules)
    with pytest.raises(ValueError, match="--counterfactual-cache-root"):
        LAUNCHER._development_cache_roots(
            counterfactual_cache_root=None,
            source_rule_cache_root=source_rules,
        )
    with pytest.raises(ValueError, match="--source-rule-cache-root"):
        LAUNCHER._development_cache_roots(
            counterfactual_cache_root=counterfactual,
            source_rule_cache_root=None,
        )


def test_cache_specs_can_reuse_an_explicit_content_addressed_root(
    tmp_path: Path,
) -> None:
    spec = json.loads(LAUNCHER.DEFAULT_SPEC.read_text(encoding="utf-8"))
    cache_root = Path("/shared/cache/recovery")
    commands = LAUNCHER._cache_specs(
        spec=spec,
        snapshot={"snapshot_root": "/snapshot"},
        scheduler=Path("/scheduler.py"),
        run_id="recovery-test",
        remote_results=Path("/remote/results"),
        local_results=tmp_path / "results",
        cache_root=cache_root,
        selected_assignments=["node004"],
    )

    assert len(commands) == 1
    command = commands[0][commands[0].index("--cmd") + 1]
    assert f"--cache-root {cache_root}" in command


def test_development_worker_shape_matches_target_process_parallelism() -> None:
    cluster = {
        "workers_per_job": 96,
        "development_workers_per_job": 48,
        "cpu_cores_per_job": 100,
        "ram_mb_per_job": 65536,
        "min_ram_mb_per_job": 16384,
    }

    assert LAUNCHER._parallel_job_resources(
        cluster,
        worker_key="development_workers_per_job",
        parallel_items=48,
    ) == (48, 52, 32768)
