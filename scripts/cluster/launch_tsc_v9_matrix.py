#!/usr/bin/env python3
"""Submit audited CFCMT TSC preflight, safety, cache, rule, and model jobs."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULER = Path("/home/erzhu419/mine_code/scheduleurm/skill/scheduler.py")
DEFAULT_SPEC = (
    PROJECT_ROOT
    / "cf_h2o/config/traffic_signal_tsc_v16_benchmark_aware_development.json"
)
MATRIX_PROTOCOL_TAG = "v20"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _matrix_protocol_tag(spec: dict[str, Any]) -> str:
    tag = str(spec.get("matrix_protocol_tag", MATRIX_PROTOCOL_TAG)).strip()
    if re.fullmatch(r"v[0-9]+", tag) is None:
        raise ValueError("matrix_protocol_tag must match v<integer>")
    return tag


def _current_counterfactual_cache_version() -> str:
    sys.path.insert(0, str(PROJECT_ROOT))
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
        COUNTERFACTUAL_CACHE_VERSION,
    )

    return str(COUNTERFACTUAL_CACHE_VERSION)


def _validate_spec(spec: dict[str, Any]) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
        COUNTERFACTUAL_CACHE_VERSION,
        SOURCE_RULE_CACHE_VERSION,
        SUPPORTED_COUNTERFACTUAL_CACHE_VERSIONS,
    )
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
        CFCMT_HIERARCHY_PROTOCOL_V3,
    )
    from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
        SUMO_EXECUTION_PROTOCOL,
    )
    from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest

    spec_version = int(spec.get("version", 0))
    if spec_version not in {1, 2, 3, 4, 5, 6, 7}:
        raise ValueError("unsupported TSC matrix specification version")
    protocol_tag = _matrix_protocol_tag(spec)
    if spec_version >= 6 and protocol_tag == MATRIX_PROTOCOL_TAG:
        raise ValueError(
            "version 6+ matrices require an explicit non-default matrix_protocol_tag"
        )
    if spec.get("counterfactual_cache_version") not in (
        SUPPORTED_COUNTERFACTUAL_CACHE_VERSIONS
    ):
        raise ValueError("matrix cache version is unsupported by executable code")
    if spec.get("source_rule_cache_version") != SOURCE_RULE_CACHE_VERSION:
        raise ValueError("matrix source-rule cache version does not match executable code")
    if spec.get("sumo_execution_protocol") != SUMO_EXECUTION_PROTOCOL:
        raise ValueError("matrix SUMO execution protocol does not match executable code")
    hierarchy_protocol = spec.get("cfcmt_hierarchy_protocol")
    if spec_version >= 3 and hierarchy_protocol != (
        CFCMT_HIERARCHY_PROTOCOL_V3
    ):
        raise ValueError("matrix CFCMT hierarchy protocol does not match executable code")
    if spec_version >= 5:
        expected_safety = {
            "raw_junction_collisions": "audit_events_and_unique_incidents",
            "starting_and_ending_teleports": "hard_failure",
            "counterfactual_collision_handling": (
                "censor_entire_matched_action_group_on_any_branch_event"
            ),
            "counterfactual_collision_deduplication": (
                "scenario_seed_time_vehicle_pair_type_lane_sha256"
            ),
            "deployment_primary_collision_admission": (
                "paired_total_incidents_no_worse_than_selected_source_prior"
            ),
            "deployment_collision_margin_incidents": 0,
            "synthetic_extension_admission": (
                "zero_raw_collision_and_zero_teleport"
            ),
        }
        if spec.get("safety") != expected_safety:
            raise ValueError("matrix benchmark-aware safety protocol mismatch")
    manifest = load_traffic_signal_manifest(PROJECT_ROOT / spec["manifest"])
    nodes = list(spec["cluster"]["nodes"])
    assignments = spec["cache_assignments"]
    if set(assignments) != set(nodes):
        raise ValueError("cache assignments must cover every configured node exactly once")
    assigned = [scenario for node in nodes for scenario in assignments[node]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("cache assignments contain duplicate scenarios")
    if set(assigned) != set(manifest.sumocfgs):
        raise ValueError("cache assignments do not exactly cover the benchmark manifest")

    counterfactual = spec["counterfactual"]
    development = spec["development"]
    source_seeds = set(int(seed) for seed in counterfactual["source_seeds"])
    evaluation_seeds = set(int(seed) for seed in development["evaluation_seeds"])
    if source_seeds & evaluation_seeds:
        raise ValueError("source and evaluation seeds must be disjoint")
    calibration_seeds = set(int(seed) for seed in development["target_calibration_seeds"])
    if not calibration_seeds or not calibration_seeds <= source_seeds:
        raise ValueError("target calibration seeds must be a non-empty source-seed subset")
    closed_loop_seeds = set(
        int(seed) for seed in development.get("target_closed_loop_selection_seeds", [])
    )
    if closed_loop_seeds & evaluation_seeds:
        raise ValueError("closed-loop selection and evaluation seeds must be disjoint")
    if closed_loop_seeds & source_seeds:
        raise ValueError("closed-loop selection and source/calibration seeds must be disjoint")
    if len(closed_loop_seeds) != len(
        development.get("target_closed_loop_selection_seeds", [])
    ):
        raise ValueError("closed-loop selection seeds contain duplicates")
    if closed_loop_seeds:
        closed_loop_duration = float(
            development.get("target_closed_loop_selection_duration_sec", 0.0)
        )
        if closed_loop_duration <= float(counterfactual["warmup_sec"]):
            raise ValueError(
                "closed-loop selection duration must exceed the warmup"
            )
        for key in (
            "target_closed_loop_min_improvement",
            "target_closed_loop_refinement_min_improvement",
            "target_closed_loop_worst_tolerance",
        ):
            if key not in development or float(development[key]) < 0.0:
                raise ValueError(
                    f"closed-loop selection requires a nonnegative {key}"
                )
    eligible_intervals = int(
        (float(counterfactual["source_duration_sec"]) - float(counterfactual["warmup_sec"]))
        // int(counterfactual["control_interval_sec"])
    )
    if not 1 <= int(counterfactual["collection_shards"]) <= eligible_intervals:
        raise ValueError("collection shard count exceeds eligible control intervals")
    budgets = [int(value) for value in development["target_group_budgets"]]
    if budgets != sorted(set(budgets)) or not budgets or budgets[0] != 0:
        raise ValueError("target budgets must be unique, sorted, and start at zero")
    policies = list(development["policies"])
    if len(policies) != len(set(policies)):
        raise ValueError("development policy list contains duplicates")
    primary_policy = development.get("primary_policy")
    if spec_version >= 4 and primary_policy is None:
        raise ValueError("version 4+ matrices require an explicit development primary_policy")
    if primary_policy is not None:
        primary_policy = str(primary_policy).strip()
        if not primary_policy:
            raise ValueError("development primary_policy must be non-empty")
        if primary_policy not in policies:
            raise ValueError("development primary_policy is absent from policies")
    if spec_version >= 6:
        acceptance = spec.get("method_acceptance")
        if not isinstance(acceptance, dict):
            raise ValueError("version 6+ matrices require method_acceptance")
        required_acceptance_policies = {
            "primary_policy",
            "selected_source_prior",
            "target_only_comparator",
            "exact_mechanism_ablation",
        }
        if spec_version >= 7:
            required_acceptance_policies.update(
                {
                    "mechanism_primary_policy",
                    "mechanism_target_only_comparator",
                    "exact_mechanism_model_ablation",
                }
            )
        missing_acceptance = required_acceptance_policies - set(acceptance)
        if missing_acceptance:
            raise ValueError(
                "method_acceptance is missing policy keys: "
                + ", ".join(sorted(missing_acceptance))
            )
        if str(acceptance["primary_policy"]) != primary_policy:
            raise ValueError(
                "method_acceptance primary_policy must match development primary_policy"
            )
        missing_policies = {
            str(acceptance[key]) for key in required_acceptance_policies
        } - set(policies)
        if missing_policies:
            raise ValueError(
                "method_acceptance policies are absent from evaluation matrix: "
                + ", ".join(sorted(missing_policies))
            )
    city_group_count = len(set(manifest.city_groups.values()))
    selector_min_domains = int(
        development.get("source_selector_min_context_domains", 5)
    )
    if not 2 <= selector_min_domains <= city_group_count - 1:
        raise ValueError(
            "source context selector threshold must be feasible in every held-out fold"
        )
    if development["source_rule_grid"] not in {
        "generalized",
        "legacy",
        "fixed_max",
        "fixed_phase",
    }:
        raise ValueError("unknown source pressure rule grid")
    cluster = spec["cluster"]
    workers = int(cluster["workers_per_job"])
    cpu_cores = int(cluster["cpu_cores_per_job"])
    if not 1 <= workers <= cpu_cores <= 192:
        raise ValueError("cluster worker and CPU reservations are inconsistent")
    for key in (
        "safety_survey_workers_per_job",
        "cache_workers_per_job",
        "source_rule_workers_per_job",
        "development_workers_per_job",
    ):
        phase_workers = int(cluster.get(key, workers))
        if not 1 <= phase_workers <= workers:
            raise ValueError(f"{key} must be in [1, workers_per_job]")
    cache_overrides = cluster.get("cache_resource_overrides", {})
    if not isinstance(cache_overrides, dict):
        raise ValueError("cache_resource_overrides must be an object")
    unknown_override_nodes = set(cache_overrides) - set(nodes)
    if unknown_override_nodes:
        raise ValueError(
            "cache resource overrides contain unknown nodes: "
            + ", ".join(sorted(unknown_override_nodes))
        )
    for node, override in cache_overrides.items():
        if not isinstance(override, dict) or set(override) != {
            "workers_per_job",
            "cpu_overhead",
        }:
            raise ValueError(
                f"cache resource override for {node} must contain exactly "
                "workers_per_job and cpu_overhead"
            )
        node_workers = int(override["workers_per_job"])
        node_overhead = int(override["cpu_overhead"])
        if not 1 <= node_workers <= int(cluster["cache_workers_per_job"]):
            raise ValueError(f"invalid cache worker override for {node}")
        if node_overhead < 0 or node_workers + node_overhead > 192:
            raise ValueError(f"invalid cache CPU overhead for {node}")


def _selected_assignment_names(
    assignments: dict[str, list[str]],
    selected: Sequence[str] | None,
) -> set[str]:
    requested = set(str(value) for value in (selected or assignments))
    unknown = requested - set(assignments)
    if unknown:
        raise ValueError(f"unknown cache assignments: {sorted(unknown)}")
    return requested


def _development_cache_roots(
    *,
    counterfactual_cache_root: Path | None,
    source_rule_cache_root: Path | None,
) -> tuple[Path, Path]:
    """Require deliberate, read-only cache inputs for model/evaluation jobs."""

    missing = []
    if counterfactual_cache_root is None:
        missing.append("--counterfactual-cache-root")
    if source_rule_cache_root is None:
        missing.append("--source-rule-cache-root")
    if missing:
        raise ValueError(
            "development requires explicit audited cache roots: "
            + ", ".join(missing)
        )
    return Path(counterfactual_cache_root), Path(source_rule_cache_root)


def _parallel_job_resources(
    cluster: dict[str, Any],
    *,
    worker_key: str,
    parallel_items: int,
) -> tuple[int, int, int]:
    if parallel_items < 1:
        raise ValueError("parallel item count must be positive")
    baseline_workers = int(cluster["workers_per_job"])
    worker_limit = int(cluster.get(worker_key, baseline_workers))
    workers = min(worker_limit, int(parallel_items))
    cpu_overhead = max(
        0,
        int(cluster["cpu_cores_per_job"]) - baseline_workers,
    )
    cpu_cores = workers + cpu_overhead
    scaled_ram = math.ceil(
        int(cluster["ram_mb_per_job"]) * workers / baseline_workers / 1024.0
    ) * 1024
    ram_mb = max(int(cluster.get("min_ram_mb_per_job", 8192)), scaled_ram)
    return workers, cpu_cores, ram_mb


def _cache_job_resources(
    cluster: dict[str, Any],
    *,
    node: str,
    parallel_items: int,
) -> tuple[int, int, int]:
    """Resolve an audited per-node cache limit without oversubscribing workers."""

    override = cluster.get("cache_resource_overrides", {}).get(node)
    if override is None:
        return _parallel_job_resources(
            cluster,
            worker_key="cache_workers_per_job",
            parallel_items=parallel_items,
        )
    baseline_workers = int(cluster["workers_per_job"])
    workers = min(int(override["workers_per_job"]), int(parallel_items))
    cpu_cores = workers + int(override["cpu_overhead"])
    scaled_ram = math.ceil(
        int(cluster["ram_mb_per_job"]) * workers / baseline_workers / 1024.0
    ) * 1024
    ram_mb = max(int(cluster.get("min_ram_mb_per_job", 8192)), scaled_ram)
    return workers, cpu_cores, ram_mb


def _module_command(wrapper: Path, module: str, arguments: Iterable[object]) -> str:
    return shlex.join([str(wrapper), "-m", module, *(str(value) for value in arguments)])


def _base_submit(
    *,
    scheduler: Path,
    protocol_tag: str,
    description: str,
    command: str,
    signature: str,
    node: str | None,
    allowed_nodes: Sequence[str] | None = None,
    cpu: int,
    ram_mb: int,
    remote_result_dir: Path,
    local_result_dir: Path,
    source_root: Path,
) -> list[str]:
    if bool(node) == bool(allowed_nodes):
        raise ValueError("submit placement requires exactly one of node or allowed_nodes")
    command = [
        sys.executable,
        str(scheduler),
        "submit",
        "--description",
        description,
        "--cmd",
        command,
        "--cwd",
        str(PROJECT_ROOT),
        "--signature",
        signature,
        "--project",
        f"CFCMT-TSC-{protocol_tag}",
        "--vram",
        "0",
        "--cpu",
        str(cpu),
        "--ram-mb",
        str(ram_mb),
        "--result-dir",
        str(remote_result_dir),
        "--local-result-dir",
        str(local_result_dir),
        "--skip-launch-staging",
        "--env",
        f"CFCMT_SOURCE_ROOT={source_root}",
    ]
    if node:
        command.extend(["--require-node", node])
    else:
        for allowed_node in allowed_nodes or ():
            command.extend(["--allowed-node", str(allowed_node)])
    return command


def _preflight_specs(
    *,
    spec: dict[str, Any],
    snapshot: dict[str, Any],
    scheduler: Path,
    run_id: str,
    remote_results: Path,
    local_results: Path,
) -> list[list[str]]:
    protocol_tag = _matrix_protocol_tag(spec)
    source_root = Path(snapshot["snapshot_root"])
    wrapper = source_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = source_root / spec["manifest"]
    result = []
    for node in spec["cluster"]["nodes"]:
        remote_dir = remote_results / "preflight" / node
        command = _module_command(
            wrapper,
            "cf_h2o.eval.traffic_signal_cluster_preflight",
            (
                "--manifest",
                manifest,
                "--expected-source-sha256",
                snapshot["source_tree_sha256"],
                "--expected-sumo-version",
                spec["expected_sumo_version"],
                "--snapshot-manifest",
                source_root / ".cfcmt_snapshot.json",
                "--expected-snapshot-sha256",
                snapshot["snapshot_sha256"],
                "--out",
                remote_dir / "preflight.json",
            ),
        )
        result.append(
            _base_submit(
                scheduler=scheduler,
                protocol_tag=protocol_tag,
                description=f"CFCMT TSC {protocol_tag} preflight {node}",
                command=command,
                signature=f"CFCMT/tsc-{protocol_tag}/{run_id}/preflight/{node}",
                node=node,
                cpu=2,
                ram_mb=4096,
                remote_result_dir=remote_dir,
                local_result_dir=local_results / "preflight" / node,
                source_root=source_root,
            )
        )
    return result


def _safety_survey_specs(
    *,
    spec: dict[str, Any],
    snapshot: dict[str, Any],
    scheduler: Path,
    run_id: str,
    remote_results: Path,
    local_results: Path,
    pool_placement: bool = False,
    selected_assignments: Sequence[str] | None = None,
) -> list[list[str]]:
    protocol_tag = _matrix_protocol_tag(spec)
    source_root = Path(snapshot["snapshot_root"])
    wrapper = source_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = source_root / spec["manifest"]
    survey = spec["safety_survey"]
    counterfactual = spec["counterfactual"]
    cluster = spec["cluster"]
    requested = _selected_assignment_names(
        spec["cache_assignments"], selected_assignments
    )
    result = []
    for node, scenarios in spec["cache_assignments"].items():
        if node not in requested:
            continue
        parallel_items = (
            len(scenarios) * len(survey["seeds"]) * len(survey["policies"])
        )
        workers, cpu_cores, ram_mb = _parallel_job_resources(
            cluster,
            worker_key="safety_survey_workers_per_job",
            parallel_items=parallel_items,
        )
        remote_dir = remote_results / "safety_survey" / node
        arguments: list[object] = [
            "--manifest",
            manifest,
            "--scenarios",
            *scenarios,
            "--policies",
            *survey["policies"],
            "--seeds",
            *survey["seeds"],
            "--duration-sec",
            survey["duration_sec"],
            "--control-interval-sec",
            counterfactual["control_interval_sec"],
            "--warmup-sec",
            counterfactual["warmup_sec"],
            "--workers",
            workers,
            "--tripinfo-root",
            remote_dir / "tripinfo",
            "--expected-sumo-version",
            spec["expected_sumo_version"],
            "--out",
            remote_dir / "safety_survey.json",
        ]
        command = _module_command(
            wrapper,
            "cf_h2o.eval.traffic_signal_safety_survey",
            arguments,
        )
        result.append(
            _base_submit(
                scheduler=scheduler,
                protocol_tag=protocol_tag,
                description=f"CFCMT TSC {protocol_tag} safety survey {node}",
                command=command,
                signature=f"CFCMT/tsc-{protocol_tag}/{run_id}/safety/{node}",
                node=None if pool_placement else node,
                allowed_nodes=cluster["nodes"] if pool_placement else None,
                cpu=cpu_cores,
                ram_mb=ram_mb,
                remote_result_dir=remote_dir,
                local_result_dir=local_results / "safety_survey" / node,
                source_root=source_root,
            )
        )
    return result


def _cache_specs(
    *,
    spec: dict[str, Any],
    snapshot: dict[str, Any],
    scheduler: Path,
    run_id: str,
    remote_results: Path,
    local_results: Path,
    cache_root: Path,
    pool_placement: bool = False,
    selected_assignments: Sequence[str] | None = None,
) -> list[list[str]]:
    protocol_tag = _matrix_protocol_tag(spec)
    source_root = Path(snapshot["snapshot_root"])
    wrapper = source_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = source_root / spec["manifest"]
    counterfactual = spec["counterfactual"]
    cluster = spec["cluster"]
    requested = _selected_assignment_names(
        spec["cache_assignments"], selected_assignments
    )
    result = []
    for node, scenarios in spec["cache_assignments"].items():
        if node not in requested:
            continue
        parallel_items = (
            len(scenarios)
            * len(counterfactual["source_seeds"])
            * int(counterfactual["collection_shards"])
        )
        workers, cpu_cores, ram_mb = _cache_job_resources(
            cluster,
            node=node,
            parallel_items=parallel_items,
        )
        remote_dir = remote_results / "cache" / node
        arguments: list[object] = [
            "--manifest",
            manifest,
            "--scenarios",
            *scenarios,
            "--seeds",
            *counterfactual["source_seeds"],
            "--duration-sec",
            counterfactual["source_duration_sec"],
            "--control-interval-sec",
            counterfactual["control_interval_sec"],
            "--warmup-sec",
            counterfactual["warmup_sec"],
            "--max-focal-tls",
            counterfactual["max_focal_tls"],
            "--counterfactual-horizon-intervals",
            counterfactual["horizon_intervals"],
            "--collection-shards",
            counterfactual["collection_shards"],
            "--workers",
            workers,
            "--cache-root",
            cache_root,
            "--behavior-policy",
            counterfactual["behavior_policy"],
            "--min-tls-coverage",
            counterfactual["min_tls_coverage"],
            "--expected-sumo-version",
            spec["expected_sumo_version"],
            "--out",
            remote_dir / "cache.json",
        ]
        command = _module_command(
            wrapper,
            "cf_h2o.eval.traffic_signal_counterfactual_cache",
            arguments,
        )
        result.append(
            _base_submit(
                scheduler=scheduler,
                protocol_tag=protocol_tag,
                description=f"CFCMT TSC {protocol_tag} source cache {node}",
                command=command,
                signature=f"CFCMT/tsc-{protocol_tag}/{run_id}/cache/{node}",
                node=None if pool_placement else node,
                allowed_nodes=cluster["nodes"] if pool_placement else None,
                cpu=cpu_cores,
                ram_mb=ram_mb,
                remote_result_dir=remote_dir,
                local_result_dir=local_results / "cache" / node,
                source_root=source_root,
            )
        )
    return result


def _source_rule_specs(
    *,
    spec: dict[str, Any],
    snapshot: dict[str, Any],
    scheduler: Path,
    run_id: str,
    remote_results: Path,
    local_results: Path,
    cache_root: Path,
    pool_placement: bool = False,
    selected_assignments: Sequence[str] | None = None,
) -> list[list[str]]:
    protocol_tag = _matrix_protocol_tag(spec)
    source_root = Path(snapshot["snapshot_root"])
    wrapper = source_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = source_root / spec["manifest"]
    counterfactual = spec["counterfactual"]
    development = spec["development"]
    cluster = spec["cluster"]
    requested = _selected_assignment_names(
        spec["cache_assignments"], selected_assignments
    )
    sys.path.insert(0, str(PROJECT_ROOT))
    from cf_h2o.eval.traffic_signal_source_rule_cache import _policy_grid

    policy_count = len(_policy_grid(development["source_rule_grid"]))
    result = []
    for node, scenarios in spec["cache_assignments"].items():
        if node not in requested:
            continue
        parallel_items = (
            len(scenarios)
            * len(development["source_policy_seeds"])
            * policy_count
        )
        workers, cpu_cores, ram_mb = _parallel_job_resources(
            cluster,
            worker_key="source_rule_workers_per_job",
            parallel_items=parallel_items,
        )
        remote_dir = remote_results / "source_rules" / node
        arguments: list[object] = [
            "--manifest",
            manifest,
            "--scenarios",
            *scenarios,
            "--seeds",
            *development["source_policy_seeds"],
            "--duration-sec",
            counterfactual["source_duration_sec"],
            "--control-interval-sec",
            counterfactual["control_interval_sec"],
            "--warmup-sec",
            counterfactual["warmup_sec"],
            "--workers",
            workers,
            "--source-rule-grid",
            development["source_rule_grid"],
            "--cache-root",
            cache_root / "source_rules",
            "--tripinfo-root",
            remote_dir / "tripinfo",
            "--expected-sumo-version",
            spec["expected_sumo_version"],
            "--out",
            remote_dir / "source_rules.json",
        ]
        command = _module_command(
            wrapper,
            "cf_h2o.eval.traffic_signal_source_rule_cache",
            arguments,
        )
        result.append(
            _base_submit(
                scheduler=scheduler,
                protocol_tag=protocol_tag,
                description=f"CFCMT TSC {protocol_tag} source rules {node}",
                command=command,
                signature=f"CFCMT/tsc-{protocol_tag}/{run_id}/source-rules/{node}",
                node=None if pool_placement else node,
                allowed_nodes=cluster["nodes"] if pool_placement else None,
                cpu=cpu_cores,
                ram_mb=ram_mb,
                remote_result_dir=remote_dir,
                local_result_dir=local_results / "source_rules" / node,
                source_root=source_root,
            )
        )
    return result


def _development_specs(
    *,
    spec: dict[str, Any],
    snapshot: dict[str, Any],
    scheduler: Path,
    run_id: str,
    remote_results: Path,
    local_results: Path,
    cache_root: Path,
    source_rule_cache_root: Path,
    pool_placement: bool = False,
    selected_budgets: Sequence[int] | None = None,
) -> list[list[str]]:
    protocol_tag = _matrix_protocol_tag(spec)
    source_root = Path(snapshot["snapshot_root"])
    wrapper = source_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = source_root / spec["manifest"]
    counterfactual = spec["counterfactual"]
    development = spec["development"]
    cluster = spec["cluster"]
    development_workers = int(
        cluster.get("development_workers_per_job", cluster["workers_per_job"])
    )
    workers, cpu_cores, ram_mb = _parallel_job_resources(
        cluster,
        worker_key="development_workers_per_job",
        parallel_items=development_workers,
    )
    budgets = development["target_group_budgets"]
    nodes = cluster["nodes"]
    if len(budgets) != len(nodes):
        raise ValueError("development budgets must map one-to-one onto cluster nodes")
    requested = set(int(value) for value in selected_budgets or budgets)
    unknown_budgets = requested - {int(value) for value in budgets}
    if unknown_budgets:
        raise ValueError(f"development budgets are absent from spec: {sorted(unknown_budgets)}")
    result = []
    for node, budget in zip(nodes, budgets):
        if int(budget) not in requested:
            continue
        remote_dir = remote_results / "development" / f"budget_{int(budget):03d}"
        arguments: list[object] = [
            "--manifest",
            manifest,
            "--policies",
            *development["policies"],
            "--seeds",
            *development["evaluation_seeds"],
            "--source-duration-sec",
            counterfactual["source_duration_sec"],
            "--duration-sec",
            development["evaluation_duration_sec"],
            "--control-interval-sec",
            counterfactual["control_interval_sec"],
            "--warmup-sec",
            counterfactual["warmup_sec"],
            "--source-seeds",
            *counterfactual["source_seeds"],
            "--source-policy-seeds",
            *development["source_policy_seeds"],
            "--source-rule-grid",
            development["source_rule_grid"],
            "--source-selector-min-context-domains",
            development.get("source_selector_min_context_domains", 5),
            "--max-focal-tls",
            counterfactual["max_focal_tls"],
            "--counterfactual-horizon-intervals",
            counterfactual["horizon_intervals"],
            "--collection-shards",
            counterfactual["collection_shards"],
            "--workers",
            workers,
            "--cache-root",
            cache_root,
            "--source-rule-cache-root",
            source_rule_cache_root,
            "--tripinfo-root",
            remote_dir / "tripinfo",
            "--target-group-budget",
            budget,
            "--target-adaptation-selection-seed",
            20260803,
            "--target-calibration-seeds",
            *development["target_calibration_seeds"],
            "--min-target-calibration-groups",
            development["min_target_calibration_groups"],
            "--target-calibration-fraction",
            development["target_calibration_fraction"],
            "--target-guard-selection-mode",
            development["target_guard_selection_mode"],
            "--source-behavior-policy",
            counterfactual["behavior_policy"],
            "--min-source-tls-coverage",
            counterfactual["min_tls_coverage"],
            "--out",
            remote_dir / "result.json",
            "--md-out",
            remote_dir / "result.md",
        ]
        primary_policy = development.get("primary_policy")
        if primary_policy is not None:
            arguments.extend(
                ["--primary-reference-policy", str(primary_policy)]
            )
        selection_seeds = development.get("target_closed_loop_selection_seeds", [])
        if selection_seeds:
            arguments.extend(["--target-closed-loop-selection-seeds", *selection_seeds])
            arguments.extend(
                [
                    "--target-closed-loop-selection-duration-sec",
                    development["target_closed_loop_selection_duration_sec"],
                    "--target-closed-loop-min-improvement",
                    development["target_closed_loop_min_improvement"],
                    "--target-closed-loop-refinement-min-improvement",
                    development[
                        "target_closed_loop_refinement_min_improvement"
                    ],
                    "--target-closed-loop-worst-tolerance",
                    development["target_closed_loop_worst_tolerance"],
                ]
            )
        command = _module_command(
            wrapper,
            "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite",
            arguments,
        )
        result.append(
            _base_submit(
                scheduler=scheduler,
                protocol_tag=protocol_tag,
                description=(
                    f"CFCMT TSC {protocol_tag} development "
                    f"budget={budget} on {node}"
                ),
                command=command,
                signature=(
                    f"CFCMT/tsc-{protocol_tag}/{run_id}/development/"
                    f"budget-{int(budget):03d}"
                ),
                node=None if pool_placement else node,
                allowed_nodes=cluster["nodes"] if pool_placement else None,
                cpu=cpu_cores,
                ram_mb=ram_mb,
                remote_result_dir=remote_dir,
                local_result_dir=local_results / "development" / f"budget_{int(budget):03d}",
                source_root=source_root,
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "safety-survey",
            "cache",
            "source-rules",
            "development",
        ),
        required=True,
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--counterfactual-cache-root",
        type=Path,
        help=(
            "explicit counterfactual cache root for cache recovery or "
            "read-only development input"
        ),
    )
    parser.add_argument(
        "--source-rule-cache-root",
        type=Path,
        help="explicit read-only source-rule cache root for development",
    )
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument(
        "--assignments",
        nargs="+",
        help="cache/source-rule assignment names to submit (for example node003 node004)",
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument(
        "--scheduler-home",
        type=Path,
        help="HOME used only by scheduler subprocesses, for an isolated queue",
    )
    parser.add_argument(
        "--pool-placement",
        action="store_true",
        help="let cache/development jobs run on any configured cluster node",
    )
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    spec = _load_json(args.spec)
    _validate_spec(spec)
    if (
        args.phase == "cache"
        and spec.get("counterfactual_cache_version")
        != _current_counterfactual_cache_version()
    ):
        raise ValueError(
            "historical matrix specifications are read-only; cache generation "
            "requires the current counterfactual cache version"
        )
    snapshot = _load_json(args.snapshot_manifest)
    remote_results = Path("/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS") / args.run_id
    local_results = PROJECT_ROOT / "cf_h2o/results/cluster" / args.run_id
    run_cache_root = (
        Path("/home/zhengliang01/scheduleurm_work/CFCMT_CACHE") / args.run_id
    )
    factory = {
        "preflight": _preflight_specs,
        "safety-survey": _safety_survey_specs,
        "cache": _cache_specs,
        "source-rules": _source_rule_specs,
        "development": _development_specs,
    }[args.phase]
    kwargs = {
        "spec": spec,
        "snapshot": snapshot,
        "scheduler": args.scheduler,
        "run_id": args.run_id,
        "remote_results": remote_results,
        "local_results": local_results,
    }
    if args.phase == "safety-survey":
        kwargs["pool_placement"] = bool(args.pool_placement)
        kwargs["selected_assignments"] = args.assignments
    elif args.phase == "cache":
        kwargs["cache_root"] = args.counterfactual_cache_root or run_cache_root
        kwargs["pool_placement"] = bool(args.pool_placement)
        kwargs["selected_assignments"] = args.assignments
    elif args.phase == "source-rules":
        kwargs["cache_root"] = run_cache_root
        kwargs["pool_placement"] = bool(args.pool_placement)
        kwargs["selected_assignments"] = args.assignments
    elif args.phase == "development":
        (
            kwargs["cache_root"],
            kwargs["source_rule_cache_root"],
        ) = _development_cache_roots(
            counterfactual_cache_root=args.counterfactual_cache_root,
            source_rule_cache_root=args.source_rule_cache_root,
        )
        kwargs["selected_budgets"] = args.budgets
        kwargs["pool_placement"] = bool(args.pool_placement)
    elif args.pool_placement:
        raise ValueError("preflight must remain pinned to each audited node")
    if args.budgets is not None and args.phase != "development":
        raise ValueError("--budgets is valid only for the development phase")
    if args.assignments is not None and args.phase not in {
        "safety-survey",
        "cache",
        "source-rules",
    }:
        raise ValueError(
            "--assignments is valid only for safety/cache/source-rules phases"
        )
    if args.source_rule_cache_root is not None and args.phase != "development":
        raise ValueError(
            "--source-rule-cache-root is valid only for the development phase"
        )
    if args.counterfactual_cache_root is not None and args.phase not in {
        "cache",
        "development",
    }:
        raise ValueError(
            "--counterfactual-cache-root is valid only for cache or development"
        )
    commands = factory(**kwargs)
    scheduler_env = os.environ.copy()
    if args.scheduler_home is not None:
        scheduler_home = args.scheduler_home.expanduser().resolve()
        scheduler_home.mkdir(parents=True, exist_ok=True)
        scheduler_env["HOME"] = str(scheduler_home)
    for command in commands:
        print(shlex.join(command), flush=True)
        if args.submit:
            subprocess.run(command, check=True, env=scheduler_env)
    if args.submit:
        subprocess.run(
            [sys.executable, str(args.scheduler), "dispatch"],
            check=True,
            env=scheduler_env,
        )


if __name__ == "__main__":
    main()
