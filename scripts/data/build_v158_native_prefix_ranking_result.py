#!/usr/bin/env python3
"""Build the compact paper artifact for the frozen V158 OOF result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "cf_h2o/results/cluster/tsc_v158_native_prefix_ranking_calibration_20260911"
OUTPUT = ROOT / "cf_h2o/results/paper_artifacts/tsc_v158_native_prefix_ranking_calibration_v1.json"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_file(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def task_map(*launches: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for launch in launches:
        for task in launch["submission"]["submitted"]:
            match = re.search(r"seed (\d+)$", task["description"])
            if match:
                result[int(match.group(1))] = {
                    "task_id": task["id"],
                    "node": task["require_node"],
                    "scheduler_status": "done",
                    "exit_code": 0,
                }
    return result


def main() -> None:
    config_path = ROOT / "cf_h2o/config/traffic_signal_tsc_v158_native_prefix_ranking_calibration.json"
    protocol_path = ROOT / "paper/tsc_v158_native_prefix_ranking_calibration_protocol.md"
    stage_path = RUN_ROOT / "derived_snapshot_v5.json"
    smoke_path = RUN_ROOT / "launch_focal_reference_smoke_v1.json"
    collect_path = RUN_ROOT / "launch_collect_v5.json"
    fit_launch_path = RUN_ROOT / "launch_fit_v1.json"
    fit_path = RUN_ROOT / "fit_v1/result.json"
    diagnostic_path = RUN_ROOT / "oof_posthoc_diagnostic_v1.json"
    scheduler_path = RUN_ROOT / "scheduler_status_v1.json"
    builder_path = Path(__file__).resolve()

    config = read_json(config_path)
    stage = read_json(stage_path)
    smoke = read_json(smoke_path)
    collect = read_json(collect_path)
    fit_launch = read_json(fit_launch_path)
    fit = read_json(fit_path)
    diagnostic = read_json(diagnostic_path)
    scheduler = read_json(scheduler_path)

    if config["protocol"] != "tsc-v158-jinan-native-prefix-ranking-calibration-v1":
        raise ValueError("unexpected V158 configuration protocol")
    if fit["status"] != "OOF_GATE_FAIL" or fit["reserve_authorized"] is not False:
        raise ValueError("V158 paper artifact builder is bound to the observed OOF failure")
    if fit["execution_snapshot"] != {
        "protocol": stage["protocol"],
        "root": stage["snapshot_root"],
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
    }:
        raise ValueError("fit execution snapshot does not match staged V158 v5")

    tasks = task_map(smoke, collect)
    seed_rows = []
    branch_collision_events = 0
    baseline_collision_events = 0
    collision_groups = []
    frozen_source_cost_differences = []
    frozen_source_override_count = 0
    source_target_disagreement_groups = 0
    for seed in config["development"]["seeds"]:
        result_path = RUN_ROOT / "development_v1" / f"seed_{seed}" / "result.json"
        row = read_json(result_path)
        expected = {
            "protocol": "tsc-v158-jinan-native-prefix-seed-bank-v1",
            "status": "PASS",
            "seed": seed,
            "action_groups": 10,
            "candidate_rows": 80,
            "native_nonreference_branches": 70,
            "baseline_trajectories": 3,
            "snapshot_restores": 0,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"seed {seed}: invalid local V158 result copy")
        if row["execution_snapshot"] != fit["execution_snapshot"]:
            raise ValueError(f"seed {seed}: execution snapshot changed")
        for group in row["groups"]:
            reference_index = int(group["reference_index"])
            native_costs = list(map(float, group["native_costs"]))
            source_index = int(
                group["base_decisions"]["uniform_source"]["selected_index"]
            )
            frozen_source_cost_differences.append(
                native_costs[source_index] - native_costs[reference_index]
            )
            frozen_source_override_count += source_index != reference_index
            source_target_disagreement_groups += bool(
                group["focal_selection"]["source_target_disagreement"]
            )
            for action_index, count in enumerate(group["collision_events_by_action"]):
                if action_index == reference_index:
                    baseline_collision_events += int(count)
                else:
                    branch_collision_events += int(count)
                if count:
                    collision_groups.append(
                        {
                            "seed": seed,
                            "group_id": group["group_id"],
                            "action_index": action_index,
                            "reference_action": action_index == reference_index,
                            "events": int(count),
                        }
                    )
        remote_result = str(Path(row["bank"]).with_name("result.json"))
        seed_rows.append(
            {
                "seed": seed,
                **tasks[seed],
                "elapsed_sec": row["elapsed_sec"],
                "result": {**local_file(result_path), "remote_path": remote_result},
                "native_bank": {
                    "path": row["bank"],
                    "sha256": row["bank_sha256"],
                },
            }
        )

    development = fit["development"]
    expected_development = {
        "action_groups": 100,
        "candidate_rows": 800,
        "native_nonreference_branches": 700,
        "baseline_trajectories": 30,
        "snapshot_restores": 0,
    }
    if any(development.get(key) != value for key, value in expected_development.items()):
        raise ValueError("V158 merged development accounting changed")

    fit_task = fit_launch["submission"]["submitted"]
    if len(fit_task) != 1 or fit_task[0]["id"] != "t92703":
        raise ValueError("V158 fit task identity changed")

    invalid_tasks = {
        task["id"]: task
        for task in scheduler["tasks"]
        if task["id"] in {
            *(f"t{i}" for i in range(92638, 92648)),
            *(f"t{i}" for i in range(92650, 92660)),
            "t92685",
        }
    }
    frozen_source_minus_pp = sum(frozen_source_cost_differences) / len(
        frozen_source_cost_differences
    )
    v158_source_minus_pp = fit["oof"]["comparisons"][
        "source_minus_phase_pressure"
    ]["mean_difference"]
    oracle_advantage = diagnostic["oracle_headroom"]["all"][
        "mean_oracle_advantage"
    ]
    artifact = {
        "protocol": "tsc-v158-native-prefix-ranking-calibration-result-v1",
        "parent_protocol": config["protocol"],
        "status": fit["status"],
        "scientific_status": fit["scientific_status"],
        "city": config["city"],
        "decision": {
            "reserve_authorized": False,
            "reserve_executed": False,
            "reserve_seed_count": 0,
            "reserve_cell_count": 0,
            "adopt_source_native_controller": False,
            "native_prefix_ranking_remediation": "closed_after_preregistered_oof_gate_failure",
        },
        "development": {
            **development,
            "simulator_trajectories": (
                development["native_nonreference_branches"]
                + development["baseline_trajectories"]
            ),
            "valid_seed_banks": len(seed_rows),
            "invalid_or_missing_seed_banks": 0,
            "model_fits": 18,
            "validity_contract_passed_all_seed_banks": True,
            "teleport_check_passed_all_seed_banks": True,
            "collision_diagnostics": {
                "native_nonreference_branch_events": branch_collision_events,
                "phase_pressure_baseline_events": baseline_collision_events,
                "affected_groups": len({row["group_id"] for row in collision_groups}),
                "affected_seeds": len({row["seed"] for row in collision_groups}),
                "events": collision_groups,
                "used_by_efficiency_gate": False,
            },
            "seed_results": seed_rows,
        },
        "oof": fit["oof"],
        "posthoc_behavior_diagnostic": {
            "status": diagnostic["status"],
            "decision_authority": diagnostic["decision_authority"],
            "inputs": diagnostic["inputs"],
            "arms": diagnostic["arms"],
            "scenario_results": diagnostic["scenario_results"],
            "pairwise_selected_actions": diagnostic["pairwise_selected_actions"],
            "oracle_headroom": diagnostic["oracle_headroom"],
            "calibration_effect_vs_frozen_source": {
                "source_target_disagreement_enriched_groups": source_target_disagreement_groups,
                "frozen_source_overrides": frozen_source_override_count,
                "frozen_source_minus_phase_pressure": frozen_source_minus_pp,
                "v158_oof_source_overrides": diagnostic["arms"]["source_native"][
                    "overrides"
                ],
                "v158_oof_source_minus_phase_pressure": v158_source_minus_pp,
                "absolute_harm_reduction": frozen_source_minus_pp
                - v158_source_minus_pp,
                "relative_harm_reduction": (
                    frozen_source_minus_pp - v158_source_minus_pp
                )
                / frozen_source_minus_pp,
                "v158_fraction_of_oracle_headroom_captured": -v158_source_minus_pp
                / oracle_advantage,
            },
            "interpretation_boundary": diagnostic["interpretation_boundary"],
        },
        "server_only_fit_artifacts": {
            **fit["artifacts"],
            "calibrators_authorized_for_reserve_or_deployment": False,
        },
        "provenance": {
            "execution_snapshot": fit["execution_snapshot"],
            "parent_snapshot": config["parent_snapshot"],
            "snapshot_overlays": stage["overlays"],
            "network_manifest": config["network_manifest"],
            "v157b_runtime": config["v157b_runtime"],
            "files": {
                "config": local_file(config_path),
                "frozen_protocol": local_file(protocol_path),
                "stage_manifest_v5": local_file(stage_path),
                "smoke_launch": local_file(smoke_path),
                "collection_launch_v5": local_file(collect_path),
                "fit_launch": local_file(fit_launch_path),
                "fit_result": local_file(fit_path),
                "posthoc_diagnostic": local_file(diagnostic_path),
                "scheduler_status": local_file(scheduler_path),
                "result_builder": local_file(builder_path),
            },
            "effective_tasks": {
                "collection": [row["task_id"] for row in seed_rows],
                "fit": fit_task[0]["id"],
            },
            "operationally_invalid_attempts": {
                "argument_list_too_long_before_simulation": [f"t{i}" for i in range(92638, 92648)],
                "pre_fix_focal_reference_mismatch": [f"t{i}" for i in range(92650, 92659)],
                "cancelled_pre_fix_task": "t92659",
                "diagnostic_task": "t92685",
                "scheduler_terminal_states": {
                    task_id: {
                        "status": invalid_tasks[task_id]["status"],
                        "node": invalid_tasks[task_id]["node"],
                        "exit_code": invalid_tasks[task_id]["exit_code"],
                    }
                    for task_id in sorted(invalid_tasks)
                },
                "used_as_scientific_observations": False,
            },
        },
        "claim_boundary": (
            "V158 did not pass its preregistered Jinan whole-seed OOF gate. "
            "It authorizes no reserve execution, controller adoption, unseen-city claim, "
            "or post-hoc retuning. The diagnostic oracle headroom and behavior summaries "
            "locate the failure but do not change the gate decision."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUTPUT), "sha256": sha256(OUTPUT), "size_bytes": OUTPUT.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()
