#!/usr/bin/env python3
"""Audit v38 Salt Lake shards and apply the frozen global pairwise gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    CANDIDATE_FAMILY,
    CONFIRMATION_FAMILIES,
    EXPECTED_COMBINED_FILE_COUNT,
    FALLBACK_FAMILY,
    TARGET_CITY_GROUP,
    TARGET_GROUP_BUDGET,
    TARGETS,
    _load_protocol,
    summarize_saltlake_confirmation,
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def audit_confirmation(
    *,
    results_root: Path,
    protocol_spec_path: Path,
    preregistration_path: Path,
    salt_cache_audit_path: Path,
    combined_cache_manifest_path: Path,
    expected_source_tree_sha256: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
    expected_salt_cache_sha256: str,
    expected_source_cache_sha256: str,
) -> dict[str, Any]:
    _load_protocol(protocol_spec_path)
    salt_audit = _read_json(salt_cache_audit_path)
    combined_manifest = _read_json(combined_cache_manifest_path)
    errors: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    result_provenance: list[dict[str, Any]] = []
    regrets: dict[str, dict[str, float]] = {}

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(salt_audit.get("passed") is True, "Salt-only cache audit did not pass")
    require(
        salt_audit.get("cache_aggregate_sha256") == expected_salt_cache_sha256,
        "Salt-only cache aggregate mismatch",
    )
    require(
        int(salt_audit.get("cache_file_count", -1)) == 96,
        "Salt-only cache file count changed",
    )
    require(
        set(salt_audit.get("scenario_group_counts", {})) == set(TARGETS),
        "Salt-only cache target set changed",
    )
    require(
        all(
            int(count) >= TARGET_GROUP_BUDGET
            for count in salt_audit.get("scenario_group_counts", {}).values()
        ),
        "Salt-only cache has fewer than 60 groups for a target",
    )
    require(
        combined_manifest.get("protocol")
        == "hard-linked-disjoint-counterfactual-cache-union-v1",
        "combined cache construction protocol changed",
    )
    require(
        combined_manifest.get("source_aggregate_sha256")
        == expected_source_cache_sha256,
        "combined cache source partition hash mismatch",
    )
    require(
        combined_manifest.get("target_aggregate_sha256")
        == expected_salt_cache_sha256,
        "combined cache target partition hash mismatch",
    )
    require(
        combined_manifest.get("combined_aggregate_sha256")
        == expected_cache_sha256,
        "combined cache aggregate mismatch",
    )
    require(
        int(combined_manifest.get("combined_file_count", -1))
        == EXPECTED_COMBINED_FILE_COUNT,
        "combined cache file count changed",
    )

    observed_result_dirs = {
        path.parent.name for path in Path(results_root).glob("*/result.json")
    }
    require(
        observed_result_dirs == set(TARGETS),
        "confirmation result target partition changed",
    )
    protocol_sha256 = _sha256(protocol_spec_path)
    for target in TARGETS:
        target_root = Path(results_root) / target
        result_path = target_root / "result.json"
        shard_path = target_root / "shard_result.json"
        launch_path = target_root / "launch_manifest.json"
        for path in (result_path, shard_path, launch_path):
            require(path.is_file(), f"{target}: missing {path.name}")
        if not all(path.is_file() for path in (result_path, shard_path, launch_path)):
            continue
        result = _read_json(result_path)
        shard = _read_json(shard_path)
        launch = _read_json(launch_path)
        result_sha256 = _sha256(result_path)
        require(shard.get("passed") is True, f"{target}: shard did not pass")
        require(int(shard.get("returncode", -1)) == 0, f"{target}: nonzero return code")
        require(
            shard.get("result_sha256") == result_sha256,
            f"{target}: result hash differs from shard record",
        )
        require(result.get("target") == target, f"{target}: target identity changed")
        require(
            result.get("protocol")
            == "tsc-v38r34-global-pairwise-salt-lake-target-confirmation-v1",
            f"{target}: result protocol changed",
        )
        require(
            result.get("protocol_spec_sha256") == protocol_sha256,
            f"{target}: protocol-spec hash mismatch",
        )
        require(
            launch.get("protocol_spec_sha256") == protocol_sha256,
            f"{target}: launch protocol-spec hash mismatch",
        )
        require(
            result.get("runtime", {}).get("source_tree_sha256")
            == expected_source_tree_sha256,
            f"{target}: source-tree hash mismatch",
        )
        require(
            result.get("cache_aggregate_sha256") == expected_cache_sha256,
            f"{target}: result cache hash mismatch",
        )
        require(
            int(result.get("cache_file_count", -1))
            == EXPECTED_COMBINED_FILE_COUNT,
            f"{target}: result cache file count changed",
        )
        require(
            result.get("baseline_sha256") == expected_baseline_sha256,
            f"{target}: baseline hash mismatch",
        )
        require(
            result.get("target_city_group") == TARGET_CITY_GROUP,
            f"{target}: target city group changed",
        )
        require(
            result.get("heldout_city_scenarios") == sorted(TARGETS),
            f"{target}: same-city holdout changed",
        )
        source_scenarios = result.get("source_scenarios", [])
        require(
            isinstance(source_scenarios, list)
            and len(source_scenarios) == 16
            and not set(source_scenarios) & set(TARGETS),
            f"{target}: Salt Lake leaked into source scenarios",
        )
        require(
            len(result.get("source_city_groups", [])) == 6
            and TARGET_CITY_GROUP not in result.get("source_city_groups", []),
            f"{target}: source city-group contract changed",
        )
        selected = [str(value) for value in result.get("selected_group_ids", ())]
        require(
            len(selected) == TARGET_GROUP_BUDGET
            and len(set(selected)) == TARGET_GROUP_BUDGET,
            f"{target}: target group budget changed",
        )
        require(
            result.get("model_families") == list(CONFIRMATION_FAMILIES),
            f"{target}: family order/set changed",
        )
        diagnostics = result.get("model_diagnostics", {})
        require(
            diagnostics.get("target_adaptation_protocol")
            == "explicit-complete-action-group-list-v1"
            and int(diagnostics.get("target_adaptation_groups", -1))
            == TARGET_GROUP_BUDGET
            and diagnostics.get("target_adaptation_group_ids") == selected,
            f"{target}: explicit 60-group fit contract failed",
        )
        evaluation = result.get("offline_evaluation", {})
        evaluation_groups = int(evaluation.get("evaluation_group_count", -1))
        require(evaluation_groups > 0, f"{target}: no evaluation groups")
        require(
            int(evaluation.get("excluded_group_count", -1))
            == TARGET_GROUP_BUDGET,
            f"{target}: adaptation groups entered evaluation",
        )
        family_rows = evaluation.get("families", {})
        require(
            set(family_rows) == set(CONFIRMATION_FAMILIES),
            f"{target}: evaluation family set changed",
        )
        target_regrets: dict[str, float] = {}
        for family in CONFIRMATION_FAMILIES:
            family_row = family_rows.get(family, {})
            value = float(family_row.get("mean_normalized_action_regret", float("nan")))
            require(
                math.isfinite(value) and value >= 0.0,
                f"{target}/{family}: invalid regret",
            )
            require(
                int(family_row.get("group_count", -1)) == evaluation_groups,
                f"{target}/{family}: group count mismatch",
            )
            target_regrets[family] = value
        regrets[target] = target_regrets
        rows[target] = {
            "target": target,
            "adaptation_group_count": len(selected),
            "evaluation_group_count": evaluation_groups,
            "fallback_regret": target_regrets.get(FALLBACK_FAMILY),
            "candidate_regret": target_regrets.get(CANDIDATE_FAMILY),
            "candidate_improvement": (
                target_regrets.get(FALLBACK_FAMILY, float("nan"))
                - target_regrets.get(CANDIDATE_FAMILY, float("nan"))
            ),
        }
        result_provenance.append(
            {
                "target": target,
                "result_path": str(result_path.resolve()),
                "result_sha256": result_sha256,
                "shard_path": str(shard_path.resolve()),
                "shard_sha256": _sha256(shard_path),
                "launch_path": str(launch_path.resolve()),
                "launch_sha256": _sha256(launch_path),
            }
        )

    integrity_passed = not errors and set(regrets) == set(TARGETS)
    stage = summarize_saltlake_confirmation(regrets) if integrity_passed else None
    return {
        "audit": "tsc_v38r34_global_pairwise_salt_lake_confirmation",
        "integrity_passed": integrity_passed,
        "errors": errors,
        "status": (
            "PASS"
            if integrity_passed and stage and stage["passed"]
            else "FAIL"
        ),
        "protocol_spec": {
            "path": str(protocol_spec_path.resolve()),
            "sha256": protocol_sha256,
        },
        "preregistration": {
            "path": str(preregistration_path.resolve()),
            "sha256": _sha256(preregistration_path),
        },
        "salt_cache_audit": {
            "path": str(salt_cache_audit_path.resolve()),
            "sha256": _sha256(salt_cache_audit_path),
            "aggregate_sha256": expected_salt_cache_sha256,
        },
        "combined_cache": {
            "manifest_path": str(combined_cache_manifest_path.resolve()),
            "manifest_sha256": _sha256(combined_cache_manifest_path),
            "source_aggregate_sha256": expected_source_cache_sha256,
            "target_aggregate_sha256": expected_salt_cache_sha256,
            "aggregate_sha256": expected_cache_sha256,
            "file_count": EXPECTED_COMBINED_FILE_COUNT,
        },
        "source_tree_sha256": expected_source_tree_sha256,
        "baseline_sha256": expected_baseline_sha256,
        "target_rows": [rows[target] for target in TARGETS if target in rows],
        "result_provenance": result_provenance,
        "confirmation_gate": stage,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# TSC v38/r34 Global Pairwise Salt Lake Outcome",
        "",
        "This document reports the frozen v38/r34 outcome without target-specific family switching.",
        "",
        f"- integrity: `{'PASS' if payload['integrity_passed'] else 'FAIL'}`",
        f"- preregistered confirmation: `{payload['status']}`",
        f"- combined cache SHA-256: `{payload['combined_cache']['aggregate_sha256']}`",
        f"- source tree SHA-256: `{payload['source_tree_sha256']}`",
        "",
        "| Salt Lake network | B | Evaluation groups | Group-normalized fallback | Pairwise candidate | Absolute improvement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("target_rows", []):
        lines.append(
            f"| {row['target']} | {row['adaptation_group_count']} | "
            f"{row['evaluation_group_count']} | {row['fallback_regret']:.6f} | "
            f"{row['candidate_regret']:.6f} | {row['candidate_improvement']:.6f} |"
        )
    gate = payload.get("confirmation_gate")
    if gate is not None:
        lines.extend(
            [
                "",
                "## Frozen gate",
                "",
                f"- fallback macro regret: `{gate['fallback_macro_regret']:.9f}`",
                f"- pairwise macro regret: `{gate['candidate_macro_regret']:.9f}`",
                f"- relative improvement: `{gate['macro_relative_improvement_pct']:.4f}%`",
                f"- improved networks: `{gate['improved_network_count']}/2`",
                f"- maximum absolute regression: `{gate['maximum_absolute_network_regression']:.9f}`",
                f"- decision: `{gate['decision']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is target-simulator adaptation with 60 complete matched action groups per network. It is not zero-shot, passive AVL/APC few-shot learning, or real-world intervention evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--salt-cache-audit", type=Path, required=True)
    parser.add_argument("--combined-cache-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-tree-sha256", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--expected-salt-cache-sha256", required=True)
    parser.add_argument("--expected-source-cache-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit_confirmation(
        results_root=args.results_root,
        protocol_spec_path=args.protocol_spec,
        preregistration_path=args.preregistration,
        salt_cache_audit_path=args.salt_cache_audit,
        combined_cache_manifest_path=args.combined_cache_manifest,
        expected_source_tree_sha256=args.expected_source_tree_sha256,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
        expected_salt_cache_sha256=args.expected_salt_cache_sha256,
        expected_source_cache_sha256=args.expected_source_cache_sha256,
    )
    _atomic_json(args.out, payload)
    _atomic_text(args.md_out, render_markdown(payload))
    if not payload["integrity_passed"]:
        raise RuntimeError("Salt Lake confirmation integrity audit failed")
    print(
        f"Salt Lake confirmation {payload['status']}: "
        f"{payload['confirmation_gate']['macro_relative_improvement_pct']:.4f}%"
    )


if __name__ == "__main__":
    main()
