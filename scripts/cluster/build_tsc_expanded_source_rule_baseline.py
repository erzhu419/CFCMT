#!/usr/bin/env python3
"""Union an immutable source-rule baseline with an audited new-city partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL = "audited-disjoint-source-rule-baseline-union-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _validated_costs(
    values: Any,
    *,
    expected_policy_keys: set[str] | None,
    location: str,
) -> tuple[dict[str, float], set[str]]:
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{location}: missing policy costs")
    parsed = {str(key): float(value) for key, value in values.items()}
    if not all(math.isfinite(value) for value in parsed.values()):
        raise ValueError(f"{location}: non-finite policy cost")
    keys = set(parsed)
    if expected_policy_keys is not None and keys != expected_policy_keys:
        raise ValueError(f"{location}: policy keys changed")
    return parsed, keys


def build_expanded_baseline(
    *,
    base_baseline_path: Path,
    expected_base_sha256: str,
    partition_results_root: Path,
    partition_audit_path: Path,
    expected_partition_audit_sha256: str,
    expected_partition_cache_sha256: str,
) -> dict[str, Any]:
    if _sha256(base_baseline_path) != str(expected_base_sha256):
        raise ValueError("base source-rule baseline hash mismatch")
    if _sha256(partition_audit_path) != str(expected_partition_audit_sha256):
        raise ValueError("partition source-rule audit hash mismatch")
    base = _read_json(base_baseline_path)
    audit = _read_json(partition_audit_path)
    if audit.get("passed") is not True or audit.get("errors"):
        raise ValueError("partition source-rule audit did not pass")
    if audit.get("cache_aggregate_sha256") != expected_partition_cache_sha256:
        raise ValueError("partition source-rule cache hash mismatch")

    base_costs_raw = base.get("source_rule_policy_costs")
    if not isinstance(base_costs_raw, dict) or not base_costs_raw:
        raise ValueError("base baseline has no source-rule costs")
    combined: dict[str, dict[str, float]] = {}
    policy_keys: set[str] | None = None
    for scenario, values in sorted(base_costs_raw.items()):
        parsed, policy_keys = _validated_costs(
            values,
            expected_policy_keys=policy_keys,
            location=f"base:{scenario}",
        )
        combined[str(scenario)] = parsed

    result_files = sorted(Path(partition_results_root).glob("**/source_rules.json"))
    audited_files = sorted(Path(path) for path in audit.get("result_files", ()))
    if [path.resolve() for path in result_files] != [
        path.resolve() for path in audited_files
    ]:
        raise ValueError("partition result files differ from the passed audit")
    partition_scenarios: list[str] = []
    result_provenance: list[dict[str, Any]] = []
    for result_path in result_files:
        payload = _read_json(result_path)
        safety = payload.get("safety_audit", {})
        if not isinstance(safety, dict) or safety.get("passed") is not True:
            raise ValueError(f"{result_path}: safety audit did not pass")
        scenario_costs = payload.get("scenario_policy_costs")
        if not isinstance(scenario_costs, dict) or not scenario_costs:
            raise ValueError(f"{result_path}: no scenario costs")
        for scenario, values in sorted(scenario_costs.items()):
            scenario = str(scenario)
            if scenario in combined:
                raise ValueError(f"source-rule scenario overlap: {scenario}")
            parsed, _ = _validated_costs(
                values,
                expected_policy_keys=policy_keys,
                location=f"partition:{scenario}",
            )
            combined[scenario] = parsed
            partition_scenarios.append(scenario)
        result_provenance.append(
            {
                "path": str(result_path.resolve()),
                "sha256": _sha256(result_path),
                "scenario_count": len(scenario_costs),
                "row_count": int(payload.get("row_count", -1)),
                "source_tree_sha256": payload.get("runtime", {}).get(
                    "source_tree_sha256"
                ),
            }
        )
    if len(partition_scenarios) != int(audit.get("scenario_count", -1)):
        raise ValueError("partition scenario count differs from audit")

    return {
        "protocol": PROTOCOL,
        "source_rule_policy_costs": dict(sorted(combined.items())),
        "scenario_count": len(combined),
        "policy_count": len(policy_keys or ()),
        "base_baseline": {
            "path": str(Path(base_baseline_path).resolve()),
            "sha256": expected_base_sha256,
            "scenario_count": len(base_costs_raw),
            "source_rule_cache": base.get("source_rule_cache"),
        },
        "added_partition": {
            "results_root": str(Path(partition_results_root).resolve()),
            "audit_path": str(Path(partition_audit_path).resolve()),
            "audit_sha256": expected_partition_audit_sha256,
            "cache_aggregate_sha256": expected_partition_cache_sha256,
            "scenarios": sorted(partition_scenarios),
            "result_files": result_provenance,
        },
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-baseline", type=Path, required=True)
    parser.add_argument("--expected-base-sha256", required=True)
    parser.add_argument("--partition-results-root", type=Path, required=True)
    parser.add_argument("--partition-audit", type=Path, required=True)
    parser.add_argument("--expected-partition-audit-sha256", required=True)
    parser.add_argument("--expected-partition-cache-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_expanded_baseline(
        base_baseline_path=args.base_baseline,
        expected_base_sha256=args.expected_base_sha256,
        partition_results_root=args.partition_results_root,
        partition_audit_path=args.partition_audit,
        expected_partition_audit_sha256=args.expected_partition_audit_sha256,
        expected_partition_cache_sha256=args.expected_partition_cache_sha256,
    )
    _atomic_json(args.out, payload)
    print(
        f"expanded source-rule baseline: {payload['scenario_count']} scenarios, "
        f"{payload['policy_count']} policies"
    )


if __name__ == "__main__":
    main()
