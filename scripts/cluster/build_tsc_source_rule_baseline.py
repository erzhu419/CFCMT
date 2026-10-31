#!/usr/bin/env python3
"""Freeze audited source-rule costs into a deterministic baseline artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "tsc-audited-source-rule-baseline-v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_cache_sha256(paths: Iterable[Path]) -> str:
    ordered = sorted((Path(path) for path in paths), key=lambda path: path.name)
    names = [path.name for path in ordered]
    if len(names) != len(set(names)):
        raise ValueError("source-rule cache basenames are not unique")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def build_baseline(
    *,
    results_root: Path,
    audit_path: Path,
    expected_cache_aggregate_sha256: str,
) -> dict[str, Any]:
    if len(expected_cache_aggregate_sha256) != 64:
        raise ValueError("expected cache aggregate SHA-256 must have 64 characters")
    int(expected_cache_aggregate_sha256, 16)

    audit = _read_json(audit_path)
    if audit.get("passed") is not True or audit.get("errors"):
        raise ValueError(f"{audit_path}: source-rule cache audit did not pass")
    result_files = sorted(results_root.glob("**/source_rules.json"))
    audited_result_files = sorted(Path(path) for path in audit.get("result_files", ()))
    if [path.resolve() for path in result_files] != [
        path.resolve() for path in audited_result_files
    ]:
        raise ValueError("source-rule result files differ from the passed audit")

    costs: dict[str, dict[str, float]] = {}
    cache_files: list[Path] = []
    result_provenance: list[dict[str, Any]] = []
    source_hashes: set[str] = set()
    policy_keys: set[str] | None = None
    total_rows = 0
    for result_file in result_files:
        payload = _read_json(result_file)
        safety = payload.get("safety_audit", {})
        if not isinstance(safety, dict) or safety.get("passed") is not True:
            raise ValueError(f"{result_file}: safety audit did not pass")
        runtime = payload.get("runtime", {})
        source_hashes.add(str(runtime.get("source_tree_sha256", "")))
        scenario_costs = payload.get("scenario_policy_costs")
        if not isinstance(scenario_costs, dict) or not scenario_costs:
            raise ValueError(f"{result_file}: missing scenario policy costs")
        for scenario, values in scenario_costs.items():
            if scenario in costs:
                raise ValueError(f"duplicate source-rule scenario: {scenario}")
            if not isinstance(values, dict) or not values:
                raise ValueError(f"{result_file}:{scenario}: empty policy costs")
            current_keys = set(str(key) for key in values)
            if policy_keys is None:
                policy_keys = current_keys
            elif current_keys != policy_keys:
                raise ValueError(f"{result_file}:{scenario}: policy keys changed")
            parsed = {str(key): float(value) for key, value in values.items()}
            if not all(math.isfinite(value) for value in parsed.values()):
                raise ValueError(f"{result_file}:{scenario}: non-finite policy cost")
            costs[str(scenario)] = parsed
        files = [Path(path) for path in payload.get("cache_files", ())]
        cache_files.extend(files)
        rows = int(payload.get("row_count", -1))
        if rows != len(files):
            raise ValueError(f"{result_file}: row/cache-file count mismatch")
        total_rows += rows
        result_provenance.append(
            {
                "path": str(result_file.resolve()),
                "sha256": _sha256(result_file),
                "scenario_count": len(scenario_costs),
                "row_count": rows,
            }
        )

    if source_hashes != {str(audit.get("source_tree_sha256", ""))}:
        raise ValueError("source tree hashes differ from the passed audit")
    if len(cache_files) != len(set(path.resolve() for path in cache_files)):
        raise ValueError("source-rule cache file identities are not unique")
    if len(cache_files) != int(audit.get("cache_file_count", -1)):
        raise ValueError("source-rule cache file count differs from the passed audit")
    if len(costs) != int(audit.get("scenario_count", -1)):
        raise ValueError("source-rule scenario count differs from the passed audit")
    if total_rows != int(audit.get("total_rows", -1)):
        raise ValueError("source-rule row count differs from the passed audit")

    aggregate_sha256 = aggregate_cache_sha256(cache_files)
    if aggregate_sha256 != expected_cache_aggregate_sha256:
        raise ValueError(
            "source-rule cache aggregate SHA-256 mismatch: "
            f"expected={expected_cache_aggregate_sha256}, actual={aggregate_sha256}"
        )
    return {
        "protocol": PROTOCOL,
        "source_rule_policy_costs": dict(sorted(costs.items())),
        "source_rule_cache": {
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": _sha256(audit_path),
            "aggregate_sha256": aggregate_sha256,
            "cache_file_count": len(cache_files),
            "policy_count": len(policy_keys or ()),
            "result_files": result_provenance,
            "row_count": total_rows,
            "scenario_count": len(costs),
            "source_tree_sha256": next(iter(source_hashes)),
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
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--expected-cache-aggregate-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = build_baseline(
        results_root=args.results_root,
        audit_path=args.audit,
        expected_cache_aggregate_sha256=args.expected_cache_aggregate_sha256,
    )
    _atomic_json(args.out, payload)
    print(
        "source-rule baseline frozen: "
        f"{payload['source_rule_cache']['scenario_count']} scenarios, "
        f"{payload['source_rule_cache']['cache_file_count']} files"
    )


if __name__ == "__main__":
    main()
