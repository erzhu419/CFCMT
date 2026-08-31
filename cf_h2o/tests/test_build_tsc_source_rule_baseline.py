from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.cluster.build_tsc_source_rule_baseline import (
    aggregate_cache_sha256,
    build_baseline,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    source_sha = "a" * 64
    results_root = tmp_path / "results"
    cache_root = tmp_path / "cache"
    result_files = []
    for index, scenario in enumerate(("city_a", "city_b")):
        cache_file = cache_root / f"source_rule_{index}.json"
        _write(cache_file, {"identity": scenario, "value": index})
        result_file = results_root / f"node{index}" / "source_rules.json"
        _write(
            result_file,
            {
                "runtime": {"source_tree_sha256": source_sha},
                "safety_audit": {"passed": True},
                "scenario_policy_costs": {
                    scenario: {"p0": 1.0 + index, "p1": 2.0 + index}
                },
                "row_count": 1,
                "cache_files": [str(cache_file)],
            },
        )
        result_files.append(result_file)
    aggregate = aggregate_cache_sha256(sorted(cache_root.glob("*.json")))
    audit_path = tmp_path / "audit.json"
    _write(
        audit_path,
        {
            "passed": True,
            "errors": [],
            "result_files": [str(path) for path in result_files],
            "cache_file_count": 2,
            "scenario_count": 2,
            "total_rows": 2,
            "source_tree_sha256": source_sha,
        },
    )
    return results_root, audit_path, aggregate


def test_build_baseline_is_deterministic_and_audited(tmp_path: Path) -> None:
    results_root, audit_path, aggregate = _fixture(tmp_path)
    first = build_baseline(
        results_root=results_root,
        audit_path=audit_path,
        expected_cache_aggregate_sha256=aggregate,
    )
    second = build_baseline(
        results_root=results_root,
        audit_path=audit_path,
        expected_cache_aggregate_sha256=aggregate,
    )
    assert first == second
    assert first["source_rule_policy_costs"]["city_a"]["p0"] == 1.0
    assert first["source_rule_cache"]["cache_file_count"] == 2
    assert first["source_rule_cache"]["aggregate_sha256"] == aggregate


def test_build_baseline_rejects_cache_content_drift(tmp_path: Path) -> None:
    results_root, audit_path, aggregate = _fixture(tmp_path)
    cache_file = next((tmp_path / "cache").glob("*.json"))
    cache_file.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="aggregate SHA-256 mismatch"):
        build_baseline(
            results_root=results_root,
            audit_path=audit_path,
            expected_cache_aggregate_sha256=aggregate,
        )


def test_aggregate_cache_sha_rejects_duplicate_basenames(tmp_path: Path) -> None:
    left = tmp_path / "left" / "same.json"
    right = tmp_path / "right" / "same.json"
    _write(left, {"side": "left"})
    _write(right, {"side": "right"})
    with pytest.raises(ValueError, match="basenames are not unique"):
        aggregate_cache_sha256((left, right))
