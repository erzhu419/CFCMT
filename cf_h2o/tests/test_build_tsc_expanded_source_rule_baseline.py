import json
from pathlib import Path

import pytest

from scripts.cluster.build_tsc_expanded_source_rule_baseline import (
    _sha256,
    build_expanded_baseline,
)


def _dump(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_expanded_baseline_unions_audited_disjoint_scenarios(tmp_path: Path) -> None:
    base = _dump(
        tmp_path / "base.json",
        {"source_rule_policy_costs": {"source": {"p": 1.0}}},
    )
    result = _dump(
        tmp_path / "partition" / "node" / "source_rules.json",
        {
            "scenario_policy_costs": {"target": {"p": 2.0}},
            "safety_audit": {"passed": True},
            "row_count": 2,
            "runtime": {"source_tree_sha256": "a" * 64},
        },
    )
    audit = _dump(
        tmp_path / "audit.json",
        {
            "passed": True,
            "errors": [],
            "cache_aggregate_sha256": "b" * 64,
            "scenario_count": 1,
            "result_files": [str(result.resolve())],
        },
    )

    payload = build_expanded_baseline(
        base_baseline_path=base,
        expected_base_sha256=_sha256(base),
        partition_results_root=tmp_path / "partition",
        partition_audit_path=audit,
        expected_partition_audit_sha256=_sha256(audit),
        expected_partition_cache_sha256="b" * 64,
    )

    assert payload["scenario_count"] == 2
    assert set(payload["source_rule_policy_costs"]) == {"source", "target"}


def test_expanded_baseline_refuses_scenario_overlap(tmp_path: Path) -> None:
    base = _dump(
        tmp_path / "base.json",
        {"source_rule_policy_costs": {"same": {"p": 1.0}}},
    )
    result = _dump(
        tmp_path / "partition" / "source_rules.json",
        {
            "scenario_policy_costs": {"same": {"p": 2.0}},
            "safety_audit": {"passed": True},
            "row_count": 1,
            "runtime": {},
        },
    )
    audit = _dump(
        tmp_path / "audit.json",
        {
            "passed": True,
            "errors": [],
            "cache_aggregate_sha256": "b" * 64,
            "scenario_count": 1,
            "result_files": [str(result.resolve())],
        },
    )

    with pytest.raises(ValueError, match="scenario overlap"):
        build_expanded_baseline(
            base_baseline_path=base,
            expected_base_sha256=_sha256(base),
            partition_results_root=tmp_path / "partition",
            partition_audit_path=audit,
            expected_partition_audit_sha256=_sha256(audit),
            expected_partition_cache_sha256="b" * 64,
        )
