from __future__ import annotations

from pathlib import Path
import shlex

from scripts.cluster.launch_tsc_target_calibrated_source_gate import (
    SEEDS,
    SIGNATURE,
    build_spec,
)


def test_v119_spec_uses_disjoint_selector_and_b0_fit_contract() -> None:
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit/result.json"),
        fit_result_sha256="a" * 64,
        target_cache_root=Path("/cache/target"),
        target_cache_audit_remote=Path("/audit/target.json"),
        selector_cache_root=Path("/cache/selector"),
        selector_cache_audit_remote=Path("/audit/selector.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/results/v119"),
        node="node003",
        cache_workers=20,
        prediction_workers=7,
    )
    tokens = shlex.split(spec["cmd"])
    assert spec["signature"] == SIGNATURE
    assert spec["require_node"] == "node003"
    assert spec["cpu"] == 20
    assert "cf_h2o.eval.traffic_signal_target_calibrated_source_gate" in tokens
    selector_index = tokens.index("--selector-seeds") + 1
    selector_end = tokens.index("--selector-collection-shards")
    assert [int(value) for value in tokens[selector_index:selector_end]] == list(SEEDS)
    assert tokens[tokens.index("--fit-result-sha256") + 1] == "a" * 64
    assert tokens[tokens.index("--selector-cache-audit-sha256") + 1] == "b" * 64
    assert tokens[tokens.index("--gate-artifact") + 1].endswith(
        "target_calibrated_source_gate_v1.pkl"
    )
