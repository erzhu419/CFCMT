from __future__ import annotations

from pathlib import Path
import shlex

from scripts.cluster.launch_tsc_cross_fitted_b100_source_predictions import (
    SIGNATURE as V120_SIGNATURE,
    build_spec as build_v120_spec,
)
from scripts.cluster.launch_tsc_stacked_b100_source_gate import (
    SEEDS,
    SIGNATURE as V121_SIGNATURE,
    build_spec as build_v121_spec,
)


def test_v120_spec_freezes_80_of_100_group_crossfit() -> None:
    spec = build_v120_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit/result.json"),
        fit_result_sha256="a" * 64,
        source_cache_root=Path("/cache/source"),
        source_cache_audit_remote=Path("/audit/source.json"),
        target_cache_root=Path("/cache/target"),
        target_cache_audit_remote=Path("/audit/target.json"),
        remote_output_root=Path("/results/v120"),
        node="node003",
        cache_workers=20,
        fit_workers=20,
    )
    tokens = shlex.split(spec["cmd"].split(" && ", 1)[0])
    assert spec["signature"] == V120_SIGNATURE
    assert spec["cpu"] == 20
    assert spec["ram_mb"] == 98304
    assert "cf_h2o.eval.traffic_signal_target_calibrated_source_crossfit" in tokens
    assert tokens[tokens.index("--fit-workers") + 1] == "20"
    assert tokens[tokens.index("--artifact") + 1].endswith(
        "crossfit_predictions_v1.pkl"
    )
    assert spec["cmd"].endswith("printf 'TASK_DONE\\n'")


def test_v121_spec_consumes_frozen_crossfit_without_selector_fit_labels() -> None:
    spec = build_v121_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit/result.json"),
        fit_result_sha256="a" * 64,
        crossfit_artifact_remote=Path("/v120/crossfit.pkl"),
        crossfit_artifact_sha256="b" * 64,
        target_cache_root=Path("/cache/target"),
        target_cache_audit_remote=Path("/audit/target.json"),
        selector_cache_root=Path("/cache/selector"),
        selector_cache_audit_remote=Path("/audit/selector.json"),
        selector_cache_audit_sha256="c" * 64,
        remote_output_root=Path("/results/v121"),
        node="node001",
        cache_workers=20,
        prediction_workers=7,
    )
    tokens = shlex.split(spec["cmd"].split(" && ", 1)[0])
    assert spec["signature"] == V121_SIGNATURE
    assert "cf_h2o.eval.traffic_signal_stacked_b100_source_gate" in tokens
    assert tokens[tokens.index("--crossfit-artifact-sha256") + 1] == "b" * 64
    seed_start = tokens.index("--selector-seeds") + 1
    seed_stop = tokens.index("--selector-collection-shards")
    assert [int(value) for value in tokens[seed_start:seed_stop]] == list(SEEDS)
    assert spec["cmd"].endswith("printf 'TASK_DONE\\n'")
