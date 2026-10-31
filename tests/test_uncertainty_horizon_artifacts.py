from __future__ import annotations

import json
from pathlib import Path

from scripts.cluster.freeze_tsc_external_v9_uncertainty_horizon_artifacts import (
    ARTIFACT_KEYS,
    MINIMUM_CONTEXT_TRUST,
    RISK_MULTIPLIER,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v82_external_v9_"
    "uncertainty_horizon_artifacts.json"
)


def _protocol() -> dict[str, object]:
    return json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_v82_artifact_grid_is_exact_and_directionally_supported() -> None:
    protocol = _protocol()
    artifacts = protocol["artifacts"]

    assert tuple(row["artifact_key"] for row in artifacts) == ARTIFACT_KEYS
    assert tuple(row["prediction_horizon_sec"] for row in artifacts) == (
        60,
        60,
        120,
        120,
    )
    assert all(row["risk_multiplier"] == RISK_MULTIPLIER for row in artifacts)
    assert all(
        row["minimum_context_trust"] == MINIMUM_CONTEXT_TRUST
        for row in artifacts
    )
    assert all(row["selection_eligible_closed_loop"] for row in artifacts)
    assert all(
        row["development_screen_summary"]["bootstrap"]["ci95"][1] < 0.0
        for row in artifacts
    )
    assert all(
        row["development_screen_summary"]["retained_seed_count"] == 22
        for row in artifacts
    )


def test_v82_reuses_h60_and_refits_only_h120() -> None:
    artifacts = _protocol()["artifacts"]

    assert [row["model_source"]["kind"] for row in artifacts] == [
        "reuse_v79_frozen_model",
        "reuse_v79_frozen_model",
        "fit_full_development_cache",
        "fit_full_development_cache",
    ]


def test_v82_holdout_partitions_remain_disjoint() -> None:
    training = _protocol()["training"]
    development = set(training["seeds"])
    confirmatory = set(training["sealed_confirmatory_seeds"])
    prospective = set(training["sealed_prospective_seeds"])

    assert (len(development), len(confirmatory), len(prospective)) == (22, 32, 8)
    assert not development.intersection(confirmatory)
    assert not development.intersection(prospective)
    assert not confirmatory.intersection(prospective)
