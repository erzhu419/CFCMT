from __future__ import annotations

import json

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset

from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    _assert_pure_waiting_selector_dataset,
    _candidate_profiles,
    _load_pure_waiting_selector_cache_audit,
    _stable_group_rows,
    nested_leave_one_seed_pressure_selection,
    nested_leave_one_seed_selection,
    select_negative_transfer_safe_profile,
    select_pressure_safe_source_profile,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256


SEEDS = (101, 102, 103, 104, 105, 106)


def test_stable_group_rows_matches_sorted_repeated_scan() -> None:
    groups = np.asarray(["b", "a", "c", "a", "b", "b", "c"])
    expected_groups = np.unique(groups)
    expected_rows = tuple(
        np.flatnonzero(groups == group) for group in expected_groups
    )

    actual_groups, actual_rows = _stable_group_rows(groups)

    assert np.array_equal(actual_groups, expected_groups)
    assert len(actual_rows) == len(expected_rows)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(actual_rows, expected_rows, strict=True)
    )


def _selector_dataset(*, estimand: str) -> MechanismDataset:
    values = np.asarray([1.0, 2.0])
    return MechanismDataset(
        feature_names=("feature",),
        features=np.zeros((2, 1)),
        context_names=("context",),
        context=np.zeros((2, 1)),
        priors={},
        targets={
            "interval_cost": values.copy(),
            "prefix_mean_cost_450s": values.copy(),
        },
        domains=np.asarray(["jinan", "jinan"]),
        metadata={
            "counterfactual_cost_mode": "halted_queue",
            "counterfactual_estimand": estimand,
            "counterfactual_cost_population": (
                "sumo_last_step_halting_number_on_controlled_lanes"
            ),
        },
    )


def test_selector_rejects_pre_v6_composite_queue_cache() -> None:
    with pytest.raises(ValueError, match="pure waiting-aligned"):
        _assert_pure_waiting_selector_dataset(
            _selector_dataset(
                estimand=(
                    "one-control-interval-local-mechanisms-plus-"
                    "halted-queue-rollout-value-v1"
                )
            ),
            target_name="prefix_mean_cost_450s",
        )


def test_selector_accepts_exact_v6_pure_waiting_cache() -> None:
    _assert_pure_waiting_selector_dataset(
        _selector_dataset(
            estimand=(
                "one-control-interval-local-mechanisms-plus-pure-"
                "halted-queue-rollout-value-v2"
            )
        ),
        target_name="prefix_mean_cost_450s",
    )


def test_selector_cache_audit_gate_accepts_only_complete_v116_evidence(
    tmp_path,
) -> None:
    path = tmp_path / "selector_audit.json"
    payload = {
        "protocol": "tsc-v116-pure-waiting-selector-cache-audit-v1",
        "status": "PASS",
        "decision": "authorize_v116_pure_waiting_nested_source_selection",
        "gate": {"passed": True},
        "scenario": "jinan_3x4_real",
        "seeds": [101, 102],
        "seed_count": 2,
        "shards_per_seed": 3,
        "cache_file_count": 6,
        "expected_cache_file_count": 6,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    accepted = _load_pure_waiting_selector_cache_audit(
        path,
        expected_sha256=_sha256(path),
        scenario="jinan_3x4_real",
        seeds=(101, 102),
        collection_shards=3,
    )
    assert accepted["cache_file_count"] == 6

    payload["gate"]["passed"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        _load_pure_waiting_selector_cache_audit(
            path,
            expected_sha256=_sha256(path),
            scenario="jinan_3x4_real",
            seeds=(101, 102),
            collection_shards=3,
        )


def _profile(
    key: str,
    *,
    source: str | None,
    weight: float,
    deltas: tuple[float, ...],
    target_key: str | None = None,
) -> dict:
    return {
        "profile_key": key,
        "source_city_group": source,
        "source_weight": weight,
        "risk_multiplier": 0.5,
        "minimum_context_trust": 0.5,
        "minimum_source_support": 0.0,
        "target_profile_key": target_key,
        "seed_metrics": {
            str(seed): {
                "group_count": 1000,
                "accepted_override_count": 10,
                "harmful_override_count": 2,
                "mean_normalized_delta_vs_phase_pressure": delta,
            }
            for seed, delta in zip(SEEDS, deltas, strict=True)
        },
    }


def test_selector_admits_only_stable_source_contribution() -> None:
    profiles = {
        "target": _profile(
            "target",
            source=None,
            weight=0.0,
            deltas=(-0.002,) * len(SEEDS),
        ),
        "source": _profile(
            "source",
            source="hangzhou",
            weight=0.5,
            deltas=(-0.004,) * len(SEEDS),
            target_key="target",
        ),
    }

    result = select_negative_transfer_safe_profile(
        profiles, training_seeds=SEEDS
    )

    assert result["selected_profile_key"] == "source"
    assert result["selected_source_city_group"] == "hangzhou"
    assert result["selected_source_weight"] == pytest.approx(0.5)


def test_selector_suppresses_negative_source_and_keeps_target_only() -> None:
    profiles = {
        "target": _profile(
            "target",
            source=None,
            weight=0.0,
            deltas=(-0.003,) * len(SEEDS),
        ),
        "source": _profile(
            "source",
            source="hangzhou",
            weight=1.0,
            deltas=(-0.001,) * len(SEEDS),
            target_key="target",
        ),
    }

    result = select_negative_transfer_safe_profile(
        profiles, training_seeds=SEEDS
    )

    assert result["selected_profile_key"] == "target"
    assert result["selected_source_city_group"] is None
    assert result["selection_reason"] == (
        "negative_transfer_suppressed_target_only_fallback"
    )


def test_selector_falls_back_to_exact_pressure_when_all_profiles_harm() -> None:
    profiles = {
        "target": _profile(
            "target",
            source=None,
            weight=0.0,
            deltas=(0.002,) * len(SEEDS),
        ),
        "source": _profile(
            "source",
            source="hangzhou",
            weight=1.0,
            deltas=(0.003,) * len(SEEDS),
            target_key="target",
        ),
    }

    result = select_negative_transfer_safe_profile(
        profiles, training_seeds=SEEDS
    )

    assert result["selected_profile_key"] is None
    assert result["selected_source_weight"] == pytest.approx(0.0)
    assert result["selection_reason"] == (
        "negative_transfer_suppressed_phase_pressure_fallback"
    )


def test_nested_selector_scores_each_seed_only_after_holding_it_out() -> None:
    profiles = {
        "target": _profile(
            "target",
            source=None,
            weight=0.0,
            deltas=(-0.002,) * len(SEEDS),
        ),
        "source": _profile(
            "source",
            source="hangzhou",
            weight=0.5,
            deltas=(-0.004,) * len(SEEDS),
            target_key="target",
        ),
    }

    result = nested_leave_one_seed_selection(profiles, SEEDS)

    assert len(result["folds"]) == len(SEEDS)
    assert {row["heldout_seed"] for row in result["folds"]} == set(SEEDS)
    assert all(row["training_seed_count"] == len(SEEDS) - 1 for row in result["folds"])
    assert result["summary"]["source_transfer_gate_passed"] is True


def test_multisource_support_gate_requires_cross_source_agreement() -> None:
    profiles, _ = _candidate_profiles(
        source_city_group="all_sources_mean",
        source_weight=1.0,
        score=np.asarray([0.0, -1.0, 0.0, -1.0]),
        uncertainty=np.zeros(4),
        trust=np.ones(4),
        group_rows=(np.asarray([0, 1]), np.asarray([2, 3])),
        pressure_rows=np.asarray([0, 2]),
        group_seeds=np.asarray([101, 102]),
        normalized_actual=np.asarray([0.0, -0.1, 0.0, -0.1]),
        source_support_scores=np.asarray(
            [
                [0.0, -1.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, -1.0],
            ]
        ),
        minimum_source_support_values=(0.6,),
    )

    selected = profiles[
        "all_sources_mean__w1__risk0__trust0p25__support0p6"
    ]
    assert selected["seed_metrics"]["101"]["accepted_override_count"] == 0
    assert selected["seed_metrics"]["102"]["accepted_override_count"] == 1


def test_zero_shot_selector_uses_pressure_only_and_admits_stable_source() -> None:
    profiles = {
        "source": _profile(
            "source",
            source="b0_hangzhou",
            weight=1.0,
            deltas=(-0.004,) * len(SEEDS),
        )
    }
    selected = select_pressure_safe_source_profile(
        profiles, training_seeds=SEEDS
    )
    assert selected["selected_profile_key"] == "source"
    assert selected["selected_source_city_group"] == "b0_hangzhou"

    nested = nested_leave_one_seed_pressure_selection(profiles, SEEDS)
    assert nested["summary"]["source_transfer_gate_passed"] is True


def test_zero_shot_selector_falls_back_when_source_harms_pressure() -> None:
    profiles = {
        "source": _profile(
            "source",
            source="b0_hangzhou",
            weight=1.0,
            deltas=(0.004,) * len(SEEDS),
        )
    }
    selected = select_pressure_safe_source_profile(
        profiles, training_seeds=SEEDS
    )
    assert selected["selected_profile_key"] is None
    assert selected["selection_reason"] == (
        "zero_shot_source_suppressed_phase_pressure_fallback"
    )
