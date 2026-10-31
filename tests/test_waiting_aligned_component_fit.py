from __future__ import annotations

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
)

from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    ANCHOR_FAMILY,
    BASE_FAMILIES,
    CORRECTION_FAMILY,
    STRICT_TARGET_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY,
    _assert_waiting_aligned_bank,
    _contract_sha256,
    _select_source_loco_candidate,
    _source_shard_contract,
    validate_architecture_matched_target_contract,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def _dataset(interval_cost: np.ndarray, waiting: np.ndarray) -> MechanismDataset:
    rows = len(interval_cost)
    return MechanismDataset(
        features=np.zeros((rows, 1), dtype=float),
        context=np.zeros((rows, 1), dtype=float),
        priors={},
        targets={
            "interval_cost": np.asarray(interval_cost, dtype=float),
            "prefix_mean_cost_450s": np.asarray(waiting, dtype=float),
        },
        domains=np.asarray(["domain"] * rows),
        feature_names=("feature",),
        context_names=("context",),
        metadata={
            "counterfactual_cost_mode": "halted_queue",
            "counterfactual_estimand": WAITING_ALIGNED_ESTIMAND_PROTOCOL_V6,
            "counterfactual_cost_population": (
                "sumo_last_step_halting_number_on_controlled_lanes"
            ),
        },
    )


def test_architecture_matched_target_uses_the_frozen_base_family_order() -> None:
    assert BASE_FAMILIES == (ANCHOR_FAMILY, CORRECTION_FAMILY)


def test_waiting_aligned_fit_requires_exact_training_target_equivalence() -> None:
    _assert_waiting_aligned_bank(
        {"scenario": _dataset(np.asarray([1.0]), np.asarray([1.0]))},
        target_name="prefix_mean_cost_450s",
        tolerance=1e-12,
    )
    with pytest.raises(ValueError, match="differ"):
        _assert_waiting_aligned_bank(
            {"scenario": _dataset(np.asarray([1.0]), np.asarray([1.01]))},
            target_name="prefix_mean_cost_450s",
            tolerance=1e-12,
        )


def test_waiting_aligned_adaptation_contract_hash_is_order_stable() -> None:
    left = _contract_sha256({"city": "jinan", "budget": 100})
    right = _contract_sha256({"budget": 100, "city": "jinan"})
    assert left == right


def test_waiting_aligned_fit_rejects_legacy_composite_queue_metadata() -> None:
    dataset = _dataset(np.asarray([1.0]), np.asarray([1.0]))
    dataset.metadata["counterfactual_estimand"] = (
        "one-control-interval-local-mechanisms-plus-halted-queue-rollout-value-v1"
    )
    with pytest.raises(ValueError, match="pure waiting-aligned"):
        _assert_waiting_aligned_bank(
            {"scenario": dataset},
            target_name="prefix_mean_cost_450s",
            tolerance=1e-12,
        )


def test_waiting_aligned_fit_requires_audited_scenario_shard_partition() -> None:
    protocol = {
        "source_training": {
            "collection_shards": 16,
            "scenario_collection_shards": {"manhattan": 64},
        }
    }
    audit = {
        "shards_per_scenario_seed": {"grid": 16, "manhattan": 64}
    }
    assert _source_shard_contract(
        protocol, audit, ("grid", "manhattan")
    ) == (16, {"manhattan": 64})

    audit["shards_per_scenario_seed"]["manhattan"] = 16
    with pytest.raises(ValueError, match="differs"):
        _source_shard_contract(protocol, audit, ("grid", "manhattan"))


def test_architecture_matched_target_contract_rejects_source_rows() -> None:
    payload = {
        "protocol": STRICT_TARGET_MODEL_PROTOCOL,
        "family": TARGET_ONLY_FAMILY,
        "selected_candidate": ANCHOR_KEY,
        "blend_candidate_selection_sha256": "a" * 64,
        "architecture_match": {
            "protocol": "pure-waiting-architecture-matched-target-only-v1",
            "base_families": [
                "causal_group_normalized_rigid_advantage",
                "causal_antisymmetric_pairwise_advantage",
            ],
            "selected_candidate": ANCHOR_KEY,
            "blend_candidate_selection_sha256": "a" * 64,
            "source_transition_rows_consumed": 0,
        },
        "fit_diagnostics": {
            "source_row_count_consumed": 0,
            "target_group_count": 100,
        },
    }
    validate_architecture_matched_target_contract(payload)
    payload["fit_diagnostics"]["source_row_count_consumed"] = 1
    with pytest.raises(ValueError, match="architecture-matched"):
        validate_architecture_matched_target_contract(payload)


def test_source_loco_candidate_selection_falls_back_without_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bank = {
        f"scenario_{group}": _dataset(np.asarray([1.0]), np.asarray([1.0]))
        for group in DEVELOPMENT_CITY_GROUPS
    }
    scenarios = {
        group: (f"scenario_{group}",) for group in DEVELOPMENT_CITY_GROUPS
    }
    models = {group: object() for group in DEVELOPMENT_CITY_GROUPS}

    def fake_evaluate(dataset, *, model_ensemble, excluded_group_ids):
        assert len(model_ensemble) == len(DEVELOPMENT_CITY_GROUPS) - 1
        assert excluded_group_ids == ()
        return {
            "evaluation_group_count": 1,
            "model_ensemble_size": len(model_ensemble),
            "candidates": {
                candidate: {
                    "mean_normalized_action_regret": (
                        0.1 if candidate == ANCHOR_KEY else 0.2
                    ),
                    "mean_alpha": 0.0 if candidate == ANCHOR_KEY else 1.0,
                }
                for candidate in CANDIDATE_KEYS
            },
        }

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_waiting_aligned_component_fit."
        "evaluate_anchored_ensemble_candidates",
        fake_evaluate,
    )
    result = _select_source_loco_candidate(
        source_bank=source_bank,
        source_scenarios_by_group=scenarios,
        b0_models=models,
    )
    assert result["target_transition_labels_consumed"] == 0
    assert result["gate_passed"] is False
    assert result["selected_candidate"] == ANCHOR_KEY
    assert result["decision"] == "suppress_residual_and_freeze_rigid_anchor"


def test_source_loco_candidate_selection_freezes_only_after_loco_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_bank = {
        f"scenario_{group}": _dataset(np.asarray([1.0]), np.asarray([1.0]))
        for group in DEVELOPMENT_CITY_GROUPS
    }
    scenarios = {
        group: (f"scenario_{group}",) for group in DEVELOPMENT_CITY_GROUPS
    }
    models = {group: object() for group in DEVELOPMENT_CITY_GROUPS}
    preferred = next(candidate for candidate in CANDIDATE_KEYS if candidate != ANCHOR_KEY)

    def fake_evaluate(dataset, *, model_ensemble, excluded_group_ids):
        return {
            "evaluation_group_count": 1,
            "model_ensemble_size": len(model_ensemble),
            "candidates": {
                candidate: {
                    "mean_normalized_action_regret": (
                        0.05 if candidate == preferred else 0.10
                    ),
                    "mean_alpha": 0.0 if candidate == ANCHOR_KEY else 0.5,
                }
                for candidate in CANDIDATE_KEYS
            },
        }

    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_waiting_aligned_component_fit."
        "evaluate_anchored_ensemble_candidates",
        fake_evaluate,
    )
    result = _select_source_loco_candidate(
        source_bank=source_bank,
        source_scenarios_by_group=scenarios,
        b0_models=models,
        workers=2,
    )
    assert result["gate_passed"] is True
    assert result["evaluation_workers"] == 2
    assert result["selected_candidate"] == preferred
    assert result["decision"] == "freeze_source_loco_candidate"
