from __future__ import annotations

import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
)
from cf_h2o.eval.traffic_signal_causal_source_component_rollout import (
    load_pure_waiting_h2oplus_runtime_override,
    load_pure_waiting_selector_runtime_override,
    run_pure_waiting_selector_rollout,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    H2OPLUS_MODEL_PROTOCOL,
    MODEL_BUNDLE_PROTOCOL,
    STRICT_TARGET_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    PURE_WAITING_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.generalized_pressure import PHASE_PRESSURE_SPEC


class _FixedModel:
    def predict(self, dataset):
        values = np.zeros(dataset.size)
        return {
            "control_cost": {
                "mean": values,
                "uncertainty": values,
                "context_trust": np.ones(dataset.size),
            }
        }


def _write_pickle(path: Path, payload) -> str:
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return _sha256(path)


def _component_payload(*, budget: int) -> dict:
    component = SimpleNamespace(
        family_models={
            ANCHOR_FAMILY: _FixedModel(),
            CORRECTION_FAMILY: _FixedModel(),
        },
        objective_modes={
            ANCHOR_FAMILY: "control_only",
            CORRECTION_FAMILY: "control_only",
        },
        prior_spec=PHASE_PRESSURE_SPEC,
    )
    candidate = "constant_alpha_0p9"
    assert candidate in CANDIDATE_KEYS
    selection_sha256 = "b" * 64
    return {
        "protocol": MODEL_BUNDLE_PROTOCOL,
        "city": "los_angeles",
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "target_group_budget": budget,
        "selected_group_ids": () if budget == 0 else ("g1",),
        "adaptation_contract_sha256": "adaptation",
        "selected_candidate": candidate,
        "candidate_selection_sha256": selection_sha256,
        "blend_candidate_selection": {
            "protocol": (
                "pure-waiting-target-label-free-source-loco-blend-selection-v1"
            ),
            "target_transition_labels_consumed": 0,
            "source_city_groups": list(DEVELOPMENT_CITY_GROUPS),
            "candidate_count": len(CANDIDATE_KEYS),
            "gate_passed": True,
            "selected_candidate": candidate,
            "fallback_candidate": "constant_alpha_0",
            "decision": "freeze_source_loco_candidate",
            "selection_sha256": selection_sha256,
        },
        "component_models": {
            "hangzhou": component,
            "new_york": component,
        },
    }


def _strict_payload() -> dict:
    candidate = "constant_alpha_0p9"
    selection_sha256 = "b" * 64
    return {
        "protocol": STRICT_TARGET_MODEL_PROTOCOL,
        "city": "los_angeles",
        "family": TARGET_ONLY_FAMILY,
        "target_name": "prefix_mean_cost_450s",
        "target_group_budget": 100,
        "selected_group_ids": ("g1",),
        "prior_policy": "phase_pressure",
        "adaptation_contract_sha256": "adaptation",
        "selected_candidate": candidate,
        "blend_candidate_selection_sha256": selection_sha256,
        "architecture_match": {
            "protocol": "pure-waiting-architecture-matched-target-only-v1",
            "base_families": [ANCHOR_FAMILY, CORRECTION_FAMILY],
            "selected_candidate": candidate,
            "blend_candidate_selection_sha256": selection_sha256,
            "source_transition_rows_consumed": 0,
        },
        "model": _FixedModel(),
        "fit_diagnostics": {
            "source_row_count_consumed": 0,
            "target_group_count": 100,
        },
    }


def _h2oplus_payload(*, budget: int) -> dict:
    groups = tuple(f"g{index}" for index in range(budget))
    domains = ["hangzhou", "new_york"]
    if budget:
        domains.append("los_angeles")
    return {
        "protocol": H2OPLUS_MODEL_PROTOCOL,
        "city": "los_angeles",
        "family": "dense",
        "objective_mode": "control_only",
        "target_name": "prefix_mean_cost_450s",
        "prior_policy": "phase_pressure",
        "prior_spec": PHASE_PRESSURE_SPEC,
        "target_group_budget": budget,
        "selected_group_ids": groups,
        "adaptation_contract_sha256": "adaptation",
        "model": _FixedModel(),
        "fit_diagnostics": {
            "source_domains": domains,
            "target_adaptation_groups": budget,
        },
    }


def _selector_payload() -> dict:
    target_key = "target_only__w0__risk0p5__trust0p5"
    source_key = "all_sources_mean__w0p5__risk0p5__trust0p5__support0p6"
    b0_key = "b0_hangzhou__w1__risk0p5__trust0p5"
    return {
        "protocol": PURE_WAITING_RESULT_PROTOCOL,
        "city": "jinan",
        "selector_cache_audit": {
            "decision": "authorize_v116_pure_waiting_nested_source_selection"
        },
        "source_transfer_gate_passed": True,
        "full_development_selection": {
            "selected_profile_key": source_key,
            "selected_source_city_group": "all_sources_mean",
        },
        "profile_definitions": {
            target_key: {
                "profile_key": target_key,
                "source_city_group": None,
                "source_weight": 0.0,
                "risk_multiplier": 0.5,
                "minimum_context_trust": 0.5,
                "minimum_source_support": 0.0,
                "target_profile_key": None,
            },
            source_key: {
                "profile_key": source_key,
                "source_city_group": "all_sources_mean",
                "source_weight": 0.5,
                "risk_multiplier": 0.5,
                "minimum_context_trust": 0.5,
                "minimum_source_support": 0.6,
                "target_profile_key": target_key,
            },
        },
        "zero_shot_source_admission": {
            "source_transfer_gate_passed": True,
            "full_development_selection": {
                "selected_profile_key": b0_key,
                "selected_source_city_group": "b0_hangzhou",
            },
            "profile_definitions": {
                b0_key: {
                    "profile_key": b0_key,
                    "source_city_group": "b0_hangzhou",
                    "source_weight": 1.0,
                    "risk_multiplier": 0.5,
                    "minimum_context_trust": 0.5,
                    "minimum_source_support": 0.0,
                    "target_profile_key": None,
                }
            },
        },
    }


@pytest.mark.parametrize(
    ("arm", "budget", "needs_strict"),
    (
        ("target_only_b100", 100, True),
        ("source_selected_b100", 100, True),
        ("source_selected_b0", 0, False),
    ),
)
def test_runtime_loader_preserves_v116_information_budget(
    tmp_path, arm, budget, needs_strict
) -> None:
    bundle_path = tmp_path / f"bundle_{budget}.pkl"
    bundle_sha = _write_pickle(bundle_path, _component_payload(budget=budget))
    strict_path = tmp_path / "strict.pkl"
    strict_sha = _write_pickle(strict_path, _strict_payload())
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(json.dumps(_selector_payload()), encoding="utf-8")

    runtime = load_pure_waiting_selector_runtime_override(
        component_bundle_path=bundle_path,
        expected_component_bundle_sha256=bundle_sha,
        selector_result_path=selector_path,
        expected_selector_result_sha256=_sha256(selector_path),
        city="los_angeles",
        arm=arm,
        strict_target_only_model_path=strict_path if needs_strict else None,
        expected_strict_target_only_model_sha256=(
            strict_sha if needs_strict else None
        ),
    )
    assert runtime.provenance["target_transition_label_budget"] == budget
    assert runtime.provenance["arm"] == arm
    assert runtime.models.prior_policy == "phase_pressure"


def test_runtime_loader_rejects_source_when_nested_gate_failed(tmp_path) -> None:
    bundle_path = tmp_path / "bundle.pkl"
    bundle_sha = _write_pickle(bundle_path, _component_payload(budget=100))
    strict_path = tmp_path / "strict.pkl"
    strict_sha = _write_pickle(strict_path, _strict_payload())
    selector = _selector_payload()
    selector["source_transfer_gate_passed"] = False
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(json.dumps(selector), encoding="utf-8")

    with pytest.raises(ValueError, match="did not authorize"):
        load_pure_waiting_selector_runtime_override(
            component_bundle_path=bundle_path,
            expected_component_bundle_sha256=bundle_sha,
            selector_result_path=selector_path,
            expected_selector_result_sha256=_sha256(selector_path),
            city="los_angeles",
            arm="source_selected_b100",
            strict_target_only_model_path=strict_path,
            expected_strict_target_only_model_sha256=strict_sha,
        )


@pytest.mark.parametrize(
    ("arm", "budget"),
    (("h2oplus_dense_b0", 0), ("h2oplus_dense_b100", 100)),
)
def test_h2oplus_runtime_loader_preserves_equal_information_budget(
    tmp_path, arm, budget
) -> None:
    model_path = tmp_path / f"h2oplus_{budget}.pkl"
    model_sha = _write_pickle(model_path, _h2oplus_payload(budget=budget))
    runtime = load_pure_waiting_h2oplus_runtime_override(
        model_path=model_path,
        expected_model_sha256=model_sha,
        expected_adaptation_contract_sha256="adaptation",
        city="los_angeles",
        arm=arm,
    )
    assert runtime.runtime_policy == "dense_contrast_raw"
    assert runtime.provenance["target_transition_label_budget"] == budget
    assert runtime.models.prediction_horizon_sec == 450


def test_v117_rollout_rejects_selector_development_city(tmp_path) -> None:
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(
        json.dumps(
            {
                "counterfactual_cache": {
                    "control_interval_sec": 10,
                    "counterfactual_horizon_intervals": 45,
                }
            }
        ),
        encoding="utf-8",
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(
        json.dumps(
            {
                "parent_protocol": {"path": str(parent_path)},
                "target_protocol": {
                    "external_city_scenarios": {
                        "los_angeles": ["la_1x4"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(json.dumps({"city": "los_angeles"}), encoding="utf-8")

    with pytest.raises(ValueError, match="held out"):
        run_pure_waiting_selector_rollout(
            component_bundle_path=tmp_path / "unused.pkl",
            expected_component_bundle_sha256="unused",
            selector_result_path=selector_path,
            expected_selector_result_sha256="unused",
            arm="source_selected_b100",
            candidate_key="source_selected_b100",
            allowed_seeds=(1,),
            rollout_kwargs={
                "protocol_spec_path": protocol_path,
                "scenario": "la_1x4",
            },
        )
