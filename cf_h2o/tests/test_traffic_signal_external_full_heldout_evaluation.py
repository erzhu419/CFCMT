from __future__ import annotations

import hashlib
import pickle
from types import SimpleNamespace

import pytest

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import BASE_FAMILIES
from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    METHOD_MODEL_PROTOCOL,
    REFIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_full_heldout_evaluation import (
    _confirmation_gate,
    _load_full_refit_model_artifact,
)


GATE_SPEC = {
    "minimum_macro_relative_improvement": 0.02,
    "minimum_improved_external_cities": 2,
    "maximum_absolute_city_regression": 0.01,
    "minimum_non_anchor_targets": 1,
}


def _city(anchor: float, selected: float, *, candidate: str = "constant_alpha_0p6"):
    return {
        "anchor_policy": "method/constant_alpha_0",
        "selected_policy": f"method/{candidate}",
        "selected_candidate": candidate,
        "equal_scenario_summary": {
            "method/constant_alpha_0": {
                "mean_normalized_action_regret": anchor
            },
            f"method/{candidate}": {
                "mean_normalized_action_regret": selected
            },
        },
    }


def test_confirmation_gate_requires_both_external_cities_to_improve():
    passed = _confirmation_gate(
        {
            "los_angeles": _city(0.20, 0.15),
            "jinan": _city(0.40, 0.30),
        },
        GATE_SPEC,
    )
    failed = _confirmation_gate(
        {
            "los_angeles": _city(0.20, 0.205),
            "jinan": _city(0.40, 0.30),
        },
        GATE_SPEC,
    )

    assert passed["passed"] is True
    assert passed["improved_external_cities"] == 2
    assert failed["passed"] is False
    assert failed["improved_external_cities"] == 1


def test_full_refit_loader_requires_exactly_one_deployment_model(tmp_path):
    groups = ("scenario:seed5057:0:tls", "scenario:seed6067:1:tls")
    model = SimpleNamespace(
        family_models={family: object() for family in BASE_FAMILIES}
    )
    payload = {
        "protocol": METHOD_MODEL_PROTOCOL,
        "city": "los_angeles",
        "selected_group_ids": groups,
        "deployment_refit_group_ids": groups,
        "deployment_refit_protocol": REFIT_PROTOCOL,
        "model_ensemble": (model,),
    }
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    loaded = _load_full_refit_model_artifact(
        path,
        expected_sha256=digest,
        expected_protocol=METHOD_MODEL_PROTOCOL,
        city="los_angeles",
        expected_families=BASE_FAMILIES,
        expected_group_ids=groups,
    )

    assert len(loaded["model_ensemble"]) == 1


def test_full_refit_loader_rejects_cross_validation_ensemble(tmp_path):
    groups = ("scenario:seed5057:0:tls", "scenario:seed6067:1:tls")
    model = SimpleNamespace(
        family_models={family: object() for family in BASE_FAMILIES}
    )
    payload = {
        "protocol": METHOD_MODEL_PROTOCOL,
        "city": "los_angeles",
        "selected_group_ids": groups,
        "deployment_refit_group_ids": groups,
        "deployment_refit_protocol": REFIT_PROTOCOL,
        "model_ensemble": (model, model),
    }
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="artifact contract failed"):
        _load_full_refit_model_artifact(
            path,
            expected_sha256=digest,
            expected_protocol=METHOD_MODEL_PROTOCOL,
            city="los_angeles",
            expected_families=BASE_FAMILIES,
            expected_group_ids=groups,
        )
