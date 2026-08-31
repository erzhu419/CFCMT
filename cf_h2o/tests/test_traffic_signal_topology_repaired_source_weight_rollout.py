from __future__ import annotations

import pickle

import numpy as np

from cf_h2o.eval.traffic_signal_anchored_pairwise_development import (
    ANCHOR_FAMILY,
    CORRECTION_FAMILY,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_topology_repaired_external_refit import (
    MODEL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_topology_repaired_source_weight_rollout import (
    load_topology_repaired_runtime_override,
    source_weight_key,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)


class _Prior:
    key = "rule"


class _Model:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, dataset):
        return {
            "control_cost": {
                "mean": np.full(dataset.size, self.value),
                "uncertainty": np.zeros(dataset.size),
                "context_trust": np.ones(dataset.size),
            }
        }


def test_runtime_override_uses_frozen_source_weight(tmp_path):
    offline = OfflineScreeningModels(
        family_models={
            ANCHOR_FAMILY: _Model(4.0),
            CORRECTION_FAMILY: _Model(4.0),
            "causal_target_only": _Model(2.0),
        },
        prior_spec=_Prior(),
        objective_modes={
            ANCHOR_FAMILY: "control_only",
            CORRECTION_FAMILY: "control_only",
            "causal_target_only": "control_only",
        },
        diagnostics={},
    )
    path = tmp_path / "model.pkl"
    path.write_bytes(
        pickle.dumps(
            {
                "protocol": MODEL_PROTOCOL,
                "city": "los_angeles",
                "selected_candidate": "constant_alpha_0",
                "source_weight_grid": (0.0, 0.25, 0.5, 0.75, 1.0),
                "model": offline,
                "topology_repair_protocol": "repair-v1",
            }
        )
    )
    override = load_topology_repaired_runtime_override(
        model_path=path,
        expected_model_sha256=_sha256(path),
        city="los_angeles",
        source_weight=0.25,
        prediction_horizon_sec=120,
    )
    assert override.provenance["source_weight"] == 0.25
    assert override.runtime_policy.endswith("_contrast_raw")
    assert source_weight_key(0.25) == "0p25"

    guarded = load_topology_repaired_runtime_override(
        model_path=path,
        expected_model_sha256=_sha256(path),
        city="los_angeles",
        source_weight=0.25,
        prediction_horizon_sec=120,
        guard_config=ContrastGuardConfig(enabled=True, risk_multiplier=0.25),
    )
    assert guarded.runtime_policy.endswith("_contrast_guard")
    assert guarded.provenance["guard_config"]["risk_multiplier"] == 0.25
