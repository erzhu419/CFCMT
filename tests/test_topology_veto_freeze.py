from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_external_topology_veto_freeze import (
    fit_city_topology_veto,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    TARGET_SUPPORT_FEATURES,
)
from cf_h2o.traffic_signal.demand_timeline_context import (
    TLS_TOPOLOGY_FEATURE_NAMES,
    TLS_TOPOLOGY_PROTOCOL,
    TLSLocalTopologyContext,
)


MODEL_SPEC = {
    "learning_rate": 0.05,
    "max_iter": 20,
    "max_leaf_nodes": 5,
    "min_samples_leaf": 4,
    "l2_regularization": 8.0,
    "random_state": 7,
    "uncertainty_quantile": 0.9,
}


def _topology() -> TLSLocalTopologyContext:
    return TLSLocalTopologyContext(
        protocol=TLS_TOPOLOGY_PROTOCOL,
        input_sha256="d" * 64,
        feature_names=TLS_TOPOLOGY_FEATURE_NAMES,
        tls_features={
            "tls0": tuple(
                0.01 * index for index in range(len(TLS_TOPOLOGY_FEATURE_NAMES))
            )
        },
    )


def _records() -> list[dict[str, object]]:
    records = []
    for index in range(36):
        records.append(
            {
                "short_proposed": True,
                "scenario": "scenario",
                "group_id": f"group{index}:tls0",
                "simulator_seed": index % 3,
                "candidate_features": {
                    name: float((index + offset) % 11) / 10.0
                    for offset, name in enumerate(TARGET_SUPPORT_FEATURES)
                },
                "actual_raw_delta": float(index % 7 - 4) / 100.0,
            }
        )
    return records


def _decision(active: bool) -> dict[str, object]:
    return {
        "decision": "freeze_active_veto" if active else "force_phase_pressure_fallback",
        "variant": "state_topology_raw_scaled",
        "proposal_count": len(_records()),
        "selected_candidate": {"acceptance_quantile": 0.075} if active else None,
    }


def test_topology_freeze_fits_exact_deployment_causal_features() -> None:
    summary, artifact = fit_city_topology_veto(
        city="jinan",
        scenarios=("scenario",),
        records=_records(),
        topology={"scenario": _topology()},
        model_spec=MODEL_SPEC,
        city_decision=_decision(True),
    )
    assert summary["final_config"]["enabled"] is True
    assert summary["final_config"]["acceptance_quantile"] == 0.075
    assert summary["combined_feature_names"] == [
        *TARGET_SUPPORT_FEATURES,
        *TLS_TOPOLOGY_FEATURE_NAMES,
    ]
    assert artifact["scenario_id_is_model_feature"] is False
    assert artifact["veto_role"] == "static_topology_veto_only_never_action_originator"
    assert np.isfinite(float(summary["final_config"]["score_threshold"]))


def test_topology_freeze_keeps_failed_city_disabled() -> None:
    summary, artifact = fit_city_topology_veto(
        city="los_angeles",
        scenarios=("scenario",),
        records=_records(),
        topology={"scenario": _topology()},
        model_spec=MODEL_SPEC,
        city_decision=_decision(False),
    )
    assert summary["final_config"]["enabled"] is False
    assert artifact["scenario_vetoes"]["scenario"].config.enabled is False
