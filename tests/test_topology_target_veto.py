from __future__ import annotations

import numpy as np

from cf_h2o.traffic_signal.demand_timeline_context import (
    TLS_TOPOLOGY_FEATURE_NAMES,
    TLS_TOPOLOGY_PROTOCOL,
    TLSLocalTopologyContext,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
)
from cf_h2o.traffic_signal.topology_target_veto import (
    TopologyTargetVeto,
    TopologyVetoConfig,
    VETO_PROTOCOL,
)


def _topology() -> TLSLocalTopologyContext:
    return TLSLocalTopologyContext(
        protocol=TLS_TOPOLOGY_PROTOCOL,
        input_sha256="c" * 64,
        feature_names=TLS_TOPOLOGY_FEATURE_NAMES,
        tls_features={
            "tls0": tuple(
                0.02 * index for index in range(len(TLS_TOPOLOGY_FEATURE_NAMES))
            )
        },
    )


def _dataset() -> MechanismDataset:
    return MechanismDataset(
        feature_names=("x", "unused"),
        features=np.asarray([[0.0, 1.0], [1.0, 1.0]], dtype=float),
        context_names=("context",),
        context=np.zeros((2, 1)),
        priors={"cost": np.zeros(2)},
        targets={"cost": np.zeros(2)},
        domains=np.asarray(["target", "target"]),
        metadata={
            "action_group_ids": ["g", "g"],
            "is_reference": [True, False],
            "row_tls": ["tls0", "tls0"],
        },
    )


def _veto() -> TopologyTargetVeto:
    topology = _topology()
    model = ProposalConditionalRegressor(
        feature_names=("x", *topology.feature_names)
    )
    context = topology.vector_for("tls0")
    model.fit(
        np.vstack(
            [np.concatenate([[value], context]) for value in (0.0, 1.0, 2.0, 3.0)]
        ),
        np.asarray([0.2, -0.2, -0.4, -0.6]),
    )
    return TopologyTargetVeto(
        model=model,
        config=TopologyVetoConfig(
            enabled=True, acceptance_quantile=0.5, score_threshold=1.0
        ),
        candidate_feature_names=("x",),
        topology_context=topology,
        target_transform="raw_scenario_robust_scaled",
        target_scale=2.0,
    )


def test_topology_veto_uses_only_runtime_tls_context() -> None:
    veto = _veto()
    result = veto.evaluate(_dataset(), candidate_index=1, reference_index=0)
    assert result["protocol"] == VETO_PROTOCOL
    assert result["eligible"] is True
    assert result["runtime_tls_id"] == "tls0"
    assert veto.diagnostics()["action_originator"] is False


def test_topology_veto_requires_runtime_tls_metadata() -> None:
    veto = _veto()
    dataset = _dataset()
    del dataset.metadata["row_tls"]
    try:
        veto.evaluate(dataset, candidate_index=1, reference_index=0)
    except ValueError as error:
        assert "TLS metadata" in str(error)
    else:
        raise AssertionError("missing runtime TLS metadata was not rejected")
