from __future__ import annotations

import numpy as np

from cf_h2o.traffic_signal.demand_schedule_context import (
    DEMAND_SCHEDULE_FEATURE_NAMES,
    DEMAND_SCHEDULE_PROTOCOL,
    DemandScheduleProfile,
)
from cf_h2o.traffic_signal.demand_timeline_context import (
    TLS_TOPOLOGY_FEATURE_NAMES,
    TLS_TOPOLOGY_PROTOCOL,
    TLSLocalTopologyContext,
)
from cf_h2o.traffic_signal.local_demand_target_veto import (
    LocalDemandTargetVeto,
    LocalDemandVetoConfig,
    VETO_PROTOCOL,
)
from cf_h2o.traffic_signal.local_demand_timeline_context import (
    LOCAL_DEMAND_FEATURE_NAMES,
    LOCAL_DEMAND_PROTOCOL,
    LocalDemandTimelineContext,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    ProposalConditionalRegressor,
)


INPUT_SHA = "b" * 64


def _contexts() -> tuple[
    DemandScheduleProfile, TLSLocalTopologyContext, LocalDemandTimelineContext
]:
    profile = DemandScheduleProfile(
        protocol=DEMAND_SCHEDULE_PROTOCOL,
        horizon_sec=3600,
        tls_count=1,
        scheduled_vehicle_count=10.0,
        input_sha256=INPUT_SHA,
        feature_names=DEMAND_SCHEDULE_FEATURE_NAMES,
        features=tuple(0.01 * index for index in range(len(DEMAND_SCHEDULE_FEATURE_NAMES))),
    )
    topology = TLSLocalTopologyContext(
        protocol=TLS_TOPOLOGY_PROTOCOL,
        input_sha256=INPUT_SHA,
        feature_names=TLS_TOPOLOGY_FEATURE_NAMES,
        tls_features={"tls0": tuple(0.02 * index for index in range(len(TLS_TOPOLOGY_FEATURE_NAMES)))},
    )
    local = LocalDemandTimelineContext(
        protocol=LOCAL_DEMAND_PROTOCOL,
        horizon_sec=3600,
        begin_sec=0.0,
        bin_sec=300,
        input_sha256=INPUT_SHA,
        feature_names=LOCAL_DEMAND_FEATURE_NAMES,
        tls_arrivals={"tls0": tuple([1.0] * 12)},
        incoming_lane_count={"tls0": 1},
        tls_neighbors={"tls0": ()},
    )
    return profile, topology, local


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
            "row_times": [300.0, 300.0],
        },
    )


def test_local_demand_veto_uses_runtime_tls_and_time() -> None:
    profile, topology, local = _contexts()
    names = (
        "x",
        *profile.feature_names,
        *topology.feature_names,
        *local.feature_names,
    )
    model = ProposalConditionalRegressor(feature_names=names)
    context = np.concatenate(
        [
            profile.vector(),
            topology.vector_for("tls0"),
            local.vector_at("tls0", 300.0),
        ]
    )
    training = np.vstack(
        [np.concatenate([[value], context]) for value in (0.0, 1.0, 2.0, 3.0)]
    )
    model.fit(training, np.asarray([0.2, -0.2, -0.4, -0.6]))
    veto = LocalDemandTargetVeto(
        model=model,
        config=LocalDemandVetoConfig(
            enabled=True, acceptance_quantile=0.5, score_threshold=1.0
        ),
        candidate_feature_names=("x",),
        demand_profile=profile,
        topology_context=topology,
        local_demand_context=local,
        target_transform="raw_scenario_robust_scaled",
        target_scale=2.0,
    )

    result = veto.evaluate(_dataset(), candidate_index=1, reference_index=0)

    assert result["protocol"] == VETO_PROTOCOL
    assert result["eligible"] is True
    assert result["runtime_tls_id"] == "tls0"
    assert result["runtime_time_sec"] == 300.0
    assert veto.diagnostics()["action_originator"] is False


def test_local_demand_veto_requires_explicit_runtime_metadata() -> None:
    profile, topology, local = _contexts()
    model = ProposalConditionalRegressor(
        feature_names=(
            "x",
            *profile.feature_names,
            *topology.feature_names,
            *local.feature_names,
        )
    )
    context = np.concatenate(
        [
            profile.vector(),
            topology.vector_for("tls0"),
            local.vector_at("tls0", 300.0),
        ]
    )
    model.fit(
        np.vstack([np.concatenate([[value], context]) for value in (0.0, 1.0)]),
        np.asarray([0.0, 1.0]),
    )
    veto = LocalDemandTargetVeto(
        model=model,
        config=LocalDemandVetoConfig(
            enabled=False, acceptance_quantile=0.5, score_threshold=0.0
        ),
        candidate_feature_names=("x",),
        demand_profile=profile,
        topology_context=topology,
        local_demand_context=local,
        target_transform="raw_scenario_robust_scaled",
        target_scale=1.0,
    )
    dataset = _dataset()
    del dataset.metadata["row_times"]

    try:
        veto.evaluate(dataset, candidate_index=1, reference_index=0)
    except ValueError as error:
        assert "runtime metadata" in str(error)
    else:
        raise AssertionError("missing runtime time metadata was not rejected")
