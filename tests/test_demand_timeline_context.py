from __future__ import annotations

from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_external_spatiotemporal_veto_diagnostic import (
    VARIANTS,
    feature_matrix,
    feature_names,
)
from cf_h2o.eval.traffic_signal_external_local_demand_model_diagnostic import (
    VARIANT_SPECS as LOCAL_VARIANT_SPECS,
    variant_feature_matrix,
    variant_feature_names,
)
from cf_h2o.traffic_signal.demand_timeline_context import (
    TEMPORAL_DEMAND_FEATURE_NAMES,
    TLS_TOPOLOGY_FEATURE_NAMES,
    build_temporal_demand_context,
    build_tls_local_topology_context,
)
from cf_h2o.traffic_signal.local_demand_timeline_context import (
    LOCAL_DEMAND_FEATURE_NAMES,
    build_local_demand_timeline_context,
)


def _fixture(root: Path, demand_xml: str) -> Path:
    (root / "network.net.xml").write_text(
        "<net>"
        "<edge id='e0' from='tls0' to='tls1'>"
        "<lane id='e0_0' index='0' speed='10' length='100'/></edge>"
        "<edge id='e1' from='tls1' to='tls0'>"
        "<lane id='e1_0' index='0' speed='12' length='120'/></edge>"
        "<tlLogic id='tls0' type='static' programID='0' offset='0'>"
        "<phase duration='30' state='Gr'/><phase duration='30' state='rG'/>"
        "</tlLogic>"
        "<tlLogic id='tls1' type='static' programID='0' offset='0'>"
        "<phase duration='30' state='GG'/></tlLogic>"
        "<connection from='e1' to='e0' fromLane='0' toLane='0' tl='tls0' linkIndex='0'/>"
        "<connection from='e0' to='e1' fromLane='0' toLane='0' tl='tls1' linkIndex='0'/>"
        "</net>",
        encoding="utf-8",
    )
    (root / "demand.rou.xml").write_text(
        f"<routes><route id='r' edges='e0 e1'/>{demand_xml}</routes>",
        encoding="utf-8",
    )
    path = root / "scenario.sumocfg"
    path.write_text(
        "<configuration><input><net-file value='network.net.xml'/>"
        "<route-files value='demand.rou.xml'/></input>"
        "<time><begin value='0'/><end value='3600'/></time></configuration>",
        encoding="utf-8",
    )
    return path


def test_temporal_context_tracks_past_and_future_schedule(tmp_path: Path) -> None:
    demand = "".join(
        f"<vehicle id='v{index}' route='r' depart='{depart}'/>"
        for index, depart in enumerate((100, 200, 400, 700))
    )
    context = build_temporal_demand_context(_fixture(tmp_path, demand))
    values = dict(zip(context.feature_names, context.vector_at(300.0)))

    assert context.feature_names == TEMPORAL_DEMAND_FEATURE_NAMES
    assert sum(context.expected_departures) == 4.0
    assert values["scheduled_prev_300_per_tls"] == 1.0
    assert values["scheduled_next_300_per_tls"] == 0.5
    assert values["scheduled_next_450_per_tls"] == 1.0
    assert values["scheduled_next_prev_300_balance"] < 0.0
    assert values["scheduled_remaining_fraction"] == 0.5


def test_temporal_flow_is_integrated_without_sampling(tmp_path: Path) -> None:
    context = build_temporal_demand_context(
        _fixture(
            tmp_path,
            "<flow id='f' route='r' begin='0' end='600' number='60'/>",
        )
    )
    values = dict(zip(context.feature_names, context.vector_at(0.0)))

    assert np.isclose(sum(context.expected_departures), 60.0)
    assert np.isclose(values["scheduled_next_300_per_tls"], 15.0)
    assert np.isclose(values["scheduled_next_450_per_tls"], 22.5)


def test_tls_topology_uses_structure_not_tls_identifier(tmp_path: Path) -> None:
    context = build_tls_local_topology_context(
        _fixture(tmp_path, "<vehicle id='v0' route='r' depart='100'/>")
    )
    tls0 = dict(zip(context.feature_names, context.vector_for("tls0")))
    tls1 = dict(zip(context.feature_names, context.vector_for("tls1")))

    assert context.feature_names == TLS_TOPOLOGY_FEATURE_NAMES
    assert tls0["tls_graph_degree_norm"] == 1.0
    assert tls0["tls_graph_closeness"] == 1.0
    assert tls0["tls_incoming_edge_count_norm"] == tls1[
        "tls_incoming_edge_count_norm"
    ]
    assert tls0["tls_phase_count_norm"] != tls1["tls_phase_count_norm"]


def test_empty_oof_fold_keeps_explicit_two_dimensional_shape() -> None:
    for variant in VARIANTS:
        matrix = feature_matrix(
            [], variant=variant, profiles={}, temporal={}, topology={}
        )
        assert matrix.shape == (0, len(feature_names(variant)))


def test_route_projected_local_arrivals_follow_free_flow_offsets(
    tmp_path: Path,
) -> None:
    context = build_local_demand_timeline_context(
        _fixture(tmp_path, "<vehicle id='v0' route='r' depart='100'/>")
    )
    tls0 = dict(zip(context.feature_names, context.vector_at("tls0", 0.0)))
    tls1 = dict(zip(context.feature_names, context.vector_at("tls1", 0.0)))

    assert context.feature_names == LOCAL_DEMAND_FEATURE_NAMES
    assert np.isclose(sum(context.tls_arrivals["tls0"]), 1.0)
    assert np.isclose(sum(context.tls_arrivals["tls1"]), 1.0)
    assert tls0["local_arrivals_next_300_per_in_lane"] == 1.0
    assert tls1["local_arrivals_next_300_per_in_lane"] == 1.0
    assert tls0["neighbor_arrivals_next_450_ratio"] == 1.0


def test_empty_local_demand_oof_fold_keeps_explicit_matrix_shape() -> None:
    for variant in LOCAL_VARIANT_SPECS:
        matrix = variant_feature_matrix(
            [],
            variant=variant,
            profiles={},
            temporal={},
            topology={},
            local_demand={},
        )
        assert matrix.shape == (0, len(variant_feature_names(variant)))
