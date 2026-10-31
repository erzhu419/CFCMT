from __future__ import annotations

from types import SimpleNamespace

import pytest

from cf_h2o.traffic_signal.network_context import (
    _signal_adjacency,
    static_topology_context,
)


def _write_intermediate_edge_network(tmp_path):
    net = tmp_path / "network.net.xml"
    net.write_text(
        """<net>
  <edge id="a_in" from="a0" to="a"><lane id="a_in_0" index="0"/></edge>
  <edge id="mid" from="a" to="m"><lane id="mid_0" index="0"/></edge>
  <edge id="b_in" from="m" to="b"><lane id="b_in_0" index="0"/></edge>
  <edge id="b_out" from="b" to="b0"><lane id="b_out_0" index="0"/></edge>
  <connection from="mid" to="b_in" fromLane="0" toLane="0"/>
  <junction id="a" type="traffic_light" x="0" y="0"/>
  <junction id="b" type="traffic_light" x="100" y="0"/>
</net>
""",
        encoding="utf-8",
    )
    routes = tmp_path / "routes.rou.xml"
    routes.write_text("<routes/>", encoding="utf-8")
    sumocfg = tmp_path / "run.sumocfg"
    sumocfg.write_text(
        """<configuration><input>
  <net-file value="network.net.xml"/>
  <route-files value="routes.rou.xml"/>
</input></configuration>
""",
        encoding="utf-8",
    )
    return sumocfg


def _infos():
    phase = SimpleNamespace(state="G", duration=30.0, green_count=1)
    return {
        "a": SimpleNamespace(
            incoming_lanes=("a_in_0",),
            controlled_links=((('a_in_0', 'mid_0', ':a_0'),),),
            candidates=(phase,),
        ),
        "b": SimpleNamespace(
            incoming_lanes=("b_in_0",),
            controlled_links=((('b_in_0', 'b_out_0', ':b_0'),),),
            candidates=(phase,),
        ),
    }


def test_static_signal_graph_traces_through_uncontrolled_edges(tmp_path) -> None:
    sumocfg = _write_intermediate_edge_network(tmp_path)
    infos = _infos()

    assert _signal_adjacency(infos) == {"a": set(), "b": set()}
    assert _signal_adjacency(infos, sumocfg=sumocfg) == {
        "a": {"b"},
        "b": set(),
    }

    context = static_topology_context(sumocfg, infos)
    assert context[5] == pytest.approx(0.25)
    assert context[7] == pytest.approx(0.5)
