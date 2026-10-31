from __future__ import annotations

from pathlib import Path

from scripts.data.audit_sumo_collision_lane_context import (
    PROTOCOL,
    audit_collision_lane_context,
)


def test_collision_context_resolves_route_and_min_gap(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="previous" from="a" to="b">
    <lane id="previous_0" index="0" speed="13.9" length="80"/>
    <lane id="previous_1" index="1" speed="13.9" length="80"/>
  </edge>
  <edge id="current" from="b" to="c">
    <lane id="current_0" index="0" speed="13.9" length="60"/>
    <lane id="current_1" index="1" speed="13.9" length="60"/>
  </edge>
  <edge id="next" from="c" to="d">
    <lane id="next_0" index="0" speed="13.9" length="100"/>
  </edge>
  <edge id="cross" from="x" to="c">
    <lane id="cross_0" index="0" speed="13.9" length="50"/>
  </edge>
  <junction id="b" type="traffic_light" incLanes="previous_0 previous_1">
    <request index="0" response="0" foes="0" cont="0"/>
  </junction>
  <junction id="c" type="priority" incLanes="current_0 current_1 cross_0">
    <request index="0" response="000" foes="000" cont="0"/>
    <request index="1" response="100" foes="100" cont="0"/>
    <request index="2" response="000" foes="010" cont="0"/>
  </junction>
  <connection from="previous" to="current" fromLane="1" toLane="1" dir="s" state="M"/>
  <connection from="current" to="next" fromLane="0" toLane="0" dir="s" state="M"/>
  <connection from="current" to="next" fromLane="1" toLane="0" dir="s" state="m"/>
  <connection from="cross" to="next" fromLane="0" toLane="0" dir="l" state="M"/>
</net>
""",
        encoding="utf-8",
    )
    routes = tmp_path / "routes.rou.xml"
    routes.write_text(
        """<routes>
  <trip id="first" depart="0" from="previous" to="next"/>
</routes>
""",
        encoding="utf-8",
    )
    config = tmp_path / "test.sumocfg"
    config.write_text(
        """<configuration>
  <input>
    <net-file value="network.net.xml"/>
    <route-files value="routes.rou.xml"/>
  </input>
  <time><step-length value="1"/></time>
  <processing><collision.check-junctions value="true"/></processing>
</configuration>
""",
        encoding="utf-8",
    )

    result = audit_collision_lane_context(
        network,
        config=config,
        lane_id="current_1",
        previous_edge_id="previous",
        next_edge_id="next",
        collider_front_position=46.0,
        victim_front_position=52.0,
    )

    assert result["protocol"] == PROTOCOL
    assert result["route_chain"]["unique_on_both_sides"] is True
    assert result["lane_mapping"]["current_to_next_lane_merge"] is True
    classification = result["classification"]
    assert classification["observed_route_chain_resolves_uniquely"] is True
    assert classification["victim_near_downstream_boundary"] is True
    assert classification["static_current_to_next_lane_merge"] is True
    assert classification["min_gap_violation_without_body_overlap"] is True
    assert classification["downstream_route_connection_request_index"] == 1
    assert classification["downstream_route_yields_to_indices"] == [2]
    assert classification["downstream_route_yields_to_connections"] == [
        {
            "dir": "l",
            "from": "cross",
            "fromLane": "0",
            "state": "M",
            "to": "next",
            "toLane": "0",
        }
    ]
    assert result["longitudinal_geometry"]["bumper_gap_m"] == 1.0
    assert result["longitudinal_geometry"]["collision_detection_gap_threshold_m"] == 2.5
    assert result["vehicle_type"]["definition_source"] == "sumo_builtin_default"
    assert result["vehicle_type"]["input_prefix_scans"][0]["stopped_at"] == "trip"


def test_collision_context_uses_explicit_default_type(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <edge id="p" from="a" to="b"><lane id="p_0" index="0" length="30"/></edge>
  <edge id="c" from="b" to="d"><lane id="c_0" index="0" length="40"/></edge>
  <edge id="n" from="d" to="e"><lane id="n_0" index="0" length="30"/></edge>
  <junction id="b" type="priority"/><junction id="d" type="priority"/>
  <connection from="p" to="c" fromLane="0" toLane="0"/>
  <connection from="c" to="n" fromLane="0" toLane="0"/>
</net>""",
        encoding="utf-8",
    )
    routes = tmp_path / "routes.rou.xml"
    routes.write_text(
        """<routes>
  <vType id="DEFAULT_VEHTYPE" length="4" minGap="1" tau="0.8"/>
  <trip id="first" depart="0" from="p" to="n"/>
</routes>""",
        encoding="utf-8",
    )
    config = tmp_path / "test.sumocfg"
    config.write_text(
        """<configuration><input><route-files value="routes.rou.xml"/></input>
<processing><collision.mingap-factor value="0.5"/></processing></configuration>""",
        encoding="utf-8",
    )

    result = audit_collision_lane_context(
        network,
        config=config,
        lane_id="c_0",
        previous_edge_id="p",
        next_edge_id="n",
        collider_front_position=20.0,
        victim_front_position=24.25,
    )

    assert result["vehicle_type"]["definition_source"] == "input_declaration"
    assert result["longitudinal_geometry"]["bumper_gap_m"] == 0.25
    assert result["longitudinal_geometry"]["collision_detection_gap_threshold_m"] == 0.5
    assert result["classification"]["min_gap_violation_without_body_overlap"] is True
