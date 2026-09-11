from pathlib import Path

from scripts.data.audit_sumo_shared_receiving_conflicts import (
    audit_shared_receiving_conflicts,
)


def _network(signal: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="controlled_in" from="a" to="j">
    <lane id="controlled_in_0" index="0"/>
  </edge>
  <edge id="major_in" from="b" to="j">
    <lane id="major_in_0" index="0"/>
  </edge>
  <edge id="out" from="j" to="c">
    <lane id="out_0" index="0"/>
  </edge>
  <tlLogic id="joined" type="static" programID="0">
    <phase duration="30" state="{signal}r"/>
    <phase duration="4" state="yr"/>
  </tlLogic>
  <junction id="j" type="traffic_light">
    <request index="0" response="10" foes="11" cont="0"/>
    <request index="1" response="00" foes="11" cont="0"/>
  </junction>
  <connection from="controlled_in" fromLane="0" to="out" toLane="0"
              via=":j_0_0" tl="joined" linkIndex="0" state="O"/>
  <connection from="major_in" fromLane="0" to="out" toLane="0"
              state="M" uncontrolled="1"/>
</net>
"""


def test_audit_finds_protected_and_uncontrolled_shared_receiver(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(_network("G"), encoding="utf-8")
    result = audit_shared_receiving_conflicts(
        network,
        focus_receiving_lanes=("out_0",),
        focus_tls_ids=("joined",),
    )
    counts = result["counts"]
    assert counts[
        "protected_controlled_with_uncontrolled_major_group_count"
    ] == 1
    assert counts["risk_record_count"] == 1
    risk = result["risk_groups"][0]
    assert risk["receiving_lane"] == "out_0"
    assert risk["junction"]["junction_id"] == "j"
    assert risk["protected_controlled_connections"][0]["service"][
        "protected_phase_count"
    ] == 1
    assert "out_0" in result["focus_groups"]


def test_yielding_signal_is_reported_but_not_a_protected_risk(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(_network("g"), encoding="utf-8")
    result = audit_shared_receiving_conflicts(network)
    counts = result["counts"]
    assert counts[
        "yielding_controlled_with_uncontrolled_major_group_count"
    ] == 1
    assert counts[
        "protected_controlled_with_uncontrolled_major_group_count"
    ] == 0
    assert result["risk_groups"] == []


def test_invalid_tls_link_index_is_a_static_anomaly(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        _network("G").replace('linkIndex="0"', 'linkIndex="5"'),
        encoding="utf-8",
    )
    result = audit_shared_receiving_conflicts(network)
    assert result["counts"]["static_anomaly_count"] == 1
    assert {
        row["category"] for row in result["static_anomalies"]
    } == {"tls_phase_state_too_short"}
    assert result["static_anomalies"][0]["affected_phase_count"] == 2
