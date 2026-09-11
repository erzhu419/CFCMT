from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_no_tls_major_response import (
    audit_no_tls_major_response_consistency,
)
from scripts.data.repair_eth_boston_no_tls_major_response import repair_network


def test_repair_demotes_only_major_with_declared_no_tls_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    network = tmp_path / "network.net.xml"
    repaired = tmp_path / "repaired.net.xml"
    network.write_text(
        """<net>
  <edge id="yielding" from="a" to="j"><lane id="yielding_0" index="0"/></edge>
  <edge id="priority" from="b" to="j"><lane id="priority_0" index="0"/></edge>
  <edge id="tls_in" from="d" to="j"><lane id="tls_in_0" index="0"/></edge>
  <edge id="out" from="j" to="c"><lane id="out_0" index="0"/></edge>
  <edge id="other" from="j" to="e"><lane id="other_0" index="0"/></edge>
  <junction id="j" type="traffic_light_right_on_red"
            incLanes="yielding_0 priority_0 tls_in_0" intLanes="">
    <request index="0" response="110" foes="110"/>
    <request index="1" response="000" foes="101"/>
    <request index="2" response="000" foes="001"/>
  </junction>
  <connection from="yielding" to="out" fromLane="0" toLane="0"
              uncontrolled="1" state="M" dir="s"/>
  <connection from="priority" to="out" fromLane="0" toLane="0"
              uncontrolled="1" state="m" dir="l"/>
  <connection from="tls_in" to="other" fromLane="0" toLane="0"
              tl="signal" linkIndex="0" state="G" dir="s"/>
</net>
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scripts.data.repair_eth_boston_no_tls_major_response."
        "KNOWN_COLLISION_JUNCTION",
        "j",
    )
    monkeypatch.setattr(
        "scripts.data.repair_eth_boston_no_tls_major_response."
        "KNOWN_COLLISION_CONNECTION",
        ("yielding", "0", "out", "0"),
    )

    result = repair_network(network, repaired)

    assert result["passed"] is True
    assert result["repair"]["connection_state_change_count"] == 1
    states = {
        row.get("from"): row.get("state")
        for row in ET.parse(repaired).getroot().iter("connection")
    }
    assert states == {"yielding": "m", "priority": "m", "tls_in": "G"}
    audit = audit_no_tls_major_response_consistency(repaired)
    assert audit["counts"]["candidate_with_external_no_tls_response_count"] == 0


def test_repair_rejects_mutual_no_tls_response(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <edge id="a" from="x" to="j"><lane id="a_0" index="0"/></edge>
  <edge id="b" from="y" to="j"><lane id="b_0" index="0"/></edge>
  <edge id="out" from="j" to="z"><lane id="out_0" index="0"/></edge>
  <junction id="j" type="priority" incLanes="a_0 b_0" intLanes="">
    <request index="0" response="10" foes="10"/>
    <request index="1" response="01" foes="01"/>
  </junction>
  <connection from="a" to="out" fromLane="0" toLane="0" state="M"/>
  <connection from="b" to="out" fromLane="0" toLane="0" state="m"/>
</net>
""",
        encoding="utf-8",
    )

    try:
        repair_network(network, tmp_path / "repaired.net.xml")
    except ValueError as exc:
        assert "not safely actionable" in str(exc)
    else:
        raise AssertionError("mutual response must not be repaired automatically")
