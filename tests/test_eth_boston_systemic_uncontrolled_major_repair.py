from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.audit_sumo_uncontrolled_major_right_of_way import (
    IMPLICIT_NO_TLS_PROTOCOL,
    audit_uncontrolled_major_right_of_way,
)
from scripts.data.repair_eth_boston_systemic_uncontrolled_major import (
    repair_network,
)


def _network(path: Path, *, ambiguous_response: bool = False) -> Path:
    left_response = "00" if ambiguous_response else "01"
    path.write_text(
        f"""<net>
  <edge id="priority" from="a" to="j"><lane id="priority_0" index="0"/></edge>
  <edge id="yielding" from="b" to="j"><lane id="yielding_0" index="0"/></edge>
  <edge id="out" from="j" to="c"><lane id="out_0" index="0"/></edge>
  <junction id="j" type="traffic_light_right_on_red"
            incLanes="priority_0 yielding_0" intLanes="">
    <request index="0" response="00" foes="10"/>
    <request index="1" response="{left_response}" foes="01"/>
  </junction>
  <connection from="priority" to="out" fromLane="0" toLane="0"
              state="M" dir="s" uncontrolled="1"/>
  <connection from="yielding" to="out" fromLane="0" toLane="0"
              state="M" dir="l" uncontrolled="1"/>
</net>
""",
        encoding="utf-8",
    )
    return path


def _states(path: Path) -> dict[str, str]:
    return {
        str(row.get("from")): str(row.get("state"))
        for row in ET.parse(path).getroot().iter("connection")
    }


def test_systemic_uncontrolled_major_repair_uses_existing_response(
    tmp_path: Path,
) -> None:
    source = _network(tmp_path / "source.net.xml")
    before = audit_uncontrolled_major_right_of_way(source)
    assert before["counts"] == {
        "connection_count": 2,
        "duplicate_major_receiving_lane_count": 1,
        "duplicate_major_junction_count": 1,
        "duplicate_major_pair_count": 1,
        "unresolved_junction_index_pair_count": 0,
        "missing_symmetric_foe_pair_count": 0,
        "ambiguous_response_pair_count": 0,
        "actionable_pair_count": 1,
    }

    repaired = tmp_path / "repaired.net.xml"
    result = repair_network(source, repaired)
    assert result["passed"] is True
    assert result["repair"]["connection_state_change_count"] == 1
    assert _states(source) == {"priority": "M", "yielding": "M"}
    assert _states(repaired) == {"priority": "M", "yielding": "m"}
    after = audit_uncontrolled_major_right_of_way(repaired)
    assert after["counts"]["duplicate_major_pair_count"] == 0
    assert result["compatibility_audit"]["passed"] is True


def test_systemic_uncontrolled_major_repair_rejects_ambiguous_response(
    tmp_path: Path,
) -> None:
    source = _network(tmp_path / "source.net.xml", ambiguous_response=True)
    audit = audit_uncontrolled_major_right_of_way(source)
    assert audit["counts"]["ambiguous_response_pair_count"] == 1
    with pytest.raises(ValueError, match="not safely actionable"):
        repair_network(source, tmp_path / "repaired.net.xml")


def test_implicit_no_tls_mode_catches_unmarked_major_pair(tmp_path: Path) -> None:
    network = _network(tmp_path / "implicit.net.xml")
    tree = ET.parse(network)
    for connection in tree.getroot().iter("connection"):
        connection.attrib.pop("uncontrolled", None)
    tree.write(network, encoding="utf-8", xml_declaration=True)

    legacy = audit_uncontrolled_major_right_of_way(network)
    implicit = audit_uncontrolled_major_right_of_way(
        network,
        include_implicit_no_tls=True,
    )

    assert legacy["protocol"] != IMPLICIT_NO_TLS_PROTOCOL
    assert legacy["counts"]["duplicate_major_pair_count"] == 0
    assert implicit["protocol"] == IMPLICIT_NO_TLS_PROTOCOL
    assert implicit["counts"]["duplicate_major_pair_count"] == 1
    assert implicit["counts"][
        "pair_with_implicit_uncontrolled_attribute_count"
    ] == 1
    assert implicit["counts"]["actionable_pair_count"] == 1
    assert implicit["counts"]["major_multiplicity_histogram"] == {"2": 1}
    assert implicit["counts"]["unique_priority_group_count"] == 1
    assert implicit["counts"]["ambiguous_priority_group_count"] == 0
    assert implicit["counts"]["planned_connection_state_change_count"] == 1


def test_implicit_no_tls_mode_resolves_three_way_priority_group(
    tmp_path: Path,
) -> None:
    network = tmp_path / "three-way.net.xml"
    network.write_text(
        """<net>
  <edge id="priority" from="a" to="j"><lane id="priority_0" index="0"/></edge>
  <edge id="middle" from="b" to="j"><lane id="middle_0" index="0"/></edge>
  <edge id="yielding" from="c" to="j"><lane id="yielding_0" index="0"/></edge>
  <edge id="out" from="j" to="d"><lane id="out_0" index="0"/></edge>
  <junction id="j" type="traffic_light"
            incLanes="priority_0 middle_0 yielding_0" intLanes="">
    <request index="0" response="000" foes="110"/>
    <request index="1" response="001" foes="101"/>
    <request index="2" response="011" foes="011"/>
  </junction>
  <connection from="priority" to="out" fromLane="0" toLane="0"
              state="M" dir="s"/>
  <connection from="middle" to="out" fromLane="0" toLane="0"
              state="M" dir="s"/>
  <connection from="yielding" to="out" fromLane="0" toLane="0"
              state="M" dir="s"/>
</net>
""",
        encoding="utf-8",
    )

    audit = audit_uncontrolled_major_right_of_way(
        network,
        include_implicit_no_tls=True,
    )

    assert audit["counts"]["duplicate_major_pair_count"] == 3
    assert audit["counts"]["major_multiplicity_histogram"] == {"3": 1}
    assert audit["counts"]["unique_priority_group_count"] == 1
    assert audit["counts"]["ambiguous_priority_group_count"] == 0
    assert audit["counts"]["planned_connection_state_change_count"] == 2
