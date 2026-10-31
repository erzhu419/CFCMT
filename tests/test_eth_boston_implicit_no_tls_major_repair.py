from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_uncontrolled_major_right_of_way import (
    audit_uncontrolled_major_right_of_way,
)
from scripts.data.repair_eth_boston_implicit_no_tls_major import repair_network


def _network(path: Path) -> Path:
    path.write_text(
        """<net>
  <edge id="-8922897#5" from="a" to="3185278424">
    <lane id="-8922897#5_0" index="0"/>
    <lane id="-8922897#5_1" index="1"/>
  </edge>
  <edge id="312661407" from="b" to="3185278424">
    <lane id="312661407_0" index="0"/>
    <lane id="312661407_1" index="1"/>
  </edge>
  <edge id="-8922897#4" from="3185278424" to="c">
    <lane id="-8922897#4_0" index="0"/>
    <lane id="-8922897#4_1" index="1"/>
  </edge>
  <junction id="3185278424" type="traffic_light"
            incLanes="312661407_0 312661407_1 -8922897#5_0 -8922897#5_1"
            intLanes="">
    <request index="0" response="0100" foes="0100"/>
    <request index="1" response="1000" foes="1000"/>
    <request index="2" response="0000" foes="0001"/>
    <request index="3" response="0000" foes="0010"/>
  </junction>
  <connection from="-8922897#5" to="-8922897#4" fromLane="0" toLane="0"
              state="M" dir="s"/>
  <connection from="-8922897#5" to="-8922897#4" fromLane="1" toLane="1"
              state="M" dir="s"/>
  <connection from="312661407" to="-8922897#4" fromLane="0" toLane="0"
              state="M" dir="s"/>
  <connection from="312661407" to="-8922897#4" fromLane="1" toLane="1"
              state="M" dir="s"/>
</net>
""",
        encoding="utf-8",
    )
    return path


def _states(path: Path) -> dict[tuple[str, str], str]:
    return {
        (str(row.get("from")), str(row.get("fromLane"))): str(row.get("state"))
        for row in ET.parse(path).getroot().iter("connection")
    }


def test_implicit_no_tls_repair_uses_groupwise_request_priority(
    tmp_path: Path,
) -> None:
    source = _network(tmp_path / "source.net.xml")
    repaired = tmp_path / "repaired.net.xml"

    result = repair_network(source, repaired)

    assert result["passed"] is True
    assert result["repair"]["connection_state_change_count"] == 2
    assert result["repair"]["known_collision_junction"] == "3185278424"
    assert _states(repaired) == {
        ("-8922897#5", "0"): "M",
        ("-8922897#5", "1"): "M",
        ("312661407", "0"): "m",
        ("312661407", "1"): "m",
    }
    after = audit_uncontrolled_major_right_of_way(
        repaired,
        include_implicit_no_tls=True,
    )
    assert after["counts"]["duplicate_major_pair_count"] == 0
    assert result["compatibility_audit"]["passed"] is True
