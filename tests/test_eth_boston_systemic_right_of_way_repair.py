from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_shared_receiving_right_of_way import (
    audit_shared_receiving_right_of_way,
)
from scripts.data.repair_eth_boston_systemic_right_of_way import repair_network


def _network(path: Path) -> Path:
    path.write_text(
        """<net>
  <edge id="controlled" from="a" to="j"><lane id="controlled_0" index="0"/></edge>
  <edge id="major" from="b" to="j"><lane id="major_0" index="0"/></edge>
  <edge id="out" from="j" to="c"><lane id="out_0" index="0"/></edge>
  <tlLogic id="tls" programID="0" type="static" offset="0">
    <phase duration="30" state="G"/>
    <phase duration="3" state="y"/>
  </tlLogic>
  <junction id="j" type="traffic_light" incLanes="controlled_0 major_0" intLanes="">
    <request index="0" response="00" foes="10"/>
    <request index="1" response="01" foes="01"/>
  </junction>
  <connection from="major" to="out" fromLane="0" toLane="0" state="M"/>
  <connection from="controlled" to="out" fromLane="0" toLane="0" tl="tls" linkIndex="0" state="O"/>
</net>
""",
        encoding="utf-8",
    )
    return path


def test_systemic_repair_inverts_priority_and_downgrades_green(tmp_path: Path) -> None:
    source = _network(tmp_path / "source.net.xml")
    repaired = tmp_path / "repaired.net.xml"
    result = repair_network(source, repaired)
    assert result["passed"] is True
    assert result["repair"]["relation_pair_count"] == 1
    assert result["repair"]["response_bits_set"] == 1
    assert result["repair"]["response_bits_cleared"] == 1
    assert result["repair"]["phase_character_changes"] == 1
    root = ET.parse(repaired).getroot()
    requests = list(root.find("junction").findall("request"))
    assert [row.get("response") for row in requests] == ["10", "00"]
    phases = list(root.find("tlLogic").findall("phase"))
    assert [row.get("state") for row in phases] == ["g", "y"]
    audit = audit_shared_receiving_right_of_way(repaired)
    assert audit["counts"]["already_exact_pair_count"] == 1
