from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data.repair_eth_boston_tls_yellow_clearance import (
    KNOWN_FAILURE_TLS_ID,
    repair_network,
)


NETWORK = f"""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
  <edge id="in" from="a" to="{KNOWN_FAILURE_TLS_ID}">
    <lane id="in_0" index="0" speed="15.65" length="100"/>
  </edge>
  <edge id="out" from="{KNOWN_FAILURE_TLS_ID}" to="b">
    <lane id="out_0" index="0" speed="15.65" length="50"/>
  </edge>
  <tlLogic id="{KNOWN_FAILURE_TLS_ID}" type="actuated" programID="0" offset="7">
    <phase duration="20" minDur="6" maxDur="50" state="G"/>
    <phase duration="1" state="y"/>
    <phase duration="5" state="r"/>
  </tlLogic>
  <connection from="in" to="out" fromLane="0" toLane="0"
              tl="{KNOWN_FAILURE_TLS_ID}" linkIndex="0"/>
</net>
"""


def test_repair_network_changes_only_undersized_yellow_duration(tmp_path: Path) -> None:
    source = tmp_path / "source.net.xml"
    repaired = tmp_path / "repaired.net.xml"
    source.write_text(NETWORK, encoding="utf-8")

    result = repair_network(source, repaired)

    assert result["passed"] is True
    assert result["pre_repair_counts"]["undersized_yellow_phase_count"] == 1
    assert result["post_repair_counts"]["undersized_yellow_phase_count"] == 0
    assert result["repair"]["changed_phase_count"] == 1
    assert result["repair"]["required_duration_histogram_sec"] == {"3": 1}
    assert result["compatibility_audit"]["passed"] is True

    logic = ET.parse(repaired).getroot().find("tlLogic")
    assert logic is not None
    phases = logic.findall("phase")
    assert phases[0].attrib == {
        "duration": "20",
        "minDur": "6",
        "maxDur": "50",
        "state": "G",
    }
    assert phases[1].attrib == {"duration": "3", "state": "y"}
    assert phases[2].attrib == {"duration": "5", "state": "r"}
    assert logic.get("offset") == "7"
