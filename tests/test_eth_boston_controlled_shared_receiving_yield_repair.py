from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from scripts.data.audit_sumo_controlled_shared_receiving_yield import (
    audit_controlled_shared_receiving_yield,
)
from scripts.data.repair_eth_boston_controlled_shared_receiving_yield import (
    repair_network,
)


def _network(*, yielding_response: str = "00", yielding_foes: str = "01") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="in_a" from="A" to="J"><lane id="in_a_0" index="0"/></edge>
  <edge id="in_b" from="B" to="J"><lane id="in_b_0" index="0"/></edge>
  <edge id="out" from="J" to="C"><lane id="out_0" index="0"/></edge>
  <tlLogic id="tls" type="static" programID="0" offset="0">
    <phase duration="30" state="Gs"/>
  </tlLogic>
  <junction id="J" type="traffic_light_right_on_red" incLanes="in_a_0 in_b_0" intLanes="">
    <request index="0" response="00" foes="10" cont="0"/>
    <request index="1" response="{yielding_response}" foes="{yielding_foes}" cont="0"/>
  </junction>
  <connection from="in_a" to="out" fromLane="0" toLane="0" tl="tls" linkIndex="0" state="O" dir="s"/>
  <connection from="in_b" to="out" fromLane="0" toLane="0" tl="tls" linkIndex="1" state="o" dir="r"/>
</net>
"""


def test_repair_sets_only_missing_yield_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.net.xml"
    repaired = tmp_path / "repaired.net.xml"
    source.write_text(_network(), encoding="utf-8")
    monkeypatch.setattr(
        "scripts.data.repair_eth_boston_controlled_shared_receiving_yield."
        "KNOWN_COLLISION_JUNCTION",
        "J",
    )
    monkeypatch.setattr(
        "scripts.data.repair_eth_boston_controlled_shared_receiving_yield."
        "KNOWN_COLLISION_RECEIVING_LANE",
        "out_0",
    )
    monkeypatch.setattr(
        "scripts.data.repair_eth_boston_controlled_shared_receiving_yield."
        "KNOWN_YIELDING_INDEX",
        1,
    )
    monkeypatch.setattr(
        "scripts.data.repair_eth_boston_controlled_shared_receiving_yield."
        "KNOWN_PROTECTED_INDEX",
        0,
    )

    result = repair_network(source, repaired)

    assert result["passed"] is True
    assert result["repair"]["response_bits_set"] == 1
    root = ET.parse(repaired).getroot()
    assert [row.get("response") for row in root.iter("request")] == ["00", "01"]
    assert [row.get("state") for row in root.iter("phase")] == ["Gs"]
    after = audit_controlled_shared_receiving_yield(repaired)
    assert after["counts"]["missing_yield_response_count"] == 0


def test_repair_rejects_missing_symmetric_foe(tmp_path: Path) -> None:
    source = tmp_path / "source.net.xml"
    source.write_text(_network(yielding_foes="00"), encoding="utf-8")

    with pytest.raises(ValueError, match="not safely actionable"):
        repair_network(source, tmp_path / "repaired.net.xml")


def test_repair_refuses_an_already_consistent_network(tmp_path: Path) -> None:
    source = tmp_path / "source.net.xml"
    source.write_text(_network(yielding_response="01"), encoding="utf-8")

    with pytest.raises(ValueError, match="not safely actionable"):
        repair_network(source, tmp_path / "repaired.net.xml")
