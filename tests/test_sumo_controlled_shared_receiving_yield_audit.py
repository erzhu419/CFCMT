from pathlib import Path

from scripts.data.audit_sumo_controlled_shared_receiving_yield import (
    audit_controlled_shared_receiving_yield,
)


def _network(response_right: str) -> str:
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
    <request index="1" response="{response_right}" foes="01" cont="0"/>
  </junction>
  <connection from="in_a" to="out" fromLane="0" toLane="0" tl="tls" linkIndex="0" state="O" dir="s"/>
  <connection from="in_b" to="out" fromLane="0" toLane="0" tl="tls" linkIndex="1" state="o" dir="r"/>
</net>
"""


def test_audit_detects_missing_right_on_red_yield_response(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(_network("00"), encoding="utf-8")

    result = audit_controlled_shared_receiving_yield(
        network,
        focus_receiving_lanes=("out_0",),
    )

    assert result["counts"]["directed_yield_requirement_count"] == 1
    assert result["counts"]["missing_yield_response_count"] == 1
    assert result["counts"]["missing_symmetric_foe_count"] == 0
    example = result["examples"][0]
    assert example["directed_requirements"][0]["yielding_junction_index"] == 1
    assert example["directed_requirements"][0]["protected_junction_index"] == 0


def test_audit_accepts_declared_right_on_red_yield_response(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(_network("01"), encoding="utf-8")

    result = audit_controlled_shared_receiving_yield(network)

    assert result["counts"]["directed_yield_requirement_count"] == 1
    assert result["counts"]["missing_yield_response_count"] == 0
    assert result["counts"]["missing_symmetric_foe_count"] == 0
