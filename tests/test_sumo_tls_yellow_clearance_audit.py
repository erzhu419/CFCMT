from pathlib import Path
import xml.etree.ElementTree as ET

from scripts.data.audit_sumo_tls_yellow_clearance import (
    analyze_yellow_clearance,
    audit_tls_yellow_clearance,
    required_yellow_seconds,
)


NETWORK = """<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="in" from="a" to="j">
    <lane id="in_0" index="0" speed="15.65" length="100"/>
  </edge>
  <edge id="out" from="j" to="b">
    <lane id="out_0" index="0" speed="15.65" length="50"/>
  </edge>
  <tlLogic id="j" type="actuated" programID="0" offset="0">
    <phase duration="20" minDur="6" maxDur="50" state="G"/>
    <phase duration="1" state="y"/>
    <phase duration="5" state="r"/>
  </tlLogic>
  <connection from="in" to="out" fromLane="0" toLane="0" tl="j" linkIndex="0"/>
</net>
"""


def test_required_yellow_seconds_matches_sumo_boundaries() -> None:
    assert required_yellow_seconds(0.0) == 3
    assert required_yellow_seconds(15.65) == 3
    assert required_yellow_seconds(20.0) == 5
    assert required_yellow_seconds(30.0) == 6


def test_audit_finds_undersized_road_yellow(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(NETWORK, encoding="utf-8")

    payload = audit_tls_yellow_clearance(network)

    assert payload["actionable"] is True
    assert payload["passed"] is False
    assert payload["status"] == "REQUIRES_REPAIR"
    assert payload["counts"]["undersized_yellow_phase_count"] == 1
    assert payload["affected_tls_ids"] == ["j"]
    example = payload["undersized_yellow_phase_examples"][0]
    assert example["duration_sec"] == 1.0
    assert example["required_duration_sec"] == 3


def test_analysis_ignores_red_and_green_phases() -> None:
    root = ET.fromstring(NETWORK.replace('duration="1" state="y"', 'duration="3" state="y"'))
    analysis = analyze_yellow_clearance(root)

    assert analysis["counts"]["yellow_phase_count"] == 1
    assert analysis["counts"]["undersized_yellow_phase_count"] == 0
