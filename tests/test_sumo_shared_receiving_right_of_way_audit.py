from pathlib import Path

from scripts.data.audit_sumo_shared_receiving_right_of_way import (
    audit_shared_receiving_right_of_way,
)


def _network(path: Path, *, corrected: bool) -> Path:
    controlled_response = "10" if corrected else "00"
    major_response = "00" if corrected else "01"
    signal = "g" if corrected else "G"
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<net>
  <edge id="controlled" from="a" to="j"><lane id="controlled_0" index="0"/></edge>
  <edge id="major" from="b" to="j"><lane id="major_0" index="0"/></edge>
  <edge id="out" from="j" to="c"><lane id="out_0" index="0"/></edge>
  <tlLogic id="tls" programID="0" type="static" offset="0">
    <phase duration="30" state="{signal}"/>
  </tlLogic>
  <junction id="j" type="traffic_light" incLanes="controlled_0 major_0" intLanes="">
    <request index="0" response="{controlled_response}" foes="10"/>
    <request index="1" response="{major_response}" foes="01"/>
  </junction>
  <connection from="major" to="out" fromLane="0" toLane="0" state="M"/>
  <connection from="controlled" to="out" fromLane="0" toLane="0" tl="tls" linkIndex="0" state="O"/>
</net>
""",
        encoding="utf-8",
    )
    return path


def test_audit_finds_reversed_priority_and_protected_service(tmp_path: Path) -> None:
    result = audit_shared_receiving_right_of_way(
        _network(tmp_path / "network.net.xml", corrected=False)
    )
    counts = result["counts"]
    assert counts["controlled_uncontrolled_major_pair_count"] == 1
    assert counts["unresolved_junction_index_pair_count"] == 0
    assert counts["controlled_missing_yield_pair_count"] == 1
    assert counts["major_wrongly_yields_pair_count"] == 1
    assert counts["missing_symmetric_foe_pair_count"] == 0
    assert counts["protected_service_pair_count"] == 1
    assert counts["already_exact_pair_count"] == 0
    assert result["examples"][0]["controlled_junction_index"] == 0
    assert result["examples"][0]["major_junction_index"] == 1


def test_audit_accepts_executable_permissive_priority(tmp_path: Path) -> None:
    result = audit_shared_receiving_right_of_way(
        _network(tmp_path / "network.net.xml", corrected=True)
    )
    counts = result["counts"]
    assert counts["already_exact_pair_count"] == 1
    assert counts["controlled_missing_yield_pair_count"] == 0
    assert counts["major_wrongly_yields_pair_count"] == 0
    assert counts["missing_symmetric_foe_pair_count"] == 0
    assert counts["protected_service_pair_count"] == 0
    assert result["examples"] == []
