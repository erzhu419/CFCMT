from pathlib import Path

import pytest

from scripts.data.audit_sumo_minor_visibility import (
    PROTOCOL,
    audit_minor_visibility,
)


def test_minor_visibility_audit_contextualizes_focus(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <edge id="a" from="x" to="j"><lane id="a_0" index="0" speed="13.8"/></edge>
  <edge id="b" from="y" to="j"><lane id="b_0" index="0" speed="13.8"/></edge>
  <edge id="c" from="j" to="z"><lane id="c_0" index="0" speed="13.8"/></edge>
  <edge id=":j_0" function="internal"><lane id=":j_0_0" index="0" speed="8"/></edge>
  <connection from="a" to="c" fromLane="0" toLane="0" state="m" dir="s" visibility="9" via=":j_0_0"/>
  <connection from="b" to="c" fromLane="0" toLane="0" state="m" dir="s" visibility="9"/>
  <connection from=":j_0" to="c" fromLane="0" toLane="0" state="M" dir="s"/>
</net>""",
        encoding="utf-8",
    )

    result = audit_minor_visibility(
        network,
        focus_connection={"from": "a", "fromLane": "0", "to": "c", "toLane": "0"},
    )

    assert result["protocol"] == PROTOCOL
    assert result["counts"]["internal_edge_count"] == 1
    assert result["counts"]["minor_or_yield_connection_count"] == 2
    assert result["counts"]["minor_with_internal_via_count"] == 1
    assert result["explicit_visibility_histogram_m"] == {"9.00": 2}
    assert result["focus_comparison"]["same_state_direction_speed_count"] == 2
    assert result["focus_comparison"]["same_visibility_fraction"] == 1.0


def test_minor_visibility_audit_requires_unique_focus(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text("<net/>", encoding="utf-8")
    with pytest.raises(ValueError, match="match count changed"):
        audit_minor_visibility(
            network,
            focus_connection={"from": "missing"},
        )
