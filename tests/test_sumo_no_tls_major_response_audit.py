from pathlib import Path

from scripts.data.audit_sumo_no_tls_major_response import (
    audit_no_tls_major_response_consistency,
)


def test_audit_finds_major_link_that_bypasses_declared_response(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <edge id="in_major" from="a" to="j"><lane id="in_major_0" index="0"/></edge>
  <edge id="in_minor" from="b" to="j"><lane id="in_minor_0" index="0"/></edge>
  <edge id="out" from="j" to="c"><lane id="out_0" index="0"/></edge>
  <junction id="j" type="traffic_light_right_on_red"
            incLanes="in_major_0 in_minor_0" intLanes="">
    <request index="0" response="10" foes="10"/>
    <request index="1" response="00" foes="01"/>
  </junction>
  <connection from="in_major" to="out" fromLane="0" toLane="0"
              uncontrolled="1" state="M" dir="s"/>
  <connection from="in_minor" to="out" fromLane="0" toLane="0"
              uncontrolled="1" state="m" dir="l"/>
</net>
""",
        encoding="utf-8",
    )

    result = audit_no_tls_major_response_consistency(
        network,
        focus_junctions=("j",),
        maximum_examples=0,
    )

    assert result["counts"]["major_with_declared_response_connection_count"] == 1
    assert result["counts"]["declared_response_bit_count"] == 1
    assert result["counts"]["response_without_foe_bit_count"] == 0
    assert result["counts"]["response_target_kind_histogram"] == {
        "external_no_tls": 1
    }
    assert result["counts"]["response_target_state_histogram"] == {"m": 1}
    assert result["counts"][
        "candidate_with_asymmetric_external_no_tls_response_count"
    ] == 1
    assert result["counts"][
        "candidate_with_same_receiving_lane_external_no_tls_response_count"
    ] == 1
    assert result["counts"][
        "external_no_tls_response_without_symmetric_foe_bit_count"
    ] == 0
    assert result["examples"][0]["junction_index"] == 0
    assert result["examples"][0]["response_indices"] == [1]
    assert result["examples"][0]["response_connections"][0]["from"] == "in_minor"
    assert result["examples"][0]["response_relations"] == [
        {
            "junction_index": 1,
            "kind": "external_no_tls",
            "connection": {
                "from": "in_minor",
                "fromLane": "0",
                "to": "out",
                "toLane": "0",
                "uncontrolled": "1",
                "state": "m",
                "dir": "l",
            },
            "same_receiving_lane": True,
            "target_responds_back": False,
            "target_declares_foe": True,
        }
    ]


def test_audit_ignores_major_without_response_and_tls_connection(
    tmp_path: Path,
) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <edge id="in_a" from="a" to="j"><lane id="in_a_0" index="0"/></edge>
  <edge id="in_b" from="b" to="j"><lane id="in_b_0" index="0"/></edge>
  <edge id="out" from="j" to="c"><lane id="out_0" index="0"/></edge>
  <junction id="j" type="traffic_light" incLanes="in_a_0 in_b_0" intLanes="">
    <request index="0" response="00" foes="10"/>
    <request index="1" response="01" foes="01"/>
  </junction>
  <connection from="in_a" to="out" fromLane="0" toLane="0" state="M" dir="s"/>
  <connection from="in_b" to="out" fromLane="0" toLane="0"
              tl="signal" linkIndex="0" state="G" dir="s"/>
</net>
""",
        encoding="utf-8",
    )

    result = audit_no_tls_major_response_consistency(network)

    assert result["counts"]["external_no_tls_major_connection_count"] == 1
    assert result["counts"]["major_with_declared_response_connection_count"] == 0


def test_audit_classifies_tls_and_mutual_no_tls_responses(tmp_path: Path) -> None:
    network = tmp_path / "network.net.xml"
    network.write_text(
        """<net>
  <edge id="in_major" from="a" to="j"><lane id="in_major_0" index="0"/></edge>
  <edge id="in_minor" from="b" to="j"><lane id="in_minor_0" index="0"/></edge>
  <edge id="in_tls" from="d" to="j"><lane id="in_tls_0" index="0"/></edge>
  <edge id="out_a" from="j" to="c"><lane id="out_a_0" index="0"/></edge>
  <edge id="out_b" from="j" to="e"><lane id="out_b_0" index="0"/></edge>
  <junction id="j" type="traffic_light"
            incLanes="in_major_0 in_minor_0 in_tls_0" intLanes="">
    <request index="0" response="110" foes="110"/>
    <request index="1" response="001" foes="001"/>
    <request index="2" response="000" foes="001"/>
  </junction>
  <connection from="in_major" to="out_a" fromLane="0" toLane="0"
              uncontrolled="1" state="M" dir="s"/>
  <connection from="in_minor" to="out_b" fromLane="0" toLane="0"
              uncontrolled="1" state="m" dir="l"/>
  <connection from="in_tls" to="out_a" fromLane="0" toLane="0"
              tl="signal" linkIndex="0" state="G" dir="s"/>
</net>
""",
        encoding="utf-8",
    )

    result = audit_no_tls_major_response_consistency(network)

    assert result["counts"]["response_target_kind_histogram"] == {
        "external_no_tls": 1,
        "tls_controlled": 1,
    }
    assert result["counts"]["candidate_with_mixed_response_kinds_count"] == 1
    assert result["counts"][
        "candidate_with_mutual_external_no_tls_response_count"
    ] == 1
    assert result["counts"][
        "candidate_with_asymmetric_external_no_tls_response_count"
    ] == 0
