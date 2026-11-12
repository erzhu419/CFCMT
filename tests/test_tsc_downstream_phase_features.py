from copy import deepcopy

import numpy as np
import pytest

from scripts.cluster.tsc_downstream_phase_features import NetworkTopology, features_for_group


def fixture_net(tmp_path):
    # Lane identifiers deliberately do not equal edge_id + '_' + lane_index.
    edges = [
        ("i", "start", "focal", "incoming.custom"),
        ("b", "focal", "branch", "branch.root"),
        ("x", "focal", "exit", "exit.lane"),
        ("a", "branch", "A", "receive.A"),
        ("c", "branch", "B", "receive.B"),
        ("ao", "A", "endA", "out.A"),
        ("ax", "A", "endAx", "out.Ax"),
        ("bo", "B", "endB", "out.B"),
    ]
    xml = ["<net>"]
    for edge, start, end, lane in edges:
        xml.append(f'<edge id="{edge}" from="{start}" to="{end}"><lane id="{lane}" index="0"/></edge>')
    for node in {v for _, start, end, _ in edges for v in (start, end)}:
        xml.append(f'<junction id="{node}" type="priority"/>')
    for source, target, tls, index in [
        ("i", "b", "F", 0), ("i", "b", "F", 0), ("i", "x", "F", 1),
        ("b", "a", None, None), ("b", "c", None, None),
        ("a", "ao", "A", 0), ("a", "ax", "A", 0), ("a", "ax", "A", 1),
        ("c", "bo", "B", 0),
    ]:
        control = f' tl="{tls}" linkIndex="{index}"' if tls else ""
        xml.append(f'<connection from="{source}" to="{target}" fromLane="0" toLane="0"{control}/>')
    xml.append("</net>")
    path = tmp_path / "network.net.xml"
    path.write_text("".join(xml))
    return NetworkTopology.from_net(path)


def fixture_snapshot():
    def executor(state, age):
        return dict(mode="green", current_state=state, current_green_state="r" * len(state),
                    green_elapsed_sec=age, remaining_sec=0., pending_all_red_sec=0.)
    return {"executors": {"F": executor("GG", 40), "A": executor("Gg", 10), "B": executor("G", 20)}}


def test_split_weights_deduplicate_connections_and_controlled_indices(tmp_path):
    topology = fixture_net(tmp_path)
    features, summary = features_for_group(topology, fixture_snapshot(), "F", ["Gr", "gG", "rG"])
    np.testing.assert_allclose(features, [[1., .75, .25, 0., 15., 0.],
                                         [.5, .375, .125, 0., 7.5, 0.],
                                         [0., 0., 0., 0., 0., 0.]])
    assert summary["candidates"][0]["enabled_connections"] == 1
    assert summary["candidates"][1]["exit_connections"] == 1
    assert topology.receiving_entries("branch.root", "F") == (("A", "receive.A"), ("B", "receive.B"))


def test_actual_clearance_state_overrides_stale_green_and_timer_is_not_waiting_time(tmp_path):
    topology, snapshot = fixture_net(tmp_path), fixture_snapshot()
    snapshot["executors"]["A"].update(mode="yellow", current_state="yy", current_green_state="GG",
                                          remaining_sec=2., pending_all_red_sec=1.)
    snapshot["executors"]["B"].update(mode="all_red", current_state="r", current_green_state="G",
                                          remaining_sec=3., pending_all_red_sec=99.)
    features, _ = features_for_group(topology, snapshot, "F", ["Gr"])
    np.testing.assert_array_equal(features, [[1., 0., 0., 1., 0., 3.]])


def test_missing_connection_on_nonterminal_lane_is_not_an_exit(tmp_path):
    topology = fixture_net(tmp_path)
    topology.successors["branch.root"] = ()
    with pytest.raises(ValueError, match="unresolved nonterminal lane"):
        features_for_group(topology, fixture_snapshot(), "F", ["Gr"])


def test_nonterminal_cycle_is_not_silently_dropped(tmp_path):
    topology = fixture_net(tmp_path)
    topology.successors["branch.root"] = ("branch.root", "receive.A")
    with pytest.raises(ValueError, match="unresolved nonterminal cycle"):
        features_for_group(topology, fixture_snapshot(), "F", ["Gr"])


@pytest.mark.parametrize("problem", ["missing", "snapshot_width", "candidate_width"])
def test_incomplete_executor_or_signal_alignment_stops_extraction(tmp_path, problem):
    topology, snapshot = fixture_net(tmp_path), deepcopy(fixture_snapshot())
    states = ["Gr"]
    if problem == "missing":
        del snapshot["executors"]["A"]
    elif problem == "snapshot_width":
        snapshot["executors"]["A"]["current_state"] = "G"
    else:
        states = ["G"]
    with pytest.raises(ValueError, match="missing pre-action executor|state/linkIndex width mismatch"):
        features_for_group(topology, snapshot, "F", states)
