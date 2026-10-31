import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cf_h2o.traffic_signal.dynamic_graph_context import (
    SIGNAL_GRAPH_FEATURES,
    build_dynamic_signal_contexts,
    build_signal_routing_graph,
    candidate_graph_features,
)


def _state(queue, occupancy, downstream_queue=0.0):
    return SimpleNamespace(
        q_by_lane={"lane": float(queue)},
        occ_by_lane={"lane": float(occupancy)},
        down_q_by_lane={"lane": float(downstream_queue)},
    )


def test_dynamic_graph_context_tracks_directed_neighbors_and_network_load():
    infos = {
        "a": SimpleNamespace(
            incoming_lanes=("a_in",),
            controlled_links=((('a_in', 'a_to_b', ''),),),
        ),
        "b": SimpleNamespace(
            incoming_lanes=("a_to_b",),
            controlled_links=((('a_to_b', 'b_out', ''),),),
        ),
        "c": SimpleNamespace(
            incoming_lanes=("c_in",),
            controlled_links=((('c_in', 'c_out', ''),),),
        ),
    }
    states = {
        "a": _state(8.0, 20.0, 2.0),
        "b": _state(4.0, 40.0, 1.0),
        "c": _state(2.0, 10.0),
    }
    contexts = build_dynamic_signal_contexts(
        states,
        infos,
        active_vehicles=30.0,
    )
    assert contexts["a"].downstream_tls_by_lane == {"a_to_b": ("b",)}
    assert contexts["a"].values["graph_downstream_q_mean"] == 4.0
    assert contexts["b"].values["graph_upstream_q_mean"] == 8.0
    assert contexts["c"].values["graph_neighbor_q_max"] == 0.0
    assert contexts["a"].values["network_active_per_tls"] == 10.0


def test_candidate_graph_features_are_action_specific_and_finite():
    infos = {
        "a": SimpleNamespace(
            incoming_lanes=("a_in",),
            controlled_links=((('a_in', 'a_to_b', ''),),),
        ),
        "b": SimpleNamespace(
            incoming_lanes=("a_to_b",),
            controlled_links=((('a_to_b', 'b_out', ''),),),
        ),
    }
    states = {"a": _state(8.0, 20.0), "b": _state(4.0, 40.0)}
    context = build_dynamic_signal_contexts(
        states,
        infos,
        active_vehicles=12.0,
    )["a"]
    green = candidate_graph_features(context, infos["a"], "G", green_queue=8.0)
    red = candidate_graph_features(context, infos["a"], "r", green_queue=0.0)
    index = {name: idx for idx, name in enumerate(SIGNAL_GRAPH_FEATURES)}
    assert green.shape == (len(SIGNAL_GRAPH_FEATURES),)
    assert np.all(np.isfinite(green))
    assert green[index["green_signal_down_q_mean"]] == 4.0
    assert green[index["green_corridor_pressure"]] == 4.0
    assert red[index["green_signal_down_q_mean"]] == 0.0


def test_routing_graph_traces_through_unsignalized_lanes():
    infos = {
        "a": SimpleNamespace(
            incoming_lanes=("a_in",),
            controlled_links=((('a_in', 'middle_0', ''),),),
        ),
        "b": SimpleNamespace(
            incoming_lanes=("b_in",),
            controlled_links=((('b_in', 'b_out', ''),),),
        ),
    }
    links = {
        "middle_0": (("middle_1",),),
        "middle_1": (("b_in",),),
        "b_out": (),
    }
    api = SimpleNamespace(
        lane=SimpleNamespace(getLinks=lambda lane: links.get(lane, ()))
    )
    graph = build_signal_routing_graph(api, infos, max_hops=4)
    assert graph.adjacency["a"] == ("b",)
    assert graph.downstream_tls_by_lane["a"]["middle_0"] == ("b",)


def test_dynamic_graph_reductions_are_python_hash_seed_invariant():
    project_root = Path(__file__).resolve().parents[2]
    script = r'''
import json
from types import SimpleNamespace

from cf_h2o.traffic_signal.dynamic_graph_context import (
    SIGNAL_GRAPH_STATE_FEATURES,
    SignalRoutingGraph,
    build_dynamic_signal_contexts,
)

neighbors = {f"neighbor_{index}" for index in range(12)}
tls_ids = {"focal", *neighbors}
infos = {tls_id: SimpleNamespace() for tls_id in tls_ids}
states = {}
for index, tls_id in enumerate(sorted(tls_ids)):
    scale = 1.0e16 if tls_id == "neighbor_0" else float(index + 1)
    states[tls_id] = SimpleNamespace(
        q_by_lane={"lane_b": 0.25 * scale, "lane_a": 0.75 * scale},
        occ_by_lane={"lane_b": 0.5 * scale, "lane_a": 0.125 * scale},
        down_q_by_lane={"lane_b": 0.2 * scale, "lane_a": 0.1 * scale},
    )
graph = SignalRoutingGraph(
    adjacency={
        tls_id: tuple(neighbors) if tls_id == "focal" else ()
        for tls_id in tls_ids
    },
    downstream_tls_by_lane={tls_id: {} for tls_id in tls_ids},
)
values = build_dynamic_signal_contexts(
    states,
    infos,
    active_vehicles=123.0,
    routing_graph=graph,
)["focal"].values
print(json.dumps([float(values[name]).hex() for name in SIGNAL_GRAPH_STATE_FEATURES]))
'''
    outputs = set()
    for hash_seed in ("1", "2", "8675309"):
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(project_root), env.get("PYTHONPATH", "")))
        )
        outputs.add(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=project_root,
                env=env,
                text=True,
            ).strip()
        )
    assert len(outputs) == 1
