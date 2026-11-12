"""Fixed, identifier-free summaries of the preaction lane/phase graph.

Relations overlap: ``other`` contains every non-own link; foes and the two
directed yielding relations are subsets. A lane is counted once within each
relation/service set, even when several links or movements lead from it. These
are available lane supplies, never allocated movement demand. Different
relations/service sets may contain the same lane and are not additive.
"""
from __future__ import annotations

import math

import numpy as np

try:
    from .tsc_joint_response_interface import EXECUTOR_FIELDS
except ImportError:
    from tsc_joint_response_interface import EXECUTOR_FIELDS


PROTOCOL = "joint-response-fixed-graph-summary-v1"
SIGNALS = ("G", "g", "Y", "y", "s", "O", "r")
RELATIONS = ("own", "foe", "yields_to", "has_priority_over", "other")
SERVICE_SETS = (("can_serve", frozenset("GgsO")),
                ("priority", frozenset("G")), ("yielding", frozenset("g")))
LANE_FIELDS = ("lane_count", "vehicle_count", "halted_count", "occupancy_raw_mean",
               "occupied_speed_mean_mps", "length_sum_m", "length_mean_m",
               "speed_limit_mean_mps")
STATIC_NAMES = tuple(f"{role}_{name}" for role in ("incoming", "outgoing") for name in LANE_FIELDS) + (
    "incoming_vehicle_within_2s_limit_distance", "incoming_vehicle_within_5s_limit_distance",
    "incoming_vehicle_within_10s_limit_distance", "incoming_nearest_distance_m",
    "incoming_shared_turn_lane_count", "internal_lane_count", "internal_vehicle_count",
    "internal_length_sum_m", "internal_speed_limit_mean_mps",
) + EXECUTOR_FIELDS + ("nominal_request_accepted",) + tuple(f"{rel}_link_count" for rel in RELATIONS)
DYNAMIC_NAMES = tuple(name for rel in RELATIONS for step in range(10) for name in (
    *(f"{rel}_t{step:02d}_{signal}_link_fraction" for signal in SIGNALS),
    *(f"{rel}_t{step:02d}_{service}_{field}" for service, _ in SERVICE_SETS
      for field in ("vehicle_count", "near_vehicle_count", "halted_count")),
))
FEATURE_NAMES = STATIC_NAMES + DYNAMIC_NAMES


def _sum(values):
    return math.fsum(values)


def _lane_aggregate(selected):
    observations = [lane["lane_measurements"] for lane in selected]
    if any(row is None for row in observations):
        raise ValueError("Joint features require observed incoming and outgoing lane measurements")
    count = _sum(lane["vehicle_count"] for lane in selected)
    # Empty-lane speed has no weight; occupied speed must be observed.
    speed_total = _sum(lane["vehicle_count"] * row["mean_speed_mps"]
                       for lane, row in zip(selected, observations) if lane["vehicle_count"] > 0)
    lengths = _sum(lane["length_m"] for lane in selected)
    n = len(selected)
    return [n, count, _sum(row["halted"] for row in observations),
            _sum(row["occupancy_raw"] for row in observations) / n if n else 0.,
            speed_total / count if count else 0., lengths, lengths / n if n else 0.,
            _sum(lane["speed_limit_mps"] for lane in selected) / n if n else 0.]


def _relation_links(own, all_links, foes, responses):
    return {
        "own": own,
        "foe": {j for i, j in foes if i in own} | {i for i, j in foes if j in own},
        "yields_to": {j for i, j in responses if i in own},
        "has_priority_over": {i for i, j in responses if j in own},
        "other": all_links - own,
    }


def build_joint_features(graph):
    """Return [eight candidate indices, original movement order, 838 fields].

Only graph observations, topology, executor context and nominal sequences are
read. Lane/link/movement identifiers are joins only. Distances within 2/5/10
seconds at the speed limit are cumulative supply proxies, not travel forecasts;
nearest distance is zero for an empty inlet, alongside its zero vehicle count.
"""
    if graph["horizon_sec"] != 10 or not graph["conflicts"].get("full_context_complete", False):
        raise ValueError("Joint features require ten seconds and a complete static conflict context")
    views = {view["candidate_index"]: view for view in graph["candidate_views"]}
    if len(graph["candidate_views"]) != 8 or set(views) != set(range(8)):
        raise ValueError("Joint features require eight distinct candidate indices")
    lanes = {row["lane_id"]: row for row in graph["lane_nodes"]}
    movements = graph["movement_nodes"]
    all_links = {row["link_index"] for row in graph["link_nodes"]}
    link_lanes = {link: set() for link in all_links}
    for movement in movements:
        for link in movement["link_indices"]:
            link_lanes[link].update(movement["incoming_lanes"])
    sequences = {}
    for candidate, view in views.items():
        sequences[candidate] = {link: view["link_sequences"][str(link)] for link in all_links}
        for sequence in sequences[candidate].values():
            if len(sequence) != 10 or set(sequence) - set(SIGNALS):
                raise ValueError(f"Unsupported nominal signal sequence for fixed seven-character recipe: {sequence!r}")
    foes = graph["conflicts"]["foes"]
    responses = graph["conflicts"]["response"]
    result = np.empty((8, len(movements), len(FEATURE_NAMES)), dtype=float)
    for row_index, movement in enumerate(movements):
        incoming = [lanes[key] for key in set(movement["incoming_lanes"])]
        outgoing = [lanes[key] for key in set(movement["outgoing_lanes"])]
        internal = [lanes[key] for key in set(movement["internal_lanes"])]
        distances = [distance for lane in incoming for distance in lane["distances_to_lane_end_m"]]
        static = _lane_aggregate(incoming) + _lane_aggregate(outgoing)
        static += [sum(distance <= seconds * lane["speed_limit_mps"]
                       for lane in incoming for distance in lane["distances_to_lane_end_m"])
                   for seconds in (2, 5, 10)]
        static += [min(distances) if distances else 0., sum(lane["shared_turn_lane"] for lane in incoming),
                   len(internal), _sum(lane["vehicle_count"] for lane in internal),
                   _sum(lane["length_m"] for lane in internal),
                   _sum(lane["speed_limit_mps"] for lane in internal) / len(internal) if internal else 0.]
        static += [graph["executor_context"][name] for name in EXECUTOR_FIELDS]
        own = set(movement["link_indices"])
        relations = _relation_links(own, all_links, foes, responses)
        for rel in RELATIONS[1:]:
            relations[rel] -= own
        for candidate in range(8):
            values = static + [float(views[candidate]["nominal_request_accepted"])]
            values += [len(relations[rel]) for rel in RELATIONS]
            for rel in RELATIONS:
                links = relations[rel]
                for step in range(10):
                    states = {link: sequences[candidate][link][step] for link in links}
                    values += [sum(state == signal for state in states.values()) / len(links)
                               if links else 0. for signal in SIGNALS]
                    for _, allowed in SERVICE_SETS:
                        supplied = {lane for link, state in states.items() if state in allowed
                                    for lane in link_lanes[link]}
                        values += [_sum(lanes[lane]["vehicle_count"] for lane in supplied),
                                   _sum(lanes[lane]["near_stopline_vehicle_count"] for lane in supplied),
                                   _sum(lanes[lane]["lane_measurements"]["halted"] for lane in supplied)]
            result[candidate, row_index] = values
    if not np.isfinite(result).all():
        raise ValueError("Joint features contain unavailable or nonfinite preaction measurements")
    return result
