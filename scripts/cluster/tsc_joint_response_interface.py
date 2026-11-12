"""Pure preaction lane/phase graph; no labels, future observations or fitting."""
from __future__ import annotations

from copy import deepcopy


PROTOCOL = "joint-lane-phase-preaction-v3"
HORIZON_SEC = 10
EXECUTOR_FIELDS = (
    "executor_green", "executor_yellow", "executor_all_red",
    "executor_green_elapsed_sec", "executor_remaining_sec", "current_min_green_sec",
    "clearance_vehicle_count",
)
SINGLE_LANE_FIELDS = (
    "incoming_vehicle_count", "incoming_halted_count",
    "incoming_occupancy_raw_mean", "incoming_speed_mean_mps",
)


def _single_lane_observation(lane, group_data, movements):
    """A singleton movement aggregate preserves this lane's t0 measurements."""
    indices = [i for i, m in enumerate(movements.values()) if m["incoming_lanes"] == [lane]]
    if not indices:
        return None
    columns = [group_data["feature_names"].index(name) for name in SINGLE_LANE_FIELDS]
    reference = group_data["X"][int(group_data["reference_index"])]
    values = [tuple(float(reference[i][column]) for column in columns) for i in indices]
    if any(row != values[0] for row in values[1:]):
        raise ValueError(f"Repeated singleton observations disagree for lane {lane}")
    vehicles, halted, occupancy, speed = values[0]
    return dict(vehicles=int(vehicles), halted=int(halted), occupancy_raw=occupancy,
                mean_speed_mps=speed if vehicles > 0 else None)


def build_joint_response_interface(group_data, t0_frame, catalog, lane_geometry,
                                   lane_speed_limits, *, conflict_relations=None):
    """Build one shared observed graph and eight complete nominal action views.

    Caller supplies only frame zero and static per-lane speed limits. Saved
    incoming vehicle positions are preserved; the ten-second speed-limit
    distance is a proxy, not an attainable discharge count or a queue estimate.
    Conflict pairs must already be mapped from junction requests to TLS links.
    IDs and candidate_index are join keys, not predictive numerical features.
    """
    if t0_frame["offset_sec"] != 0:
        raise ValueError("The joint response interface requires the preaction frame")
    movement_ids = list(group_data["movement_ids"])
    movements = {key: catalog[key] for key in movement_ids}
    tls_ids = {m["tls_id"] for m in movements.values()}
    if len(tls_ids) != 1:
        raise ValueError("Each action group must describe one complete focal TLS")
    tls = next(iter(tls_ids))
    all_focal = {key for key, m in catalog.items() if m["tls_id"] == tls}
    if set(movement_ids) != all_focal:
        raise ValueError("Joint phase inputs require all movements of the focal TLS")
    branches = {b["candidate_index"]: b for b in group_data["branches"]}
    if len(group_data["branches"]) != 8 or set(branches) != set(range(8)):
        raise ValueError("Expected the eight distinct saved candidate indices")

    lanes = sorted({lane for m in movements.values()
                    for role in ("incoming_lanes", "outgoing_lanes", "internal_lanes")
                    for lane in m[role]})
    lane_nodes, movement_nodes = [], []
    for lane in lanes:
        incoming = [key for key, m in movements.items() if lane in m["incoming_lanes"]]
        destinations = sorted({movements[key]["outgoing_edge"] for key in incoming})
        roles = [role for role in ("incoming", "outgoing", "internal")
                 if any(lane in m[role + "_lanes"] for m in movements.values())]
        length = float(lane_geometry[lane]["length_m"])
        speed = float(lane_speed_limits[lane])
        distances = sorted(length - float(vehicle["position_m"])
                           for vehicle in t0_frame["vehicles"].values()
                           if vehicle["lane_id"] == lane)
        observation = t0_frame.get("focal_lane_measurements", {}).get(lane)
        observation_source = "direct_t0_focal_lane" if observation is not None else None
        if observation is None:
            observation = t0_frame["receiving_lanes"].get(lane)
            observation_source = "t0_receiving_lane_frame" if observation is not None else None
        if observation is None and incoming:
            observation = _single_lane_observation(lane, group_data, movements)
            if observation is not None:
                observation_source = "preaction_single_lane_row"
        lane_nodes.append(dict(
            lane_id=lane, edge_id=lane_geometry[lane]["edge_id"], roles=roles,
            length_m=length, speed_limit_mps=speed, vehicle_count=len(distances),
            distances_to_lane_end_m=distances,
            near_stopline_limit_distance_m=HORIZON_SEC * speed if incoming else None,
            near_stopline_vehicle_count=sum(d <= HORIZON_SEC * speed for d in distances)
                if incoming else None,
            incoming_movement_ids=incoming, candidate_outgoing_edges=destinations,
            shared_turn_lane=len(destinations) > 1,
            lane_measurements=deepcopy(observation) if observation is not None else None,
            lane_measurements_source=observation_source,
        ))
    for key, movement in movements.items():
        movement_nodes.append({"movement_id": key, **{
            name: deepcopy(movement[name]) for name in (
                "incoming_edge", "outgoing_edge", "incoming_lanes", "outgoing_lanes",
                "internal_lanes", "link_indices")}})
    links = sorted({i for m in movements.values() for i in m["link_indices"]})
    link_nodes = [{"link_index": i,
                   "movement_ids": [key for key, m in movements.items() if i in m["link_indices"]]}
                  for i in links]
    if conflict_relations is None:
        conflicts = {"status": "unavailable", "mapping_complete": False,
                     "foes": None, "response": None}
    else:
        conflicts = deepcopy(conflict_relations)
        for relation in ("foes", "response"):
            if any(len(pair) != 2 or any(i not in links for i in pair)
                   for pair in conflicts[relation]):
                raise ValueError("Conflict relations must use mapped focal TLS link indices")
        conflicts["status"] = "available" if conflicts["mapping_complete"] else "partial"

    candidate_views = []
    for index in range(8):
        preaction = branches[index]["preaction"]
        states = list(preaction["nominal_states"][tls])
        if len(states) != HORIZON_SEC or any(i >= len(state) for i in links for state in states):
            raise ValueError("Nominal timelines must cover every focal link for ten steps")
        candidate_views.append(dict(
            candidate_index=index, nominal_states=states,
            link_sequences={str(i): "".join(state[i] for state in states) for i in links},
            nominal_request_accepted=bool(preaction["nominal_request_accepted"][tls]),
        ))
    reference = int(group_data["reference_index"])
    names = group_data["feature_names"]
    executor = {name: float(group_data["X"][reference][0][names.index(name)])
                for name in EXECUTOR_FIELDS}
    return dict(
        protocol=PROTOCOL, case_id=group_data["case_id"], focal_tls=tls,
        horizon_sec=HORIZON_SEC, reference_index=reference,
        information_time_sec=float(t0_frame["time_sec"]),
        lane_nodes=lane_nodes, movement_nodes=movement_nodes, link_nodes=link_nodes,
        lane_to_movement=[[lane, key] for key, m in movements.items() for lane in m["incoming_lanes"]],
        movement_to_link=[[key, i] for key, m in movements.items() for i in m["link_indices"]],
        executor_context=executor, conflicts=conflicts, candidate_views=candidate_views,
        semantics={
            "demand": "Observed vehicles belong to lanes once; candidate turn destinations are topology, not an allocated demand.",
            "near_stopline": "Distance no greater than ten seconds at the static lane speed limit; not a discharge bound or calibrated queue.",
            "lane_measurements": "Prefer direct focal t0 lane observations, then saved receiving-lane observations, then exact singleton incoming-lane rows. Multi-lane aggregates are not decomposed. Empty singleton speed is null because its stored zero is an encoding convention, not an observed free-flow speed.",
            "signals": "Complete joint nominal timelines preserve G/g/Y/y/s/O/r characters and order; no future clearance extension oracle.",
            "conflicts": "Unavailable or incomplete mapping remains explicit; junction request indices are not assumed to be TLS indices.",
            "identifiers": "Join keys only; compare link_sequences rather than candidate_index when auditing information separation.",
        },
    )
