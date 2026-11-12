"""Saved observations for descriptive diagnostics only; never training inputs."""
import json
from pathlib import Path

import numpy as np


def read_context(group_data, group_dir):
    """Return (action, movement) arrays from the eight server-side observations.

    Exposure is mean seconds over a movement's controlled links. A mismatch
    means any applied link character differs from its nominal same-step value.
    The near-stopline field uses only t0 positions: distance to lane end no more
    than ten seconds at the movement's mean incoming speed limit. It is a
    reachable-at-speed-limit proxy, not a queue, turn allocation or flow bound.
    """
    group_dir = Path(group_dir)
    movement_ids = group_data["movement_ids"]
    reference = int(group_data["reference_index"])
    features = np.asarray(group_data["X"], dtype=float)
    speed_column = group_data["feature_names"].index("incoming_speed_limit_mean_mps")
    shape = (8, len(movement_ids))
    result = {"actual_" + char: np.zeros(shape) for char in "GgsOYyr"}
    result.update(nominal_Y=np.zeros(shape), nominal_y=np.zeros(shape),
                  schedule_mismatch=np.zeros(shape, dtype=bool))
    near_count = np.zeros(len(movement_ids))
    bytes_read = 0
    for action in range(8):
        branch = "first_pp" if action == reference else f"action_{action}"
        payload = (group_dir / branch / "observations_server_only.json").read_bytes()
        bytes_read += len(payload)
        observation = json.loads(payload)
        frames = observation["frames"]
        assert [frame["offset_sec"] for frame in frames] == list(range(11))
        for column, movement_id in enumerate(movement_ids):
            movement = observation["catalog"][movement_id]
            tls, links = movement["tls_id"], movement["link_indices"]
            nominal = observation["preaction"]["nominal_states"][tls]
            applied = [frame["applied_states"][tls] for frame in frames[1:]]
            assert len(nominal) == 10 and links
            assert all(0 <= link < len(state) for link in links
                       for state in nominal + applied)
            actual_chars = np.asarray([[state[link] for link in links] for state in applied])
            nominal_chars = np.asarray([[state[link] for link in links] for state in nominal])
            for char in "GgsOYyr":
                result["actual_" + char][action, column] = np.count_nonzero(actual_chars == char) / len(links)
            for char in "Yy":
                result["nominal_" + char][action, column] = np.count_nonzero(nominal_chars == char) / len(links)
            result["schedule_mismatch"][action, column] = np.any(actual_chars != nominal_chars)
            if action == reference:
                lanes = set(movement["incoming_lanes"])
                reach = 10.0 * features[reference, column, speed_column]
                near_count[column] = sum(
                    observation["lane_geometry"][vehicle["lane_id"]]["length_m"]
                    - vehicle["position_m"] <= reach
                    for vehicle in frames[0]["vehicles"].values()
                    if vehicle["lane_id"] in lanes)
    result["near_stopline_count"] = np.broadcast_to(near_count, shape).copy()
    result["obs_bytes_read"] = bytes_read
    return result
