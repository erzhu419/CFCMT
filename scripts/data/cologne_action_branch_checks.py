"""Physical state checks for the two Cologne action branches."""

EXECUTOR_FIELDS = (
    "mode", "current_state", "current_green_state", "target_green_state",
    "green_elapsed_sec", "remaining_sec", "pending_all_red_sec",
)


def physical_snapshot(api, executors):
    vehicles = {
        vehicle: {
            "lane_id": str(api.vehicle.getLaneID(vehicle)),
            "route_index": int(api.vehicle.getRouteIndex(vehicle)),
            "route": list(api.vehicle.getRoute(vehicle)),
            "lane_position": float(api.vehicle.getLanePosition(vehicle)),
            "speed": float(api.vehicle.getSpeed(vehicle)),
        }
        for vehicle in sorted(api.vehicle.getIDList())
    }
    causal_states = {}
    for tls in sorted(executors):
        snapshot = executors[tls].snapshot()
        causal_states[tls] = {key: snapshot[key] for key in EXECUTOR_FIELDS}
    return {
        "time_sec": float(api.simulation.getTime()), "vehicles": vehicles,
        "executors": causal_states,
        # SUMO can report zero pending vehicles immediately after loading a state.
        "pending_raw": len(api.simulation.getPendingVehicles()),
    }


def compare_physical(before, after, tolerance=1e-8):
    mismatches, mismatch_count = [], 0
    max_difference = 0.0

    def mismatch(path, original, restored):
        nonlocal mismatch_count
        mismatch_count += 1
        if len(mismatches) < 20:
            mismatches.append({"path": path, "before": original, "after": restored})

    if before["time_sec"] != after["time_sec"]:
        mismatch("time_sec", before["time_sec"], after["time_sec"])
    old_vehicles, new_vehicles = before["vehicles"], after["vehicles"]
    if set(old_vehicles) != set(new_vehicles):
        mismatch("vehicle_ids", sorted(old_vehicles), sorted(new_vehicles))
    for vehicle in sorted(set(old_vehicles) & set(new_vehicles)):
        original, restored = old_vehicles[vehicle], new_vehicles[vehicle]
        for key in ("lane_id", "route_index", "route"):
            if original[key] != restored[key]:
                mismatch(f"vehicles.{vehicle}.{key}", original[key], restored[key])
        for key in ("lane_position", "speed"):
            difference = abs(original[key] - restored[key])
            max_difference = max(max_difference, difference)
            if difference > tolerance:
                mismatch(f"vehicles.{vehicle}.{key}", original[key], restored[key])
    old_executors, new_executors = before["executors"], after["executors"]
    if set(old_executors) != set(new_executors):
        mismatch("executor_ids", sorted(old_executors), sorted(new_executors))
    for tls in sorted(set(old_executors) & set(new_executors)):
        for key in EXECUTOR_FIELDS:
            if old_executors[tls][key] != new_executors[tls][key]:
                mismatch(f"executors.{tls}.{key}", old_executors[tls][key], new_executors[tls][key])
    return {"passed": mismatch_count == 0, "mismatch_count": mismatch_count,
            "mismatches": mismatches, "max_position_or_speed_difference": max_difference}
