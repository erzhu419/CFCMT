"""Read-only movement crossings and receiving-lane responses from native SUMO steps.

The response unit is (TLS, incoming edge, outgoing edge). A lane disappearance
is never a crossing. Internal paths and observed road transitions identify the
movement; ambiguous or lost passages remain explicit unresolved observations.
"""
from __future__ import annotations

from collections import Counter


def movement_catalog(api, infos):
    """Expand controlled internal paths with TraCI getLinks' extended fields."""
    movements = {}
    for tls, info in sorted(infos.items()):
        for link_index, links in enumerate(info.controlled_links):
            for incoming, outgoing, via in links or ():
                incoming, outgoing, via = str(incoming), str(outgoing), str(via)
                origin = str(api.lane.getEdgeID(incoming))
                destination = str(api.lane.getEdgeID(outgoing))
                key = f"{tls}|{origin}|{destination}"
                item = movements.setdefault(key, dict(
                    movement_id=key, tls_id=str(tls), incoming_edge=origin,
                    outgoing_edge=destination, incoming_lanes=set(), outgoing_lanes=set(),
                    internal_lanes=set(), link_indices=set()))
                item["incoming_lanes"].add(incoming)
                item["outgoing_lanes"].add(outgoing)
                item["link_indices"].add(link_index)
                stack, visited = ([via] if via else []), set()
                while stack:
                    lane = stack.pop()
                    if lane in visited:
                        continue
                    visited.add(lane)
                    item["internal_lanes"].add(lane)
                    for successor in api.lane.getLinks(lane):
                        # (approachedLane, priority, open, foe, approachedInternal, ...)
                        following = str(successor[4]) or str(successor[0])
                        if following.startswith(":") and following not in visited:
                            stack.append(following)
    return {key: {name: sorted(value) if isinstance(value, set) else value
                  for name, value in item.items()} for key, item in movements.items()}


class MovementCounter:
    """Pure transition accounting, independent of SUMO and control decisions."""
    def __init__(self, movements):
        self.movements = movements
        self.previous = None
        self.pending = {}
        self.initial_pending = Counter()
        self.events = []
        self.diagnostics = Counter()

    def _internals(self, row):
        return [key for key, m in self.movements.items()
                if row["lane_id"] in m["internal_lanes"]]

    def _emit(self, kind, vehicle, key, time_sec, before, after, **extra):
        self.events.append(dict(kind=kind, vehicle_id=vehicle, movement_id=key,
                                time_sec=time_sec, before=before, after=after, **extra))

    def observe(self, time_sec, vehicles, terminal=None):
        terminal = terminal or {}
        if self.previous is None:
            for vehicle, row in vehicles.items():
                candidates = self._internals(row)
                if len(candidates) == 1:
                    key = candidates[0]
                    self.pending[vehicle] = (key, True)
                    self.initial_pending[key] += 1
                elif len(candidates) > 1:
                    self.diagnostics["ambiguous_initial_internal"] += 1
                    self._emit("unresolved_initial", vehicle, None, time_sec, None, row,
                               candidate_movements=candidates)
            self.previous = vehicles
            return
        for vehicle, before in self.previous.items():
            after = vehicles.get(vehicle)
            if vehicle in self.pending:
                key, initial = self.pending[vehicle]
                movement = self.movements[key]
                if after is not None and after["road_id"] == movement["outgoing_edge"]:
                    self._emit("receiving_entry", vehicle, key, time_sec, before, after,
                               initial_internal=initial)
                    del self.pending[vehicle]
                elif after is None or after["lane_id"] not in movement["internal_lanes"]:
                    reason = terminal.get(vehicle, "left_internal_path")
                    self.diagnostics["unresolved_pending"] += 1
                    self._emit("unresolved_exit", vehicle, key, time_sec, before, after,
                               reason=reason, initial_internal=initial)
                    del self.pending[vehicle]
                continue
            # Same-road lane changes cannot cross a controlled stop line.
            if after is not None and before["road_id"] == after["road_id"]:
                continue
            relevant = {key: m for key, m in self.movements.items()
                        if before["road_id"] == m["incoming_edge"]}
            if not relevant:
                continue
            candidates = [] if after is None else [key for key, m in relevant.items()
                if after["lane_id"] in m["internal_lanes"]
                or after["road_id"] == m["outgoing_edge"]]
            if len(candidates) == 1:
                key = candidates[0]
                self._emit("stopline_entry", vehicle, key, time_sec, before, after)
                if after["road_id"] == relevant[key]["outgoing_edge"]:
                    self._emit("receiving_entry", vehicle, key, time_sec, before, after,
                               initial_internal=False)
                    self.diagnostics["direct_incoming_to_receiving"] += 1
                else:
                    self.pending[vehicle] = (key, False)
            else:
                # Arriving at a route's final incoming edge is not a TLS crossing.
                if after is None and terminal.get(vehicle) == "arrived" and before.get("route_ends_here"):
                    self.diagnostics["arrived_before_controlled_junction"] += 1
                    continue
                self.diagnostics["unresolved_incoming_transition"] += 1
                self._emit("unresolved_transition", vehicle, None, time_sec, before, after,
                           candidate_movements=candidates,
                           reason=terminal.get(vehicle, "ambiguous_or_unobserved_path"))
        for vehicle in vehicles.keys() - self.previous.keys():
            candidates = self._internals(vehicles[vehicle])
            if candidates:
                self.diagnostics["unresolved_unobserved_entry"] += 1
                key = candidates[0] if len(candidates) == 1 else None
                self._emit("unresolved_entry", vehicle, key, time_sec, None, vehicles[vehicle],
                           candidate_movements=candidates)
                if key is not None:
                    self.pending[vehicle] = (key, False)
        self.previous = vehicles


class MovementResponseObserver:
    """Call at the pre-action checkpoint and each subsequent completed 1-s step."""
    def __init__(self, api, infos, checkpoint_sec, windows=(10, 60, 450)):
        self.api = api
        self.checkpoint_sec = float(checkpoint_sec)
        self.windows = tuple(windows)
        self.movements = movement_catalog(api, infos)
        self.counter = MovementCounter(self.movements)
        self.events = self.counter.events
        self.frames = []
        self.observed_lanes = sorted({lane for m in self.movements.values()
            for key in ("incoming_lanes", "outgoing_lanes", "internal_lanes") for lane in m[key]})
        self.receiving_lanes = sorted({lane for m in self.movements.values() for lane in m["outgoing_lanes"]})
        self.lane_geometry = {lane: dict(length_m=float(api.lane.getLength(lane)),
                                       edge_id=str(api.lane.getEdgeID(lane)))
                              for lane in self.observed_lanes}

    def sample(self, time_sec, applied_states=None):
        time_sec = float(time_sec)
        expected = self.checkpoint_sec if not self.frames else self.frames[-1]["time_sec"] + 1
        if time_sec != expected:
            raise ValueError(f"movement observer requires consecutive 1-s samples: {time_sec} != {expected}")
        api = self.api
        active = set(api.vehicle.getIDList())
        lane_ids = {lane: set(api.lane.getLastStepVehicleIDs(lane)) for lane in self.observed_lanes}
        monitored = set().union(*lane_ids.values()) if lane_ids else set()
        previous = self.counter.previous or {}
        # Previously tracked vehicles are queried for one exit step even after
        # leaving the monitored lanes; this prevents a lane-list loss proxy.
        tracked = (monitored | set(previous) | set(self.counter.pending)) & active
        vehicles = {}
        for vehicle in sorted(tracked):
            route = tuple(api.vehicle.getRoute(vehicle))
            index = int(api.vehicle.getRouteIndex(vehicle))
            vehicles[vehicle] = dict(lane_id=str(api.vehicle.getLaneID(vehicle)),
                road_id=str(api.vehicle.getRoadID(vehicle)), route_index=index,
                route_ends_here=index == len(route) - 1,
                position_m=float(api.vehicle.getLanePosition(vehicle)))
        terminal = {vehicle: "arrived" for vehicle in api.simulation.getArrivedIDList()}
        terminal.update({vehicle: "teleported" for vehicle in api.simulation.getStartingTeleportIDList()})
        self.counter.observe(time_sec, vehicles, terminal)
        self.counter.previous = {vehicle: row for vehicle, row in vehicles.items()
                                 if vehicle in monitored or vehicle in self.counter.pending}
        lanes = {lane: dict(vehicles=len(lane_ids[lane]),
                           halted=int(api.lane.getLastStepHaltingNumber(lane)),
                           occupancy_raw=float(api.lane.getLastStepOccupancy(lane)),
                           mean_speed_mps=float(api.lane.getLastStepMeanSpeed(lane)))
                 for lane in self.receiving_lanes}
        self.frames.append(dict(time_sec=time_sec, offset_sec=time_sec - self.checkpoint_sec,
            applied_states=dict(applied_states or {}), receiving_lanes=lanes,
            vehicles=vehicles, terminal=terminal,
            counter_retained_vehicle_ids=sorted(self.counter.previous),
            pending_by_movement=dict(Counter(key for key, initial in self.counter.pending.values()))))

    def summary(self):
        initial = self.frames[0] if self.frames else None
        by_offset = {row["offset_sec"]: row for row in self.frames}
        windows = {}
        for end in self.windows:
            if initial is None or end not in by_offset:
                continue
            frames = [row for row in self.frames if 0 < row["offset_sec"] <= end]
            events = [row for row in self.events
                      if self.checkpoint_sec < row["time_sec"] <= self.checkpoint_sec + end]
            movements = {}
            for key, movement in self.movements.items():
                entries = sum(row["kind"] == "stopline_entry" and row["movement_id"] == key for row in events)
                receipts = sum(row["kind"] == "receiving_entry" and row["movement_id"] == key for row in events)
                lost = sum(row["kind"] == "unresolved_exit" and row["movement_id"] == key for row in events)
                unobserved = sum(row["kind"] == "unresolved_entry" and row["movement_id"] == key for row in events)
                pending = by_offset[end]["pending_by_movement"].get(key, 0)
                started = self.counter.initial_pending[key]
                movements[key] = dict(stopline_entries=entries, receiving_entries=receipts,
                    initial_internal=started, pending_internal=pending, unresolved_exits=lost,
                    unobserved_entries=unobserved,
                    accounting_balanced=started + entries + unobserved == receipts + pending + lost)
            lanes = {}
            for lane in self.receiving_lanes:
                first, last = initial["receiving_lanes"][lane], by_offset[end]["receiving_lanes"][lane]
                lanes[lane] = dict(initial=first, final=last,
                    halted_vehicle_seconds=sum(row["receiving_lanes"][lane]["halted"] for row in frames),
                    vehicle_seconds=sum(row["receiving_lanes"][lane]["vehicles"] for row in frames),
                    mean_occupancy_raw=sum(row["receiving_lanes"][lane]["occupancy_raw"] for row in frames) / len(frames))
            windows[str(end)] = dict(movements=movements, receiving_lanes=lanes)
        unresolved = sum(value for name, value in self.counter.diagnostics.items() if name.startswith("unresolved") or name.startswith("ambiguous"))
        return dict(protocol="native-observed-movement-response-v1", movement_unit="tls,incoming_edge,outgoing_edge",
            checkpoint_sec=self.checkpoint_sec, sample_count=max(len(self.frames) - 1, 0),
            windows_sec=list(self.windows), catalog=self.movements, lane_geometry=self.lane_geometry,
            windows=windows, diagnostics=dict(self.counter.diagnostics), unresolved_count=unresolved,
            complete=len(windows) == len(self.windows) and unresolved == 0
                     and all(m["accounting_balanced"] for w in windows.values() for m in w["movements"].values()),
            semantics="Observed front crossing into internal path and receiving road; event times have one-step resolution. Lane changes do not identify movement flow.")
