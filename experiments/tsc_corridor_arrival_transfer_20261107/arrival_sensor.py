"""Causal paired road-entry and downstream approach detectors for the repaired SUMO networks."""
from collections import Counter, defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

DETECTOR_OFFSET_M = 30.0
HALTED_MPS = 0.1
QUEUE_GAP_M = 5.0
STATUSES = ("straight", "other", "terminal", "censored", "unknown")


def _catalog(cfg_path):
    cfg = Path(cfg_path)
    net = ET.parse(cfg.parent / ET.parse(cfg).find("input/net-file").attrib["value"])
    programs = {logic.attrib["id"]: [p.attrib["state"] for p in logic.findall("phase")
                                     if "G" in p.attrib["state"] and "y" not in p.attrib["state"].lower()]
                for logic in net.findall("tlLogic")}
    lane_ids, geometry = {}, {}
    for edge in net.findall("edge"):
        for lane in edge.findall("lane"):
            name = lane.attrib["id"]
            lane_ids[edge.attrib["id"], lane.attrib["index"]] = name
            geometry[name] = float(lane.attrib["length"]), float(lane.attrib["speed"])
    from_lane, to_edge = defaultdict(list), defaultdict(list)
    for conn in net.findall("connection"):
        source = lane_ids[conn.attrib["from"], conn.attrib["fromLane"]]
        from_lane[source].append(conn)
        to_edge[conn.attrib["to"]].append((source, conn))
    links, membership, predecessors = {}, {}, {}
    for edge in net.findall("edge"):
        road = edge.attrib["id"]
        up, down = edge.get("from"), edge.get("to")
        if up == down or up not in programs or down not in programs:
            continue
        incoming = [(origin, con) for origin, con in to_edge[road] if con.get("tl") == up]
        if not incoming:
            continue
        through = []
        road_lanes = [v.attrib["id"] for v in edge.findall("lane")]
        for lane in road_lanes:
            cons = from_lane[lane]
            if {c.get("dir") for c in cons} != {"s"} or {c.get("tl") for c in cons} != {down}:
                continue
            indices = sorted({int(c.attrib["linkIndex"]) for c in cons})
            phases = programs[down]
            if (any(all(s[i] == "G" for i in indices) for s in phases)
                    and any(all(s[i] == "r" for i in indices) for s in phases)):
                through.append((lane, indices))
        if not through:
            continue
        (straight, indices), = through
        length, speed = geometry[straight]
        links[road] = dict(tls=down, upstream_tls=up, lane=straight, indices=indices, length_m=length,
                           speed_mps=speed, freeflow_s=length/speed, tail_s=DETECTOR_OFFSET_M/speed)
        for lane in road_lanes:
            membership[lane] = road
        # Include direct one-step jumps as well as secondary internal via lanes.
        valid = {origin for origin, con in incoming}
        pending = [con.attrib["via"] for origin, con in incoming if con.get("via")]
        while pending:
            lane = pending.pop()
            if lane in valid:
                continue
            valid.add(lane)
            pending.extend(c.attrib["via"] for c in from_lane[lane] if c.get("via") and c.get("to") == road)
        predecessors[road] = valid
    return links, membership, geometry, predecessors


class ArrivalSensor:
    """Observe after each 1 s step. No route access, simulation setters or future observations."""
    def __init__(self, cfg_path):
        self.links, self.lane_edge, self.geometry, self.predecessors = _catalog(cfg_path)
        self.time = None
        self.previous_lane = {}
        self.active = {}
        self.unmatched = {}
        self.records = []
        self.provenance = Counter()
        self.flights = defaultdict(list)
        self.spillback_tls = set()
        self.unmatched_tls = set()

    def _complete(self, vid, t, status):
        record = self.active.pop(vid)
        self.records.append(dict(**record, end_s=t, status=status))

    def sample(self, api, time_sec):
        t = int(time_sec)
        if (time_sec != t or t != (0 if self.time is None else self.time + 1)
                or api.simulation.getTime() != t):
            raise ValueError("Expected the native clock at consecutive 1-second samples from zero")

        lanes, positions, queues = {}, {}, defaultdict(list)
        queued_edges = set()
        for vid in api.vehicle.getIDList():
            lane = api.vehicle.getLaneID(vid)
            lanes[vid] = lane
            edge = self.lane_edge.get(lane)
            if edge is None:
                continue
            length, limit = self.geometry[lane]
            speed = api.vehicle.getSpeed(vid)
            distance = length - DETECTOR_OFFSET_M - api.vehicle.getLanePosition(vid)
            positions[vid] = (edge, lane, distance, speed, limit)
            if lane == self.links[edge]["lane"]:
                if speed < HALTED_MPS and distance <= 0:
                    queued_edges.add(edge)
                queues[edge].append((distance, vid, speed))

        # A stopped body on the line, or the short gap between two adjacent
        # stopped vehicles covering it, means the detection line is queued over.
        self.spillback_tls = set()
        for edge, cars in queues.items():
            behind = [r for r in cars if r[0] > 0]
            ahead = [r for r in cars if r[0] <= 0]
            if not ahead:
                continue
            near_dist, near_id, near_speed = max(ahead)
            if near_speed >= HALTED_MPS:
                continue
            rear = near_dist + api.vehicle.getLength(near_id)
            if rear >= 0:
                self.spillback_tls.add(self.links[edge]["tls"])
            elif behind:
                dist, _, back_speed = min(behind)
                if back_speed < HALTED_MPS and dist - rear <= QUEUE_GAP_M:
                    self.spillback_tls.add(self.links[edge]["tls"])

        arrived = set(api.simulation.getArrivedIDList())
        departed = set(api.simulation.getDepartedIDList())
        for vid in list(self.active):
            old_edge = self.active[vid]["edge"]
            if vid not in positions or positions[vid][0] != old_edge:
                self._complete(vid, t, "terminal" if vid in arrived else "unknown")
        # Non-paired initial/unresolved vehicles are only relevant until they
        # leave the detector risk set. Known link insertions are not an upstream release.
        self.unmatched = {v: e for v, e in self.unmatched.items()
                          if v in positions and positions[v][0] == e and positions[v][2] > 0}
        self.flights = defaultdict(list)
        for vid, (edge, lane, distance, speed, limit) in positions.items():
            meta = self.links[edge]
            old_lane = self.previous_lane.get(vid)
            if self.lane_edge.get(old_lane) != edge:
                if old_lane in self.predecessors[edge]:
                    self.provenance["released"] += 1
                    record = dict(edge=edge, vid=vid, entry_s=t, queued_at_entry=edge in queued_edges,
                                  freeflow_s=meta["freeflow_s"], tail_s=meta["tail_s"])
                    self.active[vid] = record
                    if distance <= 0:
                        self._complete(vid, t, "unknown")
                elif vid in departed:
                    self.provenance["departed_on_link"] += 1
                elif t == 0:
                    self.provenance["initial_on_link"] += 1
                    if distance > 0:
                        self.unmatched[vid] = edge
                else:
                    self.provenance["unresolved_entries"] += 1
                    if distance > 0:
                        self.unmatched[vid] = edge
            if vid not in self.active:
                continue
            if distance <= 0:
                self._complete(vid, t, "straight" if lane == meta["lane"] else "other")
            elif speed >= HALTED_MPS:
                self.flights[meta["tls"]].append(dict(**self.active[vid], age_s=t-self.active[vid]["entry_s"],
                    on_straight_lane=lane == meta["lane"], remaining_m=distance, speed_limit_mps=limit,
                    analytic_eta_s=distance/limit))
        self.unmatched_tls = {self.links[e]["tls"] for e in self.unmatched.values()}
        self.time = t
        self.previous_lane = lanes

    def decision(self, tls):
        return dict(flights=self.flights.get(tls, []), spillback=tls in self.spillback_tls,
                    unmatched=tls in self.unmatched_tls)

    def finish(self, time_sec):
        if time_sec != self.time:
            raise ValueError("Window end differs from the most recent observed timestep")
        records = self.records + [dict(**r, status="censored", end_s=time_sec) for r in self.active.values()]
        counts = Counter(r["status"] for r in records)
        identities = {(r["vid"], r["edge"], r["entry_s"]) for r in records}
        summary = dict(entry_count=self.provenance["released"], counts_by_status={k: counts[k] for k in STATUSES},
                       unresolved_entries=self.provenance["unresolved_entries"], clock_from_s=0, clock_to_s=time_sec,
                       frames=time_sec + 1, departed_on_link=self.provenance["departed_on_link"],
                       initial_on_link=self.provenance["initial_on_link"], controlled_links=len(self.links),
                       accounting_ok=len(identities) == len(records) == self.provenance["released"])
        return records, summary
