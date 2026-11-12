"""Restore the registered PP executor from its program and saved preaction clock."""
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from xml.etree import ElementTree as ET

from cf_h2o.eval.traffic_signal_resco_phase_benchmark import _green_phase_candidates
from cf_h2o.traffic_signal.safe_phase_controller import (
    PhaseTiming, SafePhaseExecutor, YELLOW_CHARS, _is_all_red, _phase_min_green,
)
from source_pilot import action_contrast
from tsc_physical_response_data import nominal_signal_plan


def load_programs(cfg_path):
    cfg = Path(cfg_path)
    net = ET.parse(cfg.parent / ET.parse(cfg).find("input/net-file").get("value"))
    widths = {}
    for conn in net.findall("connection[@tl]"):
        tls = conn.get("tl")
        widths[tls] = max(widths.get(tls, 0), int(conn.get("linkIndex")) + 1)
    programs = {}
    for logic in net.findall("tlLogic"):
        tls = logic.get("id")
        if tls in programs or logic.get("type") != "static":
            raise ValueError(f"Expected one static repaired program for {tls}")
        # SUMO static phases with no explicit minDur expose their duration as minDur.
        phases = [SimpleNamespace(state=p.attrib["state"], duration=float(p.attrib["duration"]),
                                  minDur=float(p.get("minDur", p.attrib["duration"])))
                  for p in logic.findall("phase")]
        candidates = _green_phase_candidates(phases, widths[tls])
        by_state = {p.state: p for p in phases}
        yellow = [p.duration for p in phases if any(c in YELLOW_CHARS for c in p.state)]
        all_red = [p.duration for p in phases if _is_all_red(p.state)]
        programs[tls] = [PhaseTiming(c.state, _phase_min_green(by_state[c.state], c, 5.0),
                                     median(yellow) if yellow else 3.0, median(all_red) if all_red else 1.0)
                         for c in candidates]
    return programs


def rebuild_requests(timings, tls, clock):
    executor = SafePhaseExecutor(sumo_api=None, tls_id=tls, timings=timings,
                                 initial_state=clock["current_state"])
    executor.restore(dict(clock, tls_id=tls), write_state=False)
    states = list(executor.feasible_states_now())
    plans = [nominal_signal_plan(executor, state)["states"] for state in states]
    return states, plans


def validate_focal(states, plans, case, start, pair):
    if states != start["candidate_states"]:
        raise ValueError("Reconstructed focal requests differ from the archived feasible order")
    contrast = action_contrast(plans, pair)
    if (contrast != start["supply"]["contrast"] or contrast is None
            or any(case[k] != value for k, value in contrast.items())):
        raise ValueError("Focal high/low indices, exposure or signal sequences differ from the recorded contrast")
    return contrast
