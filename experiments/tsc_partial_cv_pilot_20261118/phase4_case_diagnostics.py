"""Case replay and native-prefix diagnostics for the Partial-CV revision."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import json
import math
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import _parse_tripinfo_metrics
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import _collision_incident_key
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    _controlled_lane_set,
    _start_sumo,
    _tls_phase_infos,
)
from cf_h2o.sumo_runtime import load_libsumo
from cf_h2o.traffic_signal.safe_phase_controller import (
    aggregate_phase_audits,
    build_safe_phase_executors,
)
from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import (
    build_1hop_tls_neighbors,
    extract_decision_group_features,
)
from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import (
    PartialCvObserver,
    select_observed_pressure_phase,
)
from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_model import (
    feasible_indices_under_max_red,
)
from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_transfer import (
    CONTROL_INTERVAL_SEC,
    P_CV,
)

CASE_SPIKES = (920104, 920105)
SALT_LAKE_SPIKE = 930108
VISITED_SEEDS = (940101, 940102)
VISITED_TIMES = (600, 1200)
SEARCH_WINDOW_SEC = 60
HORIZONS = (60, 120)
CONTINUATIONS = ("conservation_gp_10", "cfcmt_b6")
METRICS = ("mean_waiting_sec", "system_vehicle_hours", "completion_ratio", "collision_incidents", "switch_count")


class NativeEpisode:
    def __init__(self, sumo, task):
        self.sumo = sumo
        self.infos = _tls_phase_infos(sumo)
        self.lanes = _controlled_lane_set(self.infos)
        self.up, self.down, self.cones = build_1hop_tls_neighbors(self.infos)
        self.executors = build_safe_phase_executors(sumo, self.infos)
        self.observer = PartialCvObserver(sumo_api=sumo, infos=self.infos, seed=task["seed"], p_cv=P_CV)
        self.observer.observe_step(sumo, dt_sec=1.0)
        self.begin = float(sumo.simulation.getTime())
        self.policy = task["policy"]
        self.ranker = None
        if self.policy != "conservation_gp_10":
            with Path(task["models_pkl"]).open("rb") as file:
                self.ranker = pickle.load(file)[task["family"]][self.policy]

    def group(self, tls):
        info = self.infos[tls]
        ex = self.executors[tls]
        feasible, state, actions = extract_decision_group_features(
            info=info, executor=ex, observer=self.observer, infos_all=self.infos,
            up_neighbors=self.up[tls], down_neighbors=self.down[tls],
        )
        if len(feasible) == 1:
            return {"phases": list(feasible), "preferred": 0, "chosen": 0, "allowed": [0]}
        allowed = feasible_indices_under_max_red(
            info=info, feasible=feasible, observer=self.observer,
            max_red_sec=None, current_green_state=str(ex.current_green_state),
        )
        scores = self.ranker.score_group(state, actions)
        chosen = min(allowed, key=lambda i: float(scores[i]))
        return {
            "phases": list(feasible), "preferred": int(np.argmin(scores)),
            "chosen": int(chosen), "allowed": list(allowed),
            "state": state.tolist(), "actions": actions.tolist(),
            "scores": np.asarray(scores).tolist(),
        }

    def tick(self, policy, override=None):
        requests = {}
        for tls in sorted(self.infos):
            if override is not None and tls == override[0]:
                target = override[1]
            elif policy == "conservation_gp_10":
                target = select_observed_pressure_phase(
                    info=self.infos[tls], executor=self.executors[tls],
                    observer=self.observer, mode="conservation_gp",
                )
            else:
                group = self.group(tls)
                target = group["phases"][group["chosen"]]
            self.executors[tls].request(target)
            requests[tls] = target
        return requests

    def step(self):
        sim = self.sumo.simulation
        before = float(sim.getTime())
        sim.step()
        after = float(sim.getTime())
        dt = max(after - before, 0.0)
        for executor in self.executors.values():
            executor.advance(dt)
        self.observer.observe_step(self.sumo, dt_sec=max(dt, 1.0))
        return after - self.begin


@contextmanager
def episode(task, tripinfo=None):
    sumo = load_libsumo()
    _start_sumo(sumo, Path(task["sumocfg"]), task["seed"], tripinfo_output=tripinfo)
    try:
        yield NativeEpisode(sumo, task)
    finally:
        sumo.close()


def replay_case(task):
    samples = []
    vehicle_seconds = departed = completed = teleports = 0
    collisions = set()
    with TemporaryDirectory(prefix="partialcv_review_") as directory:
        tripinfo = Path(directory) / "tripinfo.xml"
        with episode(task, tripinfo) as run:
            sumo = run.sumo
            sim = sumo.simulation
            inflow_links = {
                tls: sorted({lane.rsplit("_", 1)[0] for lane in info.incoming_lanes})
                for tls, info in run.infos.items()
            }
            pending = len(sim.getPendingVehicles())
            for elapsed in range(0, 3600, CONTROL_INTERVAL_SEC):
                pending_by_edge = {
                    edge: len(sumo.edge.getPendingVehicles(edge))
                    for edge in sorted({edge for edges in inflow_links.values() for edge in edges})
                }
                signal_states = {
                    tls: str(sumo.trafficlight.getRedYellowGreenState(tls)) for tls in sorted(run.infos)
                }
                requests = run.tick(task["policy"])
                samples.append({
                    "time_sec": elapsed,
                    "pending": pending,
                    "halted": sum(sumo.lane.getLastStepHaltingNumber(lane) for lane in run.lanes),
                    "completed": completed,
                    "pending_by_edge": pending_by_edge,
                    "requests": requests,
                    "signals": signal_states,
                })
                for _ in range(CONTROL_INTERVAL_SEC):
                    clock = run.step()
                    departed += sim.getDepartedNumber()
                    completed += sim.getArrivedNumber()
                    teleports += sim.getStartingTeleportNumber()
                    pending = len(sim.getPendingVehicles())
                    collisions.update(_collision_incident_key(c) for c in sim.getCollisions())
                    if clock >= 60:
                        vehicle_seconds += sumo.vehicle.getIDCount() + pending
            samples.append({
                "time_sec": 3600, "pending": pending,
                "halted": sum(sumo.lane.getLastStepHaltingNumber(lane) for lane in run.lanes),
                "completed": completed,
            })
            switches = int(aggregate_phase_audits(run.executors).get("switches", 0))
        result = {
            "mean_waiting_sec": float(_parse_tripinfo_metrics(tripinfo)["mean_tripinfo_waiting_time"]),
            "system_vehicle_hours": vehicle_seconds / 3600,
            "completion_ratio": completed / max(departed + pending, 1),
            "collision_incidents": len(collisions), "starting_teleports": teleports,
            "switch_count": switches, "completed": completed, "pending_end": pending,
        }
    return {**task, "metrics": result, "inflow_links": inflow_links, "samples": samples}


def scan_policy_states(task):
    states = []
    multi = changed = 0
    final_time = max(VISITED_TIMES) + SEARCH_WINDOW_SEC
    with episode(task) as run:
        for elapsed in range(0, final_time + 1, CONTROL_INTERVAL_SEC):
            groups = {tls: run.group(tls) for tls in sorted(run.infos)}
            multi += sum(len(g["phases"]) > 1 for g in groups.values())
            changed += sum(g["preferred"] != g["chosen"] for g in groups.values())
            if len(states) < len(VISITED_TIMES):
                anchor = VISITED_TIMES[len(states)]
                if anchor <= elapsed <= anchor + SEARCH_WINDOW_SEC:
                    choice = next(((tls, g) for tls, g in groups.items() if len(g["phases"]) > 1), None)
                    if choice is not None:
                        tls, group = choice
                        states.append({**task, "time_sec": elapsed, "tls": tls, "group": group})
            if elapsed == final_time:
                break
            for tls, group in groups.items():
                run.executors[tls].request(group["phases"][group["chosen"]])
            for _ in range(CONTROL_INTERVAL_SEC):
                run.step()
    if len(states) != len(VISITED_TIMES):
        raise ValueError(f"No multi-action state near a comparison time in {task['scenario']}/{task['seed']}")
    return {"states": states, "multi_action_decisions": multi, "max_red_changed_preferred": changed}


def branch_at_state(task):
    tls = task["tls"]
    horizon_cost = {}
    with episode(task) as run:
        for _ in range(0, task["time_sec"], CONTROL_INTERVAL_SEC):
            run.tick(task["policy"])
            for _ in range(CONTROL_INTERVAL_SEC):
                run.step()
        if run.group(tls) != task["group"]:
            raise ValueError(f"Native-prefix mismatch at {task['scenario']}/{task['seed']}/{task['time_sec']}/{tls}")
        incoming = run.infos[tls].incoming_lanes
        edges = sorted({lane.rsplit("_", 1)[0] for lane in incoming})
        lanes = run.cones[tls]
        sumo = run.sumo
        speed_limits = {lane: max(float(sumo.lane.getMaxSpeed(lane)), 1.0) for lane in lanes}
        cost = 0.0
        candidate = task["group"]["phases"][task["candidate"]]
        for elapsed in range(0, max(HORIZONS), CONTROL_INTERVAL_SEC):
            # Hold other first-step requests fixed across both continuations.
            if elapsed == 0:
                run.tick(CONTINUATIONS[0], override=(tls, candidate))
            else:
                run.tick(task["continuation"])
            for step in range(1, CONTROL_INTERVAL_SEC + 1):
                run.step()
                cost += sum(len(sumo.edge.getPendingVehicles(edge)) for edge in edges)
                for lane in lanes:
                    halted = float(sumo.lane.getLastStepHaltingNumber(lane))
                    moving = max(0.0, float(sumo.lane.getLastStepVehicleNumber(lane)) - halted)
                    cost += halted
                    if moving > 0:
                        speed = float(sumo.lane.getLastStepMeanSpeed(lane))
                        cost += 0.25 * moving * max(0.0, 1.0 - speed / speed_limits[lane])
                clock = elapsed + step
                if clock in HORIZONS:
                    horizon_cost[str(clock)] = cost / (clock * max(len(incoming), 1))
    return horizon_cost


def paired_spike_seeds(archived):
    ordinary = sorted(
        (r["mean_waiting_sec"], r["seed"]) for r in archived
        if r["scenario"] == "la_1x4" and r["policy"] == "conservation_gp_10" and r["seed"] not in CASE_SPIKES
    )
    return (*CASE_SPIKES, ordinary[(len(ordinary) - 1) // 2][1])


def compact_case_traces(cases, archived):
    """Keep the plotted network metrics and service requests for the highest-backlog entry."""
    by_id = {(r["scenario"], r["seed"], r["policy"]): r for r in cases}
    profiles = []
    for scenario, seed, policy in by_id:
        if policy != "cfcmt_b6":
            continue
        ref = "conservation_gp_10" if scenario == "la_1x4" else "zero_shot_cfcmt_b0"
        base = by_id[scenario, seed, ref]
        controls = base["samples"][:-1]
        edge = max(
            sorted(controls[0]["pending_by_edge"]),
            key=lambda candidate: sum(row["pending_by_edge"][candidate] for row in controls),
        )
        tls = next(tls for tls, edges in base["inflow_links"].items() if edge in edges)
        cfg = Path(base["sumocfg"])
        net = cfg.parent / ElementTree.parse(cfg).find("input/net-file").attrib["value"]
        connection_ids = sorted({
            int(connection.attrib["linkIndex"])
            for connection in ElementTree.parse(net).iter("connection")
            if connection.get("tl") == tls and connection.get("from") == edge
        })
        if not connection_ids:
            raise ValueError(f"No signal links for {edge} at {tls}")

        def serves(state):
            return int(any(state[index] in "gG" for index in connection_ids))

        def phase_events(rows, field):
            changes = []
            for row in rows:
                value = row[field][tls]
                if not changes or value != changes[-1][1]:
                    changes.append([row["time_sec"], value])
            return changes

        policies = {}
        for run in (base, by_id[scenario, seed, policy]):
            series = run["samples"]
            policies[run["policy"]] = {
                "metrics": run["metrics"],
                "network_pending": [row["pending"] for row in series],
                "network_halted": [row["halted"] for row in series],
                "completed": [row["completed"] for row in series],
                "entry_pending": [row["pending_by_edge"][edge] for row in series[:-1]],
                "entry_requested_green": [serves(row["requests"][tls]) for row in series[:-1]],
                "entry_sampled_green": [serves(row["signals"][tls]) for row in series[:-1]],
                "sampled_all_red": [int(set(row["signals"][tls]) <= {"r", "R"}) for row in series[:-1]],
                "selected_signal_request_changes": phase_events(series[:-1], "requests"),
                "selected_signal_state_changes": phase_events(series[:-1], "signals"),
            }
        profiles.append({"scenario": scenario, "seed": seed, "entry_edge": edge, "tls": tls, "policies": policies})

    archive_pair_summaries = {}
    for family, reference in (("la", "conservation_gp_10"), ("saltlake", "zero_shot_cfcmt_b0")):
        ref_rows = [r for r in archived if r["family"] == family and r["policy"] == reference]
        cand_rows = [r for r in archived if r["family"] == family and r["policy"] == "cfcmt_b6"]
        waits = {policy: {} for policy in (reference, "cfcmt_b6")}
        for row in ref_rows + cand_rows:
            waits[row["policy"]].setdefault(row["seed"], []).append(row["mean_waiting_sec"])
        differences = [
            np.mean(waits["cfcmt_b6"][seed]) - np.mean(waits[reference][seed])
            for seed in sorted(waits[reference])
        ]
        archive_pair_summaries[family] = {
            "reference": reference,
            "candidate": "cfcmt_b6",
            "mean_wait_difference_sec": float(np.mean(differences)),
            "median_wait_difference_sec": float(np.median(differences)),
            "candidate_lower_wait_city_seed_count": int(sum(delta < 0 for delta in differences)),
            "reference_runs_completion_below_half": sum(r["completion_ratio"] < 0.5 for r in ref_rows),
            "candidate_runs_completion_below_half": sum(r["completion_ratio"] < 0.5 for r in cand_rows),
        }
    return {
        "time_sec": [row["time_sec"] for row in cases[0]["samples"]],
        "all_archived_case_metrics_verified": True,
        "profiles": profiles,
        "archived_pair_summaries": archive_pair_summaries,
    }


def reduce_diagnostics(scans, branches):
    states = [state for scan in scans for state in scan["states"]]
    tables = {f"{continuation}:{horizon}": [] for continuation in CONTINUATIONS for horizon in HORIZONS}
    cursor = 0
    out = []
    for state in states:
        group = state["group"]
        count = len(group["phases"])
        costs = {}
        for continuation in CONTINUATIONS:
            group_branches = branches[cursor:cursor + count]
            cursor += count
            for horizon in HORIZONS:
                costs[f"{continuation}:{horizon}"] = [item[str(horizon)] for item in group_branches]
        base = costs[f"{CONTINUATIONS[0]}:{HORIZONS[0]}"]
        for variant, values in costs.items():
            minimum = min(values)
            action = group["chosen"]
            best = {i for i, cost in enumerate(values) if math.isclose(cost, minimum, abs_tol=1e-12)}
            base_best = {i for i, cost in enumerate(base) if math.isclose(cost, min(base), abs_tol=1e-12)}
            tables[variant].append({
                "reference_top_agreement": bool(best & base_best),
                "deployed_top_accuracy": action in best,
                "deployed_normalized_regret": (values[action] - minimum) / max(max(values) - minimum, 0.15),
                "unfiltered_normalized_regret": (values[group["preferred"]] - minimum) / max(max(values) - minimum, 0.15),
            })
        out.append({
            "scenario": state["scenario"], "seed": state["seed"], "time_sec": state["time_sec"],
            "tls": state["tls"], "phases": group["phases"], "preferred": group["preferred"],
            "chosen": group["chosen"], "allowed": group["allowed"], "costs": costs,
        })
    assert cursor == len(branches)
    return {
        "groups": len(states), "branch_starts": len(branches),
        "branch_simulated_hours": sum(
            len(state["group"]["phases"]) * len(CONTINUATIONS) * (state["time_sec"] + max(HORIZONS))
            for state in states
        ) / 3600,
        "scan_simulated_hours": len(scans) * (max(VISITED_TIMES) + SEARCH_WINDOW_SEC) / 3600,
        "state_restores": 0,
        "multi_action_decisions": sum(s["multi_action_decisions"] for s in scans),
        "max_red_changed_preferred": sum(s["max_red_changed_preferred"] for s in scans),
        "per_protocol": {variant: {k: float(np.mean([row[k] for row in rows])) for k in rows[0]} for variant, rows in tables.items()},
        "states": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.experiment_root
    la_jinan = root / "phase2_frozen_prereg"
    slc = root / "phase3_tits"
    staged = root / "phase0/staged_scenarios"
    slc_name = "saltlake_400s_200w_q1_weekday_peak"
    inputs = {
        "la": ("la_1x4", staged / "la_1x4/la_1x4.sumocfg", la_jinan / "phase2_frozen_models.pkl"),
        "jinan": ("jinan_3x4_real", staged / "jinan_3x4_real/jinan_3x4_real.sumocfg", la_jinan / "phase2_frozen_models.pkl"),
        "saltlake": ("saltlake_400s_peak", root / f"ext_targets/{slc_name}/{slc_name}.sumocfg", slc / "phase3_saltlake_models.pkl"),
    }
    archived = (
        json.loads((la_jinan / "phase2_eval_fresh_result.json").read_text())["rows"]
        + json.loads((slc / "phase3_saltlake_eval_fresh_result.json").read_text())["rows"]
    )
    stable = {(row["scenario"], row["policy"], row["seed"]): row for row in archived}

    def make_task(family, seed, policy):
        scenario, cfg, model = inputs[family]
        return {"family": family, "scenario": scenario, "seed": seed, "policy": policy, "sumocfg": str(cfg), "models_pkl": str(model)}

    la_seeds = paired_spike_seeds(archived)
    case_tasks = [make_task("la", seed, policy) for seed in la_seeds for policy in CONTINUATIONS]
    case_tasks.extend(make_task("saltlake", SALT_LAKE_SPIKE, policy) for policy in ("zero_shot_cfcmt_b0", "cfcmt_b6"))
    scan_tasks = [make_task(family, seed, "cfcmt_b6") for family in inputs for seed in VISITED_SEEDS]

    args.output.mkdir(parents=True, exist_ok=True)
    case_path = args.output / "case_traces.json"
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if case_path.exists():
            cases = json.loads(case_path.read_text())["cases"]
            identifiers = ("scenario", "seed", "policy")
            case_keys = {tuple(row[key] for key in identifiers) for row in cases}
            task_keys = {tuple(row[key] for key in identifiers) for row in case_tasks}
            if case_keys != task_keys:
                raise ValueError("Cached case traces do not match the selected cases")
        else:
            cases = list(pool.map(replay_case, case_tasks))
        for case in cases:
            previous = stable[case["scenario"], case["policy"], case["seed"]]
            for metric in METRICS:
                if not math.isclose(case["metrics"][metric], previous[metric], rel_tol=0, abs_tol=1e-9):
                    raise ValueError(f"Terminal mismatch {case['scenario']}/{case['seed']}/{case['policy']}: {metric} is {case['metrics'][metric]} instead of {previous[metric]}")
        if not case_path.exists():
            case_path.write_text(json.dumps({"la_seeds": la_seeds, "cases": cases}), encoding="utf-8")
        (args.output / "case_trace_summary.json").write_text(
            json.dumps(compact_case_traces(cases, archived)) + "\n", encoding="utf-8",
        )
        print(f"Verified {len(cases)} archived case replays. Scanning {len(scan_tasks)} policy prefixes.", flush=True)
        scans = list(pool.map(scan_policy_states, scan_tasks))
        branch_tasks = [
            {**state, "candidate": i, "continuation": continuation}
            for scan in scans for state in scan["states"]
            for continuation in CONTINUATIONS for i in range(len(state["group"]["phases"]))
        ]
        print(f"Running {len(branch_tasks)} native-prefix continuation branches.", flush=True)
        branches = []
        for i, row in enumerate(pool.map(branch_at_state, branch_tasks), 1):
            branches.append(row)
            if i % 24 == 0 or i == len(branch_tasks):
                print(f"Completed {i}/{len(branch_tasks)} branches.", flush=True)
    summary = reduce_diagnostics(scans, branches)
    (args.output / "case_diagnostic_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("Saved case-diagnostic results.", flush=True)


if __name__ == "__main__":
    main()
