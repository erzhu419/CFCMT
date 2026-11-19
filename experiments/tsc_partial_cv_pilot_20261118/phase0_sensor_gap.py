"""Phase 0 sensor-gap benchmark on repaired Jinan, Los Angeles, and repaired Hangzhou.

Measures full 3,600-second closed-loop performance under identical SUMO physics and
identical SafePhaseExecutor timing across:
- oracle_gp: 100% full-state Exact Generalized Pressure ceiling
- oracle_pp: 100% full-state PhasePressure ceiling
- fixed_program: 0% real-time sensor static signal plan
- observed_mp_10: 10% CV MaxPressure
- observed_pp_10: 10% CV PhasePressure
- observed_gp_10: 10% CV Exact GP
- shockwave_gp_10: 10% CV Shockwave-reconstructed GP (Comert-Cetin position tail)
- conservation_gp_10: 10% CV Hybrid Shockwave + Arrival-Service Conservation GP
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import (
    PartialCvObserver,
    select_observed_pressure_phase,
)


POLICIES = (
    "oracle_gp",
    "oracle_pp",
    "fixed_program",
    "observed_mp_10",
    "observed_pp_10",
    "observed_gp_10",
    "shockwave_gp_10",
    "conservation_gp_10",
)


def stage_repaired_scenarios(stage_dir: Path, v93_root: Path, repaired_root: Path) -> dict[str, Path]:
    """Assemble clean scenario directories using repaired zero-conflict Jinan/Hangzhou + LA."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    repaired_jinan_net = repaired_root / "inputs/jinan/jinan_3x4_real.net.xml"
    repaired_hz_cfg = repaired_root / "inputs/hangzhou/hangzhou_aligned.sumocfg"
    repaired_hz_net = repaired_root / "inputs/hangzhou/hangzhou_aligned.net.xml"

    if not repaired_jinan_net.is_file():
        raise FileNotFoundError(f"Missing repaired Jinan net: {repaired_jinan_net}")
    if not repaired_hz_net.is_file():
        raise FileNotFoundError(f"Missing repaired Hangzhou net: {repaired_hz_net}")

    scenarios: dict[str, Path] = {}

    # 1. Los Angeles (la_1x4)
    la_src = v93_root / "conversion/la_1x4"
    la_dst = stage_dir / "la_1x4"
    if not la_dst.exists():
        shutil.copytree(la_src, la_dst)
    scenarios["la_1x4"] = la_dst / "la_1x4.sumocfg"

    # 2. Three Jinan scenarios bound to the repaired zero-conflict jinan_3x4_real.net.xml
    for scen in ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"):
        src_dir = v93_root / "conversion" / scen
        dst_dir = stage_dir / scen
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repaired_jinan_net, dst_dir / f"{scen}.net.xml")
        shutil.copy2(src_dir / f"{scen}.rou.xml", dst_dir / f"{scen}.rou.xml")
        shutil.copy2(src_dir / f"{scen}.sumocfg", dst_dir / f"{scen}.sumocfg")
        scenarios[scen] = dst_dir / f"{scen}.sumocfg"

    # 3. Repaired Hangzhou 4x4
    hz_dst = stage_dir / "hangzhou_4x4_repaired"
    if not hz_dst.exists():
        shutil.copytree(repaired_hz_cfg.parent, hz_dst)
    scenarios["hangzhou_4x4_repaired"] = hz_dst / "hangzhou_aligned.sumocfg"

    return scenarios


def _oracle_gp_phase(sumo_api: Any, info: Any, executor: Any, mode: str) -> str:
    """Exact 100%-observability GP / PP matching manuscript Section IV-A."""
    feasible = executor.feasible_states_now()
    if len(feasible) == 1:
        return str(feasible[0])
    green_chars = frozenset("Gg")
    best_state = str(feasible[0])
    best_score = -float("inf")
    for state in feasible:
        green_lanes: list[str] = []
        lane_outs: dict[str, list[str]] = {}
        for link_idx, char in enumerate(state):
            if char not in green_chars or link_idx >= len(info.controlled_links):
                continue
            link_group = info.controlled_links[link_idx]
            if not link_group or not link_group[0]:
                continue
            in_lane = str(link_group[0][0])
            out_lane = str(link_group[0][1])
            if in_lane not in green_lanes:
                green_lanes.append(in_lane)
            if out_lane:
                lane_outs.setdefault(in_lane, [])
                if out_lane not in lane_outs[in_lane]:
                    lane_outs[in_lane].append(out_lane)
        if not green_lanes:
            score = 0.0
        else:
            q_green = 0.0
            down_means: list[float] = []
            for lane in green_lanes:
                q_green += (
                    float(sumo_api.lane.getLastStepHaltingNumber(lane))
                    + 0.35 * float(sumo_api.lane.getLastStepVehicleNumber(lane))
                    + 0.025 * float(sumo_api.lane.getLastStepOccupancy(lane))
                )
                outs = lane_outs.get(lane, [])
                if outs:
                    down_means.append(
                        float(
                            np.mean(
                                [
                                    float(sumo_api.lane.getLastStepOccupancy(o))
                                    for o in outs
                                ]
                            )
                        )
                    )
            if mode == "oracle_pp":
                score = q_green
            else:
                o_down = float(np.mean(down_means)) if down_means else 0.0
                score = q_green - 0.02 * o_down
        if score > best_score + 1e-12:
            best_score = score
            best_state = str(state)
    return best_state


def run_single_rollout(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one full 3,600-second simulation cell in an isolated process."""
    from cf_h2o.sumo_runtime import load_libsumo
    from cf_h2o.eval.traffic_signal_resco_cfcmt_benchmark import _parse_tripinfo_metrics
    from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
        _controlled_lane_set,
        _start_sumo,
        _tls_phase_infos,
    )
    from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import _collision_incident_key
    from cf_h2o.traffic_signal.safe_phase_controller import (
        aggregate_phase_audits,
        build_safe_phase_executors,
    )

    scenario = str(task["scenario"])
    sumocfg = Path(task["sumocfg"])
    seed = int(task["seed"])
    policy = str(task["policy"])
    duration_sec = float(task.get("duration_sec", 3600.0))
    warmup_sec = float(task.get("warmup_sec", 60.0))
    control_interval_sec = int(task.get("control_interval_sec", 10))

    sumo_api = load_libsumo()
    with tempfile.TemporaryDirectory(prefix="cfcmt_cv_p0_") as tmp_dir:
        tripinfo_path = Path(tmp_dir) / "tripinfo.xml"
        _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=tripinfo_path)
        departed = 0
        arrived = 0
        active_counts: list[float] = []
        pending_counts: list[float] = []
        halted_counts: list[float] = []
        starting_teleports = 0
        ending_teleports = 0
        collision_events = 0
        collision_incident_keys: set[tuple[str, str, str, str]] = set()
        executors = {}
        observer: PartialCvObserver | None = None

        try:
            infos = _tls_phase_infos(sumo_api)
            lanes = _controlled_lane_set(infos)
            if policy != "fixed_program":
                executors = build_safe_phase_executors(sumo_api, infos)
            if policy.endswith("_10"):
                observer = PartialCvObserver(
                    sumo_api=sumo_api, infos=infos, seed=seed, p_cv=0.10
                )
                observer.observe_step(sumo_api, dt_sec=1.0)

            begin_time = float(sumo_api.simulation.getTime())
            end_time = begin_time + duration_sec

            while float(sumo_api.simulation.getTime()) < end_time:
                now_t = float(sumo_api.simulation.getTime())
                if policy != "fixed_program":
                    for tls_id in sorted(infos):
                        info = infos[tls_id]
                        ex = executors[tls_id]
                        if policy in {"oracle_gp", "oracle_pp"}:
                            target_phase = _oracle_gp_phase(sumo_api, info, ex, policy)
                        elif observer is not None:
                            base_mode = policy.rsplit("_", 1)[0]
                            target_phase = select_observed_pressure_phase(
                                info=info,
                                executor=ex,
                                observer=observer,
                                mode=base_mode,
                            )
                        else:
                            raise ValueError(f"Unhandled policy: {policy}")
                        ex.request(target_phase)

                steps = min(
                    control_interval_sec,
                    max(int(math.ceil(end_time - now_t)), 0),
                )
                for _ in range(steps):
                    t_before = float(sumo_api.simulation.getTime())
                    sumo_api.simulation.step()
                    t_after = float(sumo_api.simulation.getTime())
                    dt = max(t_after - t_before, 0.0)
                    for ex in executors.values():
                        ex.advance(dt)
                    if observer is not None:
                        observer.observe_step(sumo_api, dt_sec=max(dt, 1.0))

                    departed += int(sumo_api.simulation.getDepartedNumber())
                    arrived += int(sumo_api.simulation.getArrivedNumber())
                    st_tp = int(sumo_api.simulation.getStartingTeleportNumber())
                    en_tp = int(sumo_api.simulation.getEndingTeleportNumber())
                    starting_teleports += st_tp
                    ending_teleports += en_tp
                    cols = tuple(sumo_api.simulation.getCollisions())
                    collision_events += len(cols)
                    collision_incident_keys.update(
                        _collision_incident_key(c) for c in cols
                    )
                    if t_after >= begin_time + warmup_sec:
                        halted_counts.append(
                            float(
                                sum(
                                    sumo_api.lane.getLastStepHaltingNumber(l)
                                    for l in lanes
                                )
                            )
                        )
                        active_counts.append(float(sumo_api.vehicle.getIDCount()))
                        pending_counts.append(
                            float(len(sumo_api.simulation.getPendingVehicles()))
                        )

            pending_end = int(len(sumo_api.simulation.getPendingVehicles()))
            demand_pop = departed + pending_end
            audit = aggregate_phase_audits(executors) if executors else {}
        finally:
            sumo_api.close()

        trip_metrics = _parse_tripinfo_metrics(tripinfo_path)

    return {
        "scenario": scenario,
        "seed": seed,
        "policy": policy,
        "mean_waiting_sec": float(trip_metrics["mean_tripinfo_waiting_time"]),
        "mean_time_loss_sec": float(trip_metrics["mean_tripinfo_time_loss"]),
        "mean_duration_sec": float(trip_metrics["mean_tripinfo_duration"]),
        "system_vehicle_hours": float(
            (np.sum(active_counts) + np.sum(pending_counts)) / 3600.0
        ),
        "mean_queue_per_lane": float(np.mean(halted_counts) / max(len(lanes), 1)),
        "completion_ratio": float(arrived / max(demand_pop, 1)),
        "departed": int(departed),
        "arrived": int(arrived),
        "pending_end": int(pending_end),
        "collision_events": int(collision_events),
        "collision_incidents": int(len(collision_incident_keys)),
        "teleports": int(starting_teleports + ending_teleports),
        "switches": int(audit.get("switches", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v93-root", type=Path, required=True)
    parser.add_argument("--repaired-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[5057, 6067, 7001, 7002]
    )
    args = parser.parse_args()

    scenarios = stage_repaired_scenarios(
        args.work_dir / "staged_scenarios",
        args.v93_root,
        args.repaired_root,
    )
    tasks = [
        {
            "scenario": scen,
            "sumocfg": str(cfg),
            "seed": seed,
            "policy": policy,
        }
        for scen, cfg in scenarios.items()
        for seed in args.seeds
        for policy in POLICIES
    ]

    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_single_rollout, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            rows.append(row)
            if idx % 20 == 0 or idx == len(tasks):
                print(
                    f"[{idx}/{len(tasks)}] {row['scenario']} seed={row['seed']} "
                    f"{row['policy']} wait={row['mean_waiting_sec']:.2f}s "
                    f"inc={row['collision_incidents']}",
                    flush=True,
                )

    rows.sort(key=lambda r: (r["scenario"], r["policy"], r["seed"]))
    summary: dict[str, dict[str, Any]] = {}
    for scen in scenarios:
        summary[scen] = {}
        for pol in POLICIES:
            sub = [r for r in rows if r["scenario"] == scen and r["policy"] == pol]
            summary[scen][pol] = {
                "mean_waiting_sec": float(np.mean([r["mean_waiting_sec"] for r in sub])),
                "system_vehicle_hours": float(
                    np.mean([r["system_vehicle_hours"] for r in sub])
                ),
                "completion_ratio": float(np.mean([r["completion_ratio"] for r in sub])),
                "mean_switches": float(np.mean([r["switches"] for r in sub])),
                "total_collision_incidents": int(
                    sum(r["collision_incidents"] for r in sub)
                ),
                "total_teleports": int(sum(r["teleports"] for r in sub)),
            }

    out_path = args.work_dir / "phase0_sensor_gap_result.json"
    out_path.write_text(
        json.dumps({"seeds": args.seeds, "summary": summary, "rows": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print("SAVED", out_path, flush=True)


if __name__ == "__main__":
    main()
