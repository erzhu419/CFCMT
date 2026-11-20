"""Phase 3 IEEE T-ITS Expansion Suite:
1. CV Penetration Sensitivity Curve (`p_cv in {0.05, 0.10, 0.20, 0.50, 1.00}`) with ZERO retraining
   of the `p_cv = 0.10` frozen CFCMT models on virgin seeds `920101..920110`.
2. Third External Real-World Target City (`Salt Lake City, UT`: `saltlake_400s_peak`,
   `saltlake_state_peak`, zero overlap with source bank, `state_restores = 0`) evaluated on
   brand-new virgin seeds `930101..930110`.
3. Sensor-Matched FRAP/MPLight Double-DQN Baselines (`mplight_cv_ep2`, `mplight_cv_ep16`,
   `transfer_mplight_cv_ep2`) under identical `p_cv = 0.10` `PartialCvObserver` and NEMA `Max-Red`
   shield on virgin seeds `920101..920110` (LA & Jinan) and `930101..930110` (Salt Lake City).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import (
    build_1hop_tls_neighbors,
    extract_decision_group_features,
)
from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import (
    PartialCvObserver,
    select_observed_pressure_phase,
)
from experiments.tsc_partial_cv_pilot_20261118.mplight_cv_baseline import (
    MPLightCvQNetwork,
    ReplayTransition,
    build_phase_competition_inputs,
)
from experiments.tsc_partial_cv_pilot_20261118.phase0_sensor_gap import _oracle_gp_phase
from experiments.tsc_partial_cv_pilot_20261118.phase1_transfer_pilot import (
    evaluate_single_branch_worker,
    scan_decision_candidates_worker,
)
from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_model import (
    ResidualGroupRecord,
    compute_exact_cons_gp_raw_pressures,
    feasible_indices_under_max_red,
    pressures_to_base_cost_scores,
)
from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_transfer import (
    ADAPT_SEEDS,
    CONTROL_INTERVAL_SEC,
    FRESH_PHASE2_SEEDS,
    LEARNED_POLICIES,
    SOURCE_SEEDS,
    eval_phase2_closed_loop_worker,
    fit_phase2_residual_suite,
    read_paper_label_archive,
    resolve_phase2_scenarios,
    select_maxred_pressure_phase,
    summarize_phase2_rows,
)

PENETRATION_RATES = (0.05, 0.10, 0.20, 0.50, 1.00)
FRESH_SALTLAKE_SEEDS = tuple(range(930101, 930111))
RL_POLICIES = ("mplight_cv_ep2", "mplight_cv_ep16", "transfer_mplight_cv_ep2")
SOURCE_ONLY_RL_POLICY = "transfer_mplight_cv_b0"


def eval_custom_policy_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Closed-loop worker supporting arbitrary `p_cv` and CFCMT / MPLight-CV ranker pickles."""
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
    family = str(task["family"])
    sumocfg = Path(task["sumocfg"])
    seed = int(task["seed"])
    policy = str(task["policy"])
    p_cv = float(task.get("p_cv", 0.10))
    models_pkl = Path(task["models_pkl"])

    learned_ctrl: Any = None
    if policy in LEARNED_POLICIES or policy in RL_POLICIES or policy == SOURCE_ONLY_RL_POLICY:
        with models_pkl.open("rb") as fh:
            all_models = pickle.load(fh)
        learned_ctrl = all_models[family][policy]

    sumo_api = load_libsumo()
    with tempfile.TemporaryDirectory(prefix="cfcmt_cv_p3_") as tmp_dir:
        tripinfo_path = Path(tmp_dir) / "tripinfo.xml"
        _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=tripinfo_path)

        departed = 0
        arrived = 0
        active_counts: list[float] = []
        pending_counts: list[float] = []
        halted_counts: list[float] = []
        starting_teleports = 0
        collision_incident_keys: set[tuple[str, str, str, str]] = set()
        executors = {}
        observer: PartialCvObserver | None = None
        disagree_vs_cons_gp = 0
        maxred_triggers = 0
        multi_action_decisions = 0

        try:
            infos = _tls_phase_infos(sumo_api)
            lanes = _controlled_lane_set(infos)
            up_nb, down_nb, _ = build_1hop_tls_neighbors(infos)

            if policy != "fixed_program":
                executors = build_safe_phase_executors(sumo_api, infos)
            if policy not in {"oracle_gp", "fixed_program"}:
                observer = PartialCvObserver(
                    sumo_api=sumo_api, infos=infos, seed=seed, p_cv=p_cv
                )
                observer.observe_step(sumo_api, dt_sec=1.0)

            begin_time = float(sumo_api.simulation.getTime())
            end_time = begin_time + 3600.0

            while float(sumo_api.simulation.getTime()) < end_time:
                now_t = float(sumo_api.simulation.getTime())
                if policy != "fixed_program":
                    for tls_id in sorted(infos):
                        info = infos[tls_id]
                        ex = executors[tls_id]
                        if policy == "oracle_gp":
                            target_phase = _oracle_gp_phase(
                                sumo_api, info, ex, "oracle_gp"
                            )
                        elif policy.startswith("maxred_conservation_gp") and observer is not None:
                            feas = tuple(str(s) for s in ex.feasible_states_now())
                            if len(feas) > 1:
                                multi_action_decisions += 1
                                allowed = feasible_indices_under_max_red(
                                    info=info,
                                    feasible=feas,
                                    observer=observer,
                                    max_red_sec=None,
                                    current_green_state=str(ex.current_green_state),
                                )
                                if len(allowed) < len(feas):
                                    maxred_triggers += 1
                            target_phase = select_maxred_pressure_phase(
                                info=info,
                                executor=ex,
                                observer=observer,
                                mode="conservation_gp",
                            )
                        elif learned_ctrl is not None and observer is not None:
                            feasible, s_vec, a_mat = extract_decision_group_features(
                                info=info,
                                executor=ex,
                                observer=observer,
                                infos_all=infos,
                                up_neighbors=up_nb[tls_id],
                                down_neighbors=down_nb[tls_id],
                            )
                            if len(feasible) == 1:
                                target_phase = feasible[0]
                            else:
                                multi_action_decisions += 1
                                allowed = feasible_indices_under_max_red(
                                    info=info,
                                    feasible=feasible,
                                    observer=observer,
                                    max_red_sec=None,
                                    current_green_state=str(ex.current_green_state),
                                )
                                if len(allowed) < len(feasible):
                                    maxred_triggers += 1
                                scores = learned_ctrl.score_group(s_vec, a_mat)
                                chosen_idx = min(
                                    allowed, key=lambda idx: float(scores[idx])
                                )
                                target_phase = feasible[chosen_idx]
                                raw_p = compute_exact_cons_gp_raw_pressures(
                                    info=info, feasible=feasible, observer=observer
                                )
                                b_scores = pressures_to_base_cost_scores(raw_p)
                                ref_idx = min(
                                    allowed, key=lambda idx: float(b_scores[idx])
                                )
                                if chosen_idx != ref_idx:
                                    disagree_vs_cons_gp += 1
                        elif observer is not None:
                            base_mode = (
                                policy.rsplit("_", 1)[0]
                                if policy.endswith("_10")
                                else policy
                            )
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
                    CONTROL_INTERVAL_SEC,
                    max(int(math.ceil(end_time - now_t)), 0),
                )
                for _ in range(steps):
                    t0 = float(sumo_api.simulation.getTime())
                    sumo_api.simulation.step()
                    t1 = float(sumo_api.simulation.getTime())
                    dt = max(t1 - t0, 0.0)
                    for ex in executors.values():
                        ex.advance(dt)
                    if observer is not None:
                        observer.observe_step(sumo_api, dt_sec=max(dt, 1.0))

                    departed += int(sumo_api.simulation.getDepartedNumber())
                    arrived += int(sumo_api.simulation.getArrivedNumber())
                    starting_teleports += int(
                        sumo_api.simulation.getStartingTeleportNumber()
                    )
                    cols = tuple(sumo_api.simulation.getCollisions())
                    collision_incident_keys.update(
                        _collision_incident_key(c) for c in cols
                    )
                    if t1 >= begin_time + 60.0:
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
            try:
                sumo_api.close()
            except Exception:
                pass

        trip = _parse_tripinfo_metrics(tripinfo_path)

    sys_veh_hours = (
        float(sum(a + p for a, p in zip(active_counts, pending_counts))) / 3600.0
    )
    return {
        "family": family,
        "scenario": scenario,
        "seed": seed,
        "p_cv": p_cv,
        "policy": policy,
        "mean_waiting_sec": float(trip["mean_tripinfo_waiting_time"]),
        "mean_time_loss_sec": float(trip["mean_tripinfo_time_loss"]),
        "mean_duration_sec": float(trip["mean_tripinfo_duration"]),
        "system_vehicle_hours": sys_veh_hours,
        "mean_queue_per_lane": float(np.mean(halted_counts)) / max(len(lanes), 1),
        "completion_ratio": float(arrived / max(demand_pop, 1)),
        "switch_count": int(audit.get("switches", 0)),
        "disagree_rate_vs_cons_gp": float(
            disagree_vs_cons_gp / max(multi_action_decisions, 1)
        ),
        "maxred_trigger_rate": float(maxred_triggers / max(multi_action_decisions, 1)),
        "collision_incidents": int(len(collision_incident_keys)),
        "starting_teleports": int(starting_teleports),
    }


def collect_rollout_transitions_worker(task: dict[str, Any]) -> list[ReplayTransition]:
    """Collect ten-second transitions from one fixed-hour MPLight-CV episode."""
    from cf_h2o.sumo_runtime import load_libsumo
    from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
        _start_sumo,
        _tls_phase_infos,
    )
    from cf_h2o.traffic_signal.safe_phase_controller import build_safe_phase_executors

    sumocfg = Path(task["sumocfg"])
    seed = int(task["seed"])
    epsilon = float(task["epsilon"])
    rng_seed = int(task["rng_seed"])
    net: MPLightCvQNetwork = task["net"]
    rng = np.random.default_rng(rng_seed)

    sumo_api = load_libsumo()
    _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=None)
    transitions: list[ReplayTransition] = []
    try:
        infos = _tls_phase_infos(sumo_api)
        executors = build_safe_phase_executors(sumo_api, infos)
        up_nb, down_nb, _ = build_1hop_tls_neighbors(infos)
        observer = PartialCvObserver(
            sumo_api=sumo_api, infos=infos, seed=seed, p_cv=0.10
        )
        observer.observe_step(sumo_api, dt_sec=1.0)

        begin_time = float(sumo_api.simulation.getTime())
        end_time = begin_time + 3600.0
        prev_decision: dict[str, tuple[np.ndarray, int]] = {}

        while True:
            now_t = float(sumo_api.simulation.getTime())
            done = now_t >= end_time
            obs_map = observer.observations

            for tls_id in sorted(infos):
                info = infos[tls_id]
                ex = executors[tls_id]
                if done:
                    feasible = ()
                    z_cur = np.empty((0, net.in_dim))
                    allowed = ()
                else:
                    feasible, s_vec, a_mat = extract_decision_group_features(
                        info=info,
                        executor=ex,
                        observer=observer,
                        infos_all=infos,
                        up_neighbors=up_nb[tls_id],
                        down_neighbors=down_nb[tls_id],
                    )
                    z_cur = build_phase_competition_inputs(s_vec, a_mat)
                    allowed = feasible_indices_under_max_red(
                        info=info,
                        feasible=feasible,
                        observer=observer,
                        max_red_sec=None,
                        current_green_state=str(ex.current_green_state),
                    )

                # Observe partial-CV queue reward for previous action at this TLS
                if tls_id in prev_decision and now_t >= begin_time + 60.0:
                    in_lanes = info.incoming_lanes
                    est_q = sum(
                        max(obs_map[l].shockwave_queue_est, obs_map[l].conservation_queue_est)
                        for l in in_lanes
                    ) / max(len(in_lanes), 1)
                    rew = -float(np.clip(est_q / 5.0, 0.0, 10.0))
                    z_prev, act_prev = prev_decision[tls_id]
                    transitions.append(
                        ReplayTransition(
                            z_cur=z_prev,
                            action_idx=act_prev,
                            reward=rew,
                            z_next=z_cur,
                            next_allowed=allowed,
                            done=done,
                        )
                    )

                if done:
                    continue
                if len(feasible) == 1:
                    chosen_idx = 0
                elif rng.random() < epsilon:
                    chosen_idx = int(rng.choice(allowed))
                else:
                    q_vals = net.predict_q(z_cur, use_target=False)
                    chosen_idx = max(allowed, key=lambda idx: float(q_vals[idx]))
                # A singleton is the executor's continuation action for this interval.
                prev_decision[tls_id] = (z_cur, chosen_idx)
                ex.request(feasible[chosen_idx])

            if done:
                break
            steps = min(
                CONTROL_INTERVAL_SEC, max(int(math.ceil(end_time - now_t)), 0)
            )
            for _ in range(steps):
                t0 = float(sumo_api.simulation.getTime())
                sumo_api.simulation.step()
                t1 = float(sumo_api.simulation.getTime())
                dt = max(t1 - t0, 0.0)
                for ex in executors.values():
                    ex.advance(dt)
                observer.observe_step(sumo_api, dt_sec=max(dt, 1.0))
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass
    return transitions


def train_mplight_agent(
    *,
    sumocfgs: Sequence[Path],
    seeds: Sequence[int],
    n_episodes: int,
    init_net: MPLightCvQNetwork | None,
    rng_seed: int,
    workers: int,
) -> tuple[MPLightCvQNetwork, dict[str, Any]]:
    """Train MPLightCvQNetwork across episodes using experience replay + Double-DQN updates."""
    rng = np.random.default_rng(rng_seed)
    net = init_net.clone() if init_net is not None else MPLightCvQNetwork(seed=rng_seed)
    replay: list[ReplayTransition] = []
    max_replay = 25000
    losses: list[float] = []

    schedule = [
        (sumocfgs[ep % len(sumocfgs)], seeds[(ep // len(sumocfgs)) % len(seeds)], ep)
        for ep in range(n_episodes)
    ]

    # Run episodes sequentially or in small batches of 2 so Q-policy improves online
    idx = 0
    while idx < len(schedule):
        batch_specs = schedule[idx : idx + min(2, len(schedule) - idx)]
        tasks = []
        for cfg, sd, ep_i in batch_specs:
            frac = ep_i / max(n_episodes - 1, 1)
            eps = float(0.40 - 0.35 * frac) if init_net is None else float(0.15 - 0.10 * frac)
            tasks.append(
                {
                    "sumocfg": str(cfg),
                    "seed": int(sd),
                    "epsilon": eps,
                    "rng_seed": int(rng_seed + 1009 * (ep_i + 1)),
                    "net": net.clone(),
                }
            )
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            for tr_list in pool.map(collect_rollout_transitions_worker, tasks):
                replay.extend(tr_list)
        if len(replay) > max_replay:
            replay = replay[-max_replay:]

        n_updates = 60 * len(batch_specs)
        mb_size = min(64, len(replay))
        if mb_size >= 16:
            for _ in range(n_updates):
                sel = rng.integers(0, len(replay), size=mb_size)
                losses.append(net.train_minibatch([replay[int(j)] for j in sel]))
        idx += len(batch_specs)

    return net, {
        "episodes": n_episodes,
        "replay_size": len(replay),
        "final_td_loss": float(np.mean(losses[-30:])) if losses else 0.0,
    }


def run_penetration_curve(
    *,
    work_dir: Path,
    scenarios: Mapping[str, Path],
    phase2_models_pkl: Path,
    phase2_fresh_json: Path,
    workers: int,
) -> None:
    p2_data = json.loads(phase2_fresh_json.read_text(encoding="utf-8"))
    existing_rows = p2_data["rows"]
    keep_pols = (
        "observed_mp_10",
        "shockwave_gp_10",
        "conservation_gp_10",
        "maxred_conservation_gp_10",
        "target_only_b6",
        "zero_shot_cfcmt_b0",
        "cfcmt_b6",
        "cfcmt_b36",
    )
    target_scens = (("la", "la_1x4"), ("jinan", "jinan_3x4_real"))

    all_rows: list[dict[str, Any]] = []
    for r in existing_rows:
        if r["scenario"] in {"la_1x4", "jinan_3x4_real"} and r["policy"] in keep_pols:
            rc = dict(r)
            rc["p_cv"] = 0.10
            all_rows.append(rc)

    tasks: list[dict[str, Any]] = []
    for p_cv in PENETRATION_RATES:
        if abs(p_cv - 0.10) <= 1e-9:
            continue
        for fam, scen in target_scens:
            for seed in FRESH_PHASE2_SEEDS:
                for pol in keep_pols:
                    tasks.append(
                        {
                            "family": fam,
                            "scenario": scen,
                            "sumocfg": str(scenarios[scen]),
                            "seed": seed,
                            "p_cv": p_cv,
                            "policy": pol,
                            "models_pkl": str(phase2_models_pkl),
                        }
                    )

    print(
        f"[Phase 3A Penetration] Running {len(tasks)} cells across p_cv in {PENETRATION_RATES}...",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_custom_policy_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            all_rows.append(r)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(
                    f"  [3A {done}/{len(tasks)}] p_cv={r['p_cv']:.2f} {r['scenario']} {r['policy']} wait={r['mean_waiting_sec']:.2f}s",
                    flush=True,
                )

    all_rows.sort(key=lambda d: (d["scenario"], d["p_cv"], d["seed"], d["policy"]))
    curve_summary: dict[str, Any] = {}
    for _, scen in target_scens:
        curve_summary[scen] = {}
        for p_cv in PENETRATION_RATES:
            key_p = f"{p_cv:.2f}"
            curve_summary[scen][key_p] = {}
            for pol in keep_pols:
                sub = [
                    x
                    for x in all_rows
                    if x["scenario"] == scen
                    and abs(float(x["p_cv"]) - p_cv) <= 1e-9
                    and x["policy"] == pol
                ]
                curve_summary[scen][key_p][pol] = {
                    "mean_waiting_sec": float(np.mean([x["mean_waiting_sec"] for x in sub])),
                    "median_waiting_sec": float(np.median([x["mean_waiting_sec"] for x in sub])),
                    "system_vehicle_hours": float(
                        np.mean([x["system_vehicle_hours"] for x in sub])
                    ),
                    "completion_ratio": float(np.mean([x["completion_ratio"] for x in sub])),
                    "total_collision_incidents": int(
                        sum(x["collision_incidents"] for x in sub)
                    ),
                }

    out_path = work_dir / "phase3_penetration_curve_result.json"
    out_path.write_text(
        json.dumps(
            {
                "penetration_rates": list(PENETRATION_RATES),
                "seeds": list(FRESH_PHASE2_SEEDS),
                "curve_summary": curve_summary,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SAVED_PHASE3_PENETRATION {out_path}", flush=True)


def run_third_city_saltlake(
    *,
    work_dir: Path,
    ext_targets_dir: Path,
    phase2_dataset_json: Path,
    workers: int,
) -> Path:
    sl_scens = {
        "saltlake_400s_peak": ext_targets_dir
        / "saltlake_400s_200w_q1_weekday_peak/saltlake_400s_200w_q1_weekday_peak.sumocfg",
        "saltlake_state_peak": ext_targets_dir
        / "saltlake_state_university_q1_weekday_peak/saltlake_state_university_q1_weekday_peak.sumocfg",
    }
    for k, p in sl_scens.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing Salt Lake City sumocfg {k}: {p}")

    sl_models_pkl = work_dir / "phase3_saltlake_models.pkl"
    sl_audit_json = work_dir / "phase3_saltlake_fit_audit.json"

    src_groups, _, _ = read_paper_label_archive(phase2_dataset_json)

    scan_tasks = [
        {
            "scenario": scen,
            "domain": "target_saltlake",
            "sumocfg": str(cfg),
            "seed": seed,
            "max_groups": 10,
        }
        for scen, cfg in sl_scens.items()
        for seed in ADAPT_SEEDS
    ]
    specs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(scan_decision_candidates_worker, t): t for t in scan_tasks}
        for fut in as_completed(futs):
            meta = futs[fut]
            for idx, c in enumerate(fut.result()):
                c["sumocfg"] = meta["sumocfg"]
                c["group_id"] = (
                    f"target_{c['scenario']}_s{c['seed']}_t{c['elapsed_sec']}_{c['tls_id']}_{idx}"
                )
                specs.append(c)
    specs.sort(key=lambda d: d["group_id"])
    btasks = [
        {
            "group_id": s["group_id"],
            "scenario": s["scenario"],
            "domain": s["domain"],
            "sumocfg": s["sumocfg"],
            "seed": s["seed"],
            "elapsed_sec": s["elapsed_sec"],
            "tls_id": s["tls_id"],
            "branch_idx": b_i,
            "cand_state": cst,
        }
        for s in specs
        for b_i, cst in enumerate(s["candidate_states"])
    ]
    by_gid: dict[str, list[dict[str, Any]]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(evaluate_single_branch_worker, bt) for bt in btasks]
        for fut in as_completed(futs):
            r = fut.result()
            by_gid.setdefault(r["group_id"], []).append(r)

    sl_groups: list[ResidualGroupRecord] = []
    for gid in sorted(by_gid):
        items = sorted(by_gid[gid], key=lambda x: x["branch_idx"])
        if len({x["fingerprint"] for x in items}) != 1:
            raise RuntimeError(f"Salt Lake pre-branch fingerprint mismatch: {gid}")
        h = items[0]
        am = np.asarray(h["action_mat"], dtype=np.float64)
        bs = -0.65 * (am[:, 0] - float(np.mean(am[:, 0]))) + 1e-9 * np.arange(len(am))
        sl_groups.append(
            ResidualGroupRecord(
                group_id=gid,
                domain="target_saltlake",
                scenario=str(h["scenario"]),
                seed=int(h["seed"]),
                tls_id=str(h["tls_id"]),
                time_sec=float(h["time_sec"]),
                candidate_states=tuple(str(x) for x in h["candidate_states"]),
                reference_index=int(h["reference_index"]),
                state_vec=np.asarray(h["state_vec"], dtype=np.float64),
                action_mat=am,
                base_scores=bs,
                branch_costs=np.asarray([float(x["branch_cost"]) for x in items], dtype=np.float64),
            )
        )

    frozen_models, fit_audit = fit_phase2_residual_suite(
        src_groups, {"saltlake": sl_groups}
    )
    with sl_models_pkl.open("wb") as fh:
        pickle.dump(frozen_models, fh)
    sl_audit_json.write_text(
        json.dumps(
            {
                "state_restores": 0,
                "pre_branch_fingerprint_mismatches": 0,
                "saltlake_adapt_groups": len(sl_groups),
                "fit_audit": fit_audit,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    policies = (
        "oracle_gp",
        "fixed_program",
        "observed_mp_10",
        "observed_pp_10",
        "observed_gp_10",
        "shockwave_gp_10",
        "conservation_gp_10",
        "maxred_conservation_gp_10",
        "zero_shot_cfcmt_b0",
        "target_only_b6",
        "source_placebo_b6",
        "cfcmt_b6",
        "target_only_b36",
        "source_placebo_b36",
        "pool_all_b36",
        "cfcmt_b36",
    )
    eval_tasks = [
        {
            "family": "saltlake",
            "scenario": scen,
            "sumocfg": str(cfg),
            "seed": seed,
            "policy": pol,
            "models_pkl": str(sl_models_pkl),
        }
        for scen, cfg in sl_scens.items()
        for seed in FRESH_SALTLAKE_SEEDS
        for pol in policies
    ]
    print(
        f"[Phase 3B Salt Lake City] Evaluating {len(eval_tasks)} cells across 10 virgin seeds {FRESH_SALTLAKE_SEEDS[0]}..{FRESH_SALTLAKE_SEEDS[-1]}...",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_phase2_closed_loop_worker, t) for t in eval_tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            if done % 64 == 0 or done == len(eval_tasks):
                print(
                    f"  [3B {done}/{len(eval_tasks)}] {r['scenario']} {r['policy']} wait={r['mean_waiting_sec']:.2f}s",
                    flush=True,
                )

    rows.sort(key=lambda d: (d["scenario"], d["seed"], d["policy"]))
    summary = summarize_phase2_rows(rows)
    out_path = work_dir / "phase3_saltlake_eval_fresh_result.json"
    out_path.write_text(
        json.dumps(
            {
                "p_cv": 0.10,
                "seeds": list(FRESH_SALTLAKE_SEEDS),
                **summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SAVED_PHASE3_SALTLAKE {out_path}", flush=True)
    return sl_models_pkl


def run_rl_baselines(
    *,
    work_dir: Path,
    scenarios: Mapping[str, Path],
    ext_targets_dir: Path,
    workers: int,
    source_only: bool = False,
) -> None:
    src_cfgs = [
        scenarios["hangzhou_4x4_repaired"],
        scenarios["hangzhou_4x4_hetero_repaired"],
        scenarios["atlanta_1x5"],
        scenarios["cologne3"],
        scenarios["ingolstadt7"],
    ]
    print("[Phase 3C RL] Pretraining source MPLight-CV across 5 source scenarios...", flush=True)
    pretrained_net, pre_audit = train_mplight_agent(
        sumocfgs=src_cfgs,
        seeds=SOURCE_SEEDS,
        n_episodes=10,
        init_net=None,
        rng_seed=20260301,
        workers=workers,
    )

    sl_scens = {
        "saltlake_400s_peak": ext_targets_dir
        / "saltlake_400s_200w_q1_weekday_peak/saltlake_400s_200w_q1_weekday_peak.sumocfg",
        "saltlake_state_peak": ext_targets_dir
        / "saltlake_state_university_q1_weekday_peak/saltlake_state_university_q1_weekday_peak.sumocfg",
    }

    family_cfgs: dict[str, tuple[list[Path], tuple[int, ...], dict[str, Path]]] = {
        "la": (
            [scenarios["la_1x4"]],
            FRESH_PHASE2_SEEDS,
            {"la_1x4": scenarios["la_1x4"]},
        ),
        "jinan": (
            [
                scenarios["jinan_3x4_real"],
                scenarios["jinan_3x4_real_2000"],
                scenarios["jinan_3x4_real_2500"],
            ],
            FRESH_PHASE2_SEEDS,
            {
                "jinan_3x4_real": scenarios["jinan_3x4_real"],
                "jinan_3x4_real_2000": scenarios["jinan_3x4_real_2000"],
                "jinan_3x4_real_2500": scenarios["jinan_3x4_real_2500"],
            },
        ),
        "saltlake": (
            list(sl_scens.values()),
            FRESH_SALTLAKE_SEEDS,
            sl_scens,
        ),
    }

    policies = (SOURCE_ONLY_RL_POLICY,) if source_only else RL_POLICIES
    rl_models: dict[str, dict[str, Any]] = {}
    rl_audit: dict[str, Any] = {"source_pretrain": pre_audit, "families": {}}
    for fam_i, (fam, (train_cfgs, _, _)) in enumerate(family_cfgs.items()):
        if source_only:
            rl_models[fam] = {SOURCE_ONLY_RL_POLICY: pretrained_net}
            rl_audit["families"][fam] = {SOURCE_ONLY_RL_POLICY: {"episodes": 0}}
            continue
        print(f"[Phase 3C RL] Training MPLight-CV arms for {fam}...", flush=True)
        ep2_net, a_ep2 = train_mplight_agent(
            sumocfgs=train_cfgs,
            seeds=ADAPT_SEEDS,
            n_episodes=2,
            init_net=None,
            rng_seed=20260400 + 10 * fam_i,
            workers=workers,
        )
        ep16_net, a_ep16 = train_mplight_agent(
            sumocfgs=train_cfgs,
            seeds=ADAPT_SEEDS,
            n_episodes=16,
            init_net=None,
            rng_seed=20260500 + 10 * fam_i,
            workers=workers,
        )
        tr_net, a_tr = train_mplight_agent(
            sumocfgs=train_cfgs,
            seeds=ADAPT_SEEDS,
            n_episodes=2,
            init_net=pretrained_net,
            rng_seed=20260600 + 10 * fam_i,
            workers=workers,
        )
        rl_models[fam] = {
            "mplight_cv_ep2": ep2_net,
            "mplight_cv_ep16": ep16_net,
            "transfer_mplight_cv_ep2": tr_net,
        }
        rl_audit["families"][fam] = {
            "mplight_cv_ep2": a_ep2,
            "mplight_cv_ep16": a_ep16,
            "transfer_mplight_cv_ep2": a_tr,
        }

    rl_pkl = work_dir / ("phase3_rl_source_only_models.pkl" if source_only else "phase3_rl_models.pkl")
    with rl_pkl.open("wb") as fh:
        pickle.dump(rl_models, fh)

    eval_tasks: list[dict[str, Any]] = []
    for fam, (_, eval_seeds, scen_map) in family_cfgs.items():
        for scen, cfg in scen_map.items():
            for seed in eval_seeds:
                for pol in policies:
                    eval_tasks.append(
                        {
                            "family": fam,
                            "scenario": scen,
                            "sumocfg": str(cfg),
                            "seed": seed,
                            "p_cv": 0.10,
                            "policy": pol,
                            "models_pkl": str(rl_pkl),
                        }
                    )

    print(
        f"[Phase 3C RL] Evaluating {len(eval_tasks)} virgin-seed RL cells...",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(eval_custom_policy_worker, t) for t in eval_tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            if done % 30 == 0 or done == len(eval_tasks):
                print(
                    f"  [3C {done}/{len(eval_tasks)}] {r['scenario']} {r['policy']} wait={r['mean_waiting_sec']:.2f}s",
                    flush=True,
                )

    rows.sort(key=lambda d: (d["family"], d["scenario"], d["seed"], d["policy"]))
    rl_summary: dict[str, Any] = {}
    for fam in family_cfgs:
        rl_summary[fam] = {}
        for pol in policies:
            sub = [r for r in rows if r["family"] == fam and r["policy"] == pol]
            rl_summary[fam][pol] = {
                "mean_waiting_sec": float(np.mean([x["mean_waiting_sec"] for x in sub])),
                "median_waiting_sec": float(np.median([x["mean_waiting_sec"] for x in sub])),
                "system_vehicle_hours": float(
                    np.mean([x["system_vehicle_hours"] for x in sub])
                ),
                "completion_ratio": float(np.mean([x["completion_ratio"] for x in sub])),
                "total_collision_incidents": int(
                    sum(x["collision_incidents"] for x in sub)
                ),
            }

    out_path = work_dir / (
        "phase3_rl_source_only_result.json" if source_only else "phase3_rl_baselines_result.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "training_audit": rl_audit,
                "family_summary": rl_summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SAVED_PHASE3_RL {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("all", "penetration", "saltlake", "rl", "rl_source_only"),
        default="all",
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--phase2-dir", type=Path, required=True)
    parser.add_argument("--archive-labels", type=Path, help="Label JSON for the Salt Lake source pool")
    parser.add_argument("--staged-scenarios-dir", type=Path, required=True)
    parser.add_argument("--source-scenarios-dir", type=Path, required=True)
    parser.add_argument("--ext-targets-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.archive_labels is not None and args.stage not in {"all", "saltlake"}:
        parser.error("--archive-labels is only used by all or saltlake")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    scenarios = resolve_phase2_scenarios(
        args.staged_scenarios_dir, args.source_scenarios_dir
    )

    if args.stage in {"all", "saltlake"}:
        run_third_city_saltlake(
            work_dir=args.work_dir,
            ext_targets_dir=args.ext_targets_dir,
            phase2_dataset_json=args.archive_labels or args.phase2_dir / "phase2_counterfactual_dataset.json",
            workers=args.workers,
        )
    if args.stage in {"all", "rl", "rl_source_only"}:
        run_rl_baselines(
            work_dir=args.work_dir,
            scenarios=scenarios,
            ext_targets_dir=args.ext_targets_dir,
            workers=args.workers,
            source_only=args.stage == "rl_source_only",
        )
    if args.stage in {"all", "penetration"}:
        run_penetration_curve(
            work_dir=args.work_dir,
            scenarios=scenarios,
            phase2_models_pkl=args.phase2_dir / "phase2_frozen_models.pkl",
            phase2_fresh_json=args.phase2_dir / "phase2_eval_fresh_result.json",
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
