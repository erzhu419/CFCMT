"""Phase 2 Cross-City Residual CFCMT + Symmetric Max-Red Cap + Source Admission Gate.

Pre-registered evaluation seeds: `920101..920110` (10 brand-new virgin seeds, never touched
in Phase 0 or Phase 1).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import (
    ActionGroupRecord,
    BlendedActionRanker,
    EndpointAntisymmetricActionRanker,
    build_1hop_tls_neighbors,
    evaluate_offline_regret,
    extract_decision_group_features,
    permute_source_groups_placebo,
)
from experiments.tsc_partial_cv_pilot_20261118.cv_sensor import (
    PartialCvObserver,
    select_observed_pressure_phase,
)
from experiments.tsc_partial_cv_pilot_20261118.phase0_sensor_gap import (
    _oracle_gp_phase,
)
from experiments.tsc_partial_cv_pilot_20261118.phase1_transfer_pilot import (
    compute_paired_comparison,
    evaluate_single_branch_worker,
    scan_decision_candidates_worker,
)
from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_model import (
    ResidualGroupRecord,
    compute_exact_cons_gp_raw_pressures,
    evaluate_residual_regret,
    extract_phase2_features,
    feasible_indices_under_max_red,
    pressures_to_base_cost_scores,
)

P_CV = 0.10
CONTROL_INTERVAL_SEC = 10
HORIZON_SEC = 60
SOURCE_SEEDS = (2027, 3037)
ADAPT_SEEDS = (5057, 6067)
DEV_EVAL_SEEDS = (7077, 8087)
FRESH_PHASE2_SEEDS = tuple(range(920101, 920111))

# 13 zero-conflict source scenarios across 6 diverse domains
SOURCE_SCENARIO_SPECS: dict[str, tuple[str, int]] = {
    "hangzhou_4x4_repaired": ("hangzhou_grid", 18),
    "hangzhou_4x4_hetero_repaired": ("hangzhou_grid", 18),
    "hangzhou_1x1_kn": ("hangzhou_rush_1x1", 10),
    "hangzhou_1x1_qc": ("hangzhou_rush_1x1", 10),
    "hangzhou_1x1_sb": ("hangzhou_rush_1x1", 10),
    "saltlake_400s": ("saltlake", 14),
    "saltlake_state": ("saltlake", 14),
    "atlanta_1x5": ("atlanta", 16),
    "cologne1": ("europe_urban", 10),
    "cologne3": ("europe_urban", 10),
    "ingolstadt7": ("europe_urban", 10),
    "grid4x4": ("resco_syn", 12),
    "arterial4x4": ("resco_syn", 12),
}

TARGET_FAMILIES: dict[str, tuple[str, ...]] = {
    "la": ("la_1x4",),
    "jinan": ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"),
}

OMEGA_POS_GRID = (0.08, 0.18, 0.25, 0.50, 0.75, 0.90, 1.00)

STRUCTURAL_SOURCE_POOLS: dict[str, tuple[str, ...]] = {
    "all_sources": (
        "hangzhou",
        "hangzhou_hetero",
        "atlanta",
        "cologne",
        "ingolstadt",
        "resco_grid_art",
    ),
    "real_cities": (
        "hangzhou",
        "hangzhou_hetero",
        "atlanta",
        "cologne",
        "ingolstadt",
    ),
    "corridor_sources": ("atlanta", "cologne", "ingolstadt"),
    "multi_phase_grids": ("hangzhou", "hangzhou_hetero"),
    "grid_sources": ("hangzhou", "hangzhou_hetero", "resco_grid_art"),
}

LEARNED_POLICIES = frozenset(
    {
        "zero_shot_cfcmt_b0",
        "target_only_b6",
        "source_placebo_b6",
        "cfcmt_b6",
        "target_only_b36",
        "source_placebo_b36",
        "pool_all_b36",
        "cfcmt_b36",
    }
)


def resolve_phase2_scenarios(staged_dir: Path, source_dir: Path) -> dict[str, Path]:
    """Ensure all 13 source scenarios and 4 target scenarios exist and return sumocfg map."""
    hz_rep_net = staged_dir / "hangzhou_4x4_repaired/hangzhou_aligned.net.xml"
    hz_het_rou = source_dir / "hangzhou_4x4_hetero/hangzhou_4x4_gudang_18041610_1h_m.rou.xml"
    hz_het_dst = source_dir / "hangzhou_4x4_hetero_repaired"
    if not (hz_het_dst / "hangzhou_hetero.sumocfg").is_file():
        hz_het_dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(hz_rep_net, hz_het_dst / "hangzhou_aligned.net.xml")
        shutil.copy2(hz_het_rou, hz_het_dst / "hangzhou_hetero.rou.xml")
        (hz_het_dst / "hangzhou_hetero.sumocfg").write_text(
            '<configuration><input><net-file value="hangzhou_aligned.net.xml"/>'
            '<route-files value="hangzhou_hetero.rou.xml"/></input>'
            '<time><begin value="0"/><end value="3600"/></time></configuration>\n',
            encoding="utf-8",
        )

    paths = {
        "la_1x4": staged_dir / "la_1x4/la_1x4.sumocfg",
        "jinan_3x4_real": staged_dir / "jinan_3x4_real/jinan_3x4_real.sumocfg",
        "jinan_3x4_real_2000": staged_dir / "jinan_3x4_real_2000/jinan_3x4_real_2000.sumocfg",
        "jinan_3x4_real_2500": staged_dir / "jinan_3x4_real_2500/jinan_3x4_real_2500.sumocfg",
        "hangzhou_4x4_repaired": staged_dir / "hangzhou_4x4_repaired/hangzhou_aligned.sumocfg",
        "hangzhou_4x4_hetero_repaired": hz_het_dst / "hangzhou_hetero.sumocfg",
        "hangzhou_1x1_kn": source_dir / "hangzhou_1x1_kn-hz_18041608_1h/hangzhou_1x1_kn-hz_18041608_1h.sumocfg",
        "hangzhou_1x1_qc": source_dir / "hangzhou_1x1_qc-yn_18041608_1h/hangzhou_1x1_qc-yn_18041608_1h.sumocfg",
        "hangzhou_1x1_sb": source_dir / "hangzhou_1x1_sb-sx_18041607_1h/hangzhou_1x1_sb-sx_18041607_1h.sumocfg",
        "saltlake_400s": source_dir / "saltlake2_400sX200w/saltlake2_400sX200w.sumocfg",
        "saltlake_state": source_dir / "saltlake2_stateXuniversity/saltlake2_stateXuniversity.sumocfg",
        "atlanta_1x5": source_dir / "atlanta_1x5/atlanta_1x5.sumocfg",
        "cologne1": source_dir / "cologne1/cologne1.sumocfg",
        "cologne3": source_dir / "cologne3/cologne3.sumocfg",
        "ingolstadt7": source_dir / "ingolstadt7/ingolstadt7.sumocfg",
        "grid4x4": source_dir / "grid4x4/grid4x4.sumocfg",
        "arterial4x4": source_dir / "arterial4x4/arterial4x4.sumocfg",
    }
    for name, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing sumocfg for {name}: {p}")
    return paths


def select_maxred_pressure_phase(
    *,
    info: Any,
    executor: Any,
    observer: PartialCvObserver,
    mode: str = "conservation_gp",
) -> str:
    """Analytical pressure selection subject to the symmetric NEMA Max-Red cap."""
    feasible = tuple(str(s) for s in executor.feasible_states_now())
    if len(feasible) <= 1:
        return feasible[0]
    allowed_idx = feasible_indices_under_max_red(
        info=info,
        feasible=feasible,
        observer=observer,
        max_red_sec=None,
        current_green_state=str(executor.current_green_state),
    )
    if len(allowed_idx) == len(feasible):
        return select_observed_pressure_phase(
            info=info, executor=executor, observer=observer, mode=mode
        )
    if len(allowed_idx) == 1:
        return feasible[allowed_idx[0]]
    # Rank strictly among relieving phases by conservation_gp pressure
    _, _, _, base_scores = extract_phase2_features(
        info=info,
        executor=executor,
        observer=observer,
        infos_all={info.tls_id: info} if hasattr(info, "tls_id") else {},
        up_neighbors=(),
        down_neighbors=(),
    )
    best_i = min(allowed_idx, key=lambda idx: float(base_scores[idx]))
    return feasible[best_i]


def _prefix_action_p2(
    *,
    seed: int,
    elapsed_sec: int,
    tls_id: str,
    info: Any,
    executor: Any,
    observer: PartialCvObserver,
) -> str:
    digest = hashlib.blake2b(
        f"p2:{seed}:{elapsed_sec}:{tls_id}".encode("utf-8"), digest_size=4
    ).digest()
    u = int.from_bytes(digest, "big") / 4294967296.0
    mode = "conservation_gp" if u < 0.85 else "shockwave_gp"
    return select_maxred_pressure_phase(
        info=info, executor=executor, observer=observer, mode=mode
    )


def scan_candidates_p2_worker(task: dict[str, Any]) -> list[dict[str, Any]]:
    from cf_h2o.sumo_runtime import load_libsumo
    from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
        _start_sumo,
        _tls_phase_infos,
    )
    from cf_h2o.traffic_signal.safe_phase_controller import build_safe_phase_executors

    scenario = str(task["scenario"])
    domain = str(task["domain"])
    sumocfg = Path(task["sumocfg"])
    seed = int(task["seed"])
    max_groups = int(task["max_groups"])

    sumo_api = load_libsumo()
    _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=None)
    raw_candidates: list[dict[str, Any]] = []
    try:
        infos = _tls_phase_infos(sumo_api)
        executors = build_safe_phase_executors(sumo_api, infos)
        observer = PartialCvObserver(
            sumo_api=sumo_api, infos=infos, seed=seed, p_cv=P_CV
        )
        observer.observe_step(sumo_api, dt_sec=1.0)

        begin_time = float(sumo_api.simulation.getTime())
        max_elapsed = 3420
        sample_epochs = set(range(180, max_elapsed + 1, 120))

        elapsed = 0
        while elapsed <= max_elapsed:
            if elapsed in sample_epochs:
                obs_map = observer.observations
                for tls_id in sorted(infos):
                    ex = executors[tls_id]
                    feas = tuple(str(s) for s in ex.feasible_states_now())
                    if len(feas) < 2:
                        continue
                    demand_est = sum(
                        max(obs_map[l].shockwave_queue_est, obs_map[l].conservation_queue_est)
                        for l in infos[tls_id].incoming_lanes
                    )
                    if demand_est >= 0.8:
                        raw_candidates.append(
                            {
                                "scenario": scenario,
                                "domain": domain,
                                "seed": seed,
                                "elapsed_sec": elapsed,
                                "tls_id": str(tls_id),
                                "candidate_states": feas,
                            }
                        )

            for tls_id in sorted(infos):
                act = _prefix_action_p2(
                    seed=seed,
                    elapsed_sec=elapsed,
                    tls_id=tls_id,
                    info=infos[tls_id],
                    executor=executors[tls_id],
                    observer=observer,
                )
                executors[tls_id].request(act)

            for _ in range(CONTROL_INTERVAL_SEC):
                t0 = float(sumo_api.simulation.getTime())
                sumo_api.simulation.step()
                t1 = float(sumo_api.simulation.getTime())
                dt = max(t1 - t0, 1.0)
                for ex in executors.values():
                    ex.advance(dt)
                observer.observe_step(sumo_api, dt_sec=dt)
            elapsed += CONTROL_INTERVAL_SEC
            if float(sumo_api.simulation.getTime()) >= begin_time + 3540.0:
                break
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass

    if len(raw_candidates) <= max_groups:
        return raw_candidates

    raw_candidates.sort(key=lambda d: (d["elapsed_sec"], d["tls_id"]))
    by_tls: dict[str, list[dict[str, Any]]] = {}
    for item in raw_candidates:
        by_tls.setdefault(item["tls_id"], []).append(item)
    tls_keys = sorted(by_tls)
    per_tls_quota = max(1, math.ceil(max_groups / max(len(tls_keys), 1)))
    per_tls_streams: dict[str, list[dict[str, Any]]] = {}
    for k in tls_keys:
        lst = by_tls[k]
        if len(lst) <= per_tls_quota:
            per_tls_streams[k] = list(lst)
        else:
            idxs = np.linspace(0, len(lst) - 1, per_tls_quota, dtype=int)
            seen_i: set[int] = set()
            sub: list[dict[str, Any]] = []
            for idx in idxs:
                i = int(idx)
                while i in seen_i and i + 1 < len(lst):
                    i += 1
                if i not in seen_i:
                    seen_i.add(i)
                    sub.append(lst[i])
            per_tls_streams[k] = sub

    selected: list[dict[str, Any]] = []
    for r_idx in range(per_tls_quota):
        for k in tls_keys:
            stream = per_tls_streams[k]
            if r_idx < len(stream):
                cand = stream[r_idx]
                if cand not in selected:
                    selected.append(cand)
                    if len(selected) >= max_groups:
                        break
        if len(selected) >= max_groups:
            break
    if len(selected) < max_groups:
        for item in raw_candidates:
            if item not in selected:
                selected.append(item)
                if len(selected) >= max_groups:
                    break
    selected.sort(key=lambda d: (d["elapsed_sec"], d["tls_id"]))
    return selected[:max_groups]


def evaluate_branch_p2_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Native-prefix 1-hop light-cone branch evaluation (`state_restores = 0`)."""
    from cf_h2o.sumo_runtime import load_libsumo
    from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
        _start_sumo,
        _tls_phase_infos,
    )
    from cf_h2o.traffic_signal.safe_phase_controller import build_safe_phase_executors

    group_id = str(task["group_id"])
    scenario = str(task["scenario"])
    domain = str(task["domain"])
    sumocfg = Path(task["sumocfg"])
    seed = int(task["seed"])
    target_elapsed = int(task["elapsed_sec"])
    focal_tls = str(task["tls_id"])
    branch_idx = int(task["branch_idx"])
    expected_cand = str(task["cand_state"])

    sumo_api = load_libsumo()
    _start_sumo(sumo_api, sumocfg, seed, tripinfo_output=None)
    try:
        infos = _tls_phase_infos(sumo_api)
        up_nb, down_nb, cone_map = build_1hop_tls_neighbors(infos)
        executors = build_safe_phase_executors(sumo_api, infos)
        observer = PartialCvObserver(
            sumo_api=sumo_api, infos=infos, seed=seed, p_cv=P_CV
        )
        observer.observe_step(sumo_api, dt_sec=1.0)

        elapsed = 0
        while elapsed < target_elapsed:
            for tls_id in sorted(infos):
                act = _prefix_action_p2(
                    seed=seed,
                    elapsed_sec=elapsed,
                    tls_id=tls_id,
                    info=infos[tls_id],
                    executor=executors[tls_id],
                    observer=observer,
                )
                executors[tls_id].request(act)
            for _ in range(CONTROL_INTERVAL_SEC):
                t0 = float(sumo_api.simulation.getTime())
                sumo_api.simulation.step()
                t1 = float(sumo_api.simulation.getTime())
                dt = max(t1 - t0, 1.0)
                for ex in executors.values():
                    ex.advance(dt)
                observer.observe_step(sumo_api, dt_sec=dt)
            elapsed += CONTROL_INTERVAL_SEC

        feasible, state_vec, action_mat, base_scores = extract_phase2_features(
            info=infos[focal_tls],
            executor=executors[focal_tls],
            observer=observer,
            infos_all=infos,
            up_neighbors=up_nb[focal_tls],
            down_neighbors=down_nb[focal_tls],
        )
        if branch_idx >= len(feasible) or feasible[branch_idx] != expected_cand:
            raise RuntimeError(
                f"Branch candidate mismatch on {group_id}: expected {expected_cand}, got {feasible}"
            )

        now_t = float(sumo_api.simulation.getTime())
        fp_src = (
            f"{now_t:.2f}|{sumo_api.vehicle.getIDCount()}|"
            f"{len(sumo_api.simulation.getPendingVehicles())}|"
            f"{np.round(state_vec, 8).tobytes().hex()}"
        )
        fingerprint = hashlib.blake2b(fp_src.encode("utf-8"), digest_size=8).hexdigest()
        ref_index = int(np.argmin(base_scores))

        cone_lanes = cone_map[focal_tls]
        focal_in_lanes = tuple(str(l) for l in infos[focal_tls].incoming_lanes)
        focal_in_edges = tuple(sorted({l.rsplit("_", 1)[0] for l in focal_in_lanes}))
        cone_vmax = {
            l: max(float(sumo_api.lane.getMaxSpeed(l)), 1.0) for l in cone_lanes
        }

        n_intervals = HORIZON_SEC // CONTROL_INTERVAL_SEC
        accum_cost = 0.0
        steps_counted = 0

        for step_block in range(n_intervals):
            for tls_id in sorted(infos):
                if step_block == 0 and tls_id == focal_tls:
                    chosen_phase = expected_cand
                else:
                    chosen_phase = select_maxred_pressure_phase(
                        info=infos[tls_id],
                        executor=executors[tls_id],
                        observer=observer,
                        mode="conservation_gp",
                    )
                executors[tls_id].request(chosen_phase)

            for _ in range(CONTROL_INTERVAL_SEC):
                t0 = float(sumo_api.simulation.getTime())
                sumo_api.simulation.step()
                t1 = float(sumo_api.simulation.getTime())
                dt = max(t1 - t0, 1.0)
                for ex in executors.values():
                    ex.advance(dt)
                observer.observe_step(sumo_api, dt_sec=dt)

                halted_cone = 0.0
                slow_loss_cone = 0.0
                for l in cone_lanes:
                    h = float(sumo_api.lane.getLastStepHaltingNumber(l))
                    v = float(sumo_api.lane.getLastStepVehicleNumber(l))
                    halted_cone += h
                    moving = max(0.0, v - h)
                    if moving > 0.0:
                        spd = float(sumo_api.lane.getLastStepMeanSpeed(l))
                        slow_loss_cone += moving * max(0.0, 1.0 - spd / cone_vmax[l])
                pending_focal = sum(
                    float(len(sumo_api.edge.getPendingVehicles(e)))
                    for e in focal_in_edges
                )
                accum_cost += halted_cone + pending_focal + 0.25 * slow_loss_cone
                steps_counted += 1

        branch_cost = accum_cost / float(max(steps_counted, 1) * max(len(focal_in_lanes), 1))
        result: dict[str, Any] = {
            "group_id": group_id,
            "branch_idx": branch_idx,
            "branch_cost": float(branch_cost),
            "fingerprint": fingerprint,
        }
        if branch_idx == 0:
            result.update(
                {
                    "domain": domain,
                    "scenario": scenario,
                    "seed": seed,
                    "tls_id": focal_tls,
                    "time_sec": float(target_elapsed),
                    "candidate_states": list(feasible),
                    "reference_index": ref_index,
                    "state_vec": state_vec.tolist(),
                    "action_mat": action_mat.tolist(),
                    "base_scores": base_scores.tolist(),
                }
            )
        return result
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass


def _rec_to_dict(g: ResidualGroupRecord) -> dict[str, Any]:
    return {
        "group_id": g.group_id,
        "domain": g.domain,
        "scenario": g.scenario,
        "seed": g.seed,
        "tls_id": g.tls_id,
        "time_sec": g.time_sec,
        "candidate_states": list(g.candidate_states),
        "reference_index": g.reference_index,
        "state_vec": g.state_vec.tolist(),
        "action_mat": g.action_mat.tolist(),
        "base_scores": g.base_scores.tolist(),
        "branch_costs": g.branch_costs.tolist(),
    }


def _dict_to_rec(d: Mapping[str, Any]) -> ResidualGroupRecord:
    am = np.asarray(d["action_mat"], dtype=np.float64)
    if "base_scores" in d:
        bs = np.asarray(d["base_scores"], dtype=np.float64)
    else:
        bs = -0.65 * (am[:, 0] - float(np.mean(am[:, 0]))) + 1e-9 * np.arange(len(am))
    return ResidualGroupRecord(
        group_id=str(d["group_id"]),
        domain=str(d["domain"]),
        scenario=str(d["scenario"]),
        seed=int(d["seed"]),
        tls_id=str(d["tls_id"]),
        time_sec=float(d["time_sec"]),
        candidate_states=tuple(str(x) for x in d["candidate_states"]),
        reference_index=int(d["reference_index"]),
        state_vec=np.asarray(d["state_vec"], dtype=np.float64),
        action_mat=am,
        base_scores=bs,
        branch_costs=np.asarray(d["branch_costs"], dtype=np.float64),
    )


def collect_phase2_dataset(
    scenarios: Mapping[str, Path], out_json: Path, workers: int
) -> tuple[list[ResidualGroupRecord], dict[str, list[ResidualGroupRecord]], dict[str, Any]]:
    if out_json.is_file():
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        src = [_dict_to_rec(x) for x in payload["source_groups"]]
        tgt = {
            k: [_dict_to_rec(x) for x in v]
            for k, v in payload["target_groups"].items()
        }
        audit = dict(payload["collection_audit"])
        if not any(g.domain == "hangzhou_hetero" for g in src):
            print(
                "[Phase 2 Augment] Collecting native-prefix branches for hangzhou_4x4_hetero_repaired (state_restores=0)...",
                flush=True,
            )
            het_scan_tasks = [
                {
                    "scenario": "hangzhou_4x4_hetero_repaired",
                    "domain": "hangzhou_hetero",
                    "sumocfg": str(scenarios["hangzhou_4x4_hetero_repaired"]),
                    "seed": seed,
                    "max_groups": 38,
                }
                for seed in SOURCE_SEEDS
            ]
            het_specs: list[dict[str, Any]] = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [
                    pool.submit(scan_decision_candidates_worker, t)
                    for t in het_scan_tasks
                ]
                for fut in as_completed(futs):
                    for idx, c in enumerate(fut.result()):
                        c["sumocfg"] = str(scenarios["hangzhou_4x4_hetero_repaired"])
                        c["group_id"] = (
                            f"source_hz_het_s{c['seed']}_t{c['elapsed_sec']}_{c['tls_id']}_{idx}"
                        )
                        het_specs.append(c)
            het_specs.sort(key=lambda d: d["group_id"])
            het_btasks = [
                {
                    "group_id": s["group_id"],
                    "scenario": s["scenario"],
                    "domain": s["domain"],
                    "sumocfg": s["sumocfg"],
                    "seed": s["seed"],
                    "elapsed_sec": s["elapsed_sec"],
                    "tls_id": s["tls_id"],
                    "branch_idx": b_idx,
                    "cand_state": cst,
                }
                for s in het_specs
                for b_idx, cst in enumerate(s["candidate_states"])
            ]
            by_gid: dict[str, list[dict[str, Any]]] = {}
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(evaluate_single_branch_worker, bt) for bt in het_btasks]
                for fut in as_completed(futs):
                    r = fut.result()
                    by_gid.setdefault(r["group_id"], []).append(r)
            for gid in sorted(by_gid):
                items = sorted(by_gid[gid], key=lambda x: x["branch_idx"])
                if len({x["fingerprint"] for x in items}) != 1:
                    raise RuntimeError(f"Fingerprint mismatch on {gid}")
                h = items[0]
                am = np.asarray(h["action_mat"], dtype=np.float64)
                bs = -0.65 * (am[:, 0] - float(np.mean(am[:, 0]))) + 1e-9 * np.arange(len(am))
                src.append(
                    ResidualGroupRecord(
                        group_id=gid,
                        domain="hangzhou_hetero",
                        scenario="hangzhou_4x4_hetero_repaired",
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
            audit["source_group_count"] = len(src)
            audit["state_restores"] = 0
            audit["pre_branch_fingerprint_mismatches"] = 0
            out_json.write_text(
                json.dumps(
                    {
                        "collection_audit": audit,
                        "source_groups": [_rec_to_dict(g) for g in src],
                        "target_groups": {
                            k: [_rec_to_dict(g) for g in v] for k, v in tgt.items()
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return src, tgt, audit

    scan_tasks: list[dict[str, Any]] = []
    for scen, (dom, quota) in SOURCE_SCENARIO_SPECS.items():
        for seed in SOURCE_SEEDS:
            scan_tasks.append(
                {
                    "role": "source",
                    "family": "source",
                    "scenario": scen,
                    "domain": dom,
                    "sumocfg": str(scenarios[scen]),
                    "seed": seed,
                    "max_groups": quota,
                }
            )

    for seed in ADAPT_SEEDS:
        scan_tasks.append(
            {
                "role": "target",
                "family": "la",
                "scenario": "la_1x4",
                "domain": "target_la",
                "sumocfg": str(scenarios["la_1x4"]),
                "seed": seed,
                "max_groups": 18,
            }
        )
        for j_scen in TARGET_FAMILIES["jinan"]:
            scan_tasks.append(
                {
                    "role": "target",
                    "family": "jinan",
                    "scenario": j_scen,
                    "domain": "target_jinan",
                    "sumocfg": str(scenarios[j_scen]),
                    "seed": seed,
                    "max_groups": 6,
                }
            )

    print(f"[Phase 2 Step A.1] Scanning {len(scan_tasks)} prefix rollouts...", flush=True)
    specs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fmap = {pool.submit(scan_candidates_p2_worker, t): t for t in scan_tasks}
        for fut in as_completed(fmap):
            meta = fmap[fut]
            for idx, c in enumerate(fut.result()):
                c["role"] = meta["role"]
                c["family"] = meta["family"]
                c["sumocfg"] = meta["sumocfg"]
                c["group_id"] = (
                    f"{meta['role']}_{c['scenario']}_s{c['seed']}_t{c['elapsed_sec']}_{c['tls_id']}_{idx}"
                )
                specs.append(c)

    specs.sort(key=lambda d: d["group_id"])
    branch_tasks: list[dict[str, Any]] = []
    for s in specs:
        for b_idx, cand_st in enumerate(s["candidate_states"]):
            branch_tasks.append(
                {
                    "group_id": s["group_id"],
                    "role": s["role"],
                    "family": s["family"],
                    "scenario": s["scenario"],
                    "domain": s["domain"],
                    "sumocfg": s["sumocfg"],
                    "seed": s["seed"],
                    "elapsed_sec": s["elapsed_sec"],
                    "tls_id": s["tls_id"],
                    "branch_idx": b_idx,
                    "cand_state": cand_st,
                }
            )

    print(
        f"[Phase 2 Step A.2] Evaluating {len(branch_tasks)} native-prefix branches across {len(specs)} groups (state_restores=0)...",
        flush=True,
    )
    by_group: dict[str, list[dict[str, Any]]] = {}
    role_map = {s["group_id"]: (s["role"], s["family"]) for s in specs}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(evaluate_branch_p2_worker, bt) for bt in branch_tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            by_group.setdefault(r["group_id"], []).append(r)
            done += 1
            if done % 200 == 0 or done == len(branch_tasks):
                print(f"  branches completed: {done}/{len(branch_tasks)}", flush=True)

    source_groups: list[ResidualGroupRecord] = []
    target_groups: dict[str, list[ResidualGroupRecord]] = {"la": [], "jinan": []}
    mismatches = 0
    for gid in sorted(by_group):
        items = sorted(by_group[gid], key=lambda x: x["branch_idx"])
        if len({x["fingerprint"] for x in items}) != 1:
            mismatches += 1
        h = items[0]
        rec = ResidualGroupRecord(
            group_id=gid,
            domain=str(h["domain"]),
            scenario=str(h["scenario"]),
            seed=int(h["seed"]),
            tls_id=str(h["tls_id"]),
            time_sec=float(h["time_sec"]),
            candidate_states=tuple(str(x) for x in h["candidate_states"]),
            reference_index=int(h["reference_index"]),
            state_vec=np.asarray(h["state_vec"], dtype=np.float64),
            action_mat=np.asarray(h["action_mat"], dtype=np.float64),
            base_scores=np.asarray(h["base_scores"], dtype=np.float64),
            branch_costs=np.asarray([float(x["branch_cost"]) for x in items], dtype=np.float64),
        )
        role, fam = role_map[gid]
        if role == "source":
            source_groups.append(rec)
        else:
            target_groups[fam].append(rec)

    if mismatches != 0:
        raise RuntimeError(f"Pre-branch fingerprint mismatch count: {mismatches}")

    audit = {
        "state_restores": 0,
        "pre_branch_fingerprint_mismatches": mismatches,
        "source_group_count": len(source_groups),
        "source_domain_counts": {
            dom: sum(1 for g in source_groups if g.domain == dom)
            for dom in sorted({g.domain for g in source_groups})
        },
        "target_group_counts": {k: len(v) for k, v in target_groups.items()},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "collection_audit": audit,
                "source_groups": [_rec_to_dict(g) for g in source_groups],
                "target_groups": {
                    k: [_rec_to_dict(g) for g in v] for k, v in target_groups.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return source_groups, target_groups, audit


def subset_residual_budget(
    groups: Sequence[ResidualGroupRecord], budget_per_seed: int
) -> list[ResidualGroupRecord]:
    by_seed: dict[int, list[ResidualGroupRecord]] = {}
    for g in groups:
        by_seed.setdefault(g.seed, []).append(g)
    out: list[ResidualGroupRecord] = []
    for seed in sorted(by_seed):
        lst = sorted(by_seed[seed], key=lambda g: (g.time_sec, g.scenario, g.tls_id))
        if len(lst) <= budget_per_seed:
            out.extend(lst)
        else:
            idxs = np.linspace(0, len(lst) - 1, budget_per_seed, dtype=int)
            used: set[int] = set()
            for idx in idxs:
                i = int(idx)
                while i in used and i + 1 < len(lst):
                    i += 1
                used.add(i)
                out.append(lst[i])
    return out


def _to_action_group(g: ResidualGroupRecord) -> ActionGroupRecord:
    return ActionGroupRecord(
        group_id=g.group_id,
        domain=g.domain,
        scenario=g.scenario,
        seed=g.seed,
        tls_id=g.tls_id,
        time_sec=g.time_sec,
        candidate_states=g.candidate_states,
        reference_index=g.reference_index,
        state_vec=g.state_vec,
        action_mat=g.action_mat,
        branch_costs=g.branch_costs,
    )


def _smooth_boltzmann_regret(
    groups: Sequence[ActionGroupRecord],
    score_list: Sequence[np.ndarray],
    tau: float = 0.08,
) -> float:
    """Smooth margin-sensitive Boltzmann expected normalized regret (eliminates step-function OOF ties)."""
    losses: list[float] = []
    for g, sc in zip(groups, score_list):
        if len(g.candidate_states) < 2:
            continue
        s = np.asarray(sc, dtype=np.float64)
        logits = np.clip(-(s - float(np.min(s))) / tau, -50.0, 0.0)
        probs = np.exp(logits)
        probs /= float(np.sum(probs))
        mc = float(np.min(g.branch_costs))
        sp = max(float(np.max(g.branch_costs) - mc), 0.15)
        losses.append(float(np.dot(probs, (g.branch_costs - mc) / sp)))
    return float(np.mean(losses)) if losses else 0.0


def _eval_precomputed_scores(
    groups: Sequence[ActionGroupRecord], score_list: Sequence[np.ndarray]
) -> dict[str, float]:
    norm_regs: list[float] = []
    phys_regs: list[float] = []
    hits = 0
    for g, sc in zip(groups, score_list):
        if len(g.candidate_states) < 2:
            continue
        ch = int(np.argmin(sc))
        mc = float(np.min(g.branch_costs))
        sp = max(float(np.max(g.branch_costs) - mc), 0.15)
        pr = float(g.branch_costs[ch] - mc)
        phys_regs.append(pr)
        norm_regs.append(pr / sp)
        if abs(pr) <= 1e-9:
            hits += 1
    n = max(len(norm_regs), 1)
    return {
        "group_count": len(norm_regs),
        "mean_normalized_regret": float(np.mean(norm_regs)) if norm_regs else 0.0,
        "mean_physical_regret": float(np.mean(phys_regs)) if phys_regs else 0.0,
        "smooth_boltzmann_regret": _smooth_boltzmann_regret(groups, score_list),
        "top1_accuracy": float(hits / n),
    }


def fit_phase2_residual_suite(
    source_groups: Sequence[ResidualGroupRecord],
    target_groups_by_family: Mapping[str, Sequence[ResidualGroupRecord]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit group-capacity-regularized structural multi-city pools + 2-fold seed OOF selection."""
    src_act = [_to_action_group(g) for g in source_groups]
    placebo_src_act = permute_source_groups_placebo(src_act, rng_seed=20261119)

    avail_doms = {g.domain for g in src_act}
    source_pool_rankers: dict[str, EndpointAntisymmetricActionRanker] = {}
    placebo_pool_rankers: dict[str, EndpointAntisymmetricActionRanker] = {}
    for pool_name, doms in STRUCTURAL_SOURCE_POOLS.items():
        eff_doms = tuple(d for d in doms if d in avail_doms)
        if not eff_doms:
            continue
        dom_set = set(eff_doms)
        weights = {d: 1.0 / len(eff_doms) for d in eff_doms}
        source_pool_rankers[pool_name] = EndpointAntisymmetricActionRanker(
            l2_regularization=15.0, random_state=42
        ).fit([g for g in src_act if g.domain in dom_set], domain_weights=weights)
        placebo_pool_rankers[pool_name] = EndpointAntisymmetricActionRanker(
            l2_regularization=15.0, random_state=42
        ).fit(
            [g for g in placebo_src_act if g.domain in dom_set], domain_weights=weights
        )

    zs_pool_name = "real_cities" if "real_cities" in source_pool_rankers else next(iter(source_pool_rankers))
    pa_pool_name = "all_sources" if "all_sources" in source_pool_rankers else zs_pool_name

    frozen_models: dict[str, dict[str, Any]] = {}
    fit_audit: dict[str, Any] = {
        "available_source_pools": list(source_pool_rankers.keys()),
        "zero_shot_default_pool": zs_pool_name,
    }

    for fam, full_res_groups in target_groups_by_family.items():
        full_act_groups = [_to_action_group(g) for g in full_res_groups]
        zero_shot_ctrl = BlendedActionRanker(
            target_ranker=None,
            source_ranker=source_pool_rankers[zs_pool_name],
            omega=1.0,
        )
        fam_models: dict[str, Any] = {"zero_shot_cfcmt_b0": zero_shot_ctrl}
        fam_audit: dict[str, Any] = {
            "base_conservation_gp_regret": evaluate_residual_regret(
                None, full_res_groups
            ),
            "zero_shot_consensus_regret_on_adapt": evaluate_offline_regret(
                zero_shot_ctrl, full_act_groups
            ),
            "zero_shot_pool_all_regret_on_adapt": evaluate_offline_regret(
                source_pool_rankers[pa_pool_name], full_act_groups
            ),
            "per_pool_zero_shot_regrets": {
                pname: evaluate_offline_regret(rk, full_act_groups)
                for pname, rk in source_pool_rankers.items()
            },
            "budgets": {},
        }

        budgets_res = {
            "b6": subset_residual_budget(full_res_groups, budget_per_seed=3),
            "b36": subset_residual_budget(full_res_groups, budget_per_seed=18),
        }

        for b_tag, t_res_sub in budgets_res.items():
            t_act_sub = [_to_action_group(g) for g in t_res_sub]
            s0, s1 = ADAPT_SEEDS
            a_s0 = [g for g in t_act_sub if g.seed == s0]
            a_s1 = [g for g in t_act_sub if g.seed == s1]

            fold_t0 = EndpointAntisymmetricActionRanker(
                l2_regularization=15.0, random_state=42
            ).fit(a_s0)
            fold_t1 = EndpointAntisymmetricActionRanker(
                l2_regularization=15.0, random_state=42
            ).fit(a_s1)

            val_groups = a_s1 + a_s0
            t_oof_scores = [
                fold_t0.score_group(g.state_vec, g.action_mat) for g in a_s1
            ] + [fold_t1.score_group(g.state_vec, g.action_mat) for g in a_s0]
            oof_to = _eval_precomputed_scores(val_groups, t_oof_scores)

            min_k = min((len(g.candidate_states) for g in t_act_sub), default=2)
            if min_k < 6:
                compat_pools = [
                    p for p in ("real_cities",) if p in source_pool_rankers
                ]
            else:
                compat_pools = [
                    p for p in ("multi_phase_grids",) if p in source_pool_rankers
                ]
            if not compat_pools:
                compat_pools = list(source_pool_rankers.keys())

            best_key = (float("inf"), float("inf"))
            best_pool = compat_pools[0]
            best_omega = 0.50
            best_cf_oof: dict[str, float] = oof_to

            for pool_name in compat_pools:
                s_ranker = source_pool_rankers[pool_name]
                p_scores = [
                    s_ranker.score_group(g.state_vec, g.action_mat) for g in val_groups
                ]
                for omega in OMEGA_POS_GRID:
                    comb = [
                        (1.0 - omega) * ts + omega * ps
                        for ts, ps in zip(t_oof_scores, p_scores)
                    ]
                    m = _eval_precomputed_scores(val_groups, comb)
                    cand_key = (
                        round(m["mean_normalized_regret"], 6),
                        round(m["smooth_boltzmann_regret"], 6),
                    )
                    if cand_key < best_key:
                        best_key = cand_key
                        best_pool = pool_name
                        best_omega = omega
                        best_cf_oof = m

            final_tgt_ranker = EndpointAntisymmetricActionRanker(
                l2_regularization=15.0, random_state=42
            ).fit(t_act_sub)

            oof_flat = np.concatenate(t_oof_scores) if t_oof_scores else np.zeros(1)
            full_flat = (
                np.concatenate(
                    [
                        final_tgt_ranker.score_group(g.state_vec, g.action_mat)
                        for g in val_groups
                    ]
                )
                if val_groups
                else np.zeros(1)
            )
            rms_oof = float(np.sqrt(np.mean(oof_flat ** 2)))
            rms_full = float(np.sqrt(np.mean(full_flat ** 2)))
            tau_cal = float(min(1.0, rms_oof / max(rms_full, 1e-9)))

            fam_models[f"target_only_{b_tag}"] = BlendedActionRanker(
                target_ranker=final_tgt_ranker, source_ranker=None, omega=0.0
            )
            fam_models[f"cfcmt_{b_tag}"] = BlendedActionRanker(
                target_ranker=final_tgt_ranker,
                source_ranker=source_pool_rankers[best_pool],
                omega=best_omega,
                target_scale=tau_cal,
            )
            fam_models[f"source_placebo_{b_tag}"] = BlendedActionRanker(
                target_ranker=final_tgt_ranker,
                source_ranker=placebo_pool_rankers[best_pool],
                omega=best_omega,
                target_scale=tau_cal,
            )

            audit_b: dict[str, Any] = {
                "target_group_count": len(t_act_sub),
                "selected_source_pool": best_pool,
                "cfcmt_omega": best_omega,
                "fold_to_full_target_scale": tau_cal,
                "oof_target_only": oof_to,
                "oof_cfcmt": best_cf_oof,
            }

            if b_tag == "b36":
                pa_scores = [
                    source_pool_rankers[pa_pool_name].score_group(
                        g.state_vec, g.action_mat
                    )
                    for g in val_groups
                ]
                best_pa_om = 0.50
                best_pa_key = (float("inf"), float("inf"))
                best_pa_oof: dict[str, float] = {}
                for omega in OMEGA_POS_GRID:
                    m = _eval_precomputed_scores(
                        val_groups,
                        [
                            (1.0 - omega) * ts + omega * ps
                            for ts, ps in zip(t_oof_scores, pa_scores)
                        ],
                    )
                    pa_key = (
                        round(m["mean_normalized_regret"], 6),
                        round(m["smooth_boltzmann_regret"], 6),
                    )
                    if pa_key < best_pa_key:
                        best_pa_key = pa_key
                        best_pa_om = omega
                        best_pa_oof = m
                fam_models["pool_all_b36"] = BlendedActionRanker(
                    target_ranker=final_tgt_ranker,
                    source_ranker=source_pool_rankers[pa_pool_name],
                    omega=best_pa_om,
                    target_scale=tau_cal,
                )
                audit_b["pool_all_omega"] = best_pa_om
                audit_b["oof_pool_all_b36"] = best_pa_oof

            fam_audit["budgets"][b_tag] = audit_b

        frozen_models[fam] = fam_models
        fit_audit[fam] = fam_audit

    return frozen_models, fit_audit


def _two_fold_phys_regret(
    c0: Any,
    val0: Sequence[ResidualGroupRecord],
    c1: Any,
    val1: Sequence[ResidualGroupRecord],
) -> dict[str, float]:
    r0 = evaluate_residual_regret(c0, val0)
    r1 = evaluate_residual_regret(c1, val1)
    n0, n1 = r0["group_count"], r1["group_count"]
    nt = max(n0 + n1, 1)
    return {
        "group_count": n0 + n1,
        "mean_normalized_regret": (
            r0["mean_normalized_regret"] * n0 + r1["mean_normalized_regret"] * n1
        )
        / nt,
        "mean_physical_regret": (
            r0["mean_physical_regret"] * n0 + r1["mean_physical_regret"] * n1
        )
        / nt,
        "top1_accuracy": (r0["top1_accuracy"] * n0 + r1["top1_accuracy"] * n1) / nt,
    }


def eval_phase2_closed_loop_worker(task: dict[str, Any]) -> dict[str, Any]:
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
    models_pkl = Path(task["models_pkl"])

    learned_ctrl: Any = None
    if policy in LEARNED_POLICIES:
        with models_pkl.open("rb") as fh:
            all_models = pickle.load(fh)
        learned_ctrl = all_models[family][policy]

    sumo_api = load_libsumo()
    with tempfile.TemporaryDirectory(prefix="cfcmt_cv_p2_") as tmp_dir:
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
                    sumo_api=sumo_api, infos=infos, seed=seed, p_cv=P_CV
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
                        elif policy == "maxred_conservation_gp_10" and observer is not None:
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


def summarize_phase2_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scen_pol: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    by_fam_rows: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_scen_pol.setdefault(str(r["scenario"]), {}).setdefault(
            str(r["policy"]), []
        ).append(r)
        by_fam_rows.setdefault(str(r["family"]), []).append(r)

    scenario_summary: dict[str, Any] = {}
    for scen, pmap in by_scen_pol.items():
        scenario_summary[scen] = {}
        for pol, items in pmap.items():
            scenario_summary[scen][pol] = {
                "mean_waiting_sec": float(
                    np.mean([x["mean_waiting_sec"] for x in items])
                ),
                "median_waiting_sec": float(
                    np.median([x["mean_waiting_sec"] for x in items])
                ),
                "system_vehicle_hours": float(
                    np.mean([x["system_vehicle_hours"] for x in items])
                ),
                "completion_ratio": float(
                    np.mean([x["completion_ratio"] for x in items])
                ),
                "mean_switches": float(np.mean([x["switch_count"] for x in items])),
                "mean_disagree_vs_cons_gp": float(
                    np.mean([x["disagree_rate_vs_cons_gp"] for x in items])
                ),
                "mean_maxred_trigger_rate": float(
                    np.mean([x["maxred_trigger_rate"] for x in items])
                ),
                "total_collision_incidents": int(
                    sum(x["collision_incidents"] for x in items)
                ),
                "total_teleports": int(
                    sum(x["starting_teleports"] for x in items)
                ),
            }

    paired_tests: dict[str, Any] = {}
    for fam, frows in by_fam_rows.items():
        present_pols = {str(x["policy"]) for x in frows}
        fam_tests: dict[str, Any] = {}
        for cand in (
            "zero_shot_cfcmt_b0",
            "cfcmt_b6",
            "cfcmt_b36",
        ):
            if cand not in present_pols:
                continue
            fam_tests[cand] = {}
            for base in (
                "fixed_program",
                "observed_mp_10",
                "observed_pp_10",
                "observed_gp_10",
                "shockwave_gp_10",
                "conservation_gp_10",
                "maxred_conservation_gp_10",
                "target_only_b6",
                "source_placebo_b6",
                "target_only_b36",
                "source_placebo_b36",
                "pool_all_b36",
            ):
                if base in present_pols and base != cand:
                    fam_tests[cand][f"vs_{base}"] = compute_paired_comparison(
                        frows, cand, base
                    )
        paired_tests[fam] = fam_tests

    return {
        "scenario_summary": scenario_summary,
        "paired_tests": paired_tests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("collect_and_fit", "dev_check", "eval_fresh"),
        required=True,
    )
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--staged-scenarios-dir", type=Path, required=True)
    parser.add_argument("--source-scenarios-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    scenarios = resolve_phase2_scenarios(
        args.staged_scenarios_dir, args.source_scenarios_dir
    )
    dataset_json = args.work_dir / "phase2_counterfactual_dataset.json"
    models_pkl = args.work_dir / "phase2_frozen_models.pkl"
    fit_audit_json = args.work_dir / "phase2_offline_fit_audit.json"

    if args.stage == "collect_and_fit":
        src_groups, tgt_groups, col_audit = collect_phase2_dataset(
            scenarios, dataset_json, workers=args.workers
        )
        frozen_models, fit_audit = fit_phase2_residual_suite(
            src_groups, tgt_groups
        )
        with models_pkl.open("wb") as fh:
            pickle.dump(frozen_models, fh)
        fit_audit_json.write_text(
            json.dumps(
                {
                    "collection_audit": col_audit,
                    "fit_audit": fit_audit,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"SAVED_PHASE2_FIT_AUDIT {fit_audit_json}", flush=True)
        return

    if not models_pkl.is_file():
        raise FileNotFoundError(f"Missing {models_pkl}; run collect_and_fit first")

    seeds = DEV_EVAL_SEEDS if args.stage == "dev_check" else FRESH_PHASE2_SEEDS
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

    eval_tasks: list[dict[str, Any]] = []
    for fam, scen_list in TARGET_FAMILIES.items():
        for scen in scen_list:
            for seed in seeds:
                for pol in policies:
                    eval_tasks.append(
                        {
                            "family": fam,
                            "scenario": scen,
                            "sumocfg": str(scenarios[scen]),
                            "seed": seed,
                            "policy": pol,
                            "models_pkl": str(models_pkl),
                        }
                    )

    print(
        f"[Phase 2 {args.stage}] Running {len(eval_tasks)} closed-loop evaluation cells across {len(seeds)} seeds...",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(eval_phase2_closed_loop_worker, t) for t in eval_tasks]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            done += 1
            if done % 25 == 0 or done == len(eval_tasks):
                print(
                    f"[{done}/{len(eval_tasks)}] {r['scenario']} seed={r['seed']} {r['policy']} wait={r['mean_waiting_sec']:.2f}s inc={r['collision_incidents']}",
                    flush=True,
                )

    rows.sort(key=lambda d: (d["family"], d["scenario"], d["seed"], d["policy"]))
    summary = summarize_phase2_rows(rows)
    out_file = args.work_dir / f"phase2_{args.stage}_result.json"
    out_file.write_text(
        json.dumps(
            {
                "stage": args.stage,
                "p_cv": P_CV,
                "max_red_cap_sec": "demand_confirmed_two_clause_cycle_proportional",
                "seeds": list(seeds),
                **summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SAVED_PHASE2_EVAL {out_file}", flush=True)


if __name__ == "__main__":
    main()
