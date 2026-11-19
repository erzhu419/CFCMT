"""Phase 1 Cross-City Partial-CV Counterfactual Transfer Pilot (B1).

Strict methodological guarantees:
1. Partial-CV non-leakage: all partial-CV controllers (heuristic and learned) observe ONLY
   equipped CVs (`p_cv = 0.10`) via `PartialCvObserver`. `extract_decision_group_features`
   does not accept `sumo_api`.
2. Zero snapshot restoration (`state_restores = 0`): counterfactual branches replay native
   deterministic prefixes from `begin_time` to `t_0` and verify bit-identical pre-action
   state fingerprints across all candidate phases.
3. 1-hop causal light-cone ground-truth branch cost: eliminates city-wide 3-hop noise while
   capturing focal incoming/outgoing/1-hop-downstream halting + focal entry-edge pending
   spillback over `H = 60s`.
4. Strict architecture-matched arms: `zero_shot_cfcmt_b0`, `target_only`, `source_placebo`,
   and `cfcmt` all use the exact same 78-dim `EndpointAntisymmetricActionRanker` with zero
   reference-veto guard.
5. Purely offline 2-fold whole-seed (`5057 <-> 6067`) regret selection before freezing
   models for single-shot evaluation on 10 untouched fresh seeds (`910101..910110`).
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.cfcmt_cv_model import (
    ActionGroupRecord,
    BlendedActionRanker,
    EndpointAntisymmetricActionRanker,
    SimplexBlendedActionRanker,
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

P_CV = 0.10
CONTROL_INTERVAL_SEC = 10
HORIZON_SEC = 60
SOURCE_SEEDS = (2027, 3037)
ADAPT_SEEDS = (5057, 6067)
FRESH_EVAL_SEEDS = tuple(range(910101, 910111))

SOURCE_SCENARIO_DOMAINS: dict[str, str] = {
    "hangzhou_4x4_repaired": "hangzhou",
    "atlanta_1x5": "atlanta",
    "cologne1": "cologne",
    "cologne3": "cologne",
    "ingolstadt7": "ingolstadt",
    "grid4x4": "resco_grid_art",
    "arterial4x4": "resco_grid_art",
}

TARGET_FAMILIES: dict[str, tuple[str, ...]] = {
    "la": ("la_1x4",),
    "jinan": ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"),
}

OMEGA_GRID = (0.0, 0.25, 0.50, 0.75, 0.90, 1.0)

SOURCE_POOLS: dict[str, tuple[str, ...]] = {
    "all_sources": ("hangzhou", "atlanta", "cologne", "ingolstadt", "resco_grid_art"),
    "real_cities": ("hangzhou", "atlanta", "cologne", "ingolstadt"),
    "corridor_sources": ("atlanta", "cologne", "ingolstadt"),
    "grid_sources": ("hangzhou", "resco_grid_art"),
}


def resolve_all_scenarios(
    staged_dir: Path, source_dir: Path
) -> dict[str, Path]:
    """Map every source and target scenario name to its .sumocfg path."""
    paths = {
        "la_1x4": staged_dir / "la_1x4/la_1x4.sumocfg",
        "jinan_3x4_real": staged_dir / "jinan_3x4_real/jinan_3x4_real.sumocfg",
        "jinan_3x4_real_2000": staged_dir / "jinan_3x4_real_2000/jinan_3x4_real_2000.sumocfg",
        "jinan_3x4_real_2500": staged_dir / "jinan_3x4_real_2500/jinan_3x4_real_2500.sumocfg",
        "hangzhou_4x4_repaired": staged_dir / "hangzhou_4x4_repaired/hangzhou_aligned.sumocfg",
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


def _prefix_behavior_action(
    *,
    seed: int,
    elapsed_sec: int,
    tls_id: str,
    info: Any,
    executor: Any,
    observer: PartialCvObserver,
) -> str:
    """Deterministic prefix exploration policy (80% conservation_gp, 20% shockwave_gp)."""
    digest = hashlib.blake2b(
        f"{seed}:{elapsed_sec}:{tls_id}".encode("utf-8"), digest_size=4
    ).digest()
    u = int.from_bytes(digest, "big") / 4294967296.0
    mode = "conservation_gp" if u < 0.80 else "shockwave_gp"
    return select_observed_pressure_phase(
        info=info, executor=executor, observer=observer, mode=mode
    )


def scan_decision_candidates_worker(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Scan one native rollout to select stratified multi-action decision epochs."""
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
        max_elapsed = 3420  # leave >= 180s margin for 60s counterfactual horizon
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
                    in_lanes = infos[tls_id].incoming_lanes
                    demand_est = sum(
                        max(obs_map[l].shockwave_queue_est, obs_map[l].conservation_queue_est)
                        for l in in_lanes
                    )
                    if demand_est >= 1.2:
                        raw_candidates.append(
                            {
                                "scenario": scenario,
                                "domain": domain,
                                "seed": seed,
                                "elapsed_sec": elapsed,
                                "tls_id": str(tls_id),
                                "candidate_states": feas,
                                "demand_est": float(demand_est),
                            }
                        )

            for tls_id in sorted(infos):
                act = _prefix_behavior_action(
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

    # Deterministic stratified subsample across time & intersections
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


def evaluate_single_branch_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Replay native prefix (`state_restores = 0`) and integrate 1-hop light-cone cost."""
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
                act = _prefix_behavior_action(
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

        # At exact counterfactual branch point t_0: verify state & extract CV features
        feasible, state_vec, action_mat = extract_decision_group_features(
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

        ref_phase = select_observed_pressure_phase(
            info=infos[focal_tls],
            executor=executors[focal_tls],
            observer=observer,
            mode="conservation_gp",
        )
        ref_index = feasible.index(ref_phase) if ref_phase in feasible else 0

        cone_lanes = cone_map[focal_tls]
        focal_in_lanes = tuple(str(l) for l in infos[focal_tls].incoming_lanes)
        focal_in_edges = tuple(sorted({l.rsplit("_", 1)[0] for l in focal_in_lanes}))
        cone_vmax = {
            l: max(float(sumo_api.lane.getMaxSpeed(l)), 1.0) for l in cone_lanes
        }

        # Roll out H = 60s (first 10s = candidate phase at focal_tls; next 50s = conservation_gp)
        n_intervals = HORIZON_SEC // CONTROL_INTERVAL_SEC
        accum_cost = 0.0
        steps_counted = 0

        for step_block in range(n_intervals):
            for tls_id in sorted(infos):
                if step_block == 0 and tls_id == focal_tls:
                    chosen_phase = expected_cand
                else:
                    chosen_phase = select_observed_pressure_phase(
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

                # Ground-truth 1-hop causal light-cone waiting + pending + slow-platoon delay
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

        norm_denom = float(max(steps_counted, 1) * max(len(focal_in_lanes), 1))
        branch_cost = accum_cost / norm_denom

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
                    "reference_index": int(ref_index),
                    "state_vec": state_vec.tolist(),
                    "action_mat": action_mat.tolist(),
                }
            )
        return result
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass


def collect_counterfactual_dataset(
    scenarios: Mapping[str, Path], out_json: Path, workers: int
) -> tuple[list[ActionGroupRecord], dict[str, list[ActionGroupRecord]], dict[str, Any]]:
    """Collect or load cached native-prefix 1-hop counterfactual dataset."""
    if out_json.is_file():
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        src = [_dict_to_group(d) for d in payload["source_groups"]]
        tgt = {
            fam: [_dict_to_group(d) for d in lst]
            for fam, lst in payload["target_groups"].items()
        }
        return src, tgt, payload["collection_audit"]

    scan_tasks: list[dict[str, Any]] = []
    # 1. Source rollouts (7 scenarios * 2 seeds * 16 groups = up to 224 groups)
    for scen, dom in SOURCE_SCENARIO_DOMAINS.items():
        for seed in SOURCE_SEEDS:
            scan_tasks.append(
                {
                    "role": "source",
                    "family": "source",
                    "scenario": scen,
                    "domain": dom,
                    "sumocfg": str(scenarios[scen]),
                    "seed": seed,
                    "max_groups": 16,
                }
            )
    # 2. Target adaptation rollouts strictly on ADAPT_SEEDS (5057, 6067)
    for seed in ADAPT_SEEDS:
        scan_tasks.append(
            {
                "role": "target",
                "family": "la",
                "scenario": "la_1x4",
                "domain": "target_la",
                "sumocfg": str(scenarios["la_1x4"]),
                "seed": seed,
                "max_groups": 20,
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
                    "max_groups": 8,
                }
            )

    print(f"[Step A.1] Scanning {len(scan_tasks)} prefix rollouts...", flush=True)
    selected_specs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fut_map = {
            pool.submit(scan_decision_candidates_worker, t): t for t in scan_tasks
        }
        for fut in as_completed(fut_map):
            meta = fut_map[fut]
            cands = fut.result()
            for idx, c in enumerate(cands):
                c["role"] = meta["role"]
                c["family"] = meta["family"]
                c["sumocfg"] = meta["sumocfg"]
                c["group_id"] = (
                    f"{meta['role']}_{c['scenario']}_s{c['seed']}_t{c['elapsed_sec']}_{c['tls_id']}_{idx}"
                )
                selected_specs.append(c)

    selected_specs.sort(key=lambda d: d["group_id"])
    branch_tasks: list[dict[str, Any]] = []
    for spec in selected_specs:
        for b_idx, cand_state in enumerate(spec["candidate_states"]):
            branch_tasks.append(
                {
                    "group_id": spec["group_id"],
                    "role": spec["role"],
                    "family": spec["family"],
                    "scenario": spec["scenario"],
                    "domain": spec["domain"],
                    "sumocfg": spec["sumocfg"],
                    "seed": spec["seed"],
                    "elapsed_sec": spec["elapsed_sec"],
                    "tls_id": spec["tls_id"],
                    "branch_idx": b_idx,
                    "cand_state": cand_state,
                }
            )

    print(
        f"[Step A.2] Evaluating {len(branch_tasks)} native-prefix branches across {len(selected_specs)} groups (state_restores=0)...",
        flush=True,
    )
    by_group: dict[str, list[dict[str, Any]]] = {}
    role_by_group = {s["group_id"]: (s["role"], s["family"]) for s in selected_specs}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(evaluate_single_branch_worker, bt) for bt in branch_tasks]
        done = 0
        for fut in as_completed(futs):
            res = fut.result()
            by_group.setdefault(res["group_id"], []).append(res)
            done += 1
            if done % 150 == 0 or done == len(branch_tasks):
                print(f"  branches completed: {done}/{len(branch_tasks)}", flush=True)

    source_groups: list[ActionGroupRecord] = []
    target_groups: dict[str, list[ActionGroupRecord]] = {"la": [], "jinan": []}
    fingerprint_mismatches = 0

    for gid in sorted(by_group):
        items = sorted(by_group[gid], key=lambda r: r["branch_idx"])
        fps = {r["fingerprint"] for r in items}
        if len(fps) != 1:
            fingerprint_mismatches += 1
        head = items[0]
        rec = ActionGroupRecord(
            group_id=gid,
            domain=str(head["domain"]),
            scenario=str(head["scenario"]),
            seed=int(head["seed"]),
            tls_id=str(head["tls_id"]),
            time_sec=float(head["time_sec"]),
            candidate_states=tuple(str(x) for x in head["candidate_states"]),
            reference_index=int(head["reference_index"]),
            state_vec=np.asarray(head["state_vec"], dtype=np.float64),
            action_mat=np.asarray(head["action_mat"], dtype=np.float64),
            branch_costs=np.asarray(
                [float(r["branch_cost"]) for r in items], dtype=np.float64
            ),
        )
        role, fam = role_by_group[gid]
        if role == "source":
            source_groups.append(rec)
        else:
            target_groups[fam].append(rec)

    if fingerprint_mismatches != 0:
        raise RuntimeError(
            f"Pre-branch native determinism check failed: {fingerprint_mismatches} mismatches!"
        )

    audit = {
        "state_restores": 0,
        "pre_branch_fingerprint_mismatches": fingerprint_mismatches,
        "source_group_count": len(source_groups),
        "source_branch_count": int(sum(len(g.candidate_states) for g in source_groups)),
        "target_group_counts": {k: len(v) for k, v in target_groups.items()},
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "collection_audit": audit,
                "source_groups": [_group_to_dict(g) for g in source_groups],
                "target_groups": {
                    k: [_group_to_dict(g) for g in v] for k, v in target_groups.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return source_groups, target_groups, audit


def _group_to_dict(g: ActionGroupRecord) -> dict[str, Any]:
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
        "branch_costs": g.branch_costs.tolist(),
    }


def _dict_to_group(d: Mapping[str, Any]) -> ActionGroupRecord:
    return ActionGroupRecord(
        group_id=str(d["group_id"]),
        domain=str(d["domain"]),
        scenario=str(d["scenario"]),
        seed=int(d["seed"]),
        tls_id=str(d["tls_id"]),
        time_sec=float(d["time_sec"]),
        candidate_states=tuple(str(x) for x in d["candidate_states"]),
        reference_index=int(d["reference_index"]),
        state_vec=np.asarray(d["state_vec"], dtype=np.float64),
        action_mat=np.asarray(d["action_mat"], dtype=np.float64),
        branch_costs=np.asarray(d["branch_costs"], dtype=np.float64),
    )


def subset_target_budget(
    groups: Sequence[ActionGroupRecord], budget_per_seed: int
) -> list[ActionGroupRecord]:
    """Deterministically select evenly spaced target groups per (seed, scenario)."""
    by_seed: dict[int, list[ActionGroupRecord]] = {}
    for g in groups:
        by_seed.setdefault(g.seed, []).append(g)
    out: list[ActionGroupRecord] = []
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


def _reference_offline_regret(groups: Sequence[ActionGroupRecord]) -> dict[str, float]:
    norm_regs = []
    phys_regs = []
    hits = 0
    for g in groups:
        if len(g.candidate_states) < 2:
            continue
        min_c = float(np.min(g.branch_costs))
        span = max(float(np.max(g.branch_costs) - min_c), 0.15)
        reg = float(g.branch_costs[g.reference_index] - min_c)
        phys_regs.append(reg)
        norm_regs.append(reg / span)
        if abs(reg) <= 1e-9:
            hits += 1
    n = max(len(norm_regs), 1)
    return {
        "group_count": len(norm_regs),
        "mean_normalized_regret": float(np.mean(norm_regs)) if norm_regs else 0.0,
        "mean_physical_regret": float(np.mean(phys_regs)) if phys_regs else 0.0,
        "top1_accuracy": float(hits / n),
    }


LEARNED_POLICIES = frozenset(
    {
        "zero_shot_cfcmt_b0",
        "target_only_b8",
        "source_placebo_b8",
        "cfcmt_b8",
        "target_only_bfull",
        "source_placebo_bfull",
        "pool_all_bfull",
        "cfcmt_bfull",
    }
)


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
        "top1_accuracy": float(hits / n),
    }


def fit_architecture_matched_suite(
    source_groups: Sequence[ActionGroupRecord],
    target_groups_by_family: Mapping[str, Sequence[ActionGroupRecord]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit compact structural source pools, run 2-fold seed OOF selection, and freeze."""
    placebo_source_groups = permute_source_groups_placebo(
        source_groups, rng_seed=20261118
    )

    source_pool_rankers: dict[str, EndpointAntisymmetricActionRanker] = {}
    placebo_pool_rankers: dict[str, EndpointAntisymmetricActionRanker] = {}
    for pool_name, doms in SOURCE_POOLS.items():
        dom_set = set(doms)
        weights = {d: 1.0 / len(doms) for d in doms}
        source_pool_rankers[pool_name] = EndpointAntisymmetricActionRanker(
            l2_regularization=15.0, random_state=42
        ).fit([g for g in source_groups if g.domain in dom_set], domain_weights=weights)
        placebo_pool_rankers[pool_name] = EndpointAntisymmetricActionRanker(
            l2_regularization=15.0, random_state=42
        ).fit(
            [g for g in placebo_source_groups if g.domain in dom_set],
            domain_weights=weights,
        )

    frozen_models: dict[str, dict[str, Any]] = {}
    fit_audit: dict[str, Any] = {}

    for fam, full_t_groups in target_groups_by_family.items():
        fam_models: dict[str, Any] = {
            "zero_shot_cfcmt_b0": BlendedActionRanker(
                target_ranker=None,
                source_ranker=source_pool_rankers["all_sources"],
                omega=1.0,
            )
        }
        fam_audit: dict[str, Any] = {
            "full_target_reference_conservation_gp_regret": _reference_offline_regret(
                full_t_groups
            ),
            "zero_shot_pool_regrets_on_adapt": {
                pname: evaluate_offline_regret(r, full_t_groups)
                for pname, r in source_pool_rankers.items()
            },
            "budgets": {},
        }

        budgets = {
            "b8": subset_target_budget(full_t_groups, budget_per_seed=4),
            "bfull": list(full_t_groups),
        }

        for b_tag, t_sub in budgets.items():
            s0, s1 = ADAPT_SEEDS
            g_s0 = [g for g in t_sub if g.seed == s0]
            g_s1 = [g for g in t_sub if g.seed == s1]

            fold_t0 = EndpointAntisymmetricActionRanker(
                l2_regularization=15.0, random_state=42
            ).fit(g_s0)
            fold_t1 = EndpointAntisymmetricActionRanker(
                l2_regularization=15.0, random_state=42
            ).fit(g_s1)

            val_groups = g_s1 + g_s0
            t_oof_scores = [
                fold_t0.score_group(g.state_vec, g.action_mat) for g in g_s1
            ] + [fold_t1.score_group(g.state_vec, g.action_mat) for g in g_s0]

            oof_target_only = _eval_precomputed_scores(val_groups, t_oof_scores)

            best_reg = float("inf")
            best_pool = "all_sources"
            best_omega = 0.50
            best_metrics: dict[str, float] = oof_target_only
            oof_grid_rows: list[dict[str, Any]] = []

            for pool_name, s_ranker in source_pool_rankers.items():
                p_scores = [
                    s_ranker.score_group(g.state_vec, g.action_mat) for g in val_groups
                ]
                for omega in OMEGA_GRID:
                    comb = [
                        (1.0 - omega) * ts + omega * ps
                        for ts, ps in zip(t_oof_scores, p_scores)
                    ]
                    m = _eval_precomputed_scores(val_groups, comb)
                    oof_grid_rows.append(
                        {"source_pool": pool_name, "omega": omega, **m}
                    )
                    if m["mean_normalized_regret"] < best_reg - 1e-9:
                        best_reg = m["mean_normalized_regret"]
                        best_pool = pool_name
                        best_omega = omega
                        best_metrics = m

            placebo_omega = best_omega if best_omega > 0.0 else 0.50
            plac_scores = [
                placebo_pool_rankers[best_pool].score_group(g.state_vec, g.action_mat)
                for g in val_groups
            ]
            oof_placebo = _eval_precomputed_scores(
                val_groups,
                [
                    (1.0 - placebo_omega) * ts + placebo_omega * ps
                    for ts, ps in zip(t_oof_scores, plac_scores)
                ],
            )

            final_target_ranker = EndpointAntisymmetricActionRanker(
                l2_regularization=15.0, random_state=42
            ).fit(t_sub)
            final_cfcmt = BlendedActionRanker(
                target_ranker=final_target_ranker,
                source_ranker=source_pool_rankers[best_pool],
                omega=best_omega,
            )
            final_placebo = BlendedActionRanker(
                target_ranker=final_target_ranker,
                source_ranker=placebo_pool_rankers[best_pool],
                omega=placebo_omega,
            )

            fam_models[f"target_only_{b_tag}"] = final_target_ranker
            fam_models[f"source_placebo_{b_tag}"] = final_placebo
            fam_models[f"cfcmt_{b_tag}"] = final_cfcmt

            audit_entry: dict[str, Any] = {
                "target_group_count": len(t_sub),
                "selected_source_pool": best_pool,
                "selected_omega": best_omega,
                "placebo_omega": placebo_omega,
                "oof_target_only": oof_target_only,
                "oof_selected_cfcmt": best_metrics,
                "oof_matched_placebo": oof_placebo,
                "oof_grid": oof_grid_rows,
            }

            if b_tag == "bfull":
                # Naive Pool-All + Target ablation (uniform all_sources with OOF positive omega)
                pool_scores_all = [
                    source_pool_rankers["all_sources"].score_group(
                        g.state_vec, g.action_mat
                    )
                    for g in val_groups
                ]
                best_pool_om = 0.50
                best_pool_m: dict[str, float] = {}
                best_pool_reg = float("inf")
                for om_p in (0.25, 0.50, 0.75, 0.90, 1.0):
                    pm = _eval_precomputed_scores(
                        val_groups,
                        [
                            (1.0 - om_p) * ts + om_p * ps
                            for ts, ps in zip(t_oof_scores, pool_scores_all)
                        ],
                    )
                    if pm["mean_normalized_regret"] < best_pool_reg - 1e-9:
                        best_pool_reg = pm["mean_normalized_regret"]
                        best_pool_om = om_p
                        best_pool_m = pm
                fam_models["pool_all_bfull"] = BlendedActionRanker(
                    target_ranker=final_target_ranker,
                    source_ranker=source_pool_rankers["all_sources"],
                    omega=best_pool_om,
                )
                audit_entry["pool_all_omega"] = best_pool_om
                audit_entry["oof_pool_all_bfull"] = best_pool_m

            fam_audit["budgets"][b_tag] = audit_entry

        frozen_models[fam] = fam_models
        fit_audit[fam] = fam_audit

    return frozen_models, fit_audit


def _combine_two_fold_regret(
    m0: Any, val0: Sequence[ActionGroupRecord], m1: Any, val1: Sequence[ActionGroupRecord]
) -> dict[str, float]:
    r0 = evaluate_offline_regret(m0, val0)
    r1 = evaluate_offline_regret(m1, val1)
    n0 = r0["group_count"]
    n1 = r1["group_count"]
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


def eval_closed_loop_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Run one 3,600-second closed-loop evaluation cell."""
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
    models_pkl = Path(task["models_pkl"]) if task.get("models_pkl") else None

    learned_ranker = None
    if models_pkl is not None and policy in LEARNED_POLICIES:
        with models_pkl.open("rb") as fh:
            all_models = pickle.load(fh)
        learned_ranker = all_models[family][policy]

    sumo_api = load_libsumo()
    with tempfile.TemporaryDirectory(prefix="cfcmt_cv_p1_") as tmp_dir:
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
        disagree_vs_cons_gp = 0
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
                        elif learned_ranker is not None and observer is not None:
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
                                scores = learned_ranker.score_group(s_vec, a_mat)
                                target_phase = feasible[int(np.argmin(scores))]
                                ref_p = select_observed_pressure_phase(
                                    info=info,
                                    executor=ex,
                                    observer=observer,
                                    mode="conservation_gp",
                                )
                                if target_phase != ref_p:
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
                    ending_teleports += int(
                        sumo_api.simulation.getEndingTeleportNumber()
                    )
                    cols = tuple(sumo_api.simulation.getCollisions())
                    collision_events += len(cols)
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
        "collision_incidents": int(len(collision_incident_keys)),
        "starting_teleports": int(starting_teleports),
    }


def compute_paired_comparison(
    rows: Sequence[Mapping[str, Any]],
    candidate_policy: str,
    baseline_policy: str,
    n_boot: int = 20000,
) -> dict[str, Any]:
    """Compute paired seed-level % waiting-time delta, bootstrap 95% CI, & exact sign-flip p."""
    cand_map: dict[tuple[str, int], float] = {}
    base_map: dict[tuple[str, int], float] = {}
    for r in rows:
        key = (str(r["scenario"]), int(r["seed"]))
        if r["policy"] == candidate_policy:
            cand_map[key] = float(r["mean_waiting_sec"])
        elif r["policy"] == baseline_policy:
            base_map[key] = float(r["mean_waiting_sec"])

    common_keys = sorted(set(cand_map) & set(base_map))
    # Aggregate per seed across scenarios in the family so seeds are independent units
    by_seed_cand: dict[int, list[float]] = {}
    by_seed_base: dict[int, list[float]] = {}
    cell_wins = 0
    for scen, seed in common_keys:
        c_val = cand_map[(scen, seed)]
        b_val = base_map[(scen, seed)]
        if c_val < b_val:
            cell_wins += 1
        by_seed_cand.setdefault(seed, []).append(c_val)
        by_seed_base.setdefault(seed, []).append(b_val)

    seeds = sorted(by_seed_cand)
    rel_pcts = np.array(
        [
            100.0
            * (float(np.mean(by_seed_cand[s])) - float(np.mean(by_seed_base[s])))
            / max(float(np.mean(by_seed_base[s])), 1e-9)
            for s in seeds
        ],
        dtype=np.float64,
    )
    abs_diffs = np.array(
        [
            float(np.mean(by_seed_cand[s])) - float(np.mean(by_seed_base[s]))
            for s in seeds
        ],
        dtype=np.float64,
    )

    rng = np.random.default_rng(20261118)
    n_s = len(rel_pcts)
    boot_idx = rng.integers(0, n_s, size=(n_boot, n_s))
    boot_means = np.mean(rel_pcts[boot_idx], axis=1)
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Exact two-sided paired sign-flip test over all 2^n_s sign assignments (exact for n_s=10)
    obs = abs(float(np.mean(rel_pcts)))
    if n_s <= 16:
        signs = (
            ((np.arange(1 << n_s)[:, None] >> np.arange(n_s)) & 1) * 2.0 - 1.0
        )
        perm_means = np.abs(np.mean(signs * rel_pcts[None, :], axis=1))
        p_val = float(np.mean(perm_means >= obs - 1e-12))
    else:
        rand_signs = rng.choice([-1.0, 1.0], size=(n_boot, n_s))
        perm_means = np.abs(np.mean(rand_signs * rel_pcts[None, :], axis=1))
        p_val = float((1 + np.sum(perm_means >= obs - 1e-12)) / (n_boot + 1))

    return {
        "n_seeds": n_s,
        "n_cells": len(common_keys),
        "seed_wins": int(np.sum(rel_pcts < 0.0)),
        "cell_wins": cell_wins,
        "mean_rel_pct": float(np.mean(rel_pcts)),
        "ci95_rel_pct": [float(ci_low), float(ci_high)],
        "mean_abs_sec": float(np.mean(abs_diffs)),
        "exact_signflip_p_two_sided": p_val,
    }


def summarize_eval_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
            "cfcmt_b8",
            "cfcmt_bfull",
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
                "target_only_b8",
                "source_placebo_b8",
                "target_only_bfull",
                "source_placebo_bfull",
                "pool_all_bfull",
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
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    scenarios = resolve_all_scenarios(
        args.staged_scenarios_dir, args.source_scenarios_dir
    )
    dataset_json = args.work_dir / "counterfactual_dataset.json"
    models_pkl = args.work_dir / "frozen_models.pkl"
    fit_audit_json = args.work_dir / "offline_fit_audit.json"

    if args.stage == "collect_and_fit":
        src_groups, tgt_groups, col_audit = collect_counterfactual_dataset(
            scenarios, dataset_json, workers=args.workers
        )
        frozen_models, fit_audit = fit_architecture_matched_suite(
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
        print(f"SAVED_FIT_AUDIT {fit_audit_json}", flush=True)
        return

    if not models_pkl.is_file():
        raise FileNotFoundError(
            f"Missing frozen models {models_pkl}; run --stage collect_and_fit first"
        )

    seeds = ADAPT_SEEDS if args.stage == "dev_check" else FRESH_EVAL_SEEDS
    policies = (
        "oracle_gp",
        "fixed_program",
        "observed_mp_10",
        "observed_pp_10",
        "observed_gp_10",
        "shockwave_gp_10",
        "conservation_gp_10",
        "zero_shot_cfcmt_b0",
        "target_only_b8",
        "source_placebo_b8",
        "cfcmt_b8",
        "target_only_bfull",
        "source_placebo_bfull",
        "pool_all_bfull",
        "cfcmt_bfull",
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
        f"[{args.stage}] Running {len(eval_tasks)} closed-loop evaluation cells across {len(seeds)} seeds...",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(eval_closed_loop_worker, t) for t in eval_tasks]
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
    summary = summarize_eval_rows(rows)
    out_file = args.work_dir / f"phase1_{args.stage}_result.json"
    out_file.write_text(
        json.dumps(
            {
                "stage": args.stage,
                "p_cv": P_CV,
                "seeds": list(seeds),
                **summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"SAVED_EVAL {out_file}", flush=True)


if __name__ == "__main__":
    main()
