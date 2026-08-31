"""RESCO SUMO phase-action benchmark.

This stage is the bridge between the synthetic Phase-2 grid and real RESCO
SUMO scenarios.  RESCO intersections do not share the synthetic NS/EW action
assumption, so actions are defined as selecting among the existing green phases
in each SUMO ``tlLogic`` program.

The benchmark currently evaluates phase-level rule baselines only.  It is
intended to validate the phase abstraction before CFCMT/H2O residual policies
are attached to RESCO scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo
from cf_h2o.eval.traffic_signal_resco_probe import (
    DEFAULT_OUT_ROOT as DEFAULT_RESC0_ROOT,
    DEFAULT_REPO_URL,
    prepare_scenarios,
)
from cf_h2o.eval.traffic_signal_transfer_feasibility import _format_table


DEFAULT_ENV_ROOT = DEFAULT_RESC0_ROOT / "environments"
DEFAULT_SCENARIOS = ("cologne1", "cologne3", "cologne8", "ingolstadt1", "ingolstadt7")
DEFAULT_POLICIES = ("fixed_program", "phase_cycle", "phase_pressure", "phase_spillback_pressure")
DEFAULT_OUT = Path("cf_h2o/results/traffic_signal_resco_phase_benchmark.json")
DEFAULT_MD_OUT = Path("cf_h2o/results/traffic_signal_resco_phase_benchmark.md")
GREEN_CHARS = frozenset("Gg")
YELLOW_CHARS = frozenset("yY")
SUMO_EXECUTION_PROTOCOL = {
    "version": "no-teleport-write-unfinished-junction-collision-audit-v2",
    "time_to_teleport_sec": -1,
    "tripinfo_write_unfinished": True,
    "eager_insert": False,
    "collision_action": "warn",
    "collision_check_junctions": True,
}


@dataclass(frozen=True)
class PhaseCandidate:
    phase_index: int
    state: str
    duration: float
    green_count: int


@dataclass(frozen=True)
class TlsPhaseInfo:
    tls_id: str
    controlled_links: Any
    candidates: tuple[PhaseCandidate, ...]
    incoming_lanes: tuple[str, ...]


def _sumocfg_files(scenario_dir: Path) -> list[Path]:
    return sorted(scenario_dir.glob("*.sumocfg"))


def _net_and_route_from_sumocfg(sumocfg: Path) -> tuple[list[str], list[str], tuple[float | None, float | None]]:
    root = ET.parse(sumocfg).getroot()
    net_files: list[str] = []
    route_files: list[str] = []
    begin: float | None = None
    end: float | None = None
    for item in root.findall(".//net-file"):
        value = item.attrib.get("value")
        if value:
            net_files.extend(part.strip() for part in value.split(",") if part.strip())
    for item in root.findall(".//route-files"):
        value = item.attrib.get("value")
        if value:
            route_files.extend(part.strip() for part in value.split(",") if part.strip())
    begin_item = root.find(".//begin")
    end_item = root.find(".//end")
    if begin_item is not None and begin_item.attrib.get("value") is not None:
        begin = float(begin_item.attrib["value"])
    if end_item is not None and end_item.attrib.get("value") is not None:
        end = float(end_item.attrib["value"])
    return net_files, route_files, (begin, end)


def _phase_green_count(state: str) -> int:
    return sum(1 for char in state if char in GREEN_CHARS)


def _green_phase_candidates(phases: Any, controlled_link_count: int) -> tuple[PhaseCandidate, ...]:
    seen: set[str] = set()
    candidates: list[PhaseCandidate] = []
    for idx, phase in enumerate(phases):
        state = str(getattr(phase, "state", ""))
        if len(state) != controlled_link_count:
            continue
        if any(char in YELLOW_CHARS for char in state):
            continue
        green_count = _phase_green_count(state)
        if green_count <= 0 or state in seen:
            continue
        seen.add(state)
        duration = float(getattr(phase, "duration", 0.0) or 0.0)
        candidates.append(
            PhaseCandidate(
                phase_index=int(idx),
                state=state,
                duration=duration,
                green_count=int(green_count),
            )
        )
    return tuple(candidates)


def _tls_phase_infos(sumo_api: Any) -> dict[str, TlsPhaseInfo]:
    infos: dict[str, TlsPhaseInfo] = {}
    for tls_id in sumo_api.trafficlight.getIDList():
        controlled_links = sumo_api.trafficlight.getControlledLinks(tls_id)
        logics = sumo_api.trafficlight.getAllProgramLogics(tls_id)
        if not logics:
            continue
        candidates = _green_phase_candidates(logics[0].phases, len(controlled_links))
        incoming_lanes: list[str] = []
        for link_group in controlled_links:
            if link_group and link_group[0]:
                lane = str(link_group[0][0])
                if lane not in incoming_lanes:
                    incoming_lanes.append(lane)
        if candidates:
            infos[str(tls_id)] = TlsPhaseInfo(
                tls_id=str(tls_id),
                controlled_links=controlled_links,
                candidates=candidates,
                incoming_lanes=tuple(incoming_lanes),
            )
    return infos


def _lane_queue(sumo_api: Any, lane: str) -> float:
    return float(
        sumo_api.lane.getLastStepHaltingNumber(lane)
        + 0.35 * sumo_api.lane.getLastStepVehicleNumber(lane)
        + 0.025 * sumo_api.lane.getLastStepOccupancy(lane)
    )


def _phase_score(sumo_api: Any, info: TlsPhaseInfo, candidate: PhaseCandidate, *, spillback: bool) -> float:
    lane_score: dict[str, float] = {}
    down_occ: dict[str, list[float]] = {}
    for link_idx, char in enumerate(candidate.state):
        if char not in GREEN_CHARS or link_idx >= len(info.controlled_links):
            continue
        link_group = info.controlled_links[link_idx]
        if not link_group or not link_group[0]:
            continue
        in_lane = str(link_group[0][0])
        out_lane = str(link_group[0][1])
        lane_score[in_lane] = max(lane_score.get(in_lane, 0.0), _lane_queue(sumo_api, in_lane))
        try:
            down_occ.setdefault(in_lane, []).append(float(sumo_api.lane.getLastStepOccupancy(out_lane)))
        except Exception:
            pass
    score = 0.0
    for lane, queue in lane_score.items():
        if spillback and down_occ.get(lane):
            block = max(0.20, 1.0 - float(np.mean(down_occ[lane])) / 120.0)
            score += queue * block
        else:
            score += queue
    return float(score)


def _choose_candidate(
    sumo_api: Any,
    info: TlsPhaseInfo,
    *,
    policy: str,
    interval_idx: int,
) -> PhaseCandidate:
    if policy == "phase_cycle":
        return info.candidates[interval_idx % len(info.candidates)]
    spillback = policy == "phase_spillback_pressure"
    return max(info.candidates, key=lambda candidate: _phase_score(sumo_api, info, candidate, spillback=spillback))


def _controlled_lane_set(infos: dict[str, TlsPhaseInfo]) -> tuple[str, ...]:
    lanes: list[str] = []
    for info in infos.values():
        for lane in info.incoming_lanes:
            if lane not in lanes:
                lanes.append(lane)
    return tuple(lanes)


def _system_queue(sumo_api: Any, lanes: tuple[str, ...]) -> float:
    return float(sum(_lane_queue(sumo_api, lane) for lane in lanes))


def _sumo_arguments(
    sumocfg: Path,
    seed: int,
    *,
    include_executable: bool,
    tripinfo_output: Path | None = None,
    load_state: Path | None = None,
    begin_time: float | None = None,
) -> list[str]:
    args = [
        "-c",
        str(sumocfg),
        "--seed",
        str(seed),
        "--no-step-log",
        "true",
        "--no-warnings",
        "true",
        "--duration-log.disable",
        "true",
        "--time-to-teleport",
        str(SUMO_EXECUTION_PROTOCOL["time_to_teleport_sec"]),
        "--collision.action",
        str(SUMO_EXECUTION_PROTOCOL["collision_action"]),
        "--collision.check-junctions",
        str(SUMO_EXECUTION_PROTOCOL["collision_check_junctions"]).lower(),
        "--save-state.rng",
        "true",
        "--save-state.precision",
        "8",
        "--thread-rngs",
        "1",
    ]
    if include_executable:
        args.insert(0, "sumo")
    if load_state is not None:
        args.extend(["--load-state", str(load_state)])
    if begin_time is not None:
        args.extend(["--begin", f"{float(begin_time):.8f}"])
    if tripinfo_output is not None:
        tripinfo_output.parent.mkdir(parents=True, exist_ok=True)
        tripinfo_output.unlink(missing_ok=True)
        args.extend(
            [
                "--tripinfo-output",
                str(tripinfo_output),
                "--tripinfo-output.write-unfinished",
                str(SUMO_EXECUTION_PROTOCOL["tripinfo_write_unfinished"]).lower(),
            ]
        )
    return args


def _start_sumo(sumo_api: Any, sumocfg: Path, seed: int, *, tripinfo_output: Path | None = None) -> None:
    try:
        sumo_api.close()
    except Exception:
        pass
    sumo_api.start(
        _sumo_arguments(
            sumocfg,
            seed,
            include_executable=True,
            tripinfo_output=tripinfo_output,
        )
    )


def _reload_sumo_state(
    sumo_api: Any,
    sumocfg: Path,
    seed: int,
    state_path: Path,
    *,
    begin_time: float,
) -> None:
    """Reload the network and state so unsaved vehicle-model internals are reset."""

    sumo_api.load(
        _sumo_arguments(
            sumocfg,
            seed,
            include_executable=False,
            load_state=state_path,
            begin_time=begin_time,
        )
    )


def _scenario_sumocfg(env_root: Path, scenario: str) -> Path:
    scenario_dir = env_root / scenario
    files = _sumocfg_files(scenario_dir)
    if not files:
        raise RuntimeError(f"missing sumocfg for RESCO scenario {scenario}: {scenario_dir}")
    return files[0]


def _evaluate_scenario_policy(
    *,
    sumo_api: Any,
    sumocfg: Path,
    policy: str,
    duration_sec: float,
    control_interval_sec: int,
    warmup_sec: float,
    seed: int,
) -> dict[str, Any]:
    _start_sumo(sumo_api, sumocfg, seed)
    infos = _tls_phase_infos(sumo_api)
    lanes = _controlled_lane_set(infos)
    begin_time = float(sumo_api.simulation.getTime())
    end_time = begin_time + float(duration_sec)
    interval_idx = 0
    queues: list[float] = []
    departed = 0
    arrived = 0
    action_counts = {tls_id: 0 for tls_id in infos}
    try:
        while float(sumo_api.simulation.getTime()) < end_time and int(sumo_api.simulation.getMinExpectedNumber()) > 0:
            if policy != "fixed_program":
                for tls_id, info in infos.items():
                    candidate = _choose_candidate(sumo_api, info, policy=policy, interval_idx=interval_idx)
                    sumo_api.trafficlight.setRedYellowGreenState(tls_id, candidate.state)
                    action_counts[tls_id] += 1
            for _ in range(int(control_interval_sec)):
                sumo_api.simulationStep()
                departed += int(sumo_api.simulation.getDepartedNumber())
                arrived += int(sumo_api.simulation.getArrivedNumber())
                if float(sumo_api.simulation.getTime()) >= begin_time + float(warmup_sec):
                    queues.append(_system_queue(sumo_api, lanes))
                if float(sumo_api.simulation.getTime()) >= end_time:
                    break
            interval_idx += 1
        return {
            "ok": True,
            "policy": policy,
            "begin_time": begin_time,
            "end_time": float(sumo_api.simulation.getTime()),
            "tls_count": len(infos),
            "controlled_lane_count": len(lanes),
            "mean_queue": float(np.mean(queues)) if queues else float("nan"),
            "p90_queue": float(np.quantile(queues, 0.90)) if queues else float("nan"),
            "departed": int(departed),
            "arrived": int(arrived),
            "remaining": int(sumo_api.vehicle.getIDCount()),
            "throughput_ratio": float(arrived / max(departed, 1)),
            "intervals": int(interval_idx),
            "action_counts": action_counts,
        }
    except Exception as exc:
        return {"ok": False, "policy": policy, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass


def _phase_summary_for_scenario(sumo_api: Any, sumocfg: Path) -> dict[str, Any]:
    _start_sumo(sumo_api, sumocfg, seed=0)
    try:
        infos = _tls_phase_infos(sumo_api)
        return {
            "tls_count": len(infos),
            "phase_candidate_count": {tls_id: len(info.candidates) for tls_id, info in infos.items()},
            "controlled_lane_count": len(_controlled_lane_set(infos)),
        }
    finally:
        try:
            sumo_api.close()
        except Exception:
            pass


def _ensure_env_root(env_root: Path, scenarios: tuple[str, ...], *, prepare: bool, repo_url: str) -> None:
    missing = [scenario for scenario in scenarios if not (env_root / scenario).exists()]
    if prepare or missing:
        prepare_scenarios(repo_url=repo_url, out_root=env_root.parent, scenarios=scenarios)


def run_benchmark(
    *,
    env_root: Path = DEFAULT_ENV_ROOT,
    scenarios: tuple[str, ...] = DEFAULT_SCENARIOS,
    policies: tuple[str, ...] = DEFAULT_POLICIES,
    duration_sec: float = 600.0,
    control_interval_sec: int = 10,
    warmup_sec: float = 60.0,
    seed: int = 101,
    prepare: bool = False,
    repo_url: str = DEFAULT_REPO_URL,
) -> dict[str, Any]:
    _ensure_env_root(env_root, scenarios, prepare=prepare, repo_url=repo_url)
    sumo_api = _load_libsumo()
    targets: list[dict[str, Any]] = []
    for scenario_idx, scenario in enumerate(scenarios):
        sumocfg = _scenario_sumocfg(env_root, scenario)
        phase_summary = _phase_summary_for_scenario(sumo_api, sumocfg)
        policy_metrics: dict[str, Any] = {}
        for policy_idx, policy in enumerate(policies):
            policy_metrics[policy] = _evaluate_scenario_policy(
                sumo_api=sumo_api,
                sumocfg=sumocfg,
                policy=policy,
                duration_sec=duration_sec,
                control_interval_sec=control_interval_sec,
                warmup_sec=warmup_sec,
                seed=seed + scenario_idx * 1009 + policy_idx * 37,
            )
        targets.append(
            {
                "scenario": scenario,
                "sumocfg": str(sumocfg),
                **phase_summary,
                "policy_metrics": policy_metrics,
            }
        )
    aggregate: dict[str, Any] = {}
    for policy in policies:
        rows = [target["policy_metrics"][policy] for target in targets if target["policy_metrics"][policy].get("ok")]
        aggregate[policy] = {
            "mean_queue": float(np.mean([row["mean_queue"] for row in rows])) if rows else float("nan"),
            "p90_queue": float(np.mean([row["p90_queue"] for row in rows])) if rows else float("nan"),
            "throughput_ratio": float(np.mean([row["throughput_ratio"] for row in rows])) if rows else float("nan"),
        }
    return {
        "experiment": "traffic_signal_resco_phase_benchmark",
        "setting": {
            "env_root": str(env_root),
            "scenarios": list(scenarios),
            "policies": list(policies),
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "seed": int(seed),
        },
        "aggregate": aggregate,
        "targets": targets,
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    aggregate_rows = [
        {
            "policy": policy,
            "mean_queue": metrics["mean_queue"],
            "p90_queue": metrics["p90_queue"],
            "throughput_ratio": metrics["throughput_ratio"],
        }
        for policy, metrics in sorted(result["aggregate"].items(), key=lambda item: item[1]["mean_queue"])
    ]
    target_rows: list[dict[str, Any]] = []
    for target in result["targets"]:
        best = min(
            (
                (policy, metrics)
                for policy, metrics in target["policy_metrics"].items()
                if metrics.get("ok") and np.isfinite(metrics.get("mean_queue", float("nan")))
            ),
            key=lambda item: item[1]["mean_queue"],
        )
        row = {
            "scenario": target["scenario"],
            "tls_count": target["tls_count"],
            "controlled_lanes": target["controlled_lane_count"],
            "best_policy": best[0],
            "best_mean_queue": best[1]["mean_queue"],
        }
        for policy in result["setting"]["policies"]:
            row[policy] = target["policy_metrics"].get(policy, {}).get("mean_queue", float("nan"))
        target_rows.append(row)

    lines = [
        "# RESCO Phase-Action Benchmark",
        "",
        "Actions select among existing green phases from each RESCO SUMO `tlLogic`; no synthetic NS/EW phase assumption is used.",
        "",
        "## Aggregate Metrics",
        "",
        _format_table(aggregate_rows, ["policy", "mean_queue", "p90_queue", "throughput_ratio"]),
        "",
        "## Scenario Metrics",
        "",
        _format_table(
            target_rows,
            ["scenario", "tls_count", "controlled_lanes", "best_policy", "best_mean_queue", *result["setting"]["policies"]],
        ),
        "",
        "## Notes",
        "",
        "- `fixed_program` leaves the original RESCO static signal programs untouched.",
        "- `phase_pressure` and `phase_spillback_pressure` overwrite signal states every control interval using one of the existing non-yellow green phases.",
        "- This validates arbitrary RESCO phase actions; CFCMT/H2O residual policies can now be attached to this phase abstraction.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--scenarios", type=_parse_csv, default=DEFAULT_SCENARIOS)
    parser.add_argument("--policies", type=_parse_csv, default=DEFAULT_POLICIES)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/traffic_signal_resco_phase_benchmark.json"))
    parser.add_argument("--md-out", type=Path, default=Path("cf_h2o/results/traffic_signal_resco_phase_benchmark.md"))
    args = parser.parse_args(argv)

    result = run_benchmark(
        env_root=args.env_root,
        scenarios=args.scenarios,
        policies=args.policies,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        seed=args.seed,
        prepare=args.prepare,
        repo_url=args.repo_url,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(result, args.md_out)
    best = min(result["aggregate"].items(), key=lambda item: item[1]["mean_queue"])
    print(f"wrote {args.out}")
    print(f"wrote {args.md_out}")
    print(f"best_policy={best[0]} mean_queue={best[1]['mean_queue']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
