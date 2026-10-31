"""Audit deterministic SUMO rollouts under fresh-process isolation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.sumo_runtime import load_libsumo as _load_libsumo
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import evaluate_policy_v3
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import _map_fresh_sumo_processes
from cf_h2o.eval.traffic_signal_resco_phase_benchmark import DEFAULT_ENV_ROOT, _scenario_sumocfg


AUDITED_FIELDS = (
    "mean_queue",
    "p90_queue",
    "departed",
    "arrived",
    "throughput_ratio",
    "phase_execution_audit",
)


def _probe_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario = str(payload["scenario"])
    repeat = int(payload["repeat"])
    metrics = evaluate_policy_v3(
        sumo_api=_load_libsumo(),
        sumocfg=_scenario_sumocfg(Path(payload["env_root"]), scenario),
        scenario=scenario,
        policy=str(payload["policy"]),
        models=None,
        duration_sec=float(payload["duration_sec"]),
        control_interval_sec=int(payload["control_interval_sec"]),
        warmup_sec=float(payload["warmup_sec"]),
        seed=int(payload["seed"]),
        tripinfo_output=Path(payload["tripinfo_root"]) / f"repeat_{repeat}.xml",
    )
    return {"repeat": repeat, "metrics": metrics}


def run_probe(
    *,
    env_root: Path = DEFAULT_ENV_ROOT,
    scenario: str = "cologne1",
    policy: str = "phase_pressure",
    seed: int = 131,
    repetitions: int = 4,
    duration_sec: float = 600.0,
    control_interval_sec: int = 10,
    warmup_sec: float = 60.0,
    workers: int = 4,
) -> dict[str, Any]:
    jobs = [
        {
            "env_root": str(env_root),
            "scenario": scenario,
            "policy": policy,
            "seed": int(seed),
            "repeat": repeat,
            "duration_sec": float(duration_sec),
            "control_interval_sec": int(control_interval_sec),
            "warmup_sec": float(warmup_sec),
            "tripinfo_root": str(
                Path("cf_h2o/results/tripinfo/traffic_signal_reproducibility_probe")
                / scenario
                / policy
                / f"seed_{seed}"
            ),
        }
        for repeat in range(int(repetitions))
    ]
    rows = sorted(_map_fresh_sumo_processes(_probe_worker, jobs, workers), key=lambda row: row["repeat"])
    failures = [row for row in rows if not row["metrics"].get("ok")]
    if failures:
        raise RuntimeError(f"reproducibility probe rollout failed: {failures}")
    signatures = [
        {field: row["metrics"][field] for field in AUDITED_FIELDS}
        for row in rows
    ]
    deterministic = all(signature == signatures[0] for signature in signatures[1:])
    return {
        "experiment": "traffic_signal_fresh_process_reproducibility_probe",
        "scenario": scenario,
        "policy": policy,
        "seed": int(seed),
        "repetitions": int(repetitions),
        "process_isolation": "spawn_maxtasksperchild_1",
        "audited_fields": list(AUDITED_FIELDS),
        "deterministic": deterministic,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--scenario", default="cologne1")
    parser.add_argument("--policy", default="phase_pressure")
    parser.add_argument("--seed", type=int, default=131)
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("cf_h2o/results/traffic_signal_reproducibility_probe.json"),
    )
    args = parser.parse_args()
    result = run_probe(
        env_root=args.env_root,
        scenario=args.scenario,
        policy=args.policy,
        seed=args.seed,
        repetitions=args.repetitions,
        duration_sec=args.duration_sec,
        control_interval_sec=args.control_interval_sec,
        warmup_sec=args.warmup_sec,
        workers=args.workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(json.dumps({"deterministic": result["deterministic"], "signatures": [
        {field: row["metrics"][field] for field in AUDITED_FIELDS}
        for row in result["rows"]
    ]}, indent=2, sort_keys=True))
    if not result["deterministic"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
