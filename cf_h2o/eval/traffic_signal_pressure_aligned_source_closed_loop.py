"""Closed-loop V150M pressure-aligned source contribution rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_pressure_aligned_source_utility import (
    RUNTIME_MODEL_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    POLICY_ARMS,
    run_rollout as _run_rollout,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v150m-pressure-aligned-source-closed-loop-rollout-v1"
ORIGINATOR_PROTOCOL = "tsc-v150m-pressure-aligned-source-originator-v1"
RUNTIME_POLICY = "cfcmt_v3_contrast_pressure_aligned_source_utility"


def run_rollout(**kwargs: Any) -> dict[str, Any]:
    return _run_rollout(
        **kwargs,
        expected_runtime_model_protocol=RUNTIME_MODEL_PROTOCOL,
        candidate_score_constraint=PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
        result_protocol=RESULT_PROTOCOL,
        originator_protocol=ORIGINATOR_PROTOCOL,
        scientific_status="seven-city-pressure-aligned-closed-loop-development-rollout",
        runtime_policy=RUNTIME_POLICY,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--city", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=POLICY_ARMS, required=True)
    parser.add_argument("--runtime-model", type=Path)
    parser.add_argument("--runtime-model-sha256")
    parser.add_argument("--duration-sec", type=float, default=3600.0)
    parser.add_argument("--warmup-sec", type=float, default=60.0)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--prediction-horizon-sec", type=int, default=450)
    parser.add_argument("--tripinfo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V150M result: {args.out}")
    result = run_rollout(
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        scenario=str(args.scenario),
        city=str(args.city),
        seed=int(args.seed),
        arm=str(args.arm),
        runtime_model_path=args.runtime_model,
        expected_runtime_model_sha256=args.runtime_model_sha256,
        duration_sec=float(args.duration_sec),
        warmup_sec=float(args.warmup_sec),
        control_interval_sec=int(args.control_interval_sec),
        prediction_horizon_sec=int(args.prediction_horizon_sec),
        tripinfo_path=args.tripinfo,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "DONE",
                "city": result["city"],
                "scenario": result["scenario"],
                "seed": result["seed"],
                "arm": result["arm"],
                "mean_waiting_time": result["metrics"].get(
                    "mean_tripinfo_waiting_time"
                ),
                "zero_incident_observed": result["zero_incident_diagnostic"][
                    "passed"
                ],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
