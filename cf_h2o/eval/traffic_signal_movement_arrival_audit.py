"""Audit candidate-specific static arrival waves without launching SUMO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from cf_h2o.traffic_signal.demand_schedule_context import _resolve_cfg_paths
from cf_h2o.traffic_signal.movement_arrival_timeline import (
    build_movement_arrival_timeline_context,
)


AUDIT_PROTOCOL = "tsc-v112-static-movement-arrival-gate-a-v1"


def _source_sha256() -> str:
    paths = (
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1]
        / "traffic_signal"
        / "movement_arrival_timeline.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _candidate_states(net_file: Path) -> dict[str, tuple[str, ...]]:
    root = ET.parse(net_file).getroot()
    result: dict[str, tuple[str, ...]] = {}
    for logic in root.findall(".//tlLogic"):
        values: list[str] = []
        for phase in logic.findall("phase"):
            state = str(phase.attrib.get("state", ""))
            if any(value in "gG" for value in state) and state not in values:
                values.append(state)
        result[str(logic.attrib["id"])] = tuple(values)
    return result


def audit_sumocfg(
    sumocfg: Path,
    *,
    horizon_sec: int = 3600,
    bin_sec: int = 5,
    sample_interval_sec: int = 15,
) -> dict[str, Any]:
    sumocfg = Path(sumocfg).resolve()
    context = build_movement_arrival_timeline_context(
        sumocfg, horizon_sec=horizon_sec, bin_sec=bin_sec
    )
    net_files = _resolve_cfg_paths(sumocfg, "net-file")
    if len(net_files) != 1:
        raise ValueError("movement arrival audit requires exactly one net file")
    states = _candidate_states(net_files[0])
    comparisons = 0
    discriminated = 0
    nonzero = 0
    invalid: list[dict[str, Any]] = []
    max_feature_spread = 0.0
    start = int(context.begin_sec)
    stop = start + int(horizon_sec)
    for time_sec in range(start, stop, int(sample_interval_sec)):
        for tls_id, candidates in states.items():
            if tls_id not in context.movement_arrivals or len(candidates) < 2:
                continue
            try:
                vectors = [
                    context.vector_at(tls_id, time_sec, candidate)
                    for candidate in candidates
                ]
            except (KeyError, ValueError) as exc:
                invalid.append(
                    {"tls_id": tls_id, "time_sec": time_sec, "error": str(exc)}
                )
                continue
            comparisons += 1
            matrix = np.vstack(vectors)
            spread = float(np.max(np.ptp(matrix, axis=0)))
            max_feature_spread = max(max_feature_spread, spread)
            nonzero += int(bool(np.any(np.abs(matrix) > 0.0)))
            discriminated += int(spread > 1e-12)
    projected_fraction = context.projected_elements / max(
        context.scheduled_elements, 1
    )
    passed = bool(
        context.projected_arrival_mass > 0.0
        and context.projected_elements > 0
        and comparisons > 0
        and discriminated > 0
        and not invalid
    )
    return {
        **context.to_dict(),
        "sumocfg": str(sumocfg),
        "projection_element_fraction": float(projected_fraction),
        "candidate_signal_time_comparisons": comparisons,
        "candidate_discriminated_count": discriminated,
        "candidate_discriminated_fraction": discriminated / max(comparisons, 1),
        "candidate_nonzero_count": nonzero,
        "candidate_nonzero_fraction": nonzero / max(comparisons, 1),
        "max_feature_spread": max_feature_spread,
        "invalid_candidate_count": len(invalid),
        "invalid_candidate_samples": invalid[:10],
        "passed": passed,
    }


def run_audit(
    sumocfgs: Sequence[Path],
    *,
    horizon_sec: int = 3600,
    bin_sec: int = 5,
    sample_interval_sec: int = 15,
) -> dict[str, Any]:
    rows = {
        Path(path).resolve().parent.name: audit_sumocfg(
            path,
            horizon_sec=horizon_sec,
            bin_sec=bin_sec,
            sample_interval_sec=sample_interval_sec,
        )
        for path in sorted((Path(value) for value in sumocfgs), key=str)
    }
    return {
        "protocol": AUDIT_PROTOCOL,
        "scientific_status": "disclosed_post-v111_development_gate_a",
        "source_sha256": _source_sha256(),
        "horizon_sec": int(horizon_sec),
        "bin_sec": int(bin_sec),
        "sample_interval_sec": int(sample_interval_sec),
        "scenario_count": len(rows),
        "rows": rows,
        "passed": bool(rows and all(row["passed"] for row in rows.values())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sumocfg", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--horizon-sec", type=int, default=3600)
    parser.add_argument("--bin-sec", type=int, default=5)
    parser.add_argument("--sample-interval-sec", type=int, default=15)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_audit(
        args.sumocfg,
        horizon_sec=args.horizon_sec,
        bin_sec=args.bin_sec,
        sample_interval_sec=args.sample_interval_sec,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(args.out), "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
