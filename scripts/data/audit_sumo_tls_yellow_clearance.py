#!/usr/bin/env python3
"""Audit loaded SUMO signal programs for undersized road yellow phases."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


PROTOCOL = "sumo-tls-yellow-clearance-audit-v1"
MINIMUM_YELLOW_SECONDS = 3
LOW_SPEED_BOUND_MPS = 71.0 / 3.6
REFERENCE_SPEED_MPS = 50.0 / 3.6
LOW_SPEED_SLOPE = 0.37
DEFAULT_MINIMUM_DECEL_MPS2 = 3.0


def _tag(element: ET.Element) -> str:
    return str(element.tag).rsplit("}", 1)[-1]


def required_yellow_seconds(
    maximum_incoming_speed_mps: float,
    *,
    minimum_decel_mps2: float = DEFAULT_MINIMUM_DECEL_MPS2,
) -> int:
    """Match SUMO NBTrafficLightDefinition::computeBrakingTime."""

    speed = float(maximum_incoming_speed_mps)
    decel = float(minimum_decel_mps2)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("maximum incoming speed must be finite and nonnegative")
    if not math.isfinite(decel) or decel <= 0.0:
        raise ValueError("minimum deceleration must be finite and positive")
    if speed < LOW_SPEED_BOUND_MPS:
        increment = math.floor((speed - REFERENCE_SPEED_MPS) * LOW_SPEED_SLOPE)
        return MINIMUM_YELLOW_SECONDS + int(max(0.0, increment))
    return int(1.8 + speed / (2.0 * decel))


def _lane_speeds(root: ET.Element) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for edge in root.iter():
        if _tag(edge) != "edge":
            continue
        edge_id = str(edge.get("id", ""))
        for lane in edge:
            if _tag(lane) != "lane":
                continue
            index = str(lane.get("index", ""))
            raw_speed = lane.get("speed")
            if edge_id and index and raw_speed is not None:
                result[(edge_id, index)] = float(raw_speed)
    return result


def _controlled_link_speeds(
    root: ET.Element,
    lane_speeds: Mapping[tuple[str, str], float],
) -> tuple[dict[str, dict[int, float]], list[dict[str, Any]]]:
    result: dict[str, dict[int, float]] = {}
    anomalies: list[dict[str, Any]] = []
    for connection in root.iter():
        if _tag(connection) != "connection" or not connection.get("tl"):
            continue
        tls_id = str(connection.get("tl"))
        try:
            link_index = int(str(connection.get("linkIndex", "")))
        except ValueError:
            anomalies.append(
                {
                    "category": "invalid_link_index",
                    "tls_id": tls_id,
                    "connection": dict(connection.attrib),
                }
            )
            continue
        lane_key = (
            str(connection.get("from", "")),
            str(connection.get("fromLane", "")),
        )
        speed = lane_speeds.get(lane_key)
        if speed is None:
            anomalies.append(
                {
                    "category": "missing_incoming_lane_speed",
                    "tls_id": tls_id,
                    "link_index": link_index,
                    "incoming_edge": lane_key[0],
                    "incoming_lane_index": lane_key[1],
                }
            )
            continue
        links = result.setdefault(tls_id, {})
        links[link_index] = max(float(speed), links.get(link_index, 0.0))
    return result, anomalies


def analyze_yellow_clearance(
    root: ET.Element,
    *,
    minimum_decel_mps2: float = DEFAULT_MINIMUM_DECEL_MPS2,
) -> dict[str, Any]:
    lane_speeds = _lane_speeds(root)
    link_speeds, anomalies = _controlled_link_speeds(root, lane_speeds)
    programs: list[dict[str, Any]] = []
    yellow_phases: list[dict[str, Any]] = []
    program_keys: set[tuple[str, str]] = set()
    tls_ids: set[str] = set()
    for logic in root.iter():
        if _tag(logic) != "tlLogic":
            continue
        tls_id = str(logic.get("id", ""))
        program_id = str(logic.get("programID", ""))
        program_key = (tls_id, program_id)
        if program_key in program_keys:
            anomalies.append(
                {
                    "category": "duplicate_tls_program_key",
                    "tls_id": tls_id,
                    "program_id": program_id,
                }
            )
        program_keys.add(program_key)
        tls_ids.add(tls_id)
        controlled = link_speeds.get(tls_id, {})
        if not controlled:
            anomalies.append(
                {
                    "category": "tls_program_without_resolved_road_links",
                    "tls_id": tls_id,
                    "program_id": program_id,
                }
            )
            continue
        maximum_speed = max(controlled.values())
        required = required_yellow_seconds(
            maximum_speed, minimum_decel_mps2=minimum_decel_mps2
        )
        phase_count = 0
        for phase in logic:
            if _tag(phase) != "phase":
                continue
            phase_index = phase_count
            phase_count += 1
            state = str(phase.get("state", ""))
            invalid_indices = sorted(index for index in controlled if index >= len(state))
            if invalid_indices:
                anomalies.append(
                    {
                        "category": "phase_state_too_short",
                        "tls_id": tls_id,
                        "program_id": program_id,
                        "phase_index": phase_index,
                        "state_width": len(state),
                        "invalid_link_indices": invalid_indices,
                    }
                )
                continue
            yellow_indices = sorted(
                index for index in controlled if state[index] in {"y", "Y"}
            )
            if not yellow_indices:
                continue
            duration = float(phase.get("duration", 0.0) or 0.0)
            record = {
                "tls_id": tls_id,
                "program_id": program_id,
                "phase_index": phase_index,
                "logic_type": str(logic.get("type", "")),
                "state": state,
                "yellow_link_indices": yellow_indices,
                "maximum_incoming_speed_mps": maximum_speed,
                "duration_sec": duration,
                "required_duration_sec": required,
                "shortfall_sec": max(float(required) - duration, 0.0),
                "min_duration_sec": (
                    float(phase.get("minDur")) if phase.get("minDur") is not None else None
                ),
                "max_duration_sec": (
                    float(phase.get("maxDur")) if phase.get("maxDur") is not None else None
                ),
            }
            yellow_phases.append(record)
        programs.append(
            {
                "tls_id": tls_id,
                "program_id": program_id,
                "logic_type": str(logic.get("type", "")),
                "phase_count": phase_count,
                "controlled_road_link_count": len(controlled),
                "maximum_incoming_speed_mps": maximum_speed,
                "required_yellow_duration_sec": required,
            }
        )
    undersized = [row for row in yellow_phases if row["shortfall_sec"] > 1e-9]
    affected_tls = sorted({str(row["tls_id"]) for row in undersized})
    return {
        "counts": {
            "lane_speed_count": len(lane_speeds),
            "traffic_light_count": len(tls_ids),
            "program_count": len(programs),
            "yellow_phase_count": len(yellow_phases),
            "undersized_yellow_phase_count": len(undersized),
            "affected_tls_count": len(affected_tls),
            "anomaly_count": len(anomalies),
        },
        "programs": programs,
        "yellow_phases": yellow_phases,
        "undersized_yellow_phases": undersized,
        "affected_tls_ids": affected_tls,
        "anomalies": anomalies,
    }


def audit_tls_yellow_clearance(
    network: Path,
    *,
    minimum_decel_mps2: float = DEFAULT_MINIMUM_DECEL_MPS2,
    maximum_examples: int = 20,
) -> dict[str, Any]:
    network = Path(network)
    if not network.is_file():
        raise FileNotFoundError(network)
    if maximum_examples < 0:
        raise ValueError("maximum examples must be nonnegative")
    analysis = analyze_yellow_clearance(
        ET.parse(network).getroot(), minimum_decel_mps2=minimum_decel_mps2
    )
    counts = dict(analysis["counts"])
    actionable = int(counts["anomaly_count"]) == 0
    passed = actionable and int(counts["undersized_yellow_phase_count"]) == 0
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "method": {
            "reference": "SUMO_NBTrafficLightDefinition_computeBrakingTime",
            "minimum_decel_mps2": float(minimum_decel_mps2),
            "scope": "road_links_with_yellow_state",
        },
        "counts": counts,
        "affected_tls_ids": analysis["affected_tls_ids"],
        "undersized_yellow_phase_examples": analysis["undersized_yellow_phases"][
            :maximum_examples
        ],
        "anomaly_examples": analysis["anomalies"][:maximum_examples],
        "actionable": actionable,
        "passed": passed,
        "status": "PASS" if passed else ("REQUIRES_REPAIR" if actionable else "FAIL"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument(
        "--minimum-decel-mps2", type=float, default=DEFAULT_MINIMUM_DECEL_MPS2
    )
    parser.add_argument("--maximum-examples", type=int, default=20)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    payload = audit_tls_yellow_clearance(
        args.network,
        minimum_decel_mps2=args.minimum_decel_mps2,
        maximum_examples=args.maximum_examples,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["actionable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
