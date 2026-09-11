#!/usr/bin/env python3
"""Audit visibility encoding for minor SUMO connections without simulation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET


PROTOCOL = "sumo-minor-connection-visibility-audit-v1"
SUMO_DEFAULT_MINOR_VISIBILITY_M = 4.5


def _connection_identity(connection: ET.Element) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in sorted(connection.attrib.items())
        if key
        in {
            "dir",
            "from",
            "fromLane",
            "state",
            "to",
            "toLane",
            "uncontrolled",
            "via",
            "visibility",
        }
    }


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)

    def value(fraction: float) -> float:
        index = int(round(fraction * (len(ordered) - 1)))
        return float(ordered[index])

    return {
        "minimum": ordered[0],
        "p25": value(0.25),
        "median": value(0.5),
        "p75": value(0.75),
        "p90": value(0.9),
        "p95": value(0.95),
        "p99": value(0.99),
        "maximum": ordered[-1],
    }


def audit_minor_visibility(
    network: Path,
    *,
    focus_connection: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    network = Path(network).resolve()
    if not network.is_file():
        raise FileNotFoundError(network)
    root = ET.parse(network).getroot()
    lane_speeds: dict[tuple[str, str], float] = {}
    internal_edges = 0
    normal_edges = 0
    for edge in root.iter("edge"):
        edge_id = str(edge.get("id", ""))
        if str(edge.get("function", "")) == "internal" or edge_id.startswith(":"):
            internal_edges += 1
        else:
            normal_edges += 1
        for offset, lane in enumerate(edge.findall("lane")):
            index = str(lane.get("index", offset))
            if lane.get("speed") is not None:
                lane_speeds[(edge_id, index)] = float(str(lane.get("speed")))

    minor_rows: list[dict[str, Any]] = []
    focus_rows: list[dict[str, Any]] = []
    state_counts: Counter[str] = Counter()
    for connection in root.iter("connection"):
        state = str(connection.get("state", ""))
        state_counts[state] += 1
        if state not in {"m", "g", "s", "y"}:
            continue
        explicit = connection.get("visibility") is not None
        visibility = float(
            str(connection.get("visibility", SUMO_DEFAULT_MINOR_VISIBILITY_M))
        )
        speed = lane_speeds.get(
            (str(connection.get("from", "")), str(connection.get("fromLane", "")))
        )
        row = {
            "connection": _connection_identity(connection),
            "state": state,
            "direction": str(connection.get("dir", "")),
            "visibility_m": visibility,
            "visibility_explicit": explicit,
            "incoming_lane_speed_mps": speed,
            "has_internal_via": connection.get("via") not in {None, ""},
        }
        minor_rows.append(row)
        if focus_connection is not None and all(
            str(connection.get(key, "")) == str(value)
            for key, value in focus_connection.items()
        ):
            focus_rows.append(row)
    if focus_connection is not None and len(focus_rows) != 1:
        raise ValueError(
            f"focus minor connection match count changed: {len(focus_rows)}"
        )

    explicit_values = [
        float(row["visibility_m"])
        for row in minor_rows
        if row["visibility_explicit"]
    ]
    effective_values = [float(row["visibility_m"]) for row in minor_rows]
    histogram = Counter(f"{value:.2f}" for value in explicit_values)
    focus = focus_rows[0] if focus_rows else None
    comparable = []
    if focus is not None:
        focus_speed = focus["incoming_lane_speed_mps"]
        comparable = [
            row
            for row in minor_rows
            if row["state"] == focus["state"]
            and row["direction"] == focus["direction"]
            and row["incoming_lane_speed_mps"] == focus_speed
        ]
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "network": str(network),
        "counts": {
            "normal_edge_count": normal_edges,
            "internal_edge_count": internal_edges,
            "connection_count": sum(state_counts.values()),
            "minor_or_yield_connection_count": len(minor_rows),
            "explicit_visibility_count": len(explicit_values),
            "default_visibility_count": len(minor_rows) - len(explicit_values),
            "minor_with_internal_via_count": sum(
                bool(row["has_internal_via"]) for row in minor_rows
            ),
        },
        "connection_state_counts": dict(sorted(state_counts.items())),
        "explicit_visibility_histogram_m": dict(sorted(histogram.items())),
        "explicit_visibility_quantiles_m": _quantiles(explicit_values),
        "effective_visibility_quantiles_m": _quantiles(effective_values),
        "focus_connection": focus,
        "focus_comparison": (
            None
            if focus is None
            else {
                "same_state_direction_speed_count": len(comparable),
                "same_visibility_count": sum(
                    float(row["visibility_m"]) == float(focus["visibility_m"])
                    for row in comparable
                ),
                "same_visibility_fraction": float(
                    sum(
                        float(row["visibility_m"])
                        == float(focus["visibility_m"])
                        for row in comparable
                    )
                    / max(len(comparable), 1)
                ),
                "visibility_quantiles_m": _quantiles(
                    [float(row["visibility_m"]) for row in comparable]
                ),
            }
        ),
        "audit_boundary": (
            "Visibility describes junction decision distance, not stopping sight "
            "distance. This audit tests whether the focus encoding is exceptional; "
            "it does not declare a safe visibility threshold."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--focus-from", required=True)
    parser.add_argument("--focus-from-lane", required=True)
    parser.add_argument("--focus-to", required=True)
    parser.add_argument("--focus-to-lane", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.out}")
    result = audit_minor_visibility(
        args.network,
        focus_connection={
            "from": args.focus_from,
            "fromLane": args.focus_from_lane,
            "to": args.focus_to,
            "toLane": args.focus_to_lane,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "DONE",
                "counts": result["counts"],
                "focus_comparison": result["focus_comparison"],
                "result": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
