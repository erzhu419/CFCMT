#!/usr/bin/env python3
"""Audit full-demand retention in a completed DLR Brunswick build attempt."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.traffic_signal.sumo_static_inputs import iterparse_xml
from scripts.data.package_dlr_brunswick_sumo import (
    EXPECTED_ROAD_TRIP_COUNT,
    _gtfs_vehicle_inventory,
    _mapped_trip_inventory,
    _road_trip_inventory,
    _tls_inventory,
)


PROTOCOL = "dlr-brunswick-sumo122-full-demand-retention-audit-v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit(*, build_root: Path, snapshot_sha256: str) -> dict[str, Any]:
    build_root = build_root.resolve()
    miv = build_root / "miv"
    source = _road_trip_inventory(miv / "bs_miv_cut.geotrips.xml.gz")
    mapped = _mapped_trip_inventory(
        miv / "miv.trips.xml", miv / "miv.net.xml.gz"
    )
    gtfs = _gtfs_vehicle_inventory(miv / "gtfs_pt_vehicles.add.xml")
    tls = _tls_inventory(miv / "miv.net.xml.gz")
    mapped_ids: set[str] = set()
    for _, element in iterparse_xml(miv / "miv.trips.xml", events=("end",)):
        if element.tag == "trip":
            mapped_ids.add(str(element.attrib.get("id", "")))
        element.clear()
    missing_ids: list[str] = []
    missing_types: Counter[str] = Counter()
    missing_departures: list[float] = []
    for _, element in iterparse_xml(
        miv / "bs_miv_cut.geotrips.xml.gz", events=("end",)
    ):
        if element.tag == "trip":
            identity = str(element.attrib.get("id", ""))
            if identity not in mapped_ids:
                missing_ids.append(identity)
                missing_types[str(element.attrib.get("type", ""))] += 1
                missing_departures.append(float(element.attrib["depart"]))
        element.clear()
    retention = float(mapped["trip_count"] / max(source["trip_count"], 1))
    checks = {
        "source_expected_trip_count": (
            int(source["trip_count"]) == EXPECTED_ROAD_TRIP_COUNT
        ),
        "exact_full_road_demand_retention": (
            int(mapped["trip_count"]) == int(source["trip_count"])
        ),
        "no_duplicate_mapped_trip_ids": int(mapped["duplicate_id_count"]) == 0,
        "no_missing_mapped_anchors": int(mapped["missing_anchor_count"]) == 0,
        "positive_full_network_gtfs": int(gtfs["vehicle_count"]) > 0,
        "positive_tls_count": int(tls["controlled_intersection_count"]) > 0,
    }
    return {
        "protocol": PROTOCOL,
        "scientific_status": "pre-simulation-build-retention-gate",
        "decision": "PASS" if all(checks.values()) else "REJECT",
        "checks": checks,
        "snapshot_sha256": snapshot_sha256,
        "build_root": str(build_root),
        "source_geotrips": source,
        "mapped_road_trips": mapped,
        "road_trip_retention": retention,
        "missing_road_trip_count": len(missing_ids),
        "missing_road_trip_type_counts": dict(sorted(missing_types.items())),
        "missing_departure_min_sec": (
            min(missing_departures) if missing_departures else None
        ),
        "missing_departure_max_sec": (
            max(missing_departures) if missing_departures else None
        ),
        "missing_id_sample": missing_ids[:12],
        "gtfs": gtfs,
        "tls": tls,
        "claim_boundary": (
            "The result rejects Brunswick before any controller or safety "
            "rollout because the frozen exact-retention gate failed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = audit(
        build_root=args.build_root,
        snapshot_sha256=args.snapshot_sha256,
    )
    _write_json(args.out, payload)
    print(json.dumps({
        "decision": payload["decision"],
        "road_trip_retention": payload["road_trip_retention"],
        "missing_road_trip_count": payload["missing_road_trip_count"],
        "out": str(args.out),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
