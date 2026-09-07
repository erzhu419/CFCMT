#!/usr/bin/env python3
"""Convert the frozen RoadnetSZ Fuhua-MetaVIM target to SUMO."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

from scripts.data.acquire_roadnetsz_shenzhen import (
    COMMIT,
    PROTOCOL as ACQUISITION_PROTOCOL,
    REPOSITORY,
)
from scripts.data.convert_libsignal_external_to_sumo import (
    _atomic_json,
    _sha256,
    convert_cityflow,
)


PROTOCOL = "cfcmt-cityflow-full-external-sumo-conversion-v10"
CITY_GROUP = "shenzhen"
SOURCE_NETWORK = "fuhua_metavim"
ROADNET_RELATIVE = "data_cityflow/fuhua_cityflow.json"
VARIANTS = (
    ("shenzhen_fuhua_real_1775", "data_cityflow/fuhua_real_1775.json"),
    ("shenzhen_fuhua_2570", "data_cityflow/fuhua_2570.json"),
    ("shenzhen_fuhua_4770", "data_cityflow/fuhua_4770.json"),
)
HORIZON_SEC = 3600.0


def _validate_manifest_file(
    *, acquisition_root: Path, manifest: dict[str, Any], relative: str
) -> Path:
    files = dict(manifest.get("files", {}))
    if relative not in files:
        raise ValueError(f"acquisition manifest is missing selected file: {relative}")
    path = acquisition_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    row = dict(files[relative])
    if path.stat().st_size != int(row.get("size_bytes", -1)):
        raise ValueError(f"acquired file size changed: {relative}")
    if _sha256(path) != str(row.get("sha256", "")):
        raise ValueError(f"acquired file identity changed: {relative}")
    return path


def _validate_explicit_one_vehicle_rows(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"RoadnetSZ flow is not a non-empty row list: {path}")
    starts: list[float] = []
    for index, row in enumerate(rows):
        start = float(row["startTime"])
        end = float(row["endTime"])
        interval = float(row["interval"])
        route = list(row["route"])
        if not route:
            raise ValueError(f"empty RoadnetSZ route at row {index}: {path}")
        if abs(end - start) > 1e-9 or abs(interval - 1.0) > 1e-9:
            raise ValueError(
                "selected RoadnetSZ conversion only accepts explicit one-vehicle "
                f"rows; row {index} has start={start}, end={end}, interval={interval}"
            )
        if start < 0.0 or start >= HORIZON_SEC:
            raise ValueError(
                f"RoadnetSZ departure is outside [0, {HORIZON_SEC:g}): "
                f"row {index} start={start} in {path}"
            )
        starts.append(start)
    return {
        "protocol": "roadnetsz-explicit-one-vehicle-row-audit-v1",
        "row_count": len(rows),
        "minimum_departure_sec": min(starts),
        "maximum_departure_sec": max(starts),
        "all_departures_within_horizon": True,
        "all_rows_encode_exactly_one_vehicle": True,
    }


def convert_shenzhen(
    *,
    acquisition_root: Path,
    output_root: Path,
    netconvert: Path,
) -> dict[str, Any]:
    acquisition_root = Path(acquisition_root).resolve()
    output_root = Path(output_root)
    netconvert = Path(netconvert).resolve()
    manifest_path = acquisition_root / "acquisition_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not netconvert.is_file():
        raise FileNotFoundError(netconvert)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != ACQUISITION_PROTOCOL:
        raise ValueError("RoadnetSZ acquisition protocol changed")
    if manifest.get("commit") != COMMIT:
        raise ValueError("RoadnetSZ acquisition commit changed")
    selected = dict(dict(manifest.get("candidate_networks", {})).get(SOURCE_NETWORK, {}))
    if selected.get("roadnet") != ROADNET_RELATIVE or tuple(
        selected.get("flows", ())
    ) != tuple(relative for _, relative in VARIANTS):
        raise ValueError("RoadnetSZ Fuhua-MetaVIM selection contract changed")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite conversion: {output_root}")

    roadnet_path = _validate_manifest_file(
        acquisition_root=acquisition_root,
        manifest=manifest,
        relative=ROADNET_RELATIVE,
    )
    flow_paths: dict[str, Path] = {}
    flow_audits: dict[str, dict[str, Any]] = {}
    for scenario, relative in VARIANTS:
        path = _validate_manifest_file(
            acquisition_root=acquisition_root,
            manifest=manifest,
            relative=relative,
        )
        flow_paths[scenario] = path
        flow_audits[scenario] = _validate_explicit_one_vehicle_rows(path)

    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        networks: dict[str, dict[str, Any]] = {}
        for scenario, _ in VARIANTS:
            network = convert_cityflow(
                roadnet_path=roadnet_path,
                flow_path=flow_paths[scenario],
                output_dir=staging / scenario,
                netconvert=netconvert,
                city=CITY_GROUP,
                network=scenario,
                prefix=scenario,
                yield_to_netconvert_priority=True,
            )
            network["source_flow_audit"] = flow_audits[scenario]
            networks[scenario] = network

        selected_relatives = [ROADNET_RELATIVE, *(relative for _, relative in VARIANTS)]
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "netconvert": str(netconvert),
            "source_repositories": {
                CITY_GROUP: {"url": REPOSITORY, "commit": COMMIT}
            },
            "source_files": {
                relative: {
                    "path": str((acquisition_root / relative).resolve()),
                    "sha256": str(manifest["files"][relative]["sha256"]),
                    "size_bytes": int(manifest["files"][relative]["size_bytes"]),
                }
                for relative in selected_relatives
            },
            "selection": {
                "source_network": SOURCE_NETWORK,
                "rule": (
                    "pre-efficacy selection: complete 0-3599 second coverage and "
                    "three demand variants; no controller outcomes inspected"
                ),
                "excluded_candidate": "fuhua_hilight",
                "exclusion_reason": (
                    "departures begin after the fixed 3600-second evaluation horizon"
                ),
            },
            "city_groups": {CITY_GROUP: [scenario for scenario, _ in VARIANTS]},
            "networks": networks,
        }
        _atomic_json(staging / "conversion_manifest.json", payload)
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--netconvert", type=Path, default=Path("/usr/bin/netconvert"))
    args = parser.parse_args(argv)
    payload = convert_shenzhen(
        acquisition_root=args.acquisition_root,
        output_root=args.output_root,
        netconvert=args.netconvert,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "protocol": payload["protocol"],
                "network_count": len(payload["networks"]),
                "vehicle_count": sum(
                    int(row["vehicle_count"])
                    for row in payload["networks"].values()
                ),
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
