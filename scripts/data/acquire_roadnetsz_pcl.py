#!/usr/bin/env python3
"""Acquire the frozen native-SUMO RoadnetSZ PCL dataset without Git."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Sequence
from urllib.request import urlopen
import xml.etree.ElementTree as ET


PROTOCOL = "roadnetsz-shenzhen-pcl-fixed-commit-acquisition-v1"
REPOSITORY = "https://github.com/zhuliwen/RoadnetSZ"
RAW_BASE = "https://raw.githubusercontent.com/zhuliwen/RoadnetSZ"
COMMIT = "80a2785961b9439b6b92637e5cecd42ab5733052"
FILES = {
    "data_sumo/pcl.con.xml": 76_343,
    "data_sumo/pcl.edg.xml": 158_580,
    "data_sumo/pcl.net.xml": 935_622,
    "data_sumo/pcl.nod.xml": 14_329,
    "data_sumo/pcl.sumocfg": 584,
    "data_sumo/pcl.tll.xml": 59_451,
    "data_sumo/pcl.trips.xml": 3_976_095,
    "data_sumo/pcl.typ.xml": 4_175,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(*, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite acquisition: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest_files = {}
        for relative, expected_size in FILES.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"{RAW_BASE}/{COMMIT}/{relative}"
            with urlopen(url, timeout=120) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out)
            observed_size = destination.stat().st_size
            if observed_size != expected_size:
                raise ValueError(
                    f"RoadnetSZ PCL file size changed: {relative}: "
                    f"{observed_size} != {expected_size}"
                )
            ET.parse(destination)
            manifest_files[relative] = {
                "url": url,
                "size_bytes": observed_size,
                "sha256": _sha256(destination),
            }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": REPOSITORY,
            "commit": COMMIT,
            "file_count": len(manifest_files),
            "total_size_bytes": sum(
                int(row["size_bytes"]) for row in manifest_files.values()
            ),
            "files": manifest_files,
            "dataset": {
                "city": "shenzhen",
                "network": "pcl",
                "platform": "SUMO",
                "net_file": "data_sumo/pcl.net.xml",
                "trip_file": "data_sumo/pcl.trips.xml",
                "source_config": "data_sumo/pcl.sumocfg",
            },
        }
        (staging / "acquisition_manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = acquire(output_root=args.output_root)
    print(
        json.dumps(
            {
                "status": "PASS",
                "commit": payload["commit"],
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
