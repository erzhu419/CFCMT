#!/usr/bin/env python3
"""Acquire the frozen RoadnetSZ Shenzhen CityFlow files without Git."""

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


PROTOCOL = "roadnetsz-shenzhen-fixed-commit-acquisition-v1"
REPOSITORY = "https://github.com/zhuliwen/RoadnetSZ"
RAW_BASE = "https://raw.githubusercontent.com/zhuliwen/RoadnetSZ"
COMMIT = "80a2785961b9439b6b92637e5cecd42ab5733052"
FILES = {
    "data_cityflow/anon_1_33_fuhua_24hto1w_2490.json": 1_383_677,
    "data_cityflow/anon_1_33_fuhua_4_27_24hto1w_4089.json": 1_958_984,
    "data_cityflow/fuhua_2570.json": 1_357_318,
    "data_cityflow/fuhua_4770.json": 2_521_756,
    "data_cityflow/fuhua_cityflow.json": 710_190,
    "data_cityflow/fuhua_real_1775.json": 837_800,
    "data_cityflow/roadnet_1_33.json": 671_333,
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
                    f"RoadnetSZ file size changed: {relative}: "
                    f"{observed_size} != {expected_size}"
                )
            json.loads(destination.read_text(encoding="utf-8"))
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
            "candidate_networks": {
                "fuhua_hilight": {
                    "roadnet": "data_cityflow/roadnet_1_33.json",
                    "flows": [
                        "data_cityflow/anon_1_33_fuhua_24hto1w_2490.json",
                        "data_cityflow/anon_1_33_fuhua_4_27_24hto1w_4089.json",
                    ],
                },
                "fuhua_metavim": {
                    "roadnet": "data_cityflow/fuhua_cityflow.json",
                    "flows": [
                        "data_cityflow/fuhua_real_1775.json",
                        "data_cityflow/fuhua_2570.json",
                        "data_cityflow/fuhua_4770.json",
                    ],
                },
            },
            "selection_status": "pre-efficacy-network-integrity-screen",
        }
        manifest_path = staging / "acquisition_manifest.json"
        manifest_path.write_text(
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
