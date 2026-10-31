#!/usr/bin/env python3
"""Build the TLS-semantic-safe LA plus Jinan SUMO confirmation package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.data.convert_libsignal_external_to_sumo import (
    _atomic_json,
    _sha256,
    convert_cityflow,
)


PROTOCOL = "cfcmt-cityflow-full-external-sumo-conversion-v9"
JINAN_VARIANTS = (
    ("jinan_3x4_real", "anon_3_4_jinan_real.json"),
    ("jinan_3x4_real_2000", "anon_3_4_jinan_real_2000.json"),
    ("jinan_3x4_real_2500", "anon_3_4_jinan_real_2500.json"),
)


def convert_external_confirmation(
    *,
    la_roadnet: Path,
    la_flow: Path,
    jinan_roadnet: Path,
    jinan_flow_dir: Path,
    jinan_repository_url: str,
    jinan_repository_commit: str,
    output_root: Path,
    netconvert: Path,
) -> dict[str, Any]:
    inputs = [la_roadnet, la_flow, jinan_roadnet, netconvert]
    inputs.extend(jinan_flow_dir / filename for _, filename in JINAN_VARIANTS)
    for path in inputs:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if len(str(jinan_repository_commit)) != 40:
        raise ValueError("Jinan repository commit must be a full 40-character hash")
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite external confirmation root: {output_root}"
        )

    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        networks: dict[str, dict[str, Any]] = {}
        networks["la_1x4"] = convert_cityflow(
            roadnet_path=la_roadnet,
            flow_path=la_flow,
            output_dir=staging / "la_1x4",
            netconvert=netconvert,
            city="los_angeles",
            network="la_1x4",
            prefix="la_1x4",
        )
        for scenario, filename in JINAN_VARIANTS:
            networks[scenario] = convert_cityflow(
                roadnet_path=jinan_roadnet,
                flow_path=jinan_flow_dir / filename,
                output_dir=staging / scenario,
                netconvert=netconvert,
                city="jinan",
                network=scenario,
                prefix=scenario,
            )

        source_files = {
            "la_flow": {
                "path": str(la_flow.resolve()),
                "sha256": _sha256(la_flow),
            },
            "la_roadnet": {
                "path": str(la_roadnet.resolve()),
                "sha256": _sha256(la_roadnet),
            },
            "jinan_roadnet": {
                "path": str(jinan_roadnet.resolve()),
                "sha256": _sha256(jinan_roadnet),
            },
        }
        for scenario, filename in JINAN_VARIANTS:
            path = jinan_flow_dir / filename
            source_files[f"{scenario}_flow"] = {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "netconvert": str(netconvert.resolve()),
            "source_repositories": {
                "jinan": {
                    "url": str(jinan_repository_url),
                    "commit": str(jinan_repository_commit),
                }
            },
            "source_files": source_files,
            "city_groups": {
                "los_angeles": ["la_1x4"],
                "jinan": [scenario for scenario, _ in JINAN_VARIANTS],
            },
            "networks": networks,
        }
        _atomic_json(staging / "conversion_manifest.json", payload)
        os.replace(staging, output_root)
        return payload
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--la-roadnet", type=Path, required=True)
    parser.add_argument("--la-flow", type=Path, required=True)
    parser.add_argument("--jinan-roadnet", type=Path, required=True)
    parser.add_argument("--jinan-flow-dir", type=Path, required=True)
    parser.add_argument(
        "--jinan-repository-url",
        default="https://github.com/wingsweihua/colight.git",
    )
    parser.add_argument("--jinan-repository-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--netconvert", type=Path, default=Path("/usr/bin/netconvert"))
    args = parser.parse_args()
    payload = convert_external_confirmation(
        la_roadnet=args.la_roadnet,
        la_flow=args.la_flow,
        jinan_roadnet=args.jinan_roadnet,
        jinan_flow_dir=args.jinan_flow_dir,
        jinan_repository_url=args.jinan_repository_url,
        jinan_repository_commit=args.jinan_repository_commit,
        output_root=args.output_root,
        netconvert=args.netconvert,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Results saved to: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
