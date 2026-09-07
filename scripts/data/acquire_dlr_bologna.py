#!/usr/bin/env python3
"""Acquire all native SUMO Bologna scenarios at one frozen DLR commit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import ssl
from typing import Sequence
from urllib.error import URLError
from urllib.request import urlopen
import xml.etree.ElementTree as ET


PROTOCOL = "dlr-bologna-fixed-commit-full-acquisition-v1"
REPOSITORY = "https://github.com/DLR-TS/sumo-scenarios"
RAW_BASE = "https://raw.githubusercontent.com/DLR-TS/sumo-scenarios"
COMMIT = "00f6eb479a9dc0fbaeb731495c11d48d7a9661d3"
SCENARIOS = ("acosta", "acosta_persontrips", "joined", "pasubio")
FILES = {
    "bologna/acosta/acosta.rou.xml": 1_728_966,
    "bologna/acosta/acosta_bus_stops.add.xml": 3_127,
    "bologna/acosta/acosta_buslanes.net.xml": 256_993,
    "bologna/acosta/acosta_busses.rou.xml": 60_748,
    "bologna/acosta/acosta_detectors.add.xml": 6_799,
    "bologna/acosta/acosta_tls.add.xml": 8_036,
    "bologna/acosta/acosta_vtypes.add.xml": 2_896,
    "bologna/acosta/run.sumocfg": 749,
    "bologna/acosta/settings.gui.xml": 71,
    "bologna/acosta_persontrips/acosta.rou.xml": 1_728_966,
    "bologna/acosta_persontrips/acosta_bus_stops.add.xml": 3_836,
    "bologna/acosta_persontrips/acosta_buslanes.net.xml": 257_226,
    "bologna/acosta_persontrips/acosta_busses.rou.xml": 8_309,
    "bologna/acosta_persontrips/acosta_detectors.add.xml": 6_799,
    "bologna/acosta_persontrips/acosta_peds.net.xml": 528_317,
    "bologna/acosta_persontrips/acosta_tls.add.xml": 8_106,
    "bologna/acosta_persontrips/acosta_tls2.add.xml": 4_237,
    "bologna/acosta_persontrips/acosta_vtypes.add.xml": 2_938,
    "bologna/acosta_persontrips/persontrips.duarcfg": 575,
    "bologna/acosta_persontrips/persontrips.rou.xml": 1_224_412,
    "bologna/acosta_persontrips/persontrips.xml": 888_869,
    "bologna/acosta_persontrips/run.sumocfg": 879,
    "bologna/acosta_persontrips/settings.gui.xml": 71,
    "bologna/joined/joined.rou.xml": 2_388_351,
    "bologna/joined/joined_bus_stops.add.xml": 5_450,
    "bologna/joined/joined_buslanes.net.xml": 428_805,
    "bologna/joined/joined_busses.rou.xml": 87_586,
    "bologna/joined/joined_detectors.add.xml": 12_390,
    "bologna/joined/joined_tls.add.xml": 14_972,
    "bologna/joined/joined_vtypes.add.xml": 2_911,
    "bologna/joined/run.sumocfg": 706,
    "bologna/pasubio/pasubio.rou.xml": 1_641_510,
    "bologna/pasubio/pasubio_bus_stops.add.xml": 2_185,
    "bologna/pasubio/pasubio_buslanes.net.xml": 196_009,
    "bologna/pasubio/pasubio_busses.rou.xml": 50_393,
    "bologna/pasubio/pasubio_detectors.add.xml": 7_408,
    "bologna/pasubio/pasubio_tls.add.xml": 9_566,
    "bologna/pasubio/pasubio_vtypes.add.xml": 2_911,
    "bologna/pasubio/run.sumocfg": 670,
}
GIT_BLOBS = {
    "bologna/acosta/acosta.rou.xml": "1c22351ccac09feac59593aadc9f21b064580de6",
    "bologna/acosta/acosta_bus_stops.add.xml": "a1e028cc892e755fecdf7dc67a91a2eedc062b8b",
    "bologna/acosta/acosta_buslanes.net.xml": "136434153cb9086a4dfa30b1141fc4c4f54d802f",
    "bologna/acosta/acosta_busses.rou.xml": "5e31a8c754aa39d0a7ab1c0e12a695c3a7f4e5c8",
    "bologna/acosta/acosta_detectors.add.xml": "c3e597ead24cf62455d73cdaca483505ffced571",
    "bologna/acosta/acosta_tls.add.xml": "cbf15caa4933bad1447af4340d452e6f5a2eb669",
    "bologna/acosta/acosta_vtypes.add.xml": "fa79f626175b1fbf96b3972a5f2a60e815086574",
    "bologna/acosta/run.sumocfg": "de7027383aab954d22b6327eb7a0173e6f639f68",
    "bologna/acosta/settings.gui.xml": "a7b5ca2c1ddd2198455a531b9807af3b9437960f",
    "bologna/acosta_persontrips/acosta.rou.xml": "1c22351ccac09feac59593aadc9f21b064580de6",
    "bologna/acosta_persontrips/acosta_bus_stops.add.xml": "2f83209dfdecea7024c44ee3cdcbbeaa7497c366",
    "bologna/acosta_persontrips/acosta_buslanes.net.xml": "d3f1e61e39c94a6d00c6442da7d8e31f36eea4d2",
    "bologna/acosta_persontrips/acosta_busses.rou.xml": "1b37196d14af122a301ef38216f903001062a335",
    "bologna/acosta_persontrips/acosta_detectors.add.xml": "c3e597ead24cf62455d73cdaca483505ffced571",
    "bologna/acosta_persontrips/acosta_peds.net.xml": "e729aff66d731f27fe60c879a6781df2aff32dda",
    "bologna/acosta_persontrips/acosta_tls.add.xml": "dbf99ea9e3dfa85fb92db52da5fd4bedbfe23de5",
    "bologna/acosta_persontrips/acosta_tls2.add.xml": "0fc1cc283be06b9e8c2bd9f185e17b50f59e06bb",
    "bologna/acosta_persontrips/acosta_vtypes.add.xml": "6c4834542a222412776f012794053aec679d6331",
    "bologna/acosta_persontrips/persontrips.duarcfg": "a0df6db3eb38ca2ecb1c5e16a02428bac48bd792",
    "bologna/acosta_persontrips/persontrips.rou.xml": "fed1f85c373d189bb61138f31464363a3f663add",
    "bologna/acosta_persontrips/persontrips.xml": "cb2503470aae3a165542de73e676a92d12838598",
    "bologna/acosta_persontrips/run.sumocfg": "5af5c784e1493c3a3fad378ce89ec7d2055141e2",
    "bologna/acosta_persontrips/settings.gui.xml": "a7b5ca2c1ddd2198455a531b9807af3b9437960f",
    "bologna/joined/joined.rou.xml": "4fdd46541ab74de0035c787b9900045ae7b0d7ef",
    "bologna/joined/joined_bus_stops.add.xml": "dd2b5a8d2abbad28617486b28c1ea74c6f16214d",
    "bologna/joined/joined_buslanes.net.xml": "3ec470d8a4cc18f6f7e3ce1f1abeaffd1e310434",
    "bologna/joined/joined_busses.rou.xml": "81f3ac1000976fc47d7e26b0d44bf97c5901b07a",
    "bologna/joined/joined_detectors.add.xml": "bde420c8b7baac080e5f170c62776ae342b25c21",
    "bologna/joined/joined_tls.add.xml": "c2235e64c800accce132ec58170cfab4b54bf8e4",
    "bologna/joined/joined_vtypes.add.xml": "207c3d437966e5d78c5b01f1e34eaed2edde8314",
    "bologna/joined/run.sumocfg": "80a72339e85c1b58a7980e865ff5ff5d11bcdf28",
    "bologna/pasubio/pasubio.rou.xml": "c07d4d1a2c8fa5c4a97761eb7b955072015f972d",
    "bologna/pasubio/pasubio_bus_stops.add.xml": "476397aa392c11be3a74d7b92c05ad2ac7f2480e",
    "bologna/pasubio/pasubio_buslanes.net.xml": "55b789aaed76e1f0d587d00bc89d0f14709d6578",
    "bologna/pasubio/pasubio_busses.rou.xml": "3dc297037d55e8d71159be51a4acffaeb55f2dea",
    "bologna/pasubio/pasubio_detectors.add.xml": "fb9435306e0def6c4f57ee83bf5eec8ce12815a4",
    "bologna/pasubio/pasubio_tls.add.xml": "e58995742a551256ec2aa205788027593814a2b3",
    "bologna/pasubio/pasubio_vtypes.add.xml": "207c3d437966e5d78c5b01f1e34eaed2edde8314",
    "bologna/pasubio/run.sumocfg": "719a216b2e870802dae52cd783f1e7a59985c7e5",
}


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _download(url: str, destination: Path) -> bool:
    insecure_retry = False
    try:
        response = urlopen(url, timeout=120)
    except URLError as exc:
        if not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        insecure_retry = True
        response = urlopen(
            url,
            timeout=120,
            context=ssl._create_unverified_context(),
        )
    with response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)
    return insecure_retry


def acquire(*, output_root: Path) -> dict:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite acquisition: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest_files = {}
        insecure_retry_count = 0
        for relative, expected_size in FILES.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"{RAW_BASE}/{COMMIT}/{relative}"
            insecure_retry_count += int(_download(url, destination))
            observed_size = destination.stat().st_size
            if observed_size != expected_size:
                raise ValueError(
                    f"DLR Bologna file size changed: {relative}: "
                    f"{observed_size} != {expected_size}"
                )
            observed_blob = _git_blob_sha1(destination)
            if observed_blob != GIT_BLOBS[relative]:
                raise ValueError(
                    f"DLR Bologna Git blob changed: {relative}: "
                    f"{observed_blob} != {GIT_BLOBS[relative]}"
                )
            ET.parse(destination)
            manifest_files[relative] = {
                "url": url,
                "size_bytes": observed_size,
                "git_blob_sha1": observed_blob,
            }
        payload = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": REPOSITORY,
            "commit": COMMIT,
            "file_count": len(manifest_files),
            "insecure_tls_retry_count": insecure_retry_count,
            "total_size_bytes": sum(
                int(row["size_bytes"]) for row in manifest_files.values()
            ),
            "files": manifest_files,
            "dataset": {
                "city": "bologna",
                "platform": "SUMO",
                "coverage": "all repository Bologna subscenarios and files",
                "scenarios": list(SCENARIOS),
                "source_configs": {
                    scenario: f"bologna/{scenario}/run.sumocfg"
                    for scenario in SCENARIOS
                },
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
                "scenarios": payload["dataset"]["scenarios"],
            },
            sort_keys=True,
        )
    )
    print(f"Results saved to: {args.output_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
