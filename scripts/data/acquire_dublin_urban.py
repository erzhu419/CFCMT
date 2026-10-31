#!/usr/bin/env python3
"""Acquire the complete Dublin Urban SUMO case at one fixed commit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import shlex
import ssl
import subprocess
import sys
import time
from typing import Sequence
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen
import xml.etree.ElementTree as ET


PROTOCOL = "dublin-urban-fixed-commit-full-acquisition-v1"
REPOSITORY = "https://github.com/maxime-gueriau/ITSC2020_CAV_impact"
RAW_BASE = "https://raw.githubusercontent.com/maxime-gueriau/ITSC2020_CAV_impact"
COMMIT = "a2fe1d2f67f546099b4b44dc81ae7ba82a60e054"
SCENARIOS = ("A", "B", "C", "D", "E", "F")
FILES = {
    "LICENSE.md": (35_149, "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"),
    "README.md": (1_088, "81bdf5e1eccf1e0d9862ea48b0694750149a9584"),
    "Urban/Simulations/Base/DCC.net.xml": (
        12_864_257,
        "312bf34cfad4aab79afb6b52b8e293369f60bc05",
    ),
    "Urban/Simulations/Base/DCC_detectors.poi.xml": (
        49_141,
        "e21a594aceecc201dc5f7eff1d1c69723ec5d152",
    ),
    "Urban/Simulations/Base/DCC_emitters.emi.xml": (
        78_552_921,
        "980fc84a1c1907cfb7a84b959931fa558255464d",
    ),
    "Urban/Simulations/Base/DCC_routes.rou.xml": (
        12_742_630,
        "86052b3c29510a773158e94440387a56b3d65447",
    ),
    "Urban/Simulations/Base/DCC_simulation.sumo.cfg": (
        627,
        "fd1667a18f5d389bf14103604a811c0ee59fe326",
    ),
    "Urban/Simulations/Base/DCC_trafficlights.add.xml": (
        164_901,
        "3a7296acfe316aa3667e9d4ffbd94bf347124d8a",
    ),
    "Urban/Simulations/Base/vtypes.add.xml": (
        1_003,
        "f09f7f4ddf1fef02cc1dd4bdb1a2321f7549ad63",
    ),
    "Urban/Simulations/Scenario A/vtypes.add.xml": (
        1_003,
        "f09f7f4ddf1fef02cc1dd4bdb1a2321f7549ad63",
    ),
    "Urban/Simulations/Scenario B/vtypes.add.xml": (
        1_026,
        "55c76756959d9542448880430a216c1e51d8b892",
    ),
    "Urban/Simulations/Scenario C/vtypes.add.xml": (
        1_020,
        "97e8c5a22db19598c78209962de9e44f410f947f",
    ),
    "Urban/Simulations/Scenario D/vtypes.add.xml": (
        1_018,
        "b38c273f95f394c9f6ebe15315243f3e3f006adc",
    ),
    "Urban/Simulations/Scenario E/vtypes.add.xml": (
        1_014,
        "b6222f053cba9af092c40661715574c8a356850a",
    ),
    "Urban/Simulations/Scenario F/vtypes.add.xml": (
        1_014,
        "aeaf8c9df1290c9c60144b1566e27f87ec30faa6",
    ),
}


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> bool:
    insecure_retry = False
    try:
        response = urlopen(url, timeout=300)
    except URLError as exc:
        if not isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise
        insecure_retry = True
        response = urlopen(
            url,
            timeout=300,
            context=ssl._create_unverified_context(),
        )
    with response, destination.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    return insecure_retry


def acquire(*, output_root: Path) -> dict:
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite Dublin acquisition: {output_root}")
    staging = output_root.with_name(f".{output_root.name}.staging-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest_files = {}
        insecure_retry_count = 0
        for relative, (expected_size, expected_blob) in FILES.items():
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            encoded = "/".join(quote(part) for part in Path(relative).parts)
            url = f"{RAW_BASE}/{COMMIT}/{encoded}"
            insecure_retry_count += int(_download(url, destination))
            observed_size = destination.stat().st_size
            observed_blob = _git_blob_sha1(destination)
            if observed_size != expected_size or observed_blob != expected_blob:
                raise ValueError(
                    f"Dublin source identity changed for {relative}: "
                    f"size={observed_size} blob={observed_blob}"
                )
            if destination.suffix == ".xml" or destination.name.endswith(
                (".sumo.cfg", ".add.xml", ".rou.xml", ".emi.xml")
            ):
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
            "license": "GPL-3.0",
            "city": "Dublin",
            "case_study": "Urban",
            "coverage": "complete Base inputs and all Urban scenarios A-F",
            "scenarios": list(SCENARIOS),
            "file_count": len(manifest_files),
            "total_size_bytes": sum(
                int(row["size_bytes"]) for row in manifest_files.values()
            ),
            "insecure_tls_retry_count": insecure_retry_count,
            "files": manifest_files,
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


def _cluster_connection(
    *, node: str, scheduler_skill_dir: Path
) -> tuple[list[str], str]:
    sys.path.insert(0, str(scheduler_skill_dir.resolve()))
    import scheduler  # type: ignore[import-not-found]

    return (
        shlex.split(scheduler._ssh_rsync_shell_for_node(node)),
        str(scheduler._ssh_target_for_node(node)),
    )


def _remote_run(
    ssh: Sequence[str], target: str, command: str, *, stdin: bytes | None = None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*ssh, target, command],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _check_remote(result: subprocess.CompletedProcess[bytes], context: str) -> None:
    if result.returncode == 0:
        return
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    raise RuntimeError(f"{context} failed ({result.returncode}): {stderr}")


def _stream_remote_file(
    *,
    relative: str,
    expected_size: int,
    expected_blob: str,
    staging: PurePosixPath,
    ssh: Sequence[str],
    target: str,
) -> None:
    destination = staging / relative
    partial = destination.parent / f".{destination.name}.part"
    encoded = "/".join(quote(part) for part in Path(relative).parts)
    url = f"{RAW_BASE}/{COMMIT}/{encoded}"
    remote_command = (
        "set -e; "
        f"mkdir -p {shlex.quote(str(destination.parent))}; "
        f"test ! -e {shlex.quote(str(destination))}; "
        f"cat > {shlex.quote(str(partial))}; "
        f"test $(wc -c < {shlex.quote(str(partial))}) -eq {expected_size}; "
        f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))}"
    )
    response = urlopen(url, timeout=300)
    transfer = subprocess.Popen(
        [*ssh, target, remote_command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert transfer.stdin is not None
    digest = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
    observed_size = 0
    with response:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            observed_size += len(block)
            digest.update(block)
            transfer.stdin.write(block)
    transfer.stdin.close()
    stdout = transfer.stdout.read() if transfer.stdout is not None else b""
    stderr = transfer.stderr.read() if transfer.stderr is not None else b""
    result = subprocess.CompletedProcess(
        transfer.args, transfer.wait(), stdout=stdout, stderr=stderr
    )
    _check_remote(result, f"Dublin remote transfer for {relative}")
    if observed_size != expected_size or digest.hexdigest() != expected_blob:
        raise ValueError(
            f"Dublin source identity changed for {relative}: "
            f"size={observed_size} blob={digest.hexdigest()}"
        )


def acquire_remote(
    *,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
) -> dict:
    ssh, target = _cluster_connection(
        node=node, scheduler_skill_dir=scheduler_skill_dir
    )
    staging = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    init = _remote_run(
        ssh,
        target,
        (
            "set -e; "
            f"test ! -e {shlex.quote(str(output_root))}; "
            f"test ! -e {shlex.quote(str(staging))}; "
            f"mkdir -p {shlex.quote(str(staging))}"
        ),
    )
    _check_remote(init, "Dublin remote acquisition initialization")
    try:
        for relative, (size, blob) in FILES.items():
            print(f"streaming {relative} ({size} bytes)", flush=True)
            for attempt in range(1, 5):
                try:
                    _stream_remote_file(
                        relative=relative,
                        expected_size=size,
                        expected_blob=blob,
                        staging=staging,
                        ssh=ssh,
                        target=target,
                    )
                    break
                except RuntimeError:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        manifest = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": REPOSITORY,
            "commit": COMMIT,
            "license": "GPL-3.0",
            "city": "Dublin",
            "case_study": "Urban",
            "coverage": "complete Base inputs and all Urban scenarios A-F",
            "scenarios": list(SCENARIOS),
            "file_count": len(FILES),
            "total_size_bytes": sum(size for size, _ in FILES.values()),
            "insecure_tls_retry_count": 0,
            "transfer_mode": "https_stream_to_shared_cluster_no_local_file",
            "files": {
                relative: {
                    "url": (
                        f"{RAW_BASE}/{COMMIT}/"
                        + "/".join(quote(part) for part in Path(relative).parts)
                    ),
                    "size_bytes": size,
                    "git_blob_sha1": blob,
                }
                for relative, (size, blob) in FILES.items()
            },
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        upload = _remote_run(
            ssh,
            target,
            "set -e; cat > "
            + shlex.quote(str(staging / "acquisition_manifest.json")),
            stdin=manifest_bytes,
        )
        _check_remote(upload, "Dublin acquisition manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"set -e; mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "Dublin acquisition finalization")
        return manifest
    except Exception:
        _remote_run(
            ssh,
            target,
            f"rm -rf -- {shlex.quote(str(staging))}",
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--remote-output-root")
    parser.add_argument("--remote-node", default="node001")
    parser.add_argument(
        "--scheduler-skill-dir",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm/skill"),
    )
    args = parser.parse_args(argv)
    if (args.output_root is None) == (args.remote_output_root is None):
        raise ValueError("select exactly one local or remote Dublin output root")
    payload = (
        acquire(output_root=args.output_root)
        if args.output_root is not None
        else acquire_remote(
            output_root=PurePosixPath(args.remote_output_root),
            node=args.remote_node,
            scheduler_skill_dir=args.scheduler_skill_dir,
        )
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "commit": payload["commit"],
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
                "scenarios": payload["scenarios"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
