#!/usr/bin/env python3
"""Acquire the complete fixed Paderborn SUMO record on shared storage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Any, Sequence
from urllib.parse import quote
from urllib.request import Request, urlopen


PROTOCOL = "cfcmt-zenodo-paderborn-v01-complete-acquisition-v1"
RECORD_ID = 4_522_059
VERSION_DOI = "10.5281/zenodo.4522059"
CONCEPT_DOI = "10.5281/zenodo.4522058"
TITLE = "Paderborn Traffic Scenario"
USER_AGENT = "Mozilla/5.0 CFCMT-research-acquisition/1.0"
FILES = (
    ("alltrips.trp.xml", 30_324_431, "d19ec5d0cc7c7e0d0f97f543a4ad9c92"),
    ("vtypes.add.xml", 220, "8781137b5cefacdbc3289ff2d8d3e7fd"),
    ("network.net.xml", 51_712_951, "6a017aad6b1690dbd61dcfa9f74042e9"),
    ("full-day.sumo.cfg", 1_123, "c87365a1351b8a2e288cf6432d364e03"),
    ("polygons.poly.xml", 14_472_505, "ef358c934cfb8b618b1beff892c4a6d3"),
    ("two-hours-ad-hoc.sumo.cfg", 1_500, "514caff1f6769609239eb76432c7b432"),
    ("README.md", 2_630, "26e69f1d103cc643e27e075c822e429b"),
    (
        "trafficlightlogic.tll.xml",
        57_039,
        "c36265310aae6ec8d7628765696ff7fd",
    ),
    ("LICENSE.md", 35_146, "4fe869ee987a340198fb0d54c55c47f1"),
    ("trips.trp.xml", 1_820_481, "ba32df8e85176f4355cc8e0ed01b8bf0"),
    ("allroutes.rou.xml", 182_571_889, "0744c9f219779bdc08ae69b76e1b9426"),
)


def _content_url(name: str) -> str:
    return (
        f"https://zenodo.org/api/records/{RECORD_ID}/files/"
        f"{quote(name)}/content"
    )


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


def _receive_command(
    *, staging: PurePosixPath, name: str, expected_size: int
) -> str:
    destination = staging / name
    partial = staging / f".{name}.part"
    return " ".join(
        [
            "set -e;",
            f"cat > {shlex.quote(str(partial))};",
            f"test $(wc -c < {shlex.quote(str(partial))}) -eq {expected_size};",
            f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))}",
        ]
    )


def _stream_file(
    *,
    ssh: Sequence[str],
    target: str,
    staging: PurePosixPath,
    name: str,
    expected_size: int,
    expected_md5: str,
) -> None:
    transfer = subprocess.Popen(
        [
            *ssh,
            target,
            _receive_command(
                staging=staging,
                name=name,
                expected_size=expected_size,
            ),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert transfer.stdin is not None
    digest = hashlib.md5(usedforsecurity=False)
    observed_size = 0
    request = Request(
        _content_url(name),
        headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
    )
    try:
        with urlopen(request, timeout=600) as response:
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                observed_size += len(block)
                digest.update(block)
                transfer.stdin.write(block)
    finally:
        transfer.stdin.close()
    stdout = transfer.stdout.read() if transfer.stdout is not None else b""
    stderr = transfer.stderr.read() if transfer.stderr is not None else b""
    result = subprocess.CompletedProcess(
        transfer.args,
        transfer.wait(),
        stdout=stdout,
        stderr=stderr,
    )
    _check_remote(result, f"Paderborn transfer for {name}")
    observed_md5 = digest.hexdigest()
    if observed_size != expected_size or observed_md5 != expected_md5:
        raise ValueError(
            f"Paderborn source identity changed for {name}: "
            f"size={observed_size} md5={observed_md5}"
        )


def _manifest() -> dict[str, Any]:
    files = {
        name: {
            "size_bytes": int(size),
            "md5": md5,
            "url": _content_url(name),
        }
        for name, size, md5 in FILES
    }
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "title": TITLE,
        "record_id": RECORD_ID,
        "version": "0.1",
        "version_doi": VERSION_DOI,
        "concept_doi": CONCEPT_DOI,
        "published_at": "2021-02-12",
        "file_count": len(files),
        "total_size_bytes": sum(row["size_bytes"] for row in files.values()),
        "coverage": "complete fixed Zenodo record",
        "transfer_mode": (
            "https_stream_through_local_to_shared_cluster_no_local_raw_file"
        ),
        "license_metadata": "GPL-2.0",
        "license_readme": "GPLv3",
        "files": files,
    }


def acquire_remote(
    *, output_root: PurePosixPath, node: str, scheduler_skill_dir: Path
) -> dict[str, Any]:
    ssh, target = _cluster_connection(
        node=node, scheduler_skill_dir=scheduler_skill_dir
    )
    staging = output_root.parent / f".{output_root.name}.staging-v1"
    initialization = _remote_run(
        ssh,
        target,
        (
            "set -e; "
            f"test ! -e {shlex.quote(str(output_root))}; "
            f"test ! -e {shlex.quote(str(staging))}; "
            f"mkdir -p {shlex.quote(str(staging))}"
        ),
    )
    _check_remote(initialization, "Paderborn acquisition initialization")
    try:
        for name, size, md5 in FILES:
            print(f"streaming {name} ({size} bytes)", flush=True)
            _stream_file(
                ssh=ssh,
                target=target,
                staging=staging,
                name=name,
                expected_size=size,
                expected_md5=md5,
            )
        payload = _manifest()
        upload = _remote_run(
            ssh,
            target,
            f"cat > {shlex.quote(str(staging / 'acquisition_manifest.json'))}",
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "Paderborn acquisition manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "Paderborn acquisition finalization")
        return payload
    except Exception:
        _remote_run(ssh, target, f"rm -rf -- {shlex.quote(str(staging))}")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-output-root", required=True)
    parser.add_argument("--remote-node", default="node001")
    parser.add_argument(
        "--scheduler-skill-dir",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm/skill"),
    )
    args = parser.parse_args(argv)
    payload = acquire_remote(
        output_root=PurePosixPath(args.remote_output_root),
        node=args.remote_node,
        scheduler_skill_dir=args.scheduler_skill_dir,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "record_id": payload["record_id"],
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
