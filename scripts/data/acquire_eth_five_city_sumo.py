#!/usr/bin/env python3
"""Acquire the fixed ETH five-city SUMO archive on shared cluster storage."""

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
from urllib.request import Request, urlopen


PROTOCOL = "cfcmt-eth-five-city-sumo-archive-acquisition-v1"
DOI = "10.3929/ethz-b-000584669"
DATASET_TITLE = (
    "Traffic Simulations for Boston, Lisbon, Los Angeles, Rio de Janeiro, "
    "San Francisco"
)
ARCHIVE_NAME = "sumo_sim_perc_mfd.zip"
BITSTREAM_URL = (
    "https://www.research-collection.ethz.ch/server/api/core/bitstreams/"
    "0275377f-baba-4865-a709-03de4b16ac92/content"
)
EXPECTED_SIZE_BYTES = 658_810_696
EXPECTED_MD5 = "802321f1d2c61de527ed579ce9c54842"
USER_AGENT = "Mozilla/5.0 CFCMT-research-acquisition/1.0"


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


def _remote_receive_command(staging: PurePosixPath) -> str:
    archive = staging / ARCHIVE_NAME
    partial = staging / f".{ARCHIVE_NAME}.part"
    listing = staging / "archive_entries.txt"
    return " ".join(
        [
            "set -e;",
            f"cat > {shlex.quote(str(partial))};",
            f"test $(wc -c < {shlex.quote(str(partial))}) -eq {EXPECTED_SIZE_BYTES};",
            f"mv {shlex.quote(str(partial))} {shlex.quote(str(archive))};",
            f"unzip -Z1 {shlex.quote(str(archive))} > {shlex.quote(str(listing))};",
            f"test -s {shlex.quote(str(listing))}",
        ]
    )


def _stream_archive_via_local_https(
    *,
    ssh: Sequence[str],
    target: str,
    staging: PurePosixPath,
) -> None:
    transfer = subprocess.Popen(
        [*ssh, target, _remote_receive_command(staging)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert transfer.stdin is not None
    digest = hashlib.md5(usedforsecurity=False)
    observed_size = 0
    request = Request(
        BITSTREAM_URL,
        headers={"User-Agent": USER_AGENT, "Accept": "application/zip"},
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
    _check_remote(result, "ETH archive stream to shared cluster")
    observed_md5 = digest.hexdigest()
    if observed_size != EXPECTED_SIZE_BYTES or observed_md5 != EXPECTED_MD5:
        raise ValueError(
            "ETH archive source identity changed: "
            f"size={observed_size} md5={observed_md5}"
        )


def _manifest(*, archive_entry_count: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_title": DATASET_TITLE,
        "doi": DOI,
        "archive_name": ARCHIVE_NAME,
        "bitstream_url": BITSTREAM_URL,
        "size_bytes": EXPECTED_SIZE_BYTES,
        "md5": EXPECTED_MD5,
        "license": "CC-BY-NC-4.0",
        "archive_entry_count": int(archive_entry_count),
        "coverage": "complete published five-city archive",
        "transfer_mode": (
            "https_stream_through_local_to_shared_cluster_no_local_raw_file"
        ),
    }


def acquire_remote(
    *,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
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
    _check_remote(initialization, "ETH archive acquisition initialization")
    try:
        _stream_archive_via_local_https(
            ssh=ssh,
            target=target,
            staging=staging,
        )
        count_result = _remote_run(
            ssh,
            target,
            f"wc -l < {shlex.quote(str(staging / 'archive_entries.txt'))}",
        )
        _check_remote(count_result, "ETH archive entry count")
        entry_count = int(count_result.stdout.decode("ascii").strip())
        if entry_count <= 0:
            raise ValueError("ETH archive contains no entries")
        payload = _manifest(archive_entry_count=entry_count)
        upload = _remote_run(
            ssh,
            target,
            f"cat > {shlex.quote(str(staging / 'acquisition_manifest.json'))}",
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "ETH archive manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "ETH archive acquisition finalization")
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
                "doi": payload["doi"],
                "archive_entry_count": payload["archive_entry_count"],
                "size_bytes": payload["size_bytes"],
                "md5": payload["md5"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
