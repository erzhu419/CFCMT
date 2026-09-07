#!/usr/bin/env python3
"""Stream the complete DLR Brunswick scenario at one fixed commit to HPC."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import time
from typing import Any, Sequence
from urllib.parse import quote
from urllib.request import urlopen


PROTOCOL = "dlr-brunswick-fixed-subtree-complete-acquisition-v1"
REPOSITORY = "https://github.com/DLR-TS/sumo-scenarios"
RAW_BASE = "https://raw.githubusercontent.com/DLR-TS/sumo-scenarios"
API_BASE = "https://api.github.com/repos/DLR-TS/sumo-scenarios/git/trees"
COMMIT = "00f6eb479a9dc0fbaeb731495c11d48d7a9661d3"
SUBTREE = "brunswick"
SUBTREE_SHA = "04bc93e619e60faa95dc5b18432fb07a5434634c"
EXPECTED_FILE_COUNT = 66
EXPECTED_TOTAL_SIZE_BYTES = 234_720_146


def _git_blob_sha1(size: int, blocks: Sequence[bytes]) -> str:
    digest = hashlib.sha1(f"blob {size}\0".encode("ascii"))
    for block in blocks:
        digest.update(block)
    return digest.hexdigest()


def _validate_tree(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("sha") != SUBTREE_SHA or payload.get("truncated") is not False:
        raise ValueError("Brunswick Git tree identity is not the frozen subtree")
    files: list[dict[str, Any]] = []
    for row in payload.get("tree", []):
        if row.get("type") != "blob":
            continue
        relative = PurePosixPath(str(row.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError(f"invalid Brunswick source path: {relative}")
        size = int(row.get("size", -1))
        blob = str(row.get("sha", ""))
        if size < 0 or len(blob) != 40:
            raise ValueError(f"invalid Brunswick blob metadata: {relative}")
        files.append(
            {
                "path": relative.as_posix(),
                "size_bytes": size,
                "git_blob_sha1": blob,
            }
        )
    files.sort(key=lambda row: str(row["path"]))
    if len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"Brunswick file count changed: {len(files)}")
    total = sum(int(row["size_bytes"]) for row in files)
    if total != EXPECTED_TOTAL_SIZE_BYTES:
        raise ValueError(f"Brunswick subtree size changed: {total}")
    return files


def _fetch_tree() -> list[dict[str, Any]]:
    with urlopen(f"{API_BASE}/{SUBTREE_SHA}?recursive=1", timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("Brunswick Git tree response is not an object")
    return _validate_tree(payload)


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
    row: dict[str, Any],
    staging: PurePosixPath,
    ssh: Sequence[str],
    target: str,
) -> None:
    relative = PurePosixPath(str(row["path"]))
    expected_size = int(row["size_bytes"])
    expected_blob = str(row["git_blob_sha1"])
    destination = staging / relative
    partial = destination.parent / f".{destination.name}.part"
    encoded = "/".join(quote(part) for part in (SUBTREE, *relative.parts))
    url = f"{RAW_BASE}/{COMMIT}/{encoded}"
    command = (
        "set -e; "
        f"mkdir -p {shlex.quote(str(destination.parent))}; "
        f"test ! -e {shlex.quote(str(destination))}; "
        f"cat > {shlex.quote(str(partial))}; "
        f"test $(wc -c < {shlex.quote(str(partial))}) -eq {expected_size}; "
        f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))}"
    )
    transfer = subprocess.Popen(
        [*ssh, target, command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert transfer.stdin is not None
    digest = hashlib.sha1(f"blob {expected_size}\0".encode("ascii"))
    observed_size = 0
    try:
        with urlopen(url, timeout=600) as response:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                observed_size += len(block)
                digest.update(block)
                transfer.stdin.write(block)
    finally:
        transfer.stdin.close()
    stdout = transfer.stdout.read() if transfer.stdout is not None else b""
    stderr = transfer.stderr.read() if transfer.stderr is not None else b""
    result = subprocess.CompletedProcess(
        transfer.args, transfer.wait(), stdout=stdout, stderr=stderr
    )
    _check_remote(result, f"Brunswick remote transfer for {relative}")
    observed_blob = digest.hexdigest()
    if observed_size != expected_size or observed_blob != expected_blob:
        raise ValueError(
            f"Brunswick source identity changed for {relative}: "
            f"size={observed_size} blob={observed_blob}"
        )


def _manifest(files: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entries = {
        str(row["path"]): {
            "url": (
                f"{RAW_BASE}/{COMMIT}/"
                + "/".join(
                    quote(part)
                    for part in (SUBTREE, *PurePosixPath(str(row["path"])).parts)
                )
            ),
            "size_bytes": int(row["size_bytes"]),
            "git_blob_sha1": str(row["git_blob_sha1"]),
        }
        for row in files
    }
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "commit": COMMIT,
        "subtree": SUBTREE,
        "subtree_git_sha1": SUBTREE_SHA,
        "city": "Brunswick",
        "coverage": "complete tracked brunswick directory",
        "licenses": {
            "sumo_scenario": "EPL-2.0",
            "bundled_gtfs": "CC-BY-SA-4.0",
        },
        "file_count": len(entries),
        "total_size_bytes": sum(
            int(row["size_bytes"]) for row in entries.values()
        ),
        "transfer_mode": "https_stream_to_shared_cluster_no_local_file",
        "files": entries,
    }


def acquire_remote(
    *,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
) -> dict[str, Any]:
    files = _fetch_tree()
    ssh, target = _cluster_connection(
        node=node, scheduler_skill_dir=scheduler_skill_dir
    )
    staging = output_root.parent / f".{output_root.name}.staging-{SUBTREE_SHA[:8]}"
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
    _check_remote(initialization, "Brunswick remote acquisition initialization")
    try:
        for row in files:
            print(
                f"streaming {row['path']} ({row['size_bytes']} bytes)",
                flush=True,
            )
            for attempt in range(1, 5):
                try:
                    _stream_remote_file(
                        row=row,
                        staging=staging,
                        ssh=ssh,
                        target=target,
                    )
                    break
                except RuntimeError:
                    if attempt == 4:
                        raise
                    time.sleep(2**attempt)
        payload = _manifest(files)
        upload = _remote_run(
            ssh,
            target,
            "set -e; cat > "
            + shlex.quote(str(staging / "acquisition_manifest.json")),
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "Brunswick acquisition manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"set -e; mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "Brunswick acquisition finalization")
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
                "commit": payload["commit"],
                "subtree_git_sha1": payload["subtree_git_sha1"],
                "file_count": payload["file_count"],
                "total_size_bytes": payload["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
