#!/usr/bin/env python3
"""Acquire the complete fixed InTAS Git tree on shared cluster storage."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shlex
import sys
import time
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.data.acquire_toronto_ptc_unseen import (
    USER_AGENT,
    _check_remote,
    _cluster_connection,
    _remote_run,
    _stream_url,
)


PROTOCOL = "cfcmt-intas-fixed-git-acquisition-v1"
REPOSITORY = "silaslobo/InTAS"
COMMIT = "0f7951ba01dda8483f0a852f2c3e4ff0d8a1c0ee"
ARCHIVE_NAME = f"InTAS-{COMMIT}.tar.gz"
ARCHIVE_URL = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{COMMIT}"
COMMIT_API = f"https://api.github.com/repos/{REPOSITORY}/commits/{COMMIT}"
TREE_API = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
EXPECTED_BLOB_COUNT = 35
EXPECTED_BLOB_BYTES = 983_190_873
REQUIRED_PATHS = {
    "LICENSE",
    "README.md",
    "scenario/InTAS_buildings.sumocfg",
    "scenario/ingolstadt.net.xml",
    "scenario/BusStations.add.xml",
    "scenario/routes/ped.rou.xml",
    "scenario/routes/BusRoutes.flow.xml",
    *{
        f"scenario/routes/InTAS_{index:03d}.rou.xml"
        for index in range(1, 23)
    },
}


def _fetch_json(url: str) -> Mapping[str, Any]:
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def _validate_git_tree(
    *, commit_payload: Mapping[str, Any], tree_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if commit_payload.get("sha") != COMMIT:
        raise ValueError(
            f"InTAS commit identity changed: {commit_payload.get('sha')}"
        )
    if tree_payload.get("truncated") is not False:
        raise ValueError("InTAS Git tree response is truncated")
    blobs = [
        {
            "path": str(row["path"]),
            "size_bytes": int(row.get("size", 0)),
            "git_sha": str(row["sha"]),
        }
        for row in tree_payload.get("tree", [])
        if row.get("type") == "blob"
    ]
    if len(blobs) != EXPECTED_BLOB_COUNT:
        raise ValueError(
            f"InTAS blob count changed: {len(blobs)} != {EXPECTED_BLOB_COUNT}"
        )
    blob_bytes = sum(row["size_bytes"] for row in blobs)
    if blob_bytes != EXPECTED_BLOB_BYTES:
        raise ValueError(
            f"InTAS blob bytes changed: {blob_bytes} != {EXPECTED_BLOB_BYTES}"
        )
    paths = {row["path"] for row in blobs}
    missing = sorted(REQUIRED_PATHS - paths)
    if missing:
        raise ValueError(f"InTAS required paths disappeared: {missing}")
    return sorted(blobs, key=lambda row: row["path"])


def _extraction_command(
    *, staging: PurePosixPath, blobs: Sequence[Mapping[str, Any]]
) -> str:
    source = staging / "source"
    archive = staging / ARCHIVE_NAME
    checks = [
        "set -e;",
        f"mkdir -p {shlex.quote(str(source))};",
        f"tar -xzf {shlex.quote(str(archive))} "
        f"-C {shlex.quote(str(source))} --strip-components=1;",
    ]
    for row in blobs:
        path = source / str(row["path"])
        checks.append(f"test -f {shlex.quote(str(path))};")
        checks.append(
            f"test $(wc -c < {shlex.quote(str(path))}) -eq {int(row['size_bytes'])};"
        )
    checks.append(
        f"test $(find {shlex.quote(str(source))} -type f | wc -l) "
        f"-eq {EXPECTED_BLOB_COUNT}"
    )
    return " ".join(checks)


def _manifest(
    *,
    archive: Mapping[str, Any],
    archive_entry_count: int,
    blobs: Sequence[Mapping[str, Any]],
    commit_payload: Mapping[str, Any],
) -> dict[str, Any]:
    commit = commit_payload.get("commit", {})
    return {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": REPOSITORY,
        "commit": COMMIT,
        "commit_url": f"https://github.com/{REPOSITORY}/commit/{COMMIT}",
        "commit_committed_at": commit.get("committer", {}).get("date"),
        "archive_name": ARCHIVE_NAME,
        "archive_url": ARCHIVE_URL,
        "archive_entry_count": int(archive_entry_count),
        "archive_size_bytes": int(archive["size_bytes"]),
        "archive_sha256": str(archive["sha256"]),
        "blob_count": len(blobs),
        "blob_bytes": sum(int(row["size_bytes"]) for row in blobs),
        "coverage": "complete fixed InTAS Git tree",
        "transfer_mode": (
            "https_stream_through_local_to_shared_cluster_no_local_raw_file"
        ),
        "source_subdirectory": "source",
        "blobs": list(blobs),
    }


def _remote_run_transport_retry(
    ssh: Sequence[str],
    target: str,
    command: str,
    *,
    stdin: bytes | None = None,
    attempts: int = 3,
) -> Any:
    result = None
    for attempt in range(attempts):
        result = _remote_run(ssh, target, command, stdin=stdin)
        if result.returncode != 255:
            return result
        if attempt + 1 < attempts:
            time.sleep(5)
    assert result is not None
    return result


def _finalize_command(
    *, staging: PurePosixPath, output_root: PurePosixPath
) -> str:
    return (
        "set -e; "
        f"if test -d {shlex.quote(str(output_root))}; then "
        f"test ! -e {shlex.quote(str(staging))}; "
        f"test -f {shlex.quote(str(output_root / 'acquisition_manifest.json'))}; "
        "else "
        f"mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}; "
        "fi"
    )


def acquire_remote(
    *, output_root: PurePosixPath, node: str, scheduler_skill_dir: Path
) -> dict[str, Any]:
    commit_payload = _fetch_json(COMMIT_API)
    tree_payload = _fetch_json(TREE_API)
    blobs = _validate_git_tree(
        commit_payload=commit_payload,
        tree_payload=tree_payload,
    )
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
    _check_remote(initialization, "InTAS acquisition initialization")
    try:
        print(f"streaming InTAS commit {COMMIT}", flush=True)
        archive = _stream_url(
            ssh=ssh,
            target=target,
            staging=staging,
            name=ARCHIVE_NAME,
            url=ARCHIVE_URL,
            expected_size=None,
        )
        count_result = _remote_run(
            ssh,
            target,
            (
                "set -o pipefail; "
                f"tar -tzf {shlex.quote(str(staging / ARCHIVE_NAME))} | wc -l"
            ),
        )
        _check_remote(count_result, "InTAS archive inventory")
        archive_entry_count = int(count_result.stdout.decode("ascii").strip())
        if archive_entry_count <= 0:
            raise ValueError("InTAS archive has no entries")

        extraction = _remote_run(
            ssh,
            target,
            _extraction_command(staging=staging, blobs=blobs),
        )
        _check_remote(extraction, "InTAS fixed-tree extraction")
        payload = _manifest(
            archive=archive,
            archive_entry_count=archive_entry_count,
            blobs=blobs,
            commit_payload=commit_payload,
        )
        upload = _remote_run_transport_retry(
            ssh,
            target,
            f"cat > {shlex.quote(str(staging / 'acquisition_manifest.json'))}",
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "InTAS acquisition manifest upload")
        finalize = _remote_run_transport_retry(
            ssh,
            target,
            _finalize_command(staging=staging, output_root=output_root),
        )
        _check_remote(finalize, "InTAS acquisition finalization")
        return payload
    except Exception:
        _remote_run_transport_retry(
            ssh,
            target,
            f"rm -rf -- {shlex.quote(str(staging))}",
        )
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
                "protocol": payload["protocol"],
                "commit": payload["commit"],
                "blob_count": payload["blob_count"],
                "blob_bytes": payload["blob_bytes"],
                "archive_size_bytes": payload["archive_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
