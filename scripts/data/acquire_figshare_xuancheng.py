#!/usr/bin/env python3
"""Stream the frozen Xuancheng Figshare dataset directly to a cluster node."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
from typing import Any, Sequence
from urllib.request import urlopen


PROTOCOL = "figshare-xuancheng-v5-streamed-acquisition-v1"
ARTICLE_ID = 29_925_824
ARTICLE_VERSION = 5
ARTICLE_DOI = "10.6084/m9.figshare.29925824"
API_URL = (
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/versions/"
    f"{ARTICLE_VERSION}"
)
STATIC_FILES = (
    "config_xuancheng_test.json",
    "roadnet_xuancheng250319.json",
    "xuancheng.net.xml",
)
CANDIDATE_DAY = "data_2023_04_03_type_filtered.json"
DAILY_PATTERN = re.compile(r"data_2023_04_(\d{2})_type_filtered\.json$")


def load_metadata() -> dict[str, Any]:
    with urlopen(API_URL, timeout=120) as response:
        return json.load(response)


def select_files(
    metadata: dict[str, Any], *, coverage: str
) -> list[dict[str, Any]]:
    if int(metadata.get("id", -1)) != ARTICLE_ID:
        raise ValueError("Figshare article ID changed")
    if int(metadata.get("version", -1)) != ARTICLE_VERSION:
        raise ValueError("Figshare article version changed")
    by_name = {str(row["name"]): dict(row) for row in metadata.get("files", [])}
    missing_static = sorted(set(STATIC_FILES) - set(by_name))
    if missing_static:
        raise ValueError(f"Xuancheng static files are missing: {missing_static}")
    daily = {
        int(match.group(1)): row
        for name, row in by_name.items()
        if (match := DAILY_PATTERN.fullmatch(name)) is not None
    }
    if set(daily) != set(range(1, 31)):
        raise ValueError(
            "Xuancheng daily coverage changed: "
            f"observed={sorted(daily)}"
        )
    if coverage == "candidate_screen":
        names = [*STATIC_FILES, CANDIDATE_DAY]
    elif coverage == "full_month":
        names = [*STATIC_FILES, *(daily[day]["name"] for day in range(1, 31))]
    else:
        raise ValueError(f"unknown Xuancheng coverage: {coverage}")
    selected = [by_name[str(name)] for name in names]
    for row in selected:
        if int(row.get("size", 0)) <= 0:
            raise ValueError(f"Figshare file has invalid size: {row.get('name')}")
        supplied = str(row.get("supplied_md5", ""))
        computed = str(row.get("computed_md5", ""))
        if not supplied or supplied != computed:
            raise ValueError(f"Figshare MD5 metadata disagrees: {row.get('name')}")
        if not str(row.get("download_url", "")).startswith("https://"):
            raise ValueError(f"Figshare download URL is invalid: {row.get('name')}")
    return selected


def _cluster_connection(
    *, node: str, scheduler_skill_dir: Path
) -> tuple[list[str], str]:
    sys.path.insert(0, str(scheduler_skill_dir.resolve()))
    import scheduler  # type: ignore[import-not-found]

    ssh = shlex.split(scheduler._ssh_rsync_shell_for_node(node))
    target = str(scheduler._ssh_target_for_node(node))
    return ssh, target


def _remote_run(
    ssh: Sequence[str], target: str, command: str, *, stdin=None
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*ssh, target, command],
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _check_remote(result: subprocess.CompletedProcess[bytes], context: str) -> None:
    if result.returncode == 0:
        return
    stderr = result.stderr.decode("utf-8", errors="replace").strip()
    raise RuntimeError(f"{context} failed ({result.returncode}): {stderr}")


def _stream_file(
    *,
    row: dict[str, Any],
    staging_root: PurePosixPath,
    ssh: Sequence[str],
    target: str,
) -> None:
    name = str(row["name"])
    destination = staging_root / name
    partial = staging_root / f".{name}.part"
    expected_size = int(row["size"])
    expected_md5 = str(row["computed_md5"])
    remote_command = (
        "set -e; "
        f"test ! -e {shlex.quote(str(destination))}; "
        f"cat > {shlex.quote(str(partial))}; "
        f"test $(wc -c < {shlex.quote(str(partial))}) -eq {expected_size}; "
        f"test $(md5sum {shlex.quote(str(partial))} | cut -d' ' -f1) = "
        f"{shlex.quote(expected_md5)}; "
        f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))}"
    )
    download = subprocess.Popen(
        [
            "curl",
            "-fsSL",
            "--retry",
            "4",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "30",
            "--max-time",
            "7200",
            str(row["download_url"]),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert download.stdout is not None
    transfer = _remote_run(
        ssh,
        target,
        remote_command,
        stdin=download.stdout,
    )
    download.stdout.close()
    download_stderr = download.stderr.read() if download.stderr is not None else b""
    download_returncode = download.wait()
    if download_returncode != 0:
        message = download_stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"download failed for {name}: {message}")
    _check_remote(transfer, f"remote validation for {name}")


def _upload_bytes(
    *,
    payload: bytes,
    destination: PurePosixPath,
    ssh: Sequence[str],
    target: str,
) -> None:
    command = f"set -e; cat > {shlex.quote(str(destination))}"
    result = subprocess.run(
        [*ssh, target, command],
        input=payload,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    _check_remote(result, f"upload {destination.name}")


def acquire_remote(
    *,
    metadata: dict[str, Any],
    coverage: str,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
) -> dict[str, Any]:
    selected = select_files(metadata, coverage=coverage)
    ssh, target = _cluster_connection(
        node=node,
        scheduler_skill_dir=scheduler_skill_dir,
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
    _check_remote(init, "remote acquisition initialization")
    try:
        for row in selected:
            print(
                f"streaming {row['name']} ({int(row['size'])} bytes)",
                flush=True,
            )
            _stream_file(
                row=row,
                staging_root=staging,
                ssh=ssh,
                target=target,
            )
        manifest = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "article_id": ARTICLE_ID,
            "article_version": ARTICLE_VERSION,
            "article_doi": ARTICLE_DOI,
            "article_title": str(metadata.get("title", "")),
            "license": str(dict(metadata.get("license", {})).get("name", "")),
            "coverage": coverage,
            "candidate_day": CANDIDATE_DAY if coverage == "candidate_screen" else None,
            "daily_day_count": 1 if coverage == "candidate_screen" else 30,
            "file_count": len(selected),
            "total_size_bytes": sum(int(row["size"]) for row in selected),
            "files": {
                str(row["name"]): {
                    "figshare_file_id": int(row["id"]),
                    "size_bytes": int(row["size"]),
                    "md5": str(row["computed_md5"]),
                    "download_url": str(row["download_url"]),
                }
                for row in selected
            },
        }
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        _upload_bytes(
            payload=manifest_bytes,
            destination=staging / "acquisition_manifest.json",
            ssh=ssh,
            target=target,
        )
        finalize = _remote_run(
            ssh,
            target,
            f"set -e; mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "remote acquisition finalization")
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
    parser.add_argument(
        "--coverage",
        choices=("candidate_screen", "full_month"),
        required=True,
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--remote-node", default="node001")
    parser.add_argument(
        "--scheduler-skill-dir",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm/skill"),
    )
    args = parser.parse_args(argv)
    metadata = load_metadata()
    manifest = acquire_remote(
        metadata=metadata,
        coverage=args.coverage,
        output_root=PurePosixPath(args.output_root),
        node=args.remote_node,
        scheduler_skill_dir=args.scheduler_skill_dir,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "coverage": manifest["coverage"],
                "file_count": manifest["file_count"],
                "total_size_bytes": manifest["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
