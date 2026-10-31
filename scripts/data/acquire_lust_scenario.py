#!/usr/bin/env python3
"""Stream the complete LuST DUE-static variant to the shared cluster."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Any, Sequence
from urllib.request import urlopen


PROTOCOL = "lust-fixed-commit-due-static-full-acquisition-v1"
REPOSITORY = "https://github.com/lcodeca/LuSTScenario"
RAW_BASE = "https://raw.githubusercontent.com/lcodeca/LuSTScenario"
COMMIT = "5edb7ecb9ad196c39172b6eb95d19ed789f4b1a6"
FILES = {
    "CHANGELOG.md": (1_969, "6f97f85dddf484d0eff179a12417f862686201da"),
    "LICENSE.md": (1_134, "6619bb029a85064aef765ec09b1c9cbe2c5bbf85"),
    "README.md": (1_776, "df9962d4a7e1bc184f4cdec1ddcfe03912b31fc8"),
    "scenario/DUERoutes/local.static.0.rou.xml": (
        41_634_764,
        "eb9ea7c75a05fe063ed618d636a9e26620a7d799",
    ),
    "scenario/DUERoutes/local.static.1.rou.xml": (
        42_862_343,
        "21f82dbdf52eee7d20807d1c0545f3df2eec37ec",
    ),
    "scenario/DUERoutes/local.static.2.rou.xml": (
        44_934_267,
        "7d2bdfdbab6a654d31d2bb44c82e0b197de15065",
    ),
    "scenario/buslines.rou.xml": (
        3_169_632,
        "884bf0cc44c6b75a4964f78603a474ee54ca6566",
    ),
    "scenario/busstops.add.xml": (
        67_058,
        "59da379c0313b69e73ecc52411991e76216298d8",
    ),
    "scenario/due.static.sumocfg": (
        1_237,
        "3dc5b9a36cd4b4edd7a90c6216d2b428a6dd15f4",
    ),
    "scenario/lust.net.xml": (
        10_940_662,
        "c507ce16eb340bede88ac0c8e74c19519e09409f",
    ),
    "scenario/tll.static.xml": (
        83_530,
        "995f29b39a187851f1f8ebd89df6096f4e925ce3",
    ),
    "scenario/transit.rou.xml": (
        35_681_718,
        "be9aa90ec293e8abad326e05b886a78c8a13bc1e",
    ),
    "scenario/vtypes.add.xml": (
        1_707,
        "3a859fe8cb2a99b24dfe4a638275bdaedfe6568b",
    ),
}
DEMAND_FILES = (
    "scenario/buslines.rou.xml",
    "scenario/DUERoutes/local.static.0.rou.xml",
    "scenario/DUERoutes/local.static.1.rou.xml",
    "scenario/DUERoutes/local.static.2.rou.xml",
    "scenario/transit.rou.xml",
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
    relative: str,
    expected_size: int,
    expected_blob: str,
    staging: PurePosixPath,
    ssh: Sequence[str],
    target: str,
) -> None:
    destination = staging / relative
    partial = destination.parent / f".{destination.name}.part"
    url = f"{RAW_BASE}/{COMMIT}/{relative}"
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
    returncode = transfer.wait()
    result = subprocess.CompletedProcess(
        transfer.args,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )
    _check_remote(result, f"remote validation for {relative}")
    if observed_size != expected_size or digest.hexdigest() != expected_blob:
        raise ValueError(
            f"LuST source identity changed for {relative}: "
            f"size={observed_size} blob={digest.hexdigest()}"
        )


def acquire_remote(
    *,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
) -> dict[str, Any]:
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
    _check_remote(init, "LuST acquisition initialization")
    try:
        for relative, (size, blob) in FILES.items():
            print(f"streaming {relative} ({size} bytes)", flush=True)
            _stream_file(
                relative=relative,
                expected_size=size,
                expected_blob=blob,
                staging=staging,
                ssh=ssh,
                target=target,
            )
        manifest = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository": REPOSITORY,
            "commit": COMMIT,
            "license": "MIT",
            "variant": "DUE static",
            "coverage": "all demand and runtime inputs named by due.static.sumocfg",
            "source_sumo_version": "0.26",
            "modern_sumo_real_data_validation_claimed": False,
            "file_count": len(FILES),
            "total_size_bytes": sum(size for size, _ in FILES.values()),
            "demand_files": list(DEMAND_FILES),
            "files": {
                relative: {
                    "size_bytes": size,
                    "git_blob_sha1": blob,
                    "url": f"{RAW_BASE}/{COMMIT}/{relative}",
                }
                for relative, (size, blob) in FILES.items()
            },
        }
        payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        upload = subprocess.run(
            [
                *ssh,
                target,
                "set -e; cat > "
                + shlex.quote(str(staging / "acquisition_manifest.json")),
            ],
            input=payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        _check_remote(upload, "LuST manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"set -e; mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "LuST acquisition finalization")
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
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--remote-node", default="node001")
    parser.add_argument(
        "--scheduler-skill-dir",
        type=Path,
        default=Path("/home/erzhu419/mine_code/scheduleurm/skill"),
    )
    args = parser.parse_args(argv)
    manifest = acquire_remote(
        output_root=PurePosixPath(args.output_root),
        node=args.remote_node,
        scheduler_skill_dir=args.scheduler_skill_dir,
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "variant": manifest["variant"],
                "file_count": manifest["file_count"],
                "total_size_bytes": manifest["total_size_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
