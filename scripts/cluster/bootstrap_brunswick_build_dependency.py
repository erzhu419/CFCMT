#!/usr/bin/env python3
"""Install the frozen pyproj wheel used only to build DLR Brunswick."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
from typing import Sequence
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.acquire_dlr_brunswick import (
    _check_remote,
    _cluster_connection,
    _remote_run,
)


PROTOCOL = "cfcmt-brunswick-build-pyproj-wheel-v1"
PYPROJ_VERSION = "3.7.1"
WHEEL = (
    "pyproj-3.7.1-cp310-cp310-manylinux_2_17_x86_64."
    "manylinux2014_x86_64.whl"
)
WHEEL_URL = (
    "https://files.pythonhosted.org/packages/76/a5/"
    "c6e11b9a99ce146741fb4d184d5c468446c6d6015b183cae82ac822a6cfa/"
    + WHEEL
)
WHEEL_SIZE = 9_259_185
WHEEL_SHA256 = "1e47c4e93b88d99dd118875ee3ca0171932444cdc0b52d493371b5d98d0f30ee"
PYTHON = PurePosixPath(
    "/home/zhengliang01/scheduleurm_work/conda_envs/"
    "freqduet-cpu-py310/bin/python3.10"
)


def _stream_wheel(
    *,
    destination: PurePosixPath,
    ssh: Sequence[str],
    target: str,
) -> None:
    partial = destination.with_name(f".{destination.name}.part")
    command = (
        "set -e; "
        f"cat > {shlex.quote(str(partial))}; "
        f"test $(wc -c < {shlex.quote(str(partial))}) -eq {WHEEL_SIZE}; "
        f"mv {shlex.quote(str(partial))} {shlex.quote(str(destination))}"
    )
    transfer = subprocess.Popen(
        [*ssh, target, command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert transfer.stdin is not None
    digest = hashlib.sha256()
    observed_size = 0
    try:
        with urlopen(WHEEL_URL, timeout=600) as response:
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
    _check_remote(result, "Brunswick pyproj wheel transfer")
    if observed_size != WHEEL_SIZE or digest.hexdigest() != WHEEL_SHA256:
        raise ValueError(
            "Brunswick pyproj wheel identity changed: "
            f"size={observed_size} sha256={digest.hexdigest()}"
        )


def bootstrap(
    *,
    output_root: PurePosixPath,
    node: str,
    scheduler_skill_dir: Path,
) -> dict[str, object]:
    ssh, target = _cluster_connection(
        node=node, scheduler_skill_dir=scheduler_skill_dir
    )
    staging = output_root.parent / f".{output_root.name}.staging-pyproj371"
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
    _check_remote(initialization, "Brunswick dependency initialization")
    try:
        wheel_path = staging / WHEEL
        _stream_wheel(destination=wheel_path, ssh=ssh, target=target)
        site_packages = staging / "site-packages"
        install = _remote_run(
            ssh,
            target,
            (
                "set -e; "
                f"{shlex.quote(str(PYTHON))} -m pip install "
                "--no-index --no-deps "
                f"--target {shlex.quote(str(site_packages))} "
                f"{shlex.quote(str(wheel_path))}"
            ),
        )
        _check_remote(install, "Brunswick pyproj offline installation")
        verify = _remote_run(
            ssh,
            target,
            (
                "set -e; "
                f"PYTHONNOUSERSITE=1 PYTHONPATH={shlex.quote(str(site_packages))} "
                f"{shlex.quote(str(PYTHON))} -c "
                + shlex.quote(
                    "import pyproj; assert pyproj.__version__ == '3.7.1'; "
                    "print(pyproj.__version__)"
                )
            ),
        )
        _check_remote(verify, "Brunswick pyproj import verification")
        observed_version = verify.stdout.decode("utf-8").strip()
        payload: dict[str, object] = {
            "protocol": PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package": "pyproj",
            "version": PYPROJ_VERSION,
            "license": "MIT",
            "source": "PyPI official release file",
            "wheel": WHEEL,
            "wheel_url": WHEEL_URL,
            "wheel_size_bytes": WHEEL_SIZE,
            "wheel_sha256": WHEEL_SHA256,
            "python": str(PYTHON),
            "site_packages": str(output_root / "site-packages"),
            "import_verified": observed_version == PYPROJ_VERSION,
        }
        upload = _remote_run(
            ssh,
            target,
            "set -e; cat > "
            + shlex.quote(str(staging / "dependency_manifest.json")),
            stdin=(json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        )
        _check_remote(upload, "Brunswick dependency manifest upload")
        finalize = _remote_run(
            ssh,
            target,
            f"set -e; mv {shlex.quote(str(staging))} {shlex.quote(str(output_root))}",
        )
        _check_remote(finalize, "Brunswick dependency finalization")
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
    payload = bootstrap(
        output_root=PurePosixPath(args.remote_output_root),
        node=args.remote_node,
        scheduler_skill_dir=args.scheduler_skill_dir,
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
