#!/usr/bin/env python3
"""Stage an immutable CFCMT source snapshot on the shared HPC filesystem."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULER_DIR = Path("/home/erzhu419/mine_code/scheduleurm/skill")


def _staged_files() -> Iterable[Path]:
    for relative_root in (
        Path("cf_h2o"),
        Path("scripts/cluster"),
        Path("scripts/data"),
    ):
        root = PROJECT_ROOT / relative_root
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(PROJECT_ROOT)
            if not path.is_file():
                continue
            if "__pycache__" in relative.parts or "results" in relative.parts:
                continue
            if path.suffix in {".pyc", ".pyo"}:
                continue
            yield path


def snapshot_sha256() -> str:
    digest = hashlib.sha256()
    for path in _staged_files():
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _load_scheduler(path: Path):
    sys.path.insert(0, str(path))
    import scheduler  # type: ignore

    return scheduler


def _run_remote(scheduler, command: str) -> str:
    code, stdout, stderr = scheduler.run_on("node001", command, timeout=120, check=True)
    if int(code) != 0:
        raise RuntimeError(f"remote command failed ({code}): {stderr}")
    return str(stdout)


def _rsync_tree(scheduler, source: Path, destination: str) -> None:
    target = scheduler._ssh_target_for_node("node001")
    for attempt in range(1, 5):
        ssh_shell = scheduler._ssh_rsync_shell_for_node("node001")
        result = subprocess.run(
            [
                "rsync",
                "-a",
                "--exclude=__pycache__/",
                "--exclude=.pytest_cache/",
                "--exclude=results/",
                "--exclude=*.pyc",
                "-e",
                ssh_shell,
                f"{source}/",
                f"{target}:{destination}/",
            ],
            check=False,
        )
        if result.returncode == 0:
            return
        if attempt == 4:
            raise subprocess.CalledProcessError(result.returncode, result.args)
        time.sleep(2**attempt)


def stage_snapshot(
    *,
    scheduler_dir: Path,
    remote_snapshot_base: Path,
    remote_data_root: Path,
) -> dict[str, object]:
    scheduler = _load_scheduler(scheduler_dir)
    sys.path.insert(0, str(PROJECT_ROOT))
    from cf_h2o.sumo_runtime import source_tree_sha256

    snapshot_digest = snapshot_sha256()
    source_digest = source_tree_sha256(PROJECT_ROOT / "cf_h2o")
    snapshot_root = remote_snapshot_base / snapshot_digest[:20]
    marker_path = snapshot_root / ".cfcmt_snapshot.json"
    existing = _run_remote(
        scheduler,
        f"if test -f {shlex.quote(str(marker_path))}; then cat {shlex.quote(str(marker_path))}; fi",
    ).strip()
    if existing:
        payload = json.loads(existing)
        if payload.get("snapshot_sha256") != snapshot_digest:
            raise RuntimeError(f"immutable snapshot collision at {snapshot_root}")
        return payload

    staging_root = Path(f"{snapshot_root}.staging-{os.getpid()}")
    _run_remote(
        scheduler,
        "mkdir -p "
        + " ".join(
            shlex.quote(str(path))
            for path in (
                staging_root / "cf_h2o",
                staging_root / "scripts/cluster",
                staging_root / "scripts/data",
                staging_root / "H2Oplus",
            )
        ),
    )
    _rsync_tree(scheduler, PROJECT_ROOT / "cf_h2o", str(staging_root / "cf_h2o"))
    _rsync_tree(
        scheduler,
        PROJECT_ROOT / "scripts/cluster",
        str(staging_root / "scripts/cluster"),
    )
    _rsync_tree(
        scheduler,
        PROJECT_ROOT / "scripts/data",
        str(staging_root / "scripts/data"),
    )
    _run_remote(
        scheduler,
        "ln -s "
        f"{shlex.quote(str(remote_data_root))} "
        f"{shlex.quote(str(staging_root / 'H2Oplus/downloads'))}",
    )

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    payload: dict[str, object] = {
        "snapshot_sha256": snapshot_digest,
        "source_tree_sha256": source_digest,
        "git_commit": git_commit,
        "snapshot_root": str(snapshot_root),
        "data_root": str(remote_data_root),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with tempfile.TemporaryDirectory(prefix="cfcmt-stage-", dir="/tmp") as temporary:
        marker = Path(temporary) / ".cfcmt_snapshot.json"
        marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _rsync_tree(scheduler, Path(temporary), str(staging_root))
    _run_remote(
        scheduler,
        "if test -e "
        f"{shlex.quote(str(snapshot_root))}; then exit 17; "
        f"else mv {shlex.quote(str(staging_root))} {shlex.quote(str(snapshot_root))}; fi",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument(
        "--remote-snapshot-base",
        type=Path,
        default=Path("/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS"),
    )
    parser.add_argument(
        "--remote-data-root",
        type=Path,
        default=Path("/home/zhengliang01/scheduleurm_work/CFCMT/H2Oplus/downloads"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "cf_h2o/results/cluster/latest_snapshot.json",
    )
    args = parser.parse_args()
    payload = stage_snapshot(
        scheduler_dir=args.scheduler_dir,
        remote_snapshot_base=args.remote_snapshot_base,
        remote_data_root=args.remote_data_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
