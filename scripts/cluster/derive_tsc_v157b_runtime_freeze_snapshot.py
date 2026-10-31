#!/usr/bin/env python3
"""Derive the V157B execution snapshot from the immutable V157A snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEDULER_DIR = Path("/home/erzhu419/mine_code/scheduleurm/skill")
PARENT_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/5a4b8d5793b359e2d16d"
)
PARENT_SNAPSHOT_SHA256 = (
    "5a4b8d5793b359e2d16dabff24c9cc606fa13f5ee7e8d288d83706587e0a63d4"
)
PARENT_SOURCE_TREE_SHA256 = (
    "dee303d7894956b01f01a0b73ec63574787a737d6b4b48b207d82c7ceead07a4"
)
REMOTE_BASE = Path("/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python3.10"
)
DERIVATION_PROTOCOL = "tsc-v157b-runtime-refit-freeze-snapshot-derivation-v2"
CONFIG_PROTOCOL = "tsc-v157b-feature-aligned-b100-runtime-refit-freeze-v2"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v157b_feature_aligned_b100_runtime_freeze.json"
)
OVERLAYS = (
    Path("cf_h2o/eval/traffic_signal_feature_aligned_b100_runtime_freeze.py"),
    CONFIG_RELATIVE,
)


def _load_scheduler(path: Path) -> Any:
    sys.path.insert(0, str(path))
    import scheduler  # type: ignore

    return scheduler


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(scheduler: Any, command: str, *, timeout: int = 180) -> str:
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=timeout, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V157B snapshot command failed: {stderr[-4000:]}")
    return str(stdout)


def _rsync_overlay(scheduler: Any, source: Path, destination: Path) -> None:
    completed = subprocess.run(
        [
            "rsync",
            "-a",
            "--timeout=300",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            str(source),
            f"{scheduler._ssh_target_for_node('node001')}:{destination}",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"V157B overlay rsync failed: {completed.stderr[-4000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V157B snapshot manifest")

    config = json.loads((PROJECT_ROOT / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    identities = dict(config["frozen_identities"])
    if (
        config.get("protocol") != CONFIG_PROTOCOL
        or identities.get("v157a_snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or identities.get("v157a_source_tree_sha256") != PARENT_SOURCE_TREE_SHA256
    ):
        raise ValueError("V157B parent snapshot contract changed")
    overlay_hashes = {
        path.as_posix(): _sha256(PROJECT_ROOT / path) for path in OVERLAYS
    }

    scheduler = _load_scheduler(args.scheduler_dir)
    parent_marker = json.loads(
        _run(scheduler, shlex.join(["cat", str(PARENT_ROOT / ".cfcmt_snapshot.json")]))
    )
    if (
        parent_marker.get("snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or parent_marker.get("source_tree_sha256") != PARENT_SOURCE_TREE_SHA256
    ):
        raise ValueError("V157A parent snapshot identity changed")

    staging = REMOTE_BASE / f".v157b-runtime-freeze-{os.getpid()}"
    committed = False
    try:
        absence_checks = [
            shlex.join(["test", "!", "-e", str(PARENT_ROOT / path)])
            for path in OVERLAYS
        ]
        _run(
            scheduler,
            " && ".join(
                [
                    shlex.join(["test", "!", "-e", str(staging)]),
                    *absence_checks,
                    shlex.join(["cp", "-a", str(PARENT_ROOT), str(staging)]),
                ]
            ),
            timeout=300,
        )
        for relative in OVERLAYS:
            _rsync_overlay(
                scheduler,
                PROJECT_ROOT / relative,
                staging / relative,
            )

        ordered_remote = tuple(staging / path for path in OVERLAYS)
        observed_remote = _run(
            scheduler,
            shlex.join(["sha256sum", *(str(path) for path in ordered_remote)]),
        )
        remote_hashes = [
            line.split()[0] for line in observed_remote.splitlines() if line.split()
        ]
        if remote_hashes != [overlay_hashes[path.as_posix()] for path in OVERLAYS]:
            raise RuntimeError("V157B remote overlay identity changed")

        digest_program = """
import hashlib
from pathlib import Path
root = Path(ROOT)
digest = hashlib.sha256()
for relative_root in ('cf_h2o', 'scripts/cluster', 'scripts/data'):
    for path in sorted((root / relative_root).rglob('*')):
        relative = path.relative_to(root)
        if not path.is_file() or '__pycache__' in relative.parts or 'results' in relative.parts:
            continue
        if path.suffix in {'.pyc', '.pyo'}:
            continue
        digest.update(relative.as_posix().encode('utf-8'))
        digest.update(b'\\0')
        digest.update(path.read_bytes())
        digest.update(b'\\0')
source = hashlib.sha256()
source_root = root / 'cf_h2o'
for path in sorted(source_root.rglob('*.py'), key=lambda item: str(item.relative_to(source_root))):
    if '__pycache__' in path.parts or 'results' in path.parts:
        continue
    source.update(path.relative_to(source_root).as_posix().encode('utf-8'))
    source.update(b'\\0')
    source.update(path.read_bytes())
    source.update(b'\\0')
print(digest.hexdigest(), source.hexdigest())
""".replace("ROOT", repr(str(staging)))
        values = _run(
            scheduler,
            shlex.join([str(REMOTE_PYTHON), "-c", digest_program]),
            timeout=300,
        ).strip().splitlines()[-1].split()
        if len(values) != 2:
            raise RuntimeError("V157B snapshot digest output changed")
        digest, source_digest = values
        destination = REMOTE_BASE / digest[:20]
        marker = {
            "protocol": DERIVATION_PROTOCOL,
            "snapshot_sha256": digest,
            "source_tree_sha256": source_digest,
            "git_commit": parent_marker["git_commit"],
            "snapshot_root": str(destination),
            "data_root": parent_marker["data_root"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "derived_from_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "derived_from_source_tree_sha256": PARENT_SOURCE_TREE_SHA256,
            "overlays": [
                {"path": path.as_posix(), "sha256": overlay_hashes[path.as_posix()]}
                for path in OVERLAYS
            ],
        }
        marker_text = json.dumps(marker, indent=2, sort_keys=True) + "\n"
        marker_program = (
            "from pathlib import Path; "
            f"Path({str(staging / '.cfcmt_snapshot.json')!r}).write_text("
            f"{marker_text!r}, encoding='utf-8')"
        )
        _run(scheduler, shlex.join([str(REMOTE_PYTHON), "-c", marker_program]))

        existing = _run(
            scheduler,
            f"if test -e {shlex.quote(str(destination))}; then "
            f"cat {shlex.quote(str(destination / '.cfcmt_snapshot.json'))}; fi",
        ).strip()
        if existing:
            _run(scheduler, shlex.join(["rm", "-rf", str(staging)]))
            if json.loads(existing).get("snapshot_sha256") != digest:
                raise RuntimeError("V157B derived snapshot path collision")
            marker = json.loads(existing)
        else:
            _run(scheduler, shlex.join(["mv", str(staging), str(destination)]))
        committed = True
    finally:
        if not committed:
            try:
                _run(scheduler, shlex.join(["rm", "-rf", str(staging)]))
            except Exception:
                pass

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
