#!/usr/bin/env python3
"""Derive the V150O snapshot by applying only the feature-binding fix."""

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
    "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/0eb4466ee3617a0cda75"
)
PARENT_SNAPSHOT_SHA256 = (
    "0eb4466ee3617a0cda755e1b2229ecccafbee76cf06acaad0d24250eab9228e0"
)
RELATIVE_PATCH = Path("cf_h2o/traffic_signal/action_ranker.py")
PARENT_FILE_SHA256 = (
    "b503a4e004ee44e6c630b0e2e7ccfb07eff418e6fe2dc7faf0cdc67a0d73253f"
)
PATCHED_FILE_SHA256 = (
    "aa70a323feff130dfbbfb750149d913c913d7ea7d8c20939212e1dd1aeef4c6d"
)
REMOTE_BASE = Path("/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python3.10"
)


def _load_scheduler(path: Path) -> Any:
    sys.path.insert(0, str(path))
    import scheduler  # type: ignore

    return scheduler


def _run(scheduler: Any, command: str) -> str:
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V156A snapshot command failed: {stderr}")
    return str(stdout)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V156A derived snapshot manifest")
    patch_path = PROJECT_ROOT / RELATIVE_PATCH
    if _sha256(patch_path) != PATCHED_FILE_SHA256:
        raise ValueError("local feature-binding patch identity changed")

    scheduler = _load_scheduler(args.scheduler_dir)
    parent_marker = json.loads(
        _run(scheduler, shlex.join(["cat", str(PARENT_ROOT / ".cfcmt_snapshot.json")]))
    )
    parent_file_hash = _run(
        scheduler, shlex.join(["sha256sum", str(PARENT_ROOT / RELATIVE_PATCH)])
    ).split()[0]
    if (
        parent_marker.get("snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or parent_file_hash != PARENT_FILE_SHA256
    ):
        raise ValueError("V150O parent snapshot identity changed")

    staging = REMOTE_BASE / f".v156a-feature-binding-{os.getpid()}"
    _run(
        scheduler,
        " && ".join(
            [
                shlex.join(["test", "!", "-e", str(staging)]),
                shlex.join(["cp", "-a", str(PARENT_ROOT), str(staging)]),
            ]
        ),
    )
    target = scheduler._ssh_target_for_node("node001")
    copied = subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            str(patch_path),
            f"{target}:{staging / RELATIVE_PATCH}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if copied.returncode != 0:
        _run(scheduler, shlex.join(["rm", "-rf", str(staging)]))
        raise RuntimeError(f"V156A patch transfer failed: {copied.stderr}")

    digest_script = """
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
    digest_values = _run(
        scheduler, shlex.join([str(REMOTE_PYTHON), "-c", digest_script])
    ).strip().splitlines()[-1].split()
    if len(digest_values) != 2:
        raise RuntimeError("V156A derived snapshot digest output changed")
    digest, source_digest = digest_values
    destination = REMOTE_BASE / digest[:20]
    marker = {
        "snapshot_sha256": digest,
        "source_tree_sha256": source_digest,
        "git_commit": parent_marker["git_commit"],
        "snapshot_root": str(destination),
        "data_root": parent_marker["data_root"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "derived_from_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
        "patch": {
            "path": RELATIVE_PATCH.as_posix(),
            "parent_sha256": PARENT_FILE_SHA256,
            "patched_sha256": PATCHED_FILE_SHA256,
            "change": "resolve PairwiseActionAdvantageRegressor inputs by stored feature names",
        },
    }
    marker_text = json.dumps(marker, indent=2, sort_keys=True) + "\n"
    marker_script = (
        "from pathlib import Path; "
        f"Path({str(staging / '.cfcmt_snapshot.json')!r}).write_text({marker_text!r}, encoding='utf-8')"
    )
    _run(scheduler, shlex.join([str(REMOTE_PYTHON), "-c", marker_script]))
    existing = _run(
        scheduler,
        f"if test -e {shlex.quote(str(destination))}; then "
        f"cat {shlex.quote(str(destination / '.cfcmt_snapshot.json'))}; fi",
    ).strip()
    if existing:
        _run(scheduler, shlex.join(["rm", "-rf", str(staging)]))
        if json.loads(existing).get("snapshot_sha256") != digest:
            raise RuntimeError("V156A derived snapshot path collision")
    else:
        _run(scheduler, shlex.join(["mv", str(staging), str(destination)]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
