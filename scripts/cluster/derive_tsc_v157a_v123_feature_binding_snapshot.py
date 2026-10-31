#!/usr/bin/env python3
"""Derive the V123 snapshot with only name-based rigid feature binding."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any


DEFAULT_SCHEDULER_DIR = Path("/home/erzhu419/mine_code/scheduleurm/skill")
PARENT_ROOT = Path(
    "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/8281502f843b7886a4fe"
)
PARENT_SNAPSHOT_SHA256 = (
    "8281502f843b7886a4fe09385ef2b30d4b04700b5cc7cf8f27deaf0c074567d2"
)
RELATIVE_PATCH = Path("cf_h2o/traffic_signal/action_ranker.py")
PARENT_FILE_SHA256 = (
    "4619ef412db1b6417208c1af88d02c03161df8978011a6bb952cd3492d11203c"
)
REMOTE_BASE = Path("/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python3.10"
)
OLD_LINE = (
    "        features = np.asarray(dataset.features[:, self.feature_indices], dtype=float)"
)
NEW_LINES = (
    "        indices = _feature_indices(dataset.feature_names, self.feature_names)\n"
    "        features = np.asarray(dataset.features[:, indices], dtype=float)"
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
        raise RuntimeError(f"remote V157A snapshot command failed: {stderr}")
    return str(stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V157A snapshot manifest")

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
        raise ValueError("V123 parent snapshot identity changed")

    staging = REMOTE_BASE / f".v157a-v123-feature-binding-{os.getpid()}"
    _run(
        scheduler,
        " && ".join(
            [
                shlex.join(["test", "!", "-e", str(staging)]),
                shlex.join(["cp", "-a", str(PARENT_ROOT), str(staging)]),
            ]
        ),
    )
    patch_program = (
        "from pathlib import Path; "
        f"p=Path({str(staging / RELATIVE_PATCH)!r}); "
        "s=p.read_text(encoding='utf-8'); "
        f"assert s.count({OLD_LINE!r}) == 2; "
        f"assert {NEW_LINES!r} not in s; "
        f"i=s.rfind({OLD_LINE!r}); assert i >= 0; "
        f"p.write_text(s[:i] + {NEW_LINES!r} + s[i + len({OLD_LINE!r}):], "
        "encoding='utf-8')"
    )
    _run(scheduler, shlex.join([str(REMOTE_PYTHON), "-c", patch_program]))

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
patched = hashlib.sha256((root / PATCH).read_bytes()).hexdigest()
print(digest.hexdigest(), source.hexdigest(), patched)
""".replace("ROOT", repr(str(staging))).replace(
        "PATCH", repr(RELATIVE_PATCH.as_posix())
    )
    values = _run(
        scheduler, shlex.join([str(REMOTE_PYTHON), "-c", digest_program])
    ).strip().splitlines()[-1].split()
    if len(values) != 3:
        raise RuntimeError("V157A snapshot digest output changed")
    digest, source_digest, patched_file_hash = values
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
            "patched_sha256": patched_file_hash,
            "change": "resolve PairwiseActionAdvantageRegressor inputs by stored feature names",
        },
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
            raise RuntimeError("V157A derived snapshot path collision")
        marker = json.loads(existing)
    else:
        _run(scheduler, shlex.join(["mv", str(staging), str(destination)]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
