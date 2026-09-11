#!/usr/bin/env python3
"""Derive the immutable V158 snapshot from the corrected V157C snapshot."""

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
    "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/102b77f2df56f9400f85"
)
PARENT_PROTOCOL = "tsc-v157c-jinan-native-one-action-snapshot-derivation-v3"
PARENT_SNAPSHOT_SHA256 = (
    "102b77f2df56f9400f85702f548fe849e0524ca0b7063f93ebc648c1e9f1be4c"
)
PARENT_SOURCE_TREE_SHA256 = (
    "5ff3c557b1dc442886020bf2e2fc38311296afca87c7282b83772b1ed9f604c2"
)
DERIVATION_PROTOCOL = "tsc-v158-native-prefix-ranking-snapshot-derivation-v1"
REMOTE_BASE = Path("/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python3.10"
)
RUNNER_RELATIVE = Path(
    "cf_h2o/eval/traffic_signal_v158_native_prefix_ranking_calibration.py"
)
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v158_native_prefix_ranking_calibration.json"
)
ACTION_RANKER_RELATIVE = Path("cf_h2o/traffic_signal/action_ranker.py")
LAUNCHER_RELATIVE = Path(
    "scripts/cluster/launch_tsc_v158_native_prefix_ranking_calibration.py"
)
OVERLAYS = (
    RUNNER_RELATIVE,
    CONFIG_RELATIVE,
    ACTION_RANKER_RELATIVE,
    LAUNCHER_RELATIVE,
)
PARENT_ABSENT_OVERLAYS = (RUNNER_RELATIVE, CONFIG_RELATIVE, LAUNCHER_RELATIVE)
PARENT_ACTION_RANKER_SHA256 = (
    "ca75447e88c7d91250449a0dc78f5d4b287bc69dd5c71f14efabdfea8eb8f4ee"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_scheduler(directory: Path) -> Any:
    sys.path.insert(0, str(directory))
    import scheduler  # type: ignore

    return scheduler


def _run(scheduler: Any, command: str, *, timeout: int = 180) -> str:
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=timeout, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V158 snapshot command failed: {stderr[-4000:]}")
    return str(stdout)


def _rsync(scheduler: Any, source: Path, destination: Path) -> None:
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
        raise RuntimeError(f"V158 overlay rsync failed: {completed.stderr[-4000:]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V158 snapshot manifest")
    overlay_hashes = {
        path.as_posix(): _sha256(PROJECT_ROOT / path) for path in OVERLAYS
    }
    scheduler = _load_scheduler(args.scheduler_dir)
    parent_marker = json.loads(
        _run(scheduler, shlex.join(["cat", str(PARENT_ROOT / ".cfcmt_snapshot.json")]))
    )
    if (
        parent_marker.get("protocol") != PARENT_PROTOCOL
        or parent_marker.get("snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or parent_marker.get("source_tree_sha256") != PARENT_SOURCE_TREE_SHA256
    ):
        raise ValueError("V158 corrected V157C parent identity changed")
    staging = REMOTE_BASE / f".v158-native-prefix-ranking-{os.getpid()}"
    committed = False
    try:
        checks = [
            shlex.join(["test", "!", "-e", str(staging)]),
            *(shlex.join(["test", "!", "-e", str(PARENT_ROOT / path)]) for path in PARENT_ABSENT_OVERLAYS),
        ]
        _run(
            scheduler,
            " && ".join(
                [*checks, shlex.join(["cp", "-a", str(PARENT_ROOT), str(staging)])]
            ),
            timeout=300,
        )
        parent_ranker = _run(
            scheduler,
            shlex.join(["sha256sum", str(staging / ACTION_RANKER_RELATIVE)]),
        ).split()[0]
        if parent_ranker != PARENT_ACTION_RANKER_SHA256:
            raise RuntimeError("V158 parent action-ranker identity changed")
        for relative in OVERLAYS:
            _rsync(scheduler, PROJECT_ROOT / relative, staging / relative)
        observed = [
            line.split()[0]
            for line in _run(
                scheduler,
                shlex.join(["sha256sum", *(str(staging / path) for path in OVERLAYS)]),
            ).splitlines()
            if line.split()
        ]
        expected = [overlay_hashes[path.as_posix()] for path in OVERLAYS]
        if observed != expected:
            raise RuntimeError("V158 remote overlays differ from reviewed local files")
        import_program = (
            "from cf_h2o.traffic_signal.action_ranker import "
            "CausalReferenceResidualRegressor; "
            "from cf_h2o.eval.traffic_signal_v158_native_prefix_ranking_calibration "
            "import _read_json,validate_protocol; "
            f"from pathlib import Path; validate_protocol(_read_json(Path({str(staging / CONFIG_RELATIVE)!r}))); "
            "print('V158_IMPORT_PASS')"
        )
        wrapper = staging / "scripts/cluster/run_cfcmt_sumo122.sh"
        smoke = _run(
            scheduler,
            shlex.join(
                [
                    "env",
                    f"CFCMT_SOURCE_ROOT={staging}",
                    str(wrapper),
                    "-c",
                    import_program,
                ]
            ),
            timeout=300,
        )
        if "V158_IMPORT_PASS" not in smoke.splitlines():
            raise RuntimeError("V158 staged module import failed")
        digest_program = """
import hashlib
from pathlib import Path
root=Path(ROOT)
digest=hashlib.sha256()
for relative_root in ('cf_h2o','scripts/cluster','scripts/data'):
    for path in sorted((root/relative_root).rglob('*')):
        relative=path.relative_to(root)
        if not path.is_file() or '__pycache__' in relative.parts or 'results' in relative.parts: continue
        if path.suffix in {'.pyc','.pyo'}: continue
        digest.update(relative.as_posix().encode()); digest.update(b'\\0'); digest.update(path.read_bytes()); digest.update(b'\\0')
source=hashlib.sha256(); source_root=root/'cf_h2o'
for path in sorted(source_root.rglob('*.py'),key=lambda p:str(p.relative_to(source_root))):
    if '__pycache__' in path.parts or 'results' in path.parts: continue
    source.update(path.relative_to(source_root).as_posix().encode()); source.update(b'\\0'); source.update(path.read_bytes()); source.update(b'\\0')
print(digest.hexdigest(),source.hexdigest())
""".replace("ROOT", repr(str(staging)))
        values = _run(
            scheduler,
            shlex.join([str(REMOTE_PYTHON), "-c", digest_program]),
            timeout=300,
        ).strip().splitlines()[-1].split()
        if len(values) != 2:
            raise RuntimeError("V158 snapshot digest output changed")
        digest, source_digest = values
        destination = REMOTE_BASE / digest[:20]
        marker = {
            "protocol": DERIVATION_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": parent_marker["git_commit"],
            "snapshot_root": str(destination),
            "snapshot_sha256": digest,
            "source_tree_sha256": source_digest,
            "data_root": parent_marker["data_root"],
            "derived_from_snapshot_root": str(PARENT_ROOT),
            "derived_from_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "derived_from_source_tree_sha256": PARENT_SOURCE_TREE_SHA256,
            "overlays": [
                {"path": path.as_posix(), "sha256": overlay_hashes[path.as_posix()]}
                for path in OVERLAYS
            ],
            "action_ranker_change": {
                "parent_sha256": PARENT_ACTION_RANKER_SHA256,
                "sha256": overlay_hashes[ACTION_RANKER_RELATIVE.as_posix()],
                "semantics": "add_CausalReferenceResidualRegressor_only",
            },
        }
        marker_text = json.dumps(marker, indent=2, sort_keys=True) + "\n"
        writer = (
            "from pathlib import Path; "
            f"Path({str(staging / '.cfcmt_snapshot.json')!r}).write_text({marker_text!r},encoding='utf-8')"
        )
        _run(scheduler, shlex.join([str(REMOTE_PYTHON), "-c", writer]))
        existing = _run(
            scheduler,
            f"if test -e {shlex.quote(str(destination))}; then "
            f"cat {shlex.quote(str(destination / '.cfcmt_snapshot.json'))}; fi",
        ).strip()
        if existing:
            _run(scheduler, shlex.join(["rm", "-rf", str(staging)]))
            existing_marker = json.loads(existing)
            comparable = dict(existing_marker)
            comparable.pop("created_at_utc", None)
            expected_marker = dict(marker)
            expected_marker.pop("created_at_utc", None)
            if comparable != expected_marker:
                raise RuntimeError("V158 content-addressed snapshot path collision")
            marker = existing_marker
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
    args.out.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
