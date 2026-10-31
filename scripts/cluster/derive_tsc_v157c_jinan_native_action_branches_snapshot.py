#!/usr/bin/env python3
"""Derive the V157C execution snapshot from the immutable V157B snapshot."""

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
    "/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS/645262f59ee4e0487f0f"
)
PARENT_SNAPSHOT_SHA256 = (
    "645262f59ee4e0487f0fd480e831fa3b09e1b6a66604f236a1531e207199235a"
)
PARENT_SOURCE_TREE_SHA256 = (
    "0dfbeb334f072431598c47dd13f4b40a36349ceb69c71d0c7b443abd0d02dc64"
)
PARENT_DERIVATION_PROTOCOL = "tsc-v157b-runtime-refit-freeze-snapshot-derivation-v2"
DERIVATION_PROTOCOL = "tsc-v157c-jinan-native-one-action-snapshot-derivation-v3"
RUNTIME_DEPENDENCY_PROTOCOL = (
    "tsc-v157c-fraction-occupancy-priority-yellow-observer-runtime-v1"
)
REMOTE_BASE = Path("/home/zhengliang01/scheduleurm_work/CFCMT_SNAPSHOTS")
REMOTE_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python3.10"
)
EVALUATOR_RELATIVE = Path("cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py")
EVALUATOR_PATCH_RELATIVE = Path(
    "scripts/cluster/traffic_signal_resco_cfcmt_v3_v157c.patch"
)
V2_EVALUATOR_RELATIVE = Path("cf_h2o/eval/traffic_signal_resco_cfcmt_v2.py")
OCCUPANCY_EQUATIONS_RELATIVE = Path(
    "cf_h2o/traffic_signal/occupancy_equations.py"
)
SAFE_PHASE_CONTROLLER_RELATIVE = Path(
    "cf_h2o/traffic_signal/safe_phase_controller.py"
)
OVERLAY_SHA256 = {
    Path("cf_h2o/eval/traffic_signal_v157c_jinan_native_action_branches.py"): (
        "a56986bff8fac992b81e37e3ca5c4ccb00c1cea653a8a9205deb54d0a9b70cbd"
    ),
    Path("cf_h2o/config/traffic_signal_tsc_v157c_jinan_native_action_branches.json"): (
        "31803300c6a31d009ca8b2160f9264c32b59089500b577e70f2a78ae94d03ff3"
    ),
    Path("scripts/cluster/launch_tsc_v157c_jinan_native_action_branches.py"): (
        "bef58eb9bf9cbefbe51213c9eadda9c9d3ad9428003bf578df39f397971caca6"
    ),
    EVALUATOR_PATCH_RELATIVE: (
        "a848475d38c3b0dc8e19e7cce98c7489a9fdaf81599249b7f542aa7cc9dd0a85"
    ),
    V2_EVALUATOR_RELATIVE: (
        "a371a1b7966bd8b6d17e9065f1bc1309dad059aab3bb2c2e68c70235b9a01be5"
    ),
    OCCUPANCY_EQUATIONS_RELATIVE: (
        "e5ebc569dc5c49befaff1e6246ea9913d72d4259c6c36ed66b41717139e2dce8"
    ),
    SAFE_PHASE_CONTROLLER_RELATIVE: (
        "15e6df85ca11e715aedfa71c750c617238d78685dfef588a38ae408a6df94179"
    ),
}
OVERLAYS = tuple(OVERLAY_SHA256)
PARENT_EVALUATOR_SHA256 = (
    "43eb6e33f20168cc4e80a6d20bfd93e1d0268e77cf486c3d81a12ee20f2c9b8b"
)
V157C_EVALUATOR_SHA256 = (
    "1d6a53c8929f6c517c178026444f4c1b5b4efd5d8484324fcc6c5d4243975888"
)
REPLACED_DEPENDENCY_PARENT_SHA256 = {
    V2_EVALUATOR_RELATIVE: (
        "171a9bb70cd3bf343beb2b70a1f9da48bfadfdcada650113bb5e42ab3a5cd115"
    ),
    SAFE_PHASE_CONTROLLER_RELATIVE: (
        "9477ed42637ddd6e7e8b2e0ea25cab830909ff4396cdf99d0bcd0966f5fc0f4d"
    ),
}
PARENT_ABSENT_OVERLAYS = tuple(
    path for path in OVERLAYS if path not in REPLACED_DEPENDENCY_PARENT_SHA256
)


def runtime_dependency_contract() -> dict[str, Any]:
    """Describe the exact runtime changes layered on the V157B parent."""

    return {
        "protocol": RUNTIME_DEPENDENCY_PROTOCOL,
        "patched_evaluator": {
            "path": EVALUATOR_RELATIVE.as_posix(),
            "parent_sha256": PARENT_EVALUATOR_SHA256,
            "patch_path": EVALUATOR_PATCH_RELATIVE.as_posix(),
            "patch_sha256": OVERLAY_SHA256[EVALUATOR_PATCH_RELATIVE],
            "sha256": V157C_EVALUATOR_SHA256,
            "semantics": [
                "action_originator_prepare_interval_before_tls_scoring",
                "action_originator_selection_layer_provenance",
                "step_observer_before_step_after_step_before_executor_and_after_executor",
            ],
        },
        "replaced_dependencies": [
            {
                "path": path.as_posix(),
                "parent_sha256": parent_sha256,
                "sha256": OVERLAY_SHA256[path],
            }
            for path, parent_sha256 in REPLACED_DEPENDENCY_PARENT_SHA256.items()
        ],
        "added_dependency": {
            "path": OCCUPANCY_EQUATIONS_RELATIVE.as_posix(),
            "sha256": OVERLAY_SHA256[OCCUPANCY_EQUATIONS_RELATIVE],
        },
    }


def _load_scheduler(path: Path) -> Any:
    sys.path.insert(0, str(path))
    import scheduler  # type: ignore

    return scheduler


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_local_overlays() -> dict[str, str]:
    """Bind the planned snapshot to the reviewed V157C execution files."""

    observed = {
        relative.as_posix(): _sha256(PROJECT_ROOT / relative)
        for relative in OVERLAYS
    }
    expected = {
        relative.as_posix(): expected_sha256
        for relative, expected_sha256 in OVERLAY_SHA256.items()
    }
    if observed != expected:
        raise ValueError(f"V157C local overlay identity changed: {observed}")
    return observed


def _run(scheduler: Any, command: str, *, timeout: int = 180) -> str:
    code, stdout, stderr = scheduler.run_on(
        "node001", command, timeout=timeout, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V157C snapshot command failed: {stderr[-4000:]}")
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
        raise RuntimeError(f"V157C overlay rsync failed: {completed.stderr[-4000:]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V157C snapshot manifest")

    overlay_hashes = validate_local_overlays()
    scheduler = _load_scheduler(args.scheduler_dir)
    parent_marker = json.loads(
        _run(scheduler, shlex.join(["cat", str(PARENT_ROOT / ".cfcmt_snapshot.json")]))
    )
    if (
        parent_marker.get("protocol") != PARENT_DERIVATION_PROTOCOL
        or parent_marker.get("snapshot_sha256") != PARENT_SNAPSHOT_SHA256
        or parent_marker.get("source_tree_sha256") != PARENT_SOURCE_TREE_SHA256
    ):
        raise ValueError("V157B parent snapshot identity changed")

    staging = REMOTE_BASE / f".v157c-jinan-native-action-branches-{os.getpid()}"
    committed = False
    try:
        absence_checks = [
            shlex.join(["test", "!", "-e", str(PARENT_ROOT / path)])
            for path in PARENT_ABSENT_OVERLAYS
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

        parent_dependency_paths = (
            EVALUATOR_RELATIVE,
            *REPLACED_DEPENDENCY_PARENT_SHA256,
        )
        observed_parent_dependencies = _run(
            scheduler,
            shlex.join(
                [
                    "sha256sum",
                    *(str(staging / path) for path in parent_dependency_paths),
                ]
            ),
        )
        observed_parent_hashes = [
            line.split()[0]
            for line in observed_parent_dependencies.splitlines()
            if line.split()
        ]
        expected_parent_hashes = [
            PARENT_EVALUATOR_SHA256,
            *REPLACED_DEPENDENCY_PARENT_SHA256.values(),
        ]
        if observed_parent_hashes != expected_parent_hashes:
            raise RuntimeError("V157C parent runtime dependency identity changed")

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
            raise RuntimeError("V157C remote overlay identity changed")

        _run(
            scheduler,
            shlex.join(
                [
                    "patch",
                    "--batch",
                    "--forward",
                    "-d",
                    str(staging),
                    "-p1",
                    "-i",
                    str(staging / EVALUATOR_PATCH_RELATIVE),
                ]
            ),
        )
        patched_evaluator_output = _run(
            scheduler,
            shlex.join(["sha256sum", str(staging / EVALUATOR_RELATIVE)]),
        )
        patched_evaluator_hash = patched_evaluator_output.split()[0]
        if patched_evaluator_hash != V157C_EVALUATOR_SHA256:
            raise RuntimeError("V157C patched evaluator identity changed")

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
            raise RuntimeError("V157C snapshot digest output changed")
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
            "derived_from_snapshot_root": str(PARENT_ROOT),
            "derived_from_snapshot_sha256": PARENT_SNAPSHOT_SHA256,
            "derived_from_source_tree_sha256": PARENT_SOURCE_TREE_SHA256,
            "overlays": [
                {"path": path.as_posix(), "sha256": overlay_hashes[path.as_posix()]}
                for path in OVERLAYS
            ],
            "runtime_dependencies": runtime_dependency_contract(),
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
            existing_marker = json.loads(existing)
            if (
                existing_marker.get("protocol") != DERIVATION_PROTOCOL
                or existing_marker.get("snapshot_sha256") != digest
                or existing_marker.get("source_tree_sha256") != source_digest
                or existing_marker.get("derived_from_snapshot_root")
                != str(PARENT_ROOT)
                or existing_marker.get("derived_from_snapshot_sha256")
                != PARENT_SNAPSHOT_SHA256
                or existing_marker.get("derived_from_source_tree_sha256")
                != PARENT_SOURCE_TREE_SHA256
                or existing_marker.get("overlays") != marker["overlays"]
                or existing_marker.get("runtime_dependencies")
                != marker["runtime_dependencies"]
            ):
                raise RuntimeError("V157C derived snapshot path collision")
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
    args.out.write_text(
        json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(marker, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
