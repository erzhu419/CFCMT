#!/usr/bin/env python3
"""Submit the correction-only V157A replay of the complete V123 curve."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
)
from scripts.cluster.launch_tsc_target_budget_source_value_curve import (  # noqa: E402
    SEEDS,
    build_spec as build_v123_spec,
)


LAUNCH_PROTOCOL = (
    "tsc-v157a-feature-aligned-target-budget-source-value-curve-launch-v1"
)
CONFIG_RELATIVE = Path(
    "cf_h2o/config/"
    "traffic_signal_tsc_v157a_feature_aligned_target_budget_source_value_curve.json"
)
ACTION_RANKER_RELATIVE = Path("cf_h2o/traffic_signal/action_ranker.py")
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
TARGET_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
FIT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v115_pure_waiting_mechanism_fit.json"
)
SIGNATURE = "CFCMT/v157a/feature-aligned-v123-source-value-curve-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derived-snapshot", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-audit-local", type=Path, required=True)
    parser.add_argument("--target-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-local", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--legacy-result", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--node", default="node005")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fit-workers", type=int, default=20)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V157A launch artifact")

    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    config = _read_json(config_path)
    identities = dict(config["frozen_identities"])
    stage = _read_json(args.derived_snapshot)
    if (
        config.get("protocol")
        != "tsc-v157a-feature-aligned-target-budget-source-value-curve-v1"
        or tuple(config.get("selector_seeds", ())) != tuple(SEEDS)
        or stage.get("snapshot_sha256")
        != identities["derived_snapshot_sha256"]
        or stage.get("source_tree_sha256")
        != identities["derived_source_tree_sha256"]
        or stage.get("derived_from_snapshot_sha256")
        != identities["parent_snapshot_sha256"]
        or stage.get("patch", {}).get("parent_sha256")
        != identities["parent_action_ranker_sha256"]
        or stage.get("patch", {}).get("patched_sha256")
        != identities["patched_action_ranker_sha256"]
        or _sha256(args.derived_snapshot)
        != identities["derived_snapshot_marker_sha256"]
    ):
        raise ValueError("V157A derived snapshot contract changed")

    local_inputs = {
        "fit_result": args.fit_result_local,
        "source_cache_audit": args.source_cache_audit_local,
        "target_cache_audit": args.target_cache_audit_local,
        "selector_cache_audit": args.selector_cache_audit_local,
        "legacy_v123_result": args.legacy_result,
    }
    expected_local = {
        key: identities[f"{key}_sha256"] for key in local_inputs
    }
    observed_local = {key: _sha256(path) for key, path in local_inputs.items()}
    if observed_local != expected_local:
        raise ValueError(f"V157A local evidence changed: {observed_local}")

    snapshot_root = Path(stage["snapshot_root"])
    remote_inputs = {
        "fit_result": args.fit_result_remote,
        "source_cache_audit": args.source_cache_audit_remote,
        "target_cache_audit": args.target_cache_audit_remote,
        "selector_cache_audit": args.selector_cache_audit_remote,
    }
    snapshot_files = {
        "action_ranker": snapshot_root / ACTION_RANKER_RELATIVE,
        "source_manifest": snapshot_root / SOURCE_MANIFEST_RELATIVE,
        "target_manifest": snapshot_root / TARGET_MANIFEST_RELATIVE,
        "fit_protocol": snapshot_root / FIT_PROTOCOL_RELATIVE,
        "snapshot_marker": snapshot_root / ".cfcmt_snapshot.json",
    }
    expected_snapshot = {
        "action_ranker": identities["patched_action_ranker_sha256"],
        "source_manifest": identities["source_manifest_sha256"],
        "target_manifest": identities["target_manifest_sha256"],
        "fit_protocol": identities["fit_protocol_sha256"],
        "snapshot_marker": identities["derived_snapshot_marker_sha256"],
    }
    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        [
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
            shlex.join(["test", "-d", str(args.source_cache_root)]),
            shlex.join(["test", "-d", str(args.target_cache_root)]),
            shlex.join(["test", "-d", str(args.selector_cache_root)]),
            *(
                shlex.join(["test", "-f", str(path)])
                for path in (*remote_inputs.values(), *snapshot_files.values())
            ),
        ]
    )
    code, _, stderr = scheduler.run_on(
        args.node, preflight, timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V157A preflight failed: {stderr}")

    ordered_paths = (*remote_inputs.values(), *snapshot_files.values())
    code, stdout, stderr = scheduler.run_on(
        args.node,
        shlex.join(["sha256sum", *(str(path) for path in ordered_paths)]),
        timeout=120,
        check=False,
    )
    hashes = [line.split()[0] for line in stdout.splitlines() if line.split()]
    expected_hashes = [
        *(identities[f"{key}_sha256"] for key in remote_inputs),
        *(expected_snapshot[key] for key in snapshot_files),
    ]
    if int(code) != 0 or hashes != expected_hashes:
        raise RuntimeError(
            f"remote V157A identities changed: {hashes} != {expected_hashes}; {stderr}"
        )

    spec = build_v123_spec(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=identities["fit_result_sha256"],
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        target_cache_root=args.target_cache_root,
        target_cache_audit_remote=args.target_cache_audit_remote,
        selector_cache_root=args.selector_cache_root,
        selector_cache_audit_remote=args.selector_cache_audit_remote,
        selector_cache_audit_sha256=identities["selector_cache_audit_sha256"],
        remote_output_root=args.remote_output_root,
        node=args.node,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
    )
    spec.update(
        {
            "description": "CFCMT V157A feature-aligned V123 replay",
            "signature": SIGNATURE,
            "resource_family": "CFCMT-v157a-feature-aligned-v123-replay",
        }
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v157a-feature-aligned-v123-replay-v1",
        ],
        input=json.dumps([spec]),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "submitted": completed.returncode == 0,
        "scheduler_returncode": completed.returncode,
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "signature": SIGNATURE,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "source_tree_sha256": stage["source_tree_sha256"],
        "snapshot_patch": stage["patch"],
        "derived_snapshot": str(args.derived_snapshot.resolve()),
        "derived_snapshot_sha256": _sha256(args.derived_snapshot),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "local_input_sha256": observed_local,
        "remote_identity_sha256": hashes,
        "remote_output_root": str(args.remote_output_root),
        "execution_node": args.node,
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V157A replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
