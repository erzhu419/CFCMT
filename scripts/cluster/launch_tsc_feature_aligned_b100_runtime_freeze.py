#!/usr/bin/env python3
"""Submit the V157B v2 domain-aligned B100 refit and gated runtime freeze."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256  # noqa: E402
from cf_h2o.eval.traffic_signal_feature_aligned_b100_runtime_freeze import (  # noqa: E402
    ARM_ORDER,
    CONFIG_PROTOCOL,
    V157A_AGGREGATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (  # noqa: E402
    RESULT_PROTOCOL as COMPONENT_FIT_RESULT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.derive_tsc_v157b_runtime_freeze_snapshot import (  # noqa: E402
    DERIVATION_PROTOCOL,
    OVERLAYS,
    REMOTE_BASE,
)


LAUNCH_PROTOCOL = "tsc-v157b-feature-aligned-b100-runtime-refit-freeze-launch-v2"
SIGNATURE = "CFCMT/v157b/feature-aligned-b100-runtime-refit-freeze-v2"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v157b_feature_aligned_b100_runtime_freeze.json"
)
FIT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v115_pure_waiting_mechanism_fit.json"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
TARGET_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_spec(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    v157a_aggregate_remote: Path,
    v157a_prediction_artifact_remote: Path,
    fit_result_remote: Path,
    source_cache_root: Path,
    source_cache_audit_remote: Path,
    target_cache_root: Path,
    target_cache_audit_remote: Path,
    selector_cache_root: Path,
    selector_cache_audit_remote: Path,
    remote_output_root: Path,
    node: str,
    cache_workers: int,
    fit_workers: int,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            f"CFCMT_EXTERNAL_CONVERSION_ROOT={conversion_root}",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
            "PYTHONUNBUFFERED=1",
            str(wrapper),
            "-c",
            (
                "from cf_h2o.eval.traffic_signal_feature_aligned_b100_runtime_freeze "
                "import main; raise SystemExit(main())"
            ),
            "--config",
            str(snapshot_root / CONFIG_RELATIVE),
            "--v157a-aggregate",
            str(v157a_aggregate_remote),
            "--v157a-prediction-artifact",
            str(v157a_prediction_artifact_remote),
            "--fit-result",
            str(fit_result_remote),
            "--fit-protocol",
            str(snapshot_root / FIT_PROTOCOL_RELATIVE),
            "--source-cache-root",
            str(source_cache_root),
            "--source-manifest",
            str(snapshot_root / SOURCE_MANIFEST_RELATIVE),
            "--source-cache-audit",
            str(source_cache_audit_remote),
            "--target-cache-root",
            str(target_cache_root),
            "--target-manifest",
            str(snapshot_root / TARGET_MANIFEST_RELATIVE),
            "--target-cache-audit",
            str(target_cache_audit_remote),
            "--selector-cache-root",
            str(selector_cache_root),
            "--selector-manifest",
            str(snapshot_root / TARGET_MANIFEST_RELATIVE),
            "--selector-cache-audit",
            str(selector_cache_audit_remote),
            "--conversion-root",
            str(conversion_root),
            "--cache-workers",
            str(cache_workers),
            "--fit-workers",
            str(fit_workers),
            "--output-root",
            str(remote_output_root),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT V157B v2 domain-aligned B100 gated runtime freeze",
        "project": "CFCMT",
        "cmd": command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v157b-feature-aligned-b100-runtime-refit-freeze-v2",
        "vram": 0,
        "ram_mb": 98304,
        "cpu": 20,
        "priority": "high",
        "require_node": str(node),
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_output_root),
    }


def _validate_local_inputs(
    *,
    config: Mapping[str, Any],
    v157a_aggregate: Path,
    fit_result: Path,
    source_cache_audit: Path,
    target_cache_audit: Path,
    selector_cache_audit: Path,
) -> dict[str, str]:
    identities = dict(config["frozen_identities"])
    paths = {
        "v157a_aggregate": v157a_aggregate,
        "fit_result": fit_result,
        "source_cache_audit": source_cache_audit,
        "target_cache_audit": target_cache_audit,
        "selector_cache_audit": selector_cache_audit,
    }
    observed = {name: _sha256(path) for name, path in paths.items()}
    expected = {name: identities[f"{name}_sha256"] for name in paths}
    if observed != expected:
        raise ValueError(f"V157B local input identity changed: {observed}")
    aggregate = _read_json(v157a_aggregate)
    fit = _read_json(fit_result)
    if (
        aggregate.get("protocol") != V157A_AGGREGATE_PROTOCOL
        or aggregate.get("b100_branch_authorization", {}).get("passed") is not True
        or fit.get("protocol") != COMPONENT_FIT_RESULT_PROTOCOL
        or fit.get("city") != "jinan"
        or tuple(config.get("arm_order", ())) != ARM_ORDER
    ):
        raise ValueError("V157B prerequisite result contract changed")
    return observed


def _validate_stage_manifest(
    stage: Mapping[str, Any], *, config: Mapping[str, Any]
) -> Path:
    """Bind the launch to the reviewed v2 overlays and their derived snapshot."""

    identities = dict(config["frozen_identities"])
    digest = str(stage.get("snapshot_sha256", ""))
    snapshot_root = Path(str(stage.get("snapshot_root", "")))
    overlay_rows = stage.get("overlays", ())
    if not isinstance(overlay_rows, list):
        raise ValueError("V157B stage manifest overlays changed")
    observed_overlays = {
        str(row.get("path")): str(row.get("sha256"))
        for row in overlay_rows
        if isinstance(row, Mapping)
    }
    expected_overlays = {
        relative.as_posix(): _sha256(PROJECT_ROOT / relative)
        for relative in OVERLAYS
    }
    if (
        stage.get("protocol") != DERIVATION_PROTOCOL
        or stage.get("derived_from_snapshot_sha256")
        != identities["v157a_snapshot_sha256"]
        or stage.get("derived_from_source_tree_sha256")
        != identities["v157a_source_tree_sha256"]
        or len(overlay_rows) != len(expected_overlays)
        or observed_overlays != expected_overlays
        or len(digest) != 64
        or snapshot_root != REMOTE_BASE / digest[:20]
    ):
        raise ValueError("V157B derived snapshot manifest contract changed")
    return snapshot_root


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--v157a-aggregate-local", type=Path, required=True)
    parser.add_argument("--v157a-aggregate-remote", type=Path, required=True)
    parser.add_argument("--v157a-prediction-artifact-remote", type=Path, required=True)
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
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--node", default="node005")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fit-workers", type=int, default=16)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V157B launch artifact")
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V157B cache workers must be in [1, 20]")
    if not 1 <= int(args.fit_workers) <= 16:
        raise ValueError("V157B v2 fit workers must be in [1, 16]")

    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    config = _read_json(config_path)
    if config.get("protocol") != CONFIG_PROTOCOL:
        raise ValueError("V157B config protocol changed")
    local_hashes = _validate_local_inputs(
        config=config,
        v157a_aggregate=args.v157a_aggregate_local,
        fit_result=args.fit_result_local,
        source_cache_audit=args.source_cache_audit_local,
        target_cache_audit=args.target_cache_audit_local,
        selector_cache_audit=args.selector_cache_audit_local,
    )
    stage = _read_json(args.stage_manifest)
    snapshot_root = _validate_stage_manifest(stage, config=config)
    snapshot_files = {
        "stage_marker": snapshot_root / ".cfcmt_snapshot.json",
        "evaluator": snapshot_root / OVERLAYS[0],
        "config": snapshot_root / CONFIG_RELATIVE,
        "fit_protocol": snapshot_root / FIT_PROTOCOL_RELATIVE,
        "source_manifest": snapshot_root / SOURCE_MANIFEST_RELATIVE,
        "target_manifest": snapshot_root / TARGET_MANIFEST_RELATIVE,
    }
    remote_inputs = {
        "v157a_aggregate": args.v157a_aggregate_remote,
        "v157a_prediction_artifact": args.v157a_prediction_artifact_remote,
        "fit_result": args.fit_result_remote,
        "source_cache_audit": args.source_cache_audit_remote,
        "target_cache_audit": args.target_cache_audit_remote,
        "selector_cache_audit": args.selector_cache_audit_remote,
    }
    identities = dict(config["frozen_identities"])
    expected_remote = [identities[f"{name}_sha256"] for name in remote_inputs]
    expected_snapshot = [
        _sha256(args.stage_manifest),
        _sha256(PROJECT_ROOT / OVERLAYS[0]),
        _sha256(config_path),
        identities["fit_protocol_sha256"],
        identities["source_manifest_sha256"],
        identities["target_manifest_sha256"],
    ]
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
        raise RuntimeError(f"remote V157B preflight failed: {stderr}")
    ordered_paths = (*remote_inputs.values(), *snapshot_files.values())
    code, stdout, stderr = scheduler.run_on(
        args.node,
        shlex.join(["sha256sum", *(str(path) for path in ordered_paths)]),
        timeout=300,
        check=False,
    )
    remote_hashes = [line.split()[0] for line in stdout.splitlines() if line.split()]
    expected_hashes = [*expected_remote, *expected_snapshot]
    if int(code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            f"remote V157B identities changed: {remote_hashes} != {expected_hashes}; {stderr}"
        )
    spec = build_spec(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        v157a_aggregate_remote=args.v157a_aggregate_remote,
        v157a_prediction_artifact_remote=args.v157a_prediction_artifact_remote,
        fit_result_remote=args.fit_result_remote,
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        target_cache_root=args.target_cache_root,
        target_cache_audit_remote=args.target_cache_audit_remote,
        selector_cache_root=args.selector_cache_root,
        selector_cache_audit_remote=args.selector_cache_audit_remote,
        remote_output_root=args.remote_output_root,
        node=args.node,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v157b-feature-aligned-b100-runtime-refit-freeze-v2",
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
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "signature": SIGNATURE,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "local_input_sha256": local_hashes,
        "remote_identity_sha256": remote_hashes,
        "remote_output_root": str(args.remote_output_root),
        "execution_node": str(args.node),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V157B runtime freeze")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
