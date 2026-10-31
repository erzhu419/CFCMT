#!/usr/bin/env python3
"""Submit the V119 B100 target-calibrated causal source gate."""

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
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (  # noqa: E402
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_pure_waiting_selector_cache_audit import (  # noqa: E402
    RESULT_PROTOCOL as SELECTOR_AUDIT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.launch_tsc_waiting_aligned_source_selector import (  # noqa: E402
    SEEDS,
)


LAUNCH_PROTOCOL = "tsc-v119-target-calibrated-source-gate-launch-v1"
SIGNATURE = "CFCMT/v119/target-calibrated-causal-source-gate-v1"
TARGET_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)


def build_spec(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    fit_result_remote: Path,
    fit_result_sha256: str,
    target_cache_root: Path,
    target_cache_audit_remote: Path,
    selector_cache_root: Path,
    selector_cache_audit_remote: Path,
    selector_cache_audit_sha256: str,
    remote_output_root: Path,
    node: str,
    cache_workers: int,
    prediction_workers: int,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / TARGET_MANIFEST_RELATIVE
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
            "-m",
            "cf_h2o.eval.traffic_signal_target_calibrated_source_gate",
            "--fit-result",
            str(fit_result_remote),
            "--fit-result-sha256",
            str(fit_result_sha256),
            "--target-cache-root",
            str(target_cache_root),
            "--target-manifest",
            str(manifest),
            "--target-cache-audit",
            str(target_cache_audit_remote),
            "--selector-cache-root",
            str(selector_cache_root),
            "--selector-manifest",
            str(manifest),
            "--selector-cache-audit",
            str(selector_cache_audit_remote),
            "--selector-cache-audit-sha256",
            str(selector_cache_audit_sha256),
            "--selector-scenario",
            "jinan_3x4_real",
            "--selector-seeds",
            *(str(value) for value in SEEDS),
            "--selector-collection-shards",
            "32",
            "--conversion-root",
            str(conversion_root),
            "--cache-workers",
            str(cache_workers),
            "--prediction-workers",
            str(prediction_workers),
            "--gate-artifact",
            str(remote_output_root / "target_calibrated_source_gate_v1.pkl"),
            "--out",
            str(remote_output_root / "result.json"),
        ]
    )
    return {
        "description": "CFCMT V119 B100 target-calibrated causal source gate",
        "project": "CFCMT",
        "cmd": command,
        "cwd": str(snapshot_root),
        "signature": SIGNATURE,
        "resource_family": "CFCMT-v119-target-calibrated-source-gate",
        "vram": 0,
        "ram_mb": 65536,
        "cpu": 20,
        "priority": "high",
        "require_node": str(node),
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_output_root),
    }


def _require_local_contracts(
    fit_result: Path,
    selector_audit: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = _read_json(fit_result)
    selector = _read_json(selector_audit)
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("local V115 Jinan fit result changed")
    if (
        selector.get("protocol") != SELECTOR_AUDIT_PROTOCOL
        or selector.get("status") != "PASS"
        or not selector.get("gate", {}).get("passed", False)
        or [int(value) for value in selector.get("seeds", ())] != list(SEEDS)
    ):
        raise ValueError("local V116 selector audit changed")
    return fit, selector


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--target-cache-root", type=Path, required=True)
    parser.add_argument("--target-cache-audit-local", type=Path, required=True)
    parser.add_argument("--target-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-local", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--node", default="node001")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--prediction-workers", type=int, default=7)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V119 launch artifact")
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V119 cache workers must be in [1, 20]")
    if not 1 <= int(args.prediction_workers) <= 7:
        raise ValueError("V119 prediction workers must be in [1, 7]")
    _, selector_audit = _require_local_contracts(
        args.fit_result_local, args.selector_cache_audit_local
    )
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    fit_sha = _sha256(args.fit_result_local)
    target_audit_sha = _sha256(args.target_cache_audit_local)
    selector_audit_sha = _sha256(args.selector_cache_audit_local)
    scheduler = _load_scheduler(args.scheduler)
    remote_inputs = (
        args.fit_result_remote,
        args.target_cache_audit_remote,
        args.selector_cache_audit_remote,
    )
    preflight = " && ".join(
        (
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
            shlex.join(["test", "-d", str(args.target_cache_root)]),
            shlex.join(["test", "-d", str(args.selector_cache_root)]),
            *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
        )
    )
    code, _, stderr = scheduler.run_on(
        args.node, preflight, timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V119 preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        args.node,
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(hash_stdout).splitlines() if line.split()
    ]
    expected_hashes = [fit_sha, target_audit_sha, selector_audit_sha]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote V119 evidence identities changed: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )
    spec = build_spec(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=fit_sha,
        target_cache_root=args.target_cache_root,
        target_cache_audit_remote=args.target_cache_audit_remote,
        selector_cache_root=args.selector_cache_root,
        selector_cache_audit_remote=args.selector_cache_audit_remote,
        selector_cache_audit_sha256=selector_audit_sha,
        remote_output_root=args.remote_output_root,
        node=args.node,
        cache_workers=args.cache_workers,
        prediction_workers=args.prediction_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v119-target-calibrated-source-gate-v1",
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
        "stage_manifest": str(args.stage_manifest.resolve()),
        "stage_manifest_sha256": _sha256(args.stage_manifest),
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "fit_result_sha256": fit_sha,
        "target_cache_audit_sha256": target_audit_sha,
        "selector_cache_audit_sha256": selector_audit_sha,
        "selector_cache_file_count": selector_audit["cache_file_count"],
        "remote_output_root": str(args.remote_output_root),
        "execution_node": str(args.node),
        "cache_workers": int(args.cache_workers),
        "prediction_workers": int(args.prediction_workers),
        "remote_evidence_sha256": remote_hashes,
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V119 source-gate submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
