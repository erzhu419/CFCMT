#!/usr/bin/env python3
"""Submit the frozen B25 V142 policy-aligned source-selection diagnostic."""

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

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_pure_waiting_selector_cache_audit import (
    RESULT_PROTOCOL as SELECTOR_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_total_budget_policy_aligned_selection import (
    PRIMARY_BUDGET,
    RESULT_PROTOCOL,
    split_policy_aligned_seeds,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
    SOURCE_AUDIT_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.launch_tsc_total_budget_causal_source_weighting import (
    FIT_PROTOCOL_RELATIVE,
    SELECTOR_MANIFEST_RELATIVE,
    SOURCE_MANIFEST_RELATIVE,
)
from scripts.cluster.launch_tsc_waiting_aligned_source_selector import SEEDS


LAUNCH_PROTOCOL = "tsc-v142-total-budget-policy-aligned-source-selection-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v142_total_budget_policy_aligned_selection.json"
)
CPU_CORES = 20
RAM_MB = 98_304
ALLOWED_NODES = ("node001", "node002", "node003", "node005", "node006")


def _validate_config(config: Mapping[str, Any]) -> None:
    adaptation, evaluation = split_policy_aligned_seeds(SEEDS)
    partition = dict(config.get("seed_partition", {}))
    candidate = dict(config.get("candidate_family", {}))
    if (
        config.get("protocol") != RESULT_PROTOCOL
        or int(config.get("target_budget", -1)) != PRIMARY_BUDGET
        or config.get("target_city") != "jinan"
        or config.get("reference_policy") != "phase_pressure"
        or tuple(int(value) for value in partition.get("adaptation_seeds", ()))
        != adaptation
        or tuple(int(value) for value in partition.get("evaluation_seeds", ()))
        != evaluation
        or int(candidate.get("candidate_count", -1)) != 33
    ):
        raise ValueError("V142 frozen protocol configuration changed")


def build_spec(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    fit_result_remote: Path,
    fit_result_sha256: str,
    source_cache_root: Path,
    source_cache_audit_remote: Path,
    selector_cache_root: Path,
    selector_cache_audit_remote: Path,
    selector_cache_audit_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
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
            "-m",
            "cf_h2o.eval.traffic_signal_total_budget_policy_aligned_selection",
            "--budget",
            str(PRIMARY_BUDGET),
            "--fit-result",
            str(fit_result_remote),
            "--fit-result-sha256",
            fit_result_sha256,
            "--fit-protocol",
            str(snapshot_root / FIT_PROTOCOL_RELATIVE),
            "--source-cache-root",
            str(source_cache_root),
            "--source-manifest",
            str(snapshot_root / SOURCE_MANIFEST_RELATIVE),
            "--source-cache-audit",
            str(source_cache_audit_remote),
            "--selector-cache-root",
            str(selector_cache_root),
            "--selector-manifest",
            str(snapshot_root / SELECTOR_MANIFEST_RELATIVE),
            "--selector-cache-audit",
            str(selector_cache_audit_remote),
            "--selector-cache-audit-sha256",
            selector_cache_audit_sha256,
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
            "--fit-workers",
            str(fit_workers),
            "--out",
            str(remote_output_root / "result.json"),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT V142 B25 policy-aligned source selection",
        "project": "CFCMT",
        "cmd": "mkdir -p " + shlex.quote(str(remote_output_root)) + " && " + command,
        "cwd": str(snapshot_root),
        "signature": "CFCMT/v142/total-budget-policy-aligned-source-selection-v1/B25",
        "resource_family": "CFCMT-v142-policy-aligned-source-selection",
        "ram_resource_family": "CFCMT-v140-total-budget-causal-source-weighting",
        "vram": 0,
        "ram_mb": RAM_MB,
        "cpu": CPU_CORES,
        "priority": "high",
        "allowed_nodes": list(ALLOWED_NODES),
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_output_root),
        "local_result_dir": str(local_output_root.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--selector-cache-root", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-local", type=Path, required=True)
    parser.add_argument("--selector-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fit-workers", type=int, default=20)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V142 launch or local result")
    if not 1 <= int(args.cache_workers) <= 20 or not 1 <= int(args.fit_workers) <= 20:
        raise ValueError("V142 workers must be in [1, 20]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    selector_audit = _read_json(args.selector_cache_audit_local)
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V142 requires the V115 Jinan fit")
    if source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL:
        raise ValueError("V142 source cache audit changed")
    if (
        selector_audit.get("protocol") != SELECTOR_AUDIT_PROTOCOL
        or selector_audit.get("status") != "PASS"
        or tuple(int(value) for value in selector_audit.get("seeds", ()))
        != tuple(SEEDS)
    ):
        raise ValueError("V142 selector cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    fit_sha = _sha256(args.fit_result_local)
    source_audit_sha = _sha256(args.source_cache_audit_local)
    selector_audit_sha = _sha256(args.selector_cache_audit_local)
    remote_inputs = (
        args.fit_result_remote,
        args.source_cache_audit_remote,
        args.selector_cache_audit_remote,
    )
    expected_hashes = [fit_sha, source_audit_sha, selector_audit_sha]
    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        (
            shlex.join(["test", "-d", str(args.source_cache_root)]),
            shlex.join(["test", "-d", str(args.selector_cache_root)]),
            shlex.join(["test", "-d", str(args.conversion_root)]),
            *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
            shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
        )
    )
    code, _, stderr = scheduler.run_on("node001", preflight, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"remote V142 preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [
        line.split()[0] for line in str(hash_stdout).splitlines() if line.split()
    ]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote V142 evidence identities changed: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )
    spec = build_spec(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=fit_sha,
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        selector_cache_root=args.selector_cache_root,
        selector_cache_audit_remote=args.selector_cache_audit_remote,
        selector_cache_audit_sha256=selector_audit_sha,
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
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
            "CFCMT-v142-policy-aligned-source-selection-v1",
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
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": stage["snapshot_sha256"],
        "remote_evidence_sha256": remote_hashes,
        "remote_output_root": str(args.remote_output_root),
        "task_spec": spec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V142 diagnostic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
