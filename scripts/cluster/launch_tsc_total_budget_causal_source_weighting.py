#!/usr/bin/env python3
"""Submit the six-budget V140 causal source-weighting matrix."""

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
from cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting import (
    RESULT_PROTOCOL,
    TARGET_BUDGETS,
    split_adaptation_evaluation_seeds,
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
from scripts.cluster.launch_tsc_waiting_aligned_source_selector import SEEDS


LAUNCH_PROTOCOL = "tsc-v140-total-budget-causal-source-weighting-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v140_total_budget_causal_source_weighting.json"
)
FIT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v115_pure_waiting_mechanism_fit.json"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
SELECTOR_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
)
ASSIGNMENTS = (
    (25, "node001"),
    (50, "node002"),
    (100, "node003"),
    (250, "node005"),
    (500, "node006"),
    (1000, "node001"),
)
CPU_CORES = 20
RAM_MB = 98_304


def _validate_config(config: Mapping[str, Any]) -> None:
    adaptation, evaluation = split_adaptation_evaluation_seeds(SEEDS)
    partition = dict(config.get("adaptation_seed_partition", {}))
    if (
        config.get("protocol") != RESULT_PROTOCOL
        or tuple(int(value) for value in config.get("target_budgets", ()))
        != TARGET_BUDGETS
        or int(config.get("primary_budget", -1)) != 25
        or tuple(int(value) for value in partition.get("adaptation_seeds", ()))
        != adaptation
        or tuple(int(value) for value in partition.get("evaluation_seeds", ()))
        != evaluation
        or config.get("target_city") != "jinan"
        or config.get("reference_policy") != "phase_pressure"
    ):
        raise ValueError("V140 frozen protocol configuration changed")


def build_specs(
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
) -> list[dict[str, Any]]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    specs = []
    for budget, node in ASSIGNMENTS:
        remote_result = remote_output_root / f"B{budget}"
        local_result = local_output_root / f"B{budget}"
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
                "cf_h2o.eval.traffic_signal_total_budget_causal_source_weighting",
                "--budget",
                str(budget),
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
                str(remote_result / "result.json"),
            ]
        ) + " && printf 'TASK_DONE\\n'"
        specs.append(
            {
                "description": f"CFCMT V140 total-budget causal source weighting B{budget}",
                "project": "CFCMT",
                "cmd": "mkdir -p " + shlex.quote(str(remote_result)) + " && " + command,
                "cwd": str(snapshot_root),
                "signature": f"CFCMT/v140/total-budget-causal-source-weighting-v1/B{budget}",
                "resource_family": "CFCMT-v140-total-budget-causal-source-weighting",
                "ram_resource_family": "CFCMT-v140-total-budget-causal-source-weighting",
                "vram": 0,
                "ram_mb": RAM_MB,
                "cpu": CPU_CORES,
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
                "result_dir": str(remote_result),
                "local_result_dir": str(local_result.resolve()),
            }
        )
    return specs


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
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V140 launch artifact")
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("V140 cache workers must be in [1, 20]")
    if not 1 <= int(args.fit_workers) <= 20:
        raise ValueError("V140 fit workers must be in [1, 20]")
    existing_local = [
        args.local_output_root / f"B{budget}"
        for budget in TARGET_BUDGETS
        if (args.local_output_root / f"B{budget}").exists()
    ]
    if existing_local:
        raise FileExistsError(f"refusing to overwrite local V140 results: {existing_local}")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    selector_audit = _read_json(args.selector_cache_audit_local)
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V140 requires the V115 Jinan fit")
    if source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL:
        raise ValueError("V140 source cache audit changed")
    if (
        selector_audit.get("protocol") != SELECTOR_AUDIT_PROTOCOL
        or selector_audit.get("status") != "PASS"
        or tuple(int(value) for value in selector_audit.get("seeds", ())) != tuple(SEEDS)
    ):
        raise ValueError("V140 selector cache audit changed")

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
    remote_budget_roots = [args.remote_output_root / f"B{budget}" for budget in TARGET_BUDGETS]
    preflight = " && ".join(
        (
            shlex.join(["test", "-d", str(args.source_cache_root)]),
            shlex.join(["test", "-d", str(args.selector_cache_root)]),
            shlex.join(["test", "-d", str(args.conversion_root)]),
            *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
            *(shlex.join(["test", "!", "-e", str(path)]) for path in remote_budget_roots),
        )
    )
    code, _, stderr = scheduler.run_on("node001", preflight, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"remote V140 preflight failed: {stderr}")
    hash_code, hash_stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [line.split()[0] for line in str(hash_stdout).splitlines() if line.split()]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            "remote V140 evidence identities changed: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )
    specs = build_specs(
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
            "CFCMT-v140-total-budget-causal-source-weighting-v1",
        ],
        input=json.dumps(specs),
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
        "assignments": [{"budget": budget, "node": node} for budget, node in ASSIGNMENTS],
        "task_specs": specs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V140 budget matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
