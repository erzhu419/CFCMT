#!/usr/bin/env python3
"""Submit the seven V150D rigid mechanism-prior integration targets."""

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
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    FOLD_COUNT,
    MINIMUM_OOF_GAIN,
    RIDGE_L2,
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
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


LAUNCH_PROTOCOL = "tsc-v150d-rigid-mechanism-prior-integration-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150d_rigid_mechanism_prior_integration.json"
)
FIT_PROTOCOL_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v115_pure_waiting_mechanism_fit.json"
)
SOURCE_MANIFEST_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"
)
MODULE = "cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration"
SIGNATURE_PREFIX = "CFCMT/v150d/rigid-mechanism-prior-integration-v1"
NODES = (
    "node001",
    "node002",
    "node003",
    "node005",
    "node006",
    "node001",
    "node002",
)


def _validate_config(config: Mapping[str, Any]) -> None:
    gate = dict(config.get("development_gate", {}))
    if (
        config.get("protocol")
        != "tsc-v150d-rigid-mechanism-prior-integration-v1"
        or config.get("authorization_protocol") != AUTHORIZATION_PROTOCOL
        or tuple(config.get("target_city_groups", ())) != EXPECTED_CITY_GROUPS
        or int(config.get("target_budget", -1)) != TARGET_BUDGET
        or int(config.get("adaptation_fold_count", -1)) != FOLD_COUNT
        or config.get("integration")
        != "rigid_score_plus_source_mechanism_score_minus_target_mechanism_score"
        or float(config.get("ridge_l2", -1.0)) != RIDGE_L2
        or float(config.get("source_prior_strength", -1.0))
        != SOURCE_PRIOR_STRENGTH
        or float(config.get("minimum_oof_gain", -1.0)) != MINIMUM_OOF_GAIN
        or config.get("inference_unit") != "city"
        or int(gate.get("minimum_improving_cities_vs_rigid", -1)) != 5
        or int(gate.get("minimum_cities_beating_same_candidate_placebo", -1))
        != 4
        or float(gate.get("maximum_city_regression", -1.0)) != 0.01
        or int(gate.get("minimum_source_admitting_cities", -1)) != 5
    ):
        raise ValueError("V150D frozen protocol configuration changed")


def build_specs(
    *,
    snapshot_root: Path,
    authorization_result_remote: Path,
    authorization_sha256: str,
    conversion_root: Path,
    fit_result_remote: Path,
    fit_result_sha256: str,
    source_cache_root: Path,
    source_cache_audit_remote: Path,
    source_cache_audit_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
    cache_workers: int,
    module: str = MODULE,
    signature_prefix: str = SIGNATURE_PREFIX,
    version_label: str = "V150D",
    resource_family: str = "CFCMT-v150d-rigid-mechanism-prior",
    extra_target_args: Sequence[str] = (),
) -> list[dict[str, Any]]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    specs = []
    for city, node in zip(EXPECTED_CITY_GROUPS, NODES, strict=True):
        remote_city = remote_output_root / city
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
                str(module),
                "target",
                "--target-city",
                city,
                "--authorization-result",
                str(authorization_result_remote),
                "--authorization-result-sha256",
                authorization_sha256,
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
                "--source-cache-audit-sha256",
                source_cache_audit_sha256,
                "--conversion-root",
                str(conversion_root),
                "--cache-workers",
                str(cache_workers),
                "--out",
                str(remote_city / "result.json"),
                *[str(value) for value in extra_target_args],
            ]
        ) + " && printf 'TASK_DONE\\n'"
        specs.append(
            {
                "description": f"CFCMT {version_label} rigid mechanism prior target {city}",
                "project": "CFCMT",
                "cmd": "mkdir -p "
                + shlex.quote(str(remote_city))
                + " && "
                + command,
                "cwd": str(snapshot_root),
                "signature": f"{signature_prefix}/{city}",
                "resource_family": str(resource_family),
                "ram_resource_family": str(resource_family),
                "vram": 0,
                "ram_mb": 65_536,
                "cpu": 8,
                "priority": "high",
                "require_node": node,
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
                "result_dir": str(remote_city),
                "local_result_dir": str((local_output_root / city).resolve()),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--authorization-result-local", type=Path, required=True)
    parser.add_argument("--authorization-result-remote", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150D launch or results")
    if not 1 <= int(args.cache_workers) <= 8:
        raise ValueError("V150D cache workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    authorization = _read_json(args.authorization_result_local)
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if (
        authorization.get("protocol") != AUTHORIZATION_PROTOCOL
        or authorization.get("development_gate", {}).get("passed") is not True
    ):
        raise ValueError("V150D requires a passing V150C authorization")
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V150D requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or source_audit.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("V150D source cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    authorization_sha = _sha256(args.authorization_result_local)
    fit_sha = _sha256(args.fit_result_local)
    source_audit_sha = _sha256(args.source_cache_audit_local)
    remote_inputs = (
        args.authorization_result_remote,
        args.fit_result_remote,
        args.source_cache_audit_remote,
    )
    expected_hashes = [authorization_sha, fit_sha, source_audit_sha]
    scheduler = _load_scheduler(args.scheduler)
    remote_city_roots = [args.remote_output_root / city for city in EXPECTED_CITY_GROUPS]
    preflight = " && ".join(
        (
            shlex.join(["test", "-d", str(args.source_cache_root)]),
            shlex.join(["test", "-d", str(args.conversion_root)]),
            *(shlex.join(["test", "-f", str(path)]) for path in remote_inputs),
            *(shlex.join(["test", "!", "-e", str(path)]) for path in remote_city_roots),
        )
    )
    code, _, stderr = scheduler.run_on("node001", preflight, timeout=120, check=False)
    if int(code) != 0:
        raise RuntimeError(f"remote V150D preflight failed: {stderr}")
    hash_code, stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", *(str(path) for path in remote_inputs)]),
        timeout=120,
        check=False,
    )
    remote_hashes = [line.split()[0] for line in str(stdout).splitlines() if line.split()]
    if int(hash_code) != 0 or remote_hashes != expected_hashes:
        raise RuntimeError(
            f"remote V150D evidence identities changed: {remote_hashes}; {hash_stderr}"
        )
    specs = build_specs(
        snapshot_root=snapshot_root,
        authorization_result_remote=args.authorization_result_remote,
        authorization_sha256=authorization_sha,
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=fit_sha,
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        source_cache_audit_sha256=source_audit_sha,
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
        cache_workers=args.cache_workers,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v150d-rigid-mechanism-prior-integration-v1",
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
        "authorization_result_sha256": authorization_sha,
        "fit_result_sha256": fit_sha,
        "source_cache_audit_sha256": source_audit_sha,
        "remote_output_root": str(args.remote_output_root),
        "task_specs": specs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150D target matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
