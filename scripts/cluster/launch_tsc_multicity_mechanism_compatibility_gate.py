#!/usr/bin/env python3
"""Submit the seven complete-city V148 mechanism-compatibility tasks."""

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
from cf_h2o.eval.traffic_signal_multicity_mechanism_compatibility_gate import (
    COMPATIBILITY_FOLD_COUNT,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    MECHANISM_COMPATIBILITY_NAMES,
    TARGET_BUDGET,
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
from scripts.cluster.launch_tsc_multicity_uniform_source_ensemble import build_specs


LAUNCH_PROTOCOL = "tsc-v148-multicity-mechanism-compatibility-gate-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v148_multicity_mechanism_compatibility_gate.json"
)
MODULE = "cf_h2o.eval.traffic_signal_multicity_mechanism_compatibility_gate"
SIGNATURE_PREFIX = "CFCMT/v148/multicity-mechanism-compatibility-gate-v1"
RESOURCE_FAMILY = "CFCMT-v148-multicity-mechanism-compatibility-gate"


def _validate_config(config: Mapping[str, Any]) -> None:
    probe = config.get("probe", {})
    rule = config.get("selection_rule", {})
    gate = config.get("development_gate", {})
    if (
        config.get("protocol")
        != "tsc-v148-multicity-mechanism-compatibility-gate-v1"
        or tuple(str(value) for value in config.get("target_city_groups", ()))
        != EXPECTED_CITY_GROUPS
        or int(config.get("target_budget", -1)) != TARGET_BUDGET
        or int(config.get("cross_fit_folds", -1)) != COMPATIBILITY_FOLD_COUNT
        or int(config.get("cross_fit_training_groups", -1)) != 20
        or int(config.get("cross_fit_scored_groups", -1)) != 5
        or config.get("scored_adaptation_labels_used") is not True
        or config.get("evaluation_features_or_labels_used") is not False
        or config.get("outer_label_exclusion")
        != "heldout_city_absent_as_source_and_target"
        or probe.get("estimator")
        != "fixed-parent-rank0-domain-balanced-residual-v1"
        or probe.get("parent_variant") != "physical"
        or tuple(probe.get("mechanisms", ())) != MECHANISM_COMPATIBILITY_NAMES
        or probe.get("source_mass")
        != "equal_mean_of_five_mechanism_trusts"
        or float(rule.get("maximum_history_regression", -1.0)) != 0.005
        or float(rule.get("minimum_history_q75_improvement", -1.0)) != 0.0005
        or rule.get("source_null_when_no_history_admission") is not True
        or int(gate.get("minimum_improving_cities", -1)) != 5
        or float(gate.get("maximum_city_regression", -1.0)) != 0.01
        or config.get("failure_rule")
        != "close_mechanism_compatibility_gate_without_threshold_retuning"
        or config.get("confirmation_target") != "boston_package_v11_v149_only"
    ):
        raise ValueError("V148 frozen protocol configuration changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--fit-result-local", type=Path, required=True)
    parser.add_argument("--fit-result-remote", type=Path, required=True)
    parser.add_argument("--source-cache-root", type=Path, required=True)
    parser.add_argument("--source-cache-audit-local", type=Path, required=True)
    parser.add_argument("--source-cache-audit-remote", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--cache-workers", type=int, default=8)
    parser.add_argument("--fit-workers", type=int, default=8)
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=EXPECTED_CITY_GROUPS,
        default=list(EXPECTED_CITY_GROUPS),
    )
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    selected_targets = tuple(str(value) for value in args.targets)
    if len(selected_targets) != len(set(selected_targets)):
        raise ValueError("V148 phased targets must be unique")
    if args.out.exists() or any(
        (args.local_output_root / city).exists() for city in selected_targets
    ):
        raise FileExistsError("refusing to overwrite V148 launch or selected results")
    if not 1 <= int(args.cache_workers) <= 8 or not 1 <= int(args.fit_workers) <= 8:
        raise ValueError("V148 workers must be in [1, 8]")
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    fit = _read_json(args.fit_result_local)
    source_audit = _read_json(args.source_cache_audit_local)
    if fit.get("protocol") != FIT_RESULT_PROTOCOL or fit.get("city") != "jinan":
        raise ValueError("V148 requires the frozen V115 architecture fit")
    if (
        source_audit.get("protocol") != SOURCE_AUDIT_PROTOCOL
        or source_audit.get("status") != "PASS"
        or not source_audit.get("gate", {}).get("passed", False)
    ):
        raise ValueError("V148 source cache audit changed")

    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    fit_sha = _sha256(args.fit_result_local)
    source_audit_sha = _sha256(args.source_cache_audit_local)
    remote_inputs = (args.fit_result_remote, args.source_cache_audit_remote)
    expected_hashes = [fit_sha, source_audit_sha]
    scheduler = _load_scheduler(args.scheduler)
    remote_city_roots = [args.remote_output_root / city for city in selected_targets]
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
        raise RuntimeError(f"remote V148 preflight failed: {stderr}")
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
            "remote V148 evidence identities changed: "
            f"{remote_hashes} != {expected_hashes}; {hash_stderr}"
        )
    specs = build_specs(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        fit_result_remote=args.fit_result_remote,
        fit_result_sha256=fit_sha,
        source_cache_root=args.source_cache_root,
        source_cache_audit_remote=args.source_cache_audit_remote,
        source_cache_audit_sha256=source_audit_sha,
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
        cache_workers=args.cache_workers,
        fit_workers=args.fit_workers,
        module=MODULE,
        description_prefix="CFCMT V148 mechanism compatibility target",
        signature_prefix=SIGNATURE_PREFIX,
        resource_family=RESOURCE_FAMILY,
    )
    specs = [
        spec
        for spec in specs
        if str(spec["signature"]).rsplit("/", 1)[-1] in set(selected_targets)
    ]
    if len(specs) != len(selected_targets):
        raise ValueError("V148 phased task coverage is incomplete")
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v148-multicity-mechanism-compatibility-gate-v1",
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
        "target_city_groups": list(selected_targets),
        "task_specs": specs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V148 mechanism compatibility gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
