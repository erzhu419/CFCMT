#!/usr/bin/env python3
"""Run LA/Jinan target-offline source and guard selection on two CPU nodes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
)


PROTOCOL = "tsc-v110-target-offline-source-and-guard-direct-launch-v1"
CITY_NODE = {"los_angeles": "node001", "jinan": "node002"}


def _command(
    *,
    snapshot_root: Path,
    input_root: Path,
    main_root: Path,
    conversion_root: Path,
    remote_result_root: Path,
    city: str,
    model_sha256: str,
    result_sha256: str,
    workers: int,
    fit_workers: int,
    minimum_mean_improvement: float,
    maximum_fold_regression: float,
    source_familywise_alpha: float,
    guard_familywise_alpha: float,
    guard_minimum_mean_improvement: float,
    guard_maximum_fold_regression: float,
    guard_min_context_trust: float,
) -> str:
    runner = snapshot_root / "scripts/cluster/run_tsc_target_offline_source_selection.py"
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    city_root = remote_result_root / city
    values = [
        str(runner),
        "--source-cache-root",
        str(input_root / "source_combined_cache_v2"),
        "--source-manifest",
        str(snapshot_root / "cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json"),
        "--source-baseline",
        str(input_root / "source_rule_baseline.json"),
        "--source-repair-audit",
        str(input_root / "source_combined_cache_repair_audit_v2.json"),
        "--external-cache-root",
        str(input_root / "external_adaptation_cache_v2"),
        "--external-manifest",
        str(snapshot_root / "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"),
        "--external-repair-audit",
        str(input_root / "external_cache_repair_audit_v2.json"),
        "--main-model",
        str(main_root / city / "model.pkl"),
        "--main-model-sha256",
        str(model_sha256),
        "--main-result",
        str(main_root / city / "result.json"),
        "--main-result-sha256",
        str(result_sha256),
        "--city",
        city,
        "--workers",
        str(workers),
        "--fit-workers",
        str(fit_workers),
        "--minimum-mean-improvement",
        str(minimum_mean_improvement),
        "--maximum-fold-regression",
        str(maximum_fold_regression),
        "--source-familywise-alpha",
        str(source_familywise_alpha),
        "--guard-familywise-alpha",
        str(guard_familywise_alpha),
        "--guard-minimum-mean-improvement",
        str(guard_minimum_mean_improvement),
        "--guard-maximum-fold-regression",
        str(guard_maximum_fold_regression),
        "--guard-min-context-trust",
        str(guard_min_context_trust),
        "--result-out",
        str(city_root / "result.json"),
    ]
    command = " ".join(
        [
            f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))}",
            f"CFCMT_EXTERNAL_CONVERSION_ROOT={shlex.quote(str(conversion_root))}",
            "PYTHONUNBUFFERED=1",
            shlex.quote(str(wrapper)),
            *(shlex.quote(value) for value in values),
        ]
    )
    return f"{command} > {shlex.quote(str(city_root / 'run.log'))} 2>&1"


def _run_city(
    scheduler: Any,
    *,
    node: str,
    city: str,
    command: str,
    remote_result_root: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    city_root = remote_result_root / city
    result_path = city_root / "result.json"
    scheduler.run_on(
        node,
        f"mkdir -p {shlex.quote(str(city_root))}",
        timeout=120,
        check=True,
    )
    exists = scheduler.run_on(
        node,
        f"test -e {shlex.quote(str(result_path))}",
        timeout=120,
        check=False,
    )[0]
    if int(exists) == 0:
        raise FileExistsError(f"remote result already exists: {result_path}")
    started = time.monotonic()
    code, stdout, stderr = scheduler.run_on(
        node, command, timeout=int(timeout_sec), check=False
    )
    tail = scheduler.run_on(
        node,
        f"tail -n 80 {shlex.quote(str(city_root / 'run.log'))}",
        timeout=120,
        check=False,
    )[1]
    return {
        "node": node,
        "city": city,
        "returncode": int(code),
        "elapsed_sec": float(time.monotonic() - started),
        "stdout_tail": str(stdout)[-2000:],
        "stderr_tail": str(stderr)[-2000:],
        "log_tail": str(tail)[-12000:],
        "remote_result": str(result_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-result-root", type=Path, required=True)
    parser.add_argument("--local-result-root", type=Path, required=True)
    parser.add_argument("--los-angeles-model-sha256", required=True)
    parser.add_argument("--los-angeles-result-sha256", required=True)
    parser.add_argument("--jinan-model-sha256", required=True)
    parser.add_argument("--jinan-result-sha256", required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--fit-workers", type=int, default=20)
    parser.add_argument("--minimum-mean-improvement", type=float, default=0.005)
    parser.add_argument("--maximum-fold-regression", type=float, default=0.01)
    parser.add_argument("--source-familywise-alpha", type=float, default=0.025)
    parser.add_argument("--guard-familywise-alpha", type=float, default=0.025)
    parser.add_argument("--guard-minimum-mean-improvement", type=float, default=0.002)
    parser.add_argument("--guard-maximum-fold-regression", type=float, default=0.01)
    parser.add_argument("--guard-min-context-trust", type=float, default=0.1)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_result_root.exists():
        raise FileExistsError("refusing to overwrite offline source launch outputs")
    if not 1 <= int(args.fit_workers) <= 20:
        raise ValueError("fit-workers must respect the 20-worker per-node limit")
    stage = json.loads(args.stage_manifest.read_text(encoding="utf-8"))
    snapshot_root = Path(stage["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)
    model_hashes = {
        "los_angeles": str(args.los_angeles_model_sha256),
        "jinan": str(args.jinan_model_sha256),
    }
    result_hashes = {
        "los_angeles": str(args.los_angeles_result_sha256),
        "jinan": str(args.jinan_result_sha256),
    }
    commands = {
        city: _command(
            snapshot_root=snapshot_root,
            input_root=args.input_root,
            main_root=args.main_root,
            conversion_root=args.conversion_root,
            remote_result_root=args.remote_result_root,
            city=city,
            model_sha256=model_hashes[city],
            result_sha256=result_hashes[city],
            workers=args.workers,
            fit_workers=args.fit_workers,
            minimum_mean_improvement=args.minimum_mean_improvement,
            maximum_fold_regression=args.maximum_fold_regression,
            source_familywise_alpha=args.source_familywise_alpha,
            guard_familywise_alpha=args.guard_familywise_alpha,
            guard_minimum_mean_improvement=(
                args.guard_minimum_mean_improvement
            ),
            guard_maximum_fold_regression=(
                args.guard_maximum_fold_regression
            ),
            guard_min_context_trust=args.guard_min_context_trust,
        )
        for city in CITY_NODE
    }
    with ThreadPoolExecutor(max_workers=len(CITY_NODE)) as pool:
        futures = {
            city: pool.submit(
                _run_city,
                scheduler,
                node=CITY_NODE[city],
                city=city,
                command=commands[city],
                remote_result_root=args.remote_result_root,
                timeout_sec=args.timeout_sec,
            )
            for city in CITY_NODE
        }
        runs = [futures[city].result() for city in CITY_NODE]
    failures = [row for row in runs if row["returncode"] != 0]
    payload = {
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_manifest": str(args.stage_manifest.resolve()),
        "snapshot_sha256": stage["snapshot_sha256"],
        "snapshot_root": str(snapshot_root),
        "input_root": str(args.input_root),
        "main_root": str(args.main_root),
        "conversion_root": str(args.conversion_root),
        "remote_result_root": str(args.remote_result_root),
        "workers": int(args.workers),
        "fit_workers": int(args.fit_workers),
        "selector_thresholds": {
            "minimum_mean_improvement": float(args.minimum_mean_improvement),
            "maximum_fold_regression": float(args.maximum_fold_regression),
            "source_familywise_alpha": float(args.source_familywise_alpha),
            "guard_familywise_alpha": float(args.guard_familywise_alpha),
            "guard_minimum_mean_improvement": float(
                args.guard_minimum_mean_improvement
            ),
            "guard_maximum_fold_regression": float(
                args.guard_maximum_fold_regression
            ),
            "guard_min_context_trust": float(args.guard_min_context_trust),
        },
        "commands": commands,
        "runs": runs,
        "passed": not failures,
    }
    _atomic_json(args.out, payload)
    if failures:
        raise RuntimeError(
            "offline source-selection run failed: "
            + ", ".join(row["city"] for row in failures)
        )
    args.local_result_root.mkdir(parents=True, exist_ok=False)
    for row in runs:
        result_text = scheduler.run_on(
            row["node"],
            f"cat {shlex.quote(row['remote_result'])}",
            timeout=120,
            check=True,
        )[1]
        _atomic_json(
            args.local_result_root / f"{row['city']}.json",
            json.loads(str(result_text)),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
