#!/usr/bin/env python3
"""Recover one interrupted v76 cache node with idempotent detached execution."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_waiting_cache import (  # noqa: E402
    PROTOCOL,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (  # noqa: E402
    _direct_remote_command,
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_multihorizon_waiting_cache import (  # noqa: E402
    LAUNCH_PROTOCOL,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    DEFAULT_SCHEDULER,
    _read_json,
)


RECOVERY_PROTOCOL = "v83r79-multihorizon-cache-durable-node-recovery-v1"


def _remote_file_count(scheduler: Any, node: str, cache_root: Path) -> int:
    code, stdout, stderr = scheduler.run_on(
        node,
        f"find {cache_root} -type f -name '*.npz' | wc -l",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise RuntimeError(f"cannot count remote cache files: {stderr}")
    return int(str(stdout).strip())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-launch", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--scheduleurm-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--transport-prefix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite recovery evidence: {args.out}")
    initial = _read_json(args.initial_launch)
    protocol = _read_json(args.protocol)
    if (
        initial.get("protocol") != LAUNCH_PROTOCOL
        or "execution_error" not in initial
        or initial.get("frozen_protocol_sha256") != _sha256(args.protocol)
        or protocol.get("protocol") != PROTOCOL
    ):
        raise ValueError("multihorizon recovery parent evidence changed")
    node = str(args.node)
    specs = [
        dict(spec)
        for spec in initial["specs"]
        if str(spec.get("require_node")) == node
    ]
    if len(specs) != 1:
        raise ValueError(f"recovery requires exactly one frozen spec for {node}")
    spec = specs[0]
    assigned = [
        int(value)
        for value in protocol["collection"]["node_assignments"].get(node, ())
    ]
    if (
        not assigned
        or f"multihorizon-cache/{node}" not in str(spec.get("signature", ""))
        or "--rollout-prefix-horizons-sec 10 30 60 120 450" not in str(spec["cmd"])
        or not str(spec["result_dir"]).endswith(f"/{node}")
    ):
        raise ValueError("multihorizon recovery frozen command changed")
    scheduler = _load_scheduler(args.scheduler)
    remote_summary = Path(str(spec["result_dir"])) / "cache_summary.json"
    code, process_count, _ = scheduler.run_on(
        node,
        "ps -eo args | grep -c [t]raffic_signal_counterfactual_cache",
        timeout=120,
        check=False,
    )
    if int(code) == 0 and int(str(process_count).strip() or 0) > 0:
        raise RuntimeError(f"collector is still running on {node}")
    code, _, _ = scheduler.run_on(
        node,
        f"test ! -e {remote_summary}",
        timeout=120,
        check=False,
    )
    if int(code) != 0:
        raise FileExistsError(f"recovery summary already exists: {remote_summary}")
    cache_root = Path(str(initial["remote_cache_root"]))
    before = _remote_file_count(scheduler, node, cache_root)
    expected = int(protocol["collection"]["expected_cache_file_count"])
    if not 0 < before < expected:
        raise ValueError(
            f"recovery requires an incomplete nonempty cache: {before}/{expected}"
        )
    scheduleurm_root = Path(args.scheduleurm_root).resolve()
    sys.path.insert(0, str(scheduleurm_root))
    from algorithm.experiments.durable_remote_command import (  # type: ignore
        run_durable_remote_capture,
    )

    returncode, stdout, stderr, durable = run_durable_remote_capture(
        node,
        _direct_remote_command(spec),
        args.transport_prefix,
        run_id=f"cfcmt-v76-multihorizon-{node}-recovery-v1",
        timeout_s=int(args.timeout_sec),
        poll_interval_s=15.0,
    )
    after = _remote_file_count(scheduler, node, cache_root)
    summary_code, _, summary_stderr = scheduler.run_on(
        node,
        f"test -f {remote_summary}",
        timeout=120,
        check=False,
    )
    summary_exists = int(summary_code) == 0
    sync_errors = []
    if int(returncode) == 0 and after == expected and summary_exists:
        for frozen_spec in initial["specs"]:
            source_node = str(frozen_spec["require_node"])
            local_result = Path(str(frozen_spec["local_result_dir"]))
            if local_result.exists():
                raise FileExistsError(
                    f"refusing to overwrite synced task result: {local_result}"
                )
            local_result.mkdir(parents=True, exist_ok=False)
            completed = subprocess.run(
                [
                    "rsync",
                    "-a",
                    "-e",
                    scheduler._ssh_rsync_shell_for_node(source_node),
                    f"{scheduler._ssh_target_for_node(source_node)}:"
                    f"{frozen_spec['result_dir']}/",
                    f"{local_result}/",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                sync_errors.append(
                    {
                        "node": source_node,
                        "returncode": completed.returncode,
                        "stderr": completed.stderr[-2000:],
                    }
                )
    payload = {
        "protocol": RECOVERY_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "initial_launch": str(args.initial_launch.resolve()),
        "initial_launch_sha256": _sha256(args.initial_launch),
        "frozen_protocol": str(args.protocol.resolve()),
        "frozen_protocol_sha256": _sha256(args.protocol),
        "node": node,
        "assigned_seeds": assigned,
        "remote_summary": str(remote_summary),
        "cache_files_before": before,
        "cache_files_after": after,
        "expected_cache_files": expected,
        "returncode": int(returncode),
        "remote_summary_exists": summary_exists,
        "remote_summary_check_stderr": str(summary_stderr)[-1000:],
        "stdout_tail": str(stdout)[-8000:],
        "stderr_tail": str(stderr)[-8000:],
        "durable_transport": durable,
        "sync_errors": sync_errors,
        "passed": bool(
            int(returncode) == 0
            and after == expected
            and summary_exists
            and not sync_errors
        ),
    }
    _atomic_json(args.out, payload)
    if not payload["passed"]:
        raise RuntimeError(
            "multihorizon durable recovery failed: "
            f"returncode={returncode}, files={after}/{expected}, "
            f"sync_errors={sync_errors}, stderr={stderr[-2000:]}"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "node": node,
                "cache_files_before": before,
                "cache_files_after": after,
                "resumed_existing": durable.get("resumed_existing"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
