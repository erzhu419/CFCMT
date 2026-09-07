#!/usr/bin/env python3
"""Run the pure-waiting causal source selector on one compute node."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (  # noqa: E402
    RESULT_PROTOCOL as WAITING_ALIGNED_FIT_RESULT_PROTOCOL,
)


DEFAULT_SCHEDULER_DIR = Path("/home/erzhu419/mine_code/scheduleurm/skill")
SEEDS = (
    80314,
    88625,
    27178,
    77524,
    63775,
    50945,
    73412,
    61481,
    50744,
    34310,
    84579,
    46535,
    84751,
    61994,
    47023,
    31657,
    51400,
    22208,
    52367,
    74744,
    74374,
    50079,
)
PURE_WAITING_SELECTOR_CACHE_AUDIT_PROTOCOL = (
    "tsc-v116-pure-waiting-selector-cache-audit-v1"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_selector_cache_audit(path: Path) -> dict[str, object]:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(audit, dict)
        or audit.get("protocol") != PURE_WAITING_SELECTOR_CACHE_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("decision")
        != "authorize_v116_pure_waiting_nested_source_selection"
        or not audit.get("gate", {}).get("passed", False)
        or audit.get("scenario") != "jinan_3x4_real"
        or [int(value) for value in audit.get("seeds", ())] != list(SEEDS)
        or int(audit.get("seed_count", -1)) != len(SEEDS)
        or int(audit.get("shards_per_seed", -1)) != 32
        or int(audit.get("cache_file_count", -1)) != len(SEEDS) * 32
    ):
        raise ValueError("selector cache audit does not authorize V116 launch")
    return audit


def _load_scheduler(path: Path):
    sys.path.insert(0, str(path))
    import scheduler  # type: ignore

    return scheduler


def _fetch(scheduler, node: str, remote: Path, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node(node),
            f"{scheduler._ssh_target_for_node(node)}:{remote}",
            str(local),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheduler-dir", type=Path, default=DEFAULT_SCHEDULER_DIR)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--remote-results-root", type=Path, required=True)
    parser.add_argument("--remote-cache-root", type=Path, required=True)
    parser.add_argument("--remote-conversion-root", type=Path, required=True)
    parser.add_argument("--component-result", type=Path)
    parser.add_argument("--strict-target-result", type=Path)
    parser.add_argument("--estimand-aligned-fit-result", type=Path)
    parser.add_argument("--selector-cache-audit-local", type=Path)
    parser.add_argument("--selector-cache-audit-remote", type=Path)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--node", default="node001")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--prediction-workers", type=int, default=7)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    args = parser.parse_args()
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("node cache workers must be in [1, 20]")
    if not 1 <= int(args.prediction_workers) <= 7:
        raise ValueError("source prediction workers must be in [1, 7]")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    legacy_inputs = (args.component_result, args.strict_target_result)
    if args.estimand_aligned_fit_result is not None:
        if any(value is not None for value in legacy_inputs):
            raise ValueError("do not mix legacy and estimand-aligned fit inputs")
        fit_result = json.loads(
            args.estimand_aligned_fit_result.read_text(encoding="utf-8")
        )
        if fit_result.get("protocol") != WAITING_ALIGNED_FIT_RESULT_PROTOCOL:
            raise ValueError("estimand-aligned fit result protocol changed")
        component_artifact = fit_result["model_artifacts"][
            "source_components_b100"
        ]
        strict_artifact = fit_result["model_artifacts"]["strict_target_b100"]
        zero_shot_artifact = fit_result["model_artifacts"][
            "source_components_b0"
        ]
        aligned = True
    else:
        if not all(value is not None for value in legacy_inputs):
            raise ValueError("legacy component and strict-target results are required")
        component = json.loads(args.component_result.read_text(encoding="utf-8"))
        strict = json.loads(args.strict_target_result.read_text(encoding="utf-8"))
        component_artifact = component["model_artifact"]
        strict_artifact = strict["model_artifact"]
        zero_shot_artifact = None
        aligned = False
    selector_audit_sha256 = None
    if aligned:
        if (
            args.selector_cache_audit_local is None
            or args.selector_cache_audit_remote is None
        ):
            raise ValueError("V116 requires local and remote selector cache audits")
        _require_selector_cache_audit(args.selector_cache_audit_local)
        selector_audit_sha256 = _sha256(args.selector_cache_audit_local)
    elif (
        args.selector_cache_audit_local is not None
        or args.selector_cache_audit_remote is not None
    ):
        raise ValueError("selector cache audit inputs are only valid for V116")
    snapshot_root = Path(snapshot["snapshot_root"])
    stem = (
        "pure_waiting_source_selector_v1"
        if aligned
        else "waiting_aligned_multisource_selector_v2"
    )
    remote_out = args.remote_results_root / f"{stem}.json"
    remote_log = args.remote_results_root / f"{stem}.log"
    remote_pid = args.remote_results_root / f"{stem}.pid"
    remote_exit = args.remote_results_root / f"{stem}.exit"
    manifest = (
        snapshot_root
        / "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
    )
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = [
        str(wrapper),
        "-m",
        "cf_h2o.eval.traffic_signal_waiting_aligned_source_selector",
        "--cache-root",
        str(args.remote_cache_root),
        "--manifest",
        str(manifest),
        "--conversion-root",
        str(args.remote_conversion_root),
        "--scenario",
        "jinan_3x4_real",
        "--seeds",
        *(str(value) for value in SEEDS),
        "--collection-shards",
        "32",
        "--target-name",
        "prefix_mean_cost_450s",
        "--cache-workers",
        str(args.cache_workers),
        "--prediction-workers",
        str(args.prediction_workers),
        "--component-bundle",
        str(component_artifact["path"]),
        "--component-bundle-sha256",
        str(component_artifact["sha256"]),
        "--strict-target-only-model",
        str(strict_artifact["path"]),
        "--strict-target-only-model-sha256",
        str(strict_artifact["sha256"]),
        "--city",
        "jinan",
    ]
    if aligned:
        command.extend(
            [
                "--selector-cache-audit",
                str(args.selector_cache_audit_remote),
                "--selector-cache-audit-sha256",
                str(selector_audit_sha256),
            ]
        )
    if zero_shot_artifact is not None:
        command.extend(
            [
                "--zero-shot-component-bundle",
                str(zero_shot_artifact["path"]),
                "--zero-shot-component-bundle-sha256",
                str(zero_shot_artifact["sha256"]),
            ]
        )
    command.extend(["--out", str(remote_out)])
    quoted = " ".join(shlex.quote(value) for value in command)
    detached_body = (
        f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))} "
        f"{quoted} > {shlex.quote(str(remote_log))} 2>&1; "
        f"printf '%s\\n' $? > {shlex.quote(str(remote_exit))}"
    )
    remote_command = (
        f"mkdir -p {shlex.quote(str(args.remote_results_root))} && "
        f"test ! -e {shlex.quote(str(remote_out))} && "
        f"test ! -e {shlex.quote(str(remote_pid))} && "
        f"test ! -e {shlex.quote(str(remote_exit))} && "
        "(nohup bash -lc "
        f"{shlex.quote(detached_body)} >/dev/null 2>&1 & "
        f"echo $! > {shlex.quote(str(remote_pid))})"
    )
    scheduler = _load_scheduler(args.scheduler_dir)
    if aligned:
        hash_code, hash_stdout, hash_stderr = scheduler.run_on(
            args.node,
            shlex.join(
                ["sha256sum", str(args.selector_cache_audit_remote)]
            ),
            timeout=120,
            check=False,
        )
        remote_hash = (
            str(hash_stdout).split()[0] if str(hash_stdout).split() else None
        )
        if int(hash_code) != 0 or remote_hash != selector_audit_sha256:
            raise RuntimeError(
                "remote selector-cache audit differs from local evidence: "
                f"{remote_hash} != {selector_audit_sha256}; {hash_stderr}"
            )
    started = datetime.now(timezone.utc)
    start_code, stdout, stderr = scheduler.run_on(
        args.node, remote_command, timeout=120, check=False
    )
    code = int(start_code)
    poll_events = []
    deadline = time.monotonic() + int(args.timeout_sec)
    while code == 0 and time.monotonic() < deadline:
        poll_command = (
            f"if test -f {shlex.quote(str(remote_exit))}; then "
            f"printf 'EXIT:'; cat {shlex.quote(str(remote_exit))}; "
            f"elif test -f {shlex.quote(str(remote_pid))} && "
            f"kill -0 $(cat {shlex.quote(str(remote_pid))}) 2>/dev/null; then "
            f"printf 'RUNNING:'; cat {shlex.quote(str(remote_pid))}; "
            "else printf 'LOST'; fi"
        )
        poll_code, poll_stdout, poll_stderr = scheduler.run_on(
            args.node, poll_command, timeout=60, check=False
        )
        event = str(poll_stdout).strip()
        poll_events.append(
            {
                "at_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": int(poll_code),
                "event": event,
                "stderr": str(poll_stderr)[-1000:],
            }
        )
        if event.startswith("EXIT:"):
            code = int(event.split(":", 1)[1].strip())
            break
        if event == "LOST":
            code = 3
            break
        time.sleep(30)
    else:
        if code == 0:
            code = 124
    local_result = args.local_output_root / remote_out.name
    local_log = args.local_output_root / remote_log.name
    _fetch(scheduler, args.node, remote_log, local_log)
    if int(code) == 0:
        _fetch(scheduler, args.node, remote_out, local_result)
    payload = {
        "protocol": (
            "tsc-v116-pure-waiting-source-selector-launch-v2"
            if aligned
            else "tsc-v112-waiting-aligned-multisource-selector-launch-v2"
        ),
        "created_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "start_returncode": int(start_code),
        "returncode": int(code),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "snapshot_root": str(snapshot_root),
        "remote_cache_root": str(args.remote_cache_root),
        "remote_result": str(remote_out),
        "remote_pid": str(remote_pid),
        "remote_exit": str(remote_exit),
        "local_result": str(local_result),
        "local_log": str(local_log),
        "component_bundle_sha256": str(component_artifact["sha256"]),
        "strict_target_only_model_sha256": str(strict_artifact["sha256"]),
        "zero_shot_component_bundle_sha256": (
            str(zero_shot_artifact["sha256"])
            if zero_shot_artifact is not None
            else None
        ),
        "estimand_aligned_fit_result": (
            str(args.estimand_aligned_fit_result.resolve())
            if args.estimand_aligned_fit_result is not None
            else None
        ),
        "selector_cache_audit": (
            {
                "local": str(args.selector_cache_audit_local.resolve()),
                "remote": str(args.selector_cache_audit_remote),
                "sha256": selector_audit_sha256,
            }
            if aligned
            else None
        ),
        "seed_count": len(SEEDS),
        "collection_shards": 32,
        "cache_workers": int(args.cache_workers),
        "prediction_workers": int(args.prediction_workers),
        "poll_events": poll_events,
        "passed": bool(int(code) == 0 and local_result.is_file()),
    }
    args.local_output_root.mkdir(parents=True, exist_ok=True)
    launch_path = args.local_output_root / f"{stem}_launch.json"
    launch_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(int(code) or 2)


if __name__ == "__main__":
    main()
