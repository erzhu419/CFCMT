#!/usr/bin/env python3
"""Run a V112 waiting-aligned arrival screen in place on node001."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
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


def _load_scheduler(path: Path):
    sys.path.insert(0, str(path))
    import scheduler  # type: ignore

    return scheduler


def _fetch(scheduler, remote: Path, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            scheduler._ssh_rsync_shell_for_node("node001"),
            f"{scheduler._ssh_target_for_node('node001')}:{remote}",
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
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--screen", choices=("rule", "model"), default="rule")
    parser.add_argument("--cache-workers", type=int, default=20)
    parser.add_argument("--fold-workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= int(args.cache_workers) <= 20:
        raise ValueError("node cache workers must be in [1, 20]")
    if not 1 <= int(args.fold_workers) <= 20:
        raise ValueError("node fold workers must be in [1, 20]")
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    snapshot_root = Path(snapshot["snapshot_root"])
    stem = f"arrival_{args.screen}_screen_v1"
    remote_out = args.remote_results_root / f"{stem}.json"
    remote_log = args.remote_results_root / f"{stem}.log"
    manifest = (
        snapshot_root
        / "cf_h2o/config/traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
    )
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = [
        str(wrapper),
        "-m",
        (
            "cf_h2o.eval.traffic_signal_movement_arrival_offline_screen"
            if args.screen == "rule"
            else "cf_h2o.eval.traffic_signal_movement_arrival_model_screen"
        ),
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
    ]
    if args.screen == "model":
        command.extend(["--fold-workers", str(args.fold_workers)])
    command.extend(["--out", str(remote_out)])
    quoted = " ".join(shlex.quote(value) for value in command)
    remote_command = (
        f"test ! -e {shlex.quote(str(remote_out))} && "
        f"mkdir -p {shlex.quote(str(args.remote_results_root))} && "
        f"CFCMT_SOURCE_ROOT={shlex.quote(str(snapshot_root))} "
        f"{quoted} > {shlex.quote(str(remote_log))} 2>&1"
    )
    scheduler = _load_scheduler(args.scheduler_dir)
    started = datetime.now(timezone.utc)
    code, stdout, stderr = scheduler.run_on(
        "node001", remote_command, timeout=7200, check=False
    )
    local_result = args.local_output_root / remote_out.name
    local_log = args.local_output_root / remote_log.name
    if Path(remote_log):
        _fetch(scheduler, remote_log, local_log)
    if int(code) == 0:
        _fetch(scheduler, remote_out, local_result)
    payload = {
        "protocol": "tsc-v112-movement-arrival-offline-screen-launch-v2",
        "created_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "node": "node001",
        "returncode": int(code),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "snapshot_root": str(snapshot_root),
        "remote_cache_root": str(args.remote_cache_root),
        "remote_result": str(remote_out),
        "local_result": str(local_result),
        "local_log": str(local_log),
        "seed_count": len(SEEDS),
        "collection_shards": 32,
        "cache_workers": int(args.cache_workers),
        "fold_workers": int(args.fold_workers),
        "screen": args.screen,
        "passed": bool(int(code) == 0 and local_result.is_file()),
    }
    args.local_output_root.mkdir(parents=True, exist_ok=True)
    launch_path = args.local_output_root / f"arrival_{args.screen}_screen_launch_v1.json"
    launch_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(int(code) or 2)


if __name__ == "__main__":
    main()
