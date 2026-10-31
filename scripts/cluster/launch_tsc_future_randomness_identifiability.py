#!/usr/bin/env python3
"""Submit the seven-city V150F fixed-state future-randomness diagnostic."""

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

from cf_h2o.eval.traffic_signal_future_randomness_identifiability import (
    MINIMUM_VALID_FUTURES_PER_CHECKPOINT,
    SNAPSHOT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.cluster.launch_tsc_rigid_mechanism_prior_integration import NODES


LAUNCH_PROTOCOL = "tsc-v150f-fixed-state-future-randomness-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150f_future_randomness_identifiability.json"
)
MODULE = "cf_h2o.eval.traffic_signal_future_randomness_identifiability"
SIGNATURE_PREFIX = "CFCMT/v150f/fixed-state-future-randomness-v1"
RESOURCE_FAMILY = "CFCMT-v150f-future-randomness"


def _validate_config(config: Mapping[str, Any]) -> None:
    scenarios = dict(config.get("city_scenarios", {}))
    integrity = dict(config.get("integrity_requirements", {}))
    if (
        config.get("protocol") != "tsc-v150f-fixed-state-future-randomness-v1"
        or config.get("scientific_role") != "development_diagnostic_only"
        or tuple(sorted(scenarios)) != EXPECTED_CITY_GROUPS
        or len(set(scenarios.values())) != len(EXPECTED_CITY_GROUPS)
        or int(config.get("base_seed", -1)) != 5057
        or tuple(config.get("future_seeds", ()))
        != (9101, 9102, 9103, 9104, 9105)
        or float(config.get("base_trace_duration_sec", -1.0)) != 1200.0
        or float(config.get("warmup_sec", -1.0)) != 300.0
        or int(config.get("checkpoint_count", -1)) != 3
        or float(config.get("checkpoint_spacing_sec", -1.0)) != 180.0
        or int(config.get("control_interval_sec", -1)) != 30
        or int(config.get("counterfactual_horizon_intervals", -1)) != 3
        or config.get("counterfactual_cost_mode") != "system_vehicle_load"
        or config.get("snapshot_protocol") != SNAPSHOT_PROTOCOL
        or integrity.get("same_seed_exact_replay") is not True
        or integrity.get("rng_state_absent_from_snapshot") is not True
        or int(
            integrity.get("minimum_valid_future_pairs_per_checkpoint", -1)
        )
        != MINIMUM_VALID_FUTURES_PER_CHECKPOINT
    ):
        raise ValueError("V150F frozen diagnostic configuration changed")


def build_specs(
    *,
    snapshot_root: Path,
    config: Mapping[str, Any],
    remote_output_root: Path,
    local_output_root: Path,
) -> list[dict[str, Any]]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = snapshot_root / str(config["source_manifest"])
    specs = []
    for city, node in zip(EXPECTED_CITY_GROUPS, NODES, strict=True):
        remote_city = remote_output_root / city
        command = shlex.join(
            [
                "env",
                f"CFCMT_SOURCE_ROOT={snapshot_root}",
                "PYTHONUNBUFFERED=1",
                str(wrapper),
                "-m",
                MODULE,
                "scenario",
                "--manifest",
                str(manifest),
                "--target-city",
                city,
                "--scenario",
                str(config["city_scenarios"][city]),
                "--base-seed",
                str(config["base_seed"]),
                "--future-seeds",
                *[str(value) for value in config["future_seeds"]],
                "--base-trace-duration-sec",
                str(config["base_trace_duration_sec"]),
                "--warmup-sec",
                str(config["warmup_sec"]),
                "--checkpoint-count",
                str(config["checkpoint_count"]),
                "--checkpoint-spacing-sec",
                str(config["checkpoint_spacing_sec"]),
                "--control-interval-sec",
                str(config["control_interval_sec"]),
                "--counterfactual-horizon-intervals",
                str(config["counterfactual_horizon_intervals"]),
                "--counterfactual-cost-mode",
                str(config["counterfactual_cost_mode"]),
                "--out",
                str(remote_city / "result.json"),
            ]
        ) + " && printf 'TASK_DONE\\n'"
        specs.append(
            {
                "description": f"CFCMT V150F future-randomness diagnostic {city}",
                "project": "CFCMT",
                "cmd": "mkdir -p "
                + shlex.quote(str(remote_city))
                + " && "
                + command,
                "cwd": str(snapshot_root),
                "signature": f"{SIGNATURE_PREFIX}/{city}",
                "resource_family": RESOURCE_FAMILY,
                "ram_resource_family": RESOURCE_FAMILY,
                "vram": 0,
                "ram_mb": 65_536,
                "cpu": 4,
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
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150F launch or results")

    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    _validate_config(config)
    stage = _read_json(args.stage_manifest)
    snapshot_root = Path(stage["snapshot_root"])
    remote_city_roots = [
        args.remote_output_root / city for city in EXPECTED_CITY_GROUPS
    ]
    scheduler = _load_scheduler(args.scheduler)
    preflight = " && ".join(
        (
            shlex.join(
                ["test", "-f", str(snapshot_root / CONFIG_RELATIVE)]
            ),
            shlex.join(
                [
                    "test",
                    "-f",
                    str(snapshot_root / str(config["source_manifest"])),
                ]
            ),
            *(shlex.join(["test", "!", "-e", str(path)]) for path in remote_city_roots),
        )
    )
    code, _, stderr = scheduler.run_on(
        "node001", preflight, timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V150F preflight failed: {stderr}")

    specs = build_specs(
        snapshot_root=snapshot_root,
        config=config,
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            "CFCMT-v150f-fixed-state-future-randomness-v1",
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
        "remote_output_root": str(args.remote_output_root),
        "task_specs": specs,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150F diagnostic matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
