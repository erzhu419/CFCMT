#!/usr/bin/env python3
"""Submit the frozen V150L state-conditioned source closed-loop matrix."""

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
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    POLICY_ARMS,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    RESULT_PROTOCOL as V150K_RESULT_PROTOCOL,
    RUNTIME_MODEL_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
)


LAUNCH_PROTOCOL = "tsc-v150l-state-conditioned-source-closed-loop-launch-v3"
CONFIG_PROTOCOL = "tsc-v150l-state-conditioned-source-closed-loop-development-v3"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150l_state_conditioned_source_closed_loop.json"
)
MANIFEST_RELATIVE = Path("cf_h2o/config/traffic_signal_cross_city_v2_saltlake18.json")
MODULE = "cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop"
SIGNATURE_ROOT = "CFCMT/v150l/state-conditioned-source-closed-loop-v3"
RESOURCE_FAMILY = "CFCMT-v150l-state-conditioned-source-closed-loop"
ALLOWED_NODES = (
    "node001",
    "node002",
    "node003",
    "node004",
    "node005",
    "node006",
)
CPU_CORES = 1
RAM_MB = 8_192


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    runtime = dict(config.get("runtime", {}))
    smoke = dict(config.get("smoke", {}))
    full = dict(config.get("full", {}))
    city_scenarios = dict(config.get("city_scenarios", {}))
    scenario_count = sum(len(values) for values in city_scenarios.values())
    if (
        config.get("protocol") != CONFIG_PROTOCOL
        or tuple(city_scenarios) != EXPECTED_CITY_GROUPS
        or scenario_count != 18
        or tuple(config.get("arms", ())) != POLICY_ARMS
        or int(config.get("target_budget_action_groups", -1)) != 100
        or runtime.get("backend") != "libsumo"
        or runtime.get("sumo_version") != "1.22.0"
        or float(runtime.get("duration_sec", -1.0)) != 3600.0
        or float(runtime.get("warmup_sec", -1.0)) != 60.0
        or int(runtime.get("control_interval_sec", -1)) != 10
        or int(runtime.get("prediction_horizon_sec", -1)) != 450
        or runtime.get("coordination_mode") != "direct"
        or int(runtime.get("residual_cooldown_intervals", -1)) != 0
        or int(smoke.get("matrix_size", -1))
        != len(POLICY_ARMS)
        * len(smoke.get("seeds", ()))
        * sum(len(values) for values in smoke.get("city_scenarios", {}).values())
        or int(full.get("matrix_size", -1))
        != len(POLICY_ARMS) * len(full.get("seeds", ())) * scenario_count
    ):
        raise ValueError("V150L frozen configuration changed")


def rollout_identities(
    config: Mapping[str, Any], *, stage: str
) -> tuple[tuple[str, str, int, str], ...]:
    if stage == "smoke":
        city_scenarios = dict(config["smoke"]["city_scenarios"])
        seeds = tuple(int(value) for value in config["smoke"]["seeds"])
        expected_size = int(config["smoke"]["matrix_size"])
    elif stage == "full":
        city_scenarios = dict(config["city_scenarios"])
        seeds = tuple(int(value) for value in config["full"]["seeds"])
        expected_size = int(config["full"]["matrix_size"])
    else:
        raise ValueError(f"unknown V150L stage: {stage!r}")
    identities = tuple(
        (str(city), str(scenario), seed, arm)
        for city, scenarios in city_scenarios.items()
        for scenario in scenarios
        for seed in seeds
        for arm in POLICY_ARMS
    )
    if len(identities) != expected_size or len(identities) != len(set(identities)):
        raise ValueError("V150L rollout matrix identity changed")
    return identities


def load_runtime_records(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    expected_admitted = set(str(value) for value in config["offline_admitted_source_cities"])
    records: dict[str, dict[str, Any]] = {}
    for city in EXPECTED_CITY_GROUPS:
        result_path = Path(root) / city / "result.json"
        result = _read_json(result_path)
        runtime = dict(result.get("runtime_model", {}))
        admitted = bool(result.get("method", {}).get("selector", {}).get("admitted"))
        if (
            result.get("protocol") != V150K_RESULT_PROTOCOL
            or str(result.get("target_city")) != city
            or runtime.get("protocol") != RUNTIME_MODEL_PROTOCOL
            or not runtime.get("path")
            or len(str(runtime.get("sha256", ""))) != 64
            or admitted != (city in expected_admitted)
        ):
            raise ValueError(f"invalid V150K runtime record for {city}")
        records[city] = {
            "result_path": str(result_path.resolve()),
            "result_sha256": _sha256(result_path),
            "model_path": str(runtime["path"]),
            "model_sha256": str(runtime["sha256"]),
            "model_size_bytes": int(runtime["size_bytes"]),
            "admitted": admitted,
        }
    return records


def build_specs(
    *,
    snapshot_root: Path,
    conversion_root: Path,
    config: Mapping[str, Any],
    runtime_records: Mapping[str, Mapping[str, Any]],
    stage: str,
    remote_output_root: Path,
    local_output_root: Path,
    remote_scratch_root: Path,
) -> list[dict[str, Any]]:
    runtime = dict(config["runtime"])
    wrapper = Path(snapshot_root) / "scripts/cluster/run_cfcmt_sumo122.sh"
    manifest = Path(snapshot_root) / MANIFEST_RELATIVE
    specs: list[dict[str, Any]] = []
    for city, scenario, seed, arm in rollout_identities(config, stage=stage):
        remote_leaf = (
            Path(remote_output_root)
            / city
            / scenario
            / f"seed_{seed}"
            / arm
        )
        local_leaf = (
            Path(local_output_root)
            / city
            / scenario
            / f"seed_{seed}"
            / arm
        )
        scratch = (
            Path(remote_scratch_root)
            / city
            / scenario
            / f"seed_{seed}"
            / f"{arm}.xml"
        )
        arguments = [
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
            MODULE,
            "--manifest",
            str(manifest),
            "--conversion-root",
            str(conversion_root),
            "--scenario",
            scenario,
            "--city",
            city,
            "--seed",
            str(seed),
            "--arm",
            arm,
            "--duration-sec",
            str(runtime["duration_sec"]),
            "--warmup-sec",
            str(runtime["warmup_sec"]),
            "--control-interval-sec",
            str(runtime["control_interval_sec"]),
            "--prediction-horizon-sec",
            str(runtime["prediction_horizon_sec"]),
            "--tripinfo",
            str(scratch),
            "--out",
            str(remote_leaf / "result.json"),
        ]
        if arm != "phase_pressure":
            model = runtime_records[city]
            arguments.extend(
                [
                    "--runtime-model",
                    str(model["model_path"]),
                    "--runtime-model-sha256",
                    str(model["model_sha256"]),
                ]
            )
        command = (
            shlex.join(["mkdir", "-p", str(remote_leaf), str(scratch.parent)])
            + " && "
            + shlex.join(arguments)
            + " && printf 'TASK_DONE\\n'"
        )
        specs.append(
            {
                "description": f"CFCMT V150L {stage} {city} {scenario} seed {seed} {arm}",
                "project": "CFCMT",
                "cmd": command,
                "cwd": str(snapshot_root),
                "signature": f"{SIGNATURE_ROOT}/{stage}/{city}/{scenario}/{seed}/{arm}",
                "resource_family": RESOURCE_FAMILY,
                "ram_resource_family": RESOURCE_FAMILY,
                "vram": 0,
                "ram_mb": RAM_MB,
                "cpu": CPU_CORES,
                "priority": "high",
                "allowed_nodes": list(ALLOWED_NODES),
                "skip_launch_staging": True,
                "env_spec": "none",
                "extra_env": {},
                "result_dir": str(remote_leaf),
                "local_result_dir": str(local_leaf.resolve()),
            }
        )
    return specs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--runtime-result-root", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--remote-scratch-root", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError("refusing to overwrite V150L launch or results")
    config_path = PROJECT_ROOT / CONFIG_RELATIVE
    config = _read_json(config_path)
    _validate_config(config)
    snapshot = _read_json(args.stage_manifest)
    snapshot_root = Path(snapshot["snapshot_root"])
    runtime_records = load_runtime_records(args.runtime_result_root, config)
    scheduler = _load_scheduler(args.scheduler)
    remote_models = [Path(row["model_path"]) for row in runtime_records.values()]
    checks = [
        shlex.join(["test", "-d", str(args.conversion_root)]),
        shlex.join(["test", "-f", str(snapshot_root / MANIFEST_RELATIVE)]),
        shlex.join(["test", "!", "-e", str(args.remote_output_root)]),
        shlex.join(["test", "!", "-e", str(args.remote_scratch_root)]),
        *(shlex.join(["test", "-f", str(path)]) for path in remote_models),
    ]
    code, _, stderr = scheduler.run_on(
        "node001", " && ".join(checks), timeout=120, check=False
    )
    if int(code) != 0:
        raise RuntimeError(f"remote V150L preflight failed: {stderr}")
    hash_code, stdout, hash_stderr = scheduler.run_on(
        "node001",
        shlex.join(["sha256sum", *(str(path) for path in remote_models)]),
        timeout=120,
        check=False,
    )
    observed_hashes = [
        line.split()[0] for line in str(stdout).splitlines() if line.split()
    ]
    expected_hashes = [
        str(runtime_records[city]["model_sha256"])
        for city in EXPECTED_CITY_GROUPS
    ]
    if int(hash_code) != 0 or observed_hashes != expected_hashes:
        raise RuntimeError(
            f"remote V150L model identities changed: {observed_hashes}; {hash_stderr}"
        )
    specs = build_specs(
        snapshot_root=snapshot_root,
        conversion_root=args.conversion_root,
        config=config,
        runtime_records=runtime_records,
        stage=str(args.stage),
        remote_output_root=args.remote_output_root,
        local_output_root=args.local_output_root,
        remote_scratch_root=args.remote_scratch_root,
    )
    completed = subprocess.run(
        [
            str(args.scheduler),
            "submit-jsonl",
            "--stdin",
            "--trusted",
            "--json",
            "--intent-label",
            f"CFCMT-v150l-state-conditioned-source-closed-loop-{args.stage}",
        ],
        input=json.dumps(specs),
        text=True,
        capture_output=True,
        check=False,
    )
    payload = {
        "protocol": LAUNCH_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": str(args.stage),
        "submitted": completed.returncode == 0,
        "scheduler_returncode": int(completed.returncode),
        "scheduler_stdout": completed.stdout,
        "scheduler_stderr": completed.stderr,
        "snapshot_root": str(snapshot_root),
        "snapshot_sha256": str(snapshot["snapshot_sha256"]),
        "config_sha256": _sha256(config_path),
        "runtime_records": runtime_records,
        "remote_output_root": str(args.remote_output_root),
        "remote_scratch_root": str(args.remote_scratch_root),
        "rollout_count": len(specs),
        "task_specs": specs,
    }
    atomic_write_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError("scheduler rejected V150L matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
