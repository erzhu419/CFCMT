#!/usr/bin/env python3
"""Launch ordered Boston v15 yellow-clearance admission stages."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from scripts.cluster.launch_eth_boston_trigger_admission_v5 import (
    EXPECTED_TRIP_COUNT,
    TRIGGER_BEGIN_TIME_SEC,
    validate_route_result,
)
from scripts.cluster.launch_eth_boston_v14_admission import (
    ALLOWED_NODES,
    CPU_CORES,
    RAM_MB,
    _common_payload,
    _manifest,
    _submit,
    _verify_remote_inputs,
    build_full_spec as _build_v14_full_spec,
    build_route_spec as _build_v14_route_spec,
)
from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    _load_scheduler,
)
from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    DEFAULT_SCHEDULER,
    _read_json,
)
from scripts.data.repair_eth_boston_systemic_uncontrolled_major import (
    PACKAGE_PROTOCOL as V14_PACKAGE_PROTOCOL,
    validate_systemic_uncontrolled_major_manifest,
)
from scripts.data.repair_eth_boston_tls_yellow_clearance import (
    PACKAGE_PROTOCOL as V15_PACKAGE_PROTOCOL,
    validate_tls_yellow_clearance_manifest,
)


LAUNCH_PROTOCOL = "tsc-v150-eth-boston-v15-admission-pipeline-launch-v1"
SIGNATURE_ROOT = "CFCMT/v150/eth-boston-v15-yellow-clearance"
TRIGGER_HORIZON_SEC = 10_800
TRIGGER_STOP_TIME_SEC = TRIGGER_BEGIN_TIME_SEC + TRIGGER_HORIZON_SEC
FORMER_FAILURE_TIMES_SEC = (
    9_020.0,
    9_518.0,
    9_735.0,
    10_019.0,
    11_304.0,
    11_504.0,
    11_871.0,
    12_279.0,
    13_146.0,
    16_431.0,
)


def validate_trigger_package(manifest: Mapping[str, Any]) -> None:
    validate_tls_yellow_clearance_manifest(manifest)
    microscopic = dict(manifest.get("microscopic_instantiation", {}))
    demand = dict(manifest.get("microscopic_demand_inventory", {}))
    gates = dict(manifest.get("gates", {}))
    begin_sec = float(microscopic.get("begin_sec", -1.0))
    end_sec = float(microscopic.get("end_sec", -1.0))
    if (
        manifest.get("protocol") != V15_PACKAGE_PROTOCOL
        or manifest.get("passed") is not True
        or manifest.get("city_code") != "BOS"
        or gates.get("boston_tls_yellow_clearance_normalized") is not True
        or microscopic.get("backend_required_by_protocol") != "libsumo"
        or begin_sec != TRIGGER_BEGIN_TIME_SEC
        or end_sec < TRIGGER_STOP_TIME_SEC
        or int(demand.get("trip_count", -1)) != EXPECTED_TRIP_COUNT
        or any(
            failure_time < begin_sec or failure_time > TRIGGER_STOP_TIME_SEC
            for failure_time in FORMER_FAILURE_TIMES_SEC
        )
    ):
        raise ValueError("Boston v15 package does not cover the trigger evidence window")


def validate_trigger_result(
    result: Mapping[str, Any], *, expected_package_manifest_sha256: str
) -> None:
    setting = dict(result.get("setting", {}))
    observed = dict(result.get("observed", {}))
    checks = dict(result.get("checks", {}))
    if (
        result.get("mode") != "operational_preflight"
        or result.get("operational_preflight_passed") is not True
        or result.get("scientific_admission_passed") is not False
        or result.get("error") is not None
        or not checks
        or not all(bool(value) for value in checks.values())
        or setting.get("package_manifest_sha256")
        != expected_package_manifest_sha256
        or setting.get("city_code") != "BOS"
        or setting.get("backend") != "libsumo"
        or setting.get("expected_sumo_version") != "1.22.0"
        or float(setting.get("begin_time_sec", -1.0)) != TRIGGER_BEGIN_TIME_SEC
        or int(setting.get("operational_horizon_sec", -1))
        != TRIGGER_HORIZON_SEC
        or float(setting.get("stop_time_sec", -1.0)) != TRIGGER_STOP_TIME_SEC
        or float(observed.get("final_time_sec", -1.0)) != TRIGGER_STOP_TIME_SEC
    ):
        raise ValueError("Boston v15 trigger-window admission did not pass")
    if any(
        float(observed["final_time_sec"]) <= failure_time
        for failure_time in FORMER_FAILURE_TIMES_SEC
    ):
        raise ValueError("Boston v15 trigger did not cross every prior failure")


def build_package_spec(
    *,
    snapshot_root: Path,
    base_root: Path,
    output_root: Path,
    expected_base_manifest_sha256: str,
    remote_result_root: Path,
    local_result_root: Path,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            "PYTHONUNBUFFERED=1",
            str(wrapper),
            "-m",
            "scripts.data.repair_eth_boston_tls_yellow_clearance",
            "--base-root",
            str(base_root),
            "--output-root",
            str(output_root),
            "--expected-base-manifest-sha256",
            str(expected_base_manifest_sha256),
            "--out",
            str(remote_result_root / "package_manifest.json"),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT Boston v15 SUMO yellow-clearance package repair",
        "project": "CFCMT",
        "cmd": "mkdir -p "
        + shlex.quote(str(remote_result_root))
        + " && "
        + command,
        "cwd": str(snapshot_root),
        "signature": f"{SIGNATURE_ROOT}/package-v1",
        "resource_family": "CFCMT-v150-eth-boston-v15-package",
        "ram_resource_family": "CFCMT-v150-eth-boston-v15-package",
        "vram": 0,
        "ram_mb": 32_768,
        "cpu": 2,
        "priority": "high",
        "require_node": "node001",
        "skip_launch_staging": True,
        "env_spec": "none",
        "extra_env": {},
        "result_dir": str(remote_result_root),
        "local_result_dir": str(local_result_root.resolve()),
    }


def _version_spec(spec: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    value = dict(spec)
    value.update(
        {
            "description": f"CFCMT Boston v15 {stage} admission",
            "signature": f"{SIGNATURE_ROOT}/{stage}-v1",
            "ram_resource_family": f"CFCMT-v150-eth-boston-{stage}-admission-v15",
        }
    )
    if stage in {"trigger", "full"}:
        value["allowed_nodes"] = list(ALLOWED_NODES)
    return value


def build_route_spec(**kwargs: Any) -> dict[str, Any]:
    return _version_spec(_build_v14_route_spec(**kwargs), stage="route")


def build_trigger_spec(
    *,
    snapshot_root: Path,
    package_root: Path,
    expected_package_manifest_sha256: str,
    remote_output_root: Path,
    local_output_root: Path,
) -> dict[str, Any]:
    wrapper = snapshot_root / "scripts/cluster/run_cfcmt_sumo122.sh"
    command = shlex.join(
        [
            "env",
            f"CFCMT_SOURCE_ROOT={snapshot_root}",
            "PYTHONUNBUFFERED=1",
            str(wrapper),
            "-m",
            "cf_h2o.eval.traffic_signal_eth_city_admission",
            "--package-root",
            str(package_root),
            "--expected-package-manifest-sha256",
            str(expected_package_manifest_sha256),
            "--expected-city-code",
            "BOS",
            "--expected-sumo-version",
            "1.22.0",
            "--operational-horizon-sec",
            str(TRIGGER_HORIZON_SEC),
            "--progress-interval-sec",
            "900",
            "--state-sample-interval-sec",
            "60",
            "--out",
            str(remote_output_root / "result.json"),
        ]
    ) + " && printf 'TASK_DONE\\n'"
    return {
        "description": "CFCMT Boston v15 10800s trigger admission",
        "project": "CFCMT",
        "cmd": "mkdir -p "
        + shlex.quote(str(remote_output_root))
        + " && "
        + command,
        "cwd": str(snapshot_root),
        "signature": f"{SIGNATURE_ROOT}/trigger-v1",
        "resource_family": "CFCMT-v118-eth-boston-trigger-admission",
        "ram_resource_family": "CFCMT-v150-eth-boston-trigger-admission-v15",
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


def build_full_spec(**kwargs: Any) -> dict[str, Any]:
    return _version_spec(_build_v14_full_spec(**kwargs), stage="full")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("package", "route", "trigger", "full"), required=True
    )
    parser.add_argument("--stage-manifest", type=Path, required=True)
    parser.add_argument("--scheduler", type=Path, default=DEFAULT_SCHEDULER)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--package-manifest-local", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--remote-output-root", type=Path, required=True)
    parser.add_argument("--local-output-root", type=Path, required=True)
    parser.add_argument("--route-result-local", type=Path)
    parser.add_argument("--route-result-remote", type=Path)
    parser.add_argument("--trigger-result-local", type=Path)
    parser.add_argument("--trigger-result-remote", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.local_output_root.exists():
        raise FileExistsError(f"refusing to overwrite Boston v15 {args.stage} evidence")
    snapshot = _read_json(args.stage_manifest)
    snapshot_root = Path(snapshot["snapshot_root"])
    scheduler = _load_scheduler(args.scheduler)

    if args.stage == "package":
        base_manifest = _manifest(
            args.package_manifest_local, args.expected_package_manifest_sha256
        )
        validate_systemic_uncontrolled_major_manifest(base_manifest)
        if base_manifest.get("protocol") != V14_PACKAGE_PROTOCOL:
            raise ValueError("Boston v15 repair requires the admitted v14 package")
        output_package_root = (
            args.package_root.parent / "boston_complete_published_package_v15"
        )
        _verify_remote_inputs(
            scheduler=scheduler,
            paths_and_hashes=(
                (
                    args.package_root / "package_manifest.json",
                    args.expected_package_manifest_sha256,
                ),
            ),
            absent_path=output_package_root,
        )
        spec = build_package_spec(
            snapshot_root=snapshot_root,
            base_root=args.package_root,
            output_root=output_package_root,
            expected_base_manifest_sha256=args.expected_package_manifest_sha256,
            remote_result_root=args.remote_output_root,
            local_result_root=args.local_output_root,
        )
    else:
        package = _manifest(
            args.package_manifest_local, args.expected_package_manifest_sha256
        )
        validate_trigger_package(package)
        prerequisites: list[tuple[Path, str]] = [
            (
                args.package_root / "package_manifest.json",
                args.expected_package_manifest_sha256,
            )
        ]
        common = {
            "snapshot_root": snapshot_root,
            "package_root": args.package_root,
            "expected_package_manifest_sha256": args.expected_package_manifest_sha256,
            "remote_output_root": args.remote_output_root,
            "local_output_root": args.local_output_root,
        }
        if args.stage == "route":
            spec = build_route_spec(**common)
        elif args.stage == "trigger":
            if args.route_result_local is None or args.route_result_remote is None:
                parser.error("trigger requires route result paths")
            route = _read_json(args.route_result_local)
            validate_route_result(
                route,
                expected_package_manifest_sha256=args.expected_package_manifest_sha256,
            )
            prerequisites.append((args.route_result_remote, _sha256(args.route_result_local)))
            spec = build_trigger_spec(**common)
        else:
            if args.trigger_result_local is None or args.trigger_result_remote is None:
                parser.error("full requires trigger result paths")
            trigger = _read_json(args.trigger_result_local)
            validate_trigger_result(
                trigger,
                expected_package_manifest_sha256=args.expected_package_manifest_sha256,
            )
            prerequisites.append(
                (args.trigger_result_remote, _sha256(args.trigger_result_local))
            )
            spec = build_full_spec(**common)
        _verify_remote_inputs(
            scheduler=scheduler,
            paths_and_hashes=prerequisites,
            absent_path=args.remote_output_root,
        )

    completed = _submit(
        scheduler_path=args.scheduler,
        spec=spec,
        intent_label=f"CFCMT-v150-eth-boston-v15-{args.stage}",
    )
    payload = _common_payload(
        stage=args.stage,
        snapshot=snapshot,
        completed=completed,
        spec=spec,
    )
    payload.update(
        {
            "protocol": LAUNCH_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_manifest_sha256": args.expected_package_manifest_sha256,
            "former_failure_times_sec": list(FORMER_FAILURE_TIMES_SEC),
            "remote_output_root": str(args.remote_output_root),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if completed.returncode != 0:
        raise RuntimeError(f"scheduler rejected Boston v15 {args.stage} admission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
