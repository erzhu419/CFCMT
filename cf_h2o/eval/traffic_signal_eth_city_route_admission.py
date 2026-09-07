"""Validate every trip in a canonicalized ETH city package with duarouter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence
import xml.etree.ElementTree as ET

from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.data.package_eth_city_sumo import (
    PROTOCOL as PACKAGE_PROTOCOL,
    ROUTE_CANONICALIZATION_PROTOCOL,
)
from scripts.data.repair_eth_boston_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_REMEDIATED_PACKAGE_PROTOCOL,
    validate_remediated_package_manifest,
)
from scripts.data.repair_eth_boston_uncontrolled_merge import (
    PACKAGE_PROTOCOL as BOSTON_V8_PACKAGE_PROTOCOL,
    validate_uncontrolled_merge_manifest,
)
from scripts.data.repair_eth_boston_ramp_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_V9_PACKAGE_PROTOCOL,
    validate_ramp_merge_manifest,
)
from scripts.data.repair_eth_boston_permissive_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_V10_PACKAGE_PROTOCOL,
    validate_permissive_merge_manifest,
)
from scripts.data.repair_eth_boston_protected_uncontrolled_merge_tls import (
    PACKAGE_PROTOCOL as BOSTON_V11_PACKAGE_PROTOCOL,
    validate_protected_uncontrolled_merge_manifest,
)


PROTOCOL = "eth-five-city-complete-demand-duarouter-admission-v3"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _config_value(root: ET.Element, section: str, tag: str) -> str | None:
    element = root.find(f"./{section}/{tag}")
    return None if element is None else element.attrib.get("value")


def validate_package(
    package_root: Path,
    *,
    expected_manifest_sha256: str,
    expected_city_code: str,
) -> dict[str, Any]:
    root = package_root.resolve()
    manifest_path = root / "package_manifest.json"
    manifest_sha256 = _sha256(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            f"ETH package manifest changed: {manifest_sha256} != "
            f"{expected_manifest_sha256}"
        )
    package = _read_json(manifest_path)
    package_protocol = package.get("protocol")
    if package_protocol == BOSTON_V11_PACKAGE_PROTOCOL:
        validate_protected_uncontrolled_merge_manifest(package)
    elif package_protocol == BOSTON_V10_PACKAGE_PROTOCOL:
        validate_permissive_merge_manifest(package)
    elif package_protocol == BOSTON_V9_PACKAGE_PROTOCOL:
        validate_ramp_merge_manifest(package)
    elif package_protocol == BOSTON_V8_PACKAGE_PROTOCOL:
        validate_uncontrolled_merge_manifest(package)
    elif package_protocol == BOSTON_REMEDIATED_PACKAGE_PROTOCOL:
        validate_remediated_package_manifest(package)
    elif package_protocol != PACKAGE_PROTOCOL:
        raise ValueError(f"unexpected package protocol: {package.get('protocol')}")
    if package.get("city_code") != expected_city_code.upper():
        raise ValueError("ETH route-admission city changed")
    if package.get("passed") is not True or not all(
        bool(value) for value in dict(package.get("gates", {})).values()
    ):
        raise ValueError("ETH package did not pass static admission")

    source_demand = dict(package.get("demand_inventory", {}))
    microscopic_demand = dict(package.get("microscopic_demand_inventory", {}))
    canonicalization = dict(
        package.get("destination_taz_schema_canonicalization", {})
    )
    trip_count = int(source_demand.get("trip_count", -1))
    if (
        trip_count <= 0
        or int(microscopic_demand.get("trip_count", -2)) != trip_count
        or int(source_demand.get("unique_trip_id_count", -1)) != trip_count
        or int(microscopic_demand.get("unique_trip_id_count", -2)) != trip_count
        or canonicalization.get("protocol") != ROUTE_CANONICALIZATION_PROTOCOL
        or int(canonicalization.get("renamed_trip_to_taz_count", -1))
        != trip_count
        or int(canonicalization.get("trip_anchor_edges_changed", -1)) != 0
        or int(canonicalization.get("trips_removed", -1)) != 0
        or canonicalization.get("route_repair_applied") is not False
        or int(
            microscopic_demand.get("custom_trip_to_taz_attribute_count", -1)
        )
        != 0
        or int(microscopic_demand.get("canonical_to_taz_attribute_count", -1))
        != trip_count
    ):
        raise ValueError("ETH destination-TAZ canonicalization is incomplete")

    strict = dict(package.get("microscopic_instantiation", {}))
    route_relative = str(strict.get("route_file", ""))
    if route_relative != "microscopic_trips.rou.xml":
        raise ValueError(f"unexpected microscopic route file: {route_relative}")
    network_relative = str(strict.get("network_file", ""))
    if network_relative != "microscopic_network.net.xml":
        raise ValueError(
            f"unexpected microscopic network file: {network_relative}"
        )
    config = root / str(strict.get("config", ""))
    config_root = ET.parse(config).getroot()
    code = expected_city_code.upper()
    expected_inputs = {
        "net-file": network_relative,
        "route-files": route_relative,
        "additional-files": f"microscopic_inputs.add.xml,source/{code}/taz.xml",
    }
    observed_inputs = {
        tag: _config_value(config_root, "input", tag) for tag in expected_inputs
    }
    if observed_inputs != expected_inputs:
        raise ValueError(
            f"ETH microscopic input boundary changed: {observed_inputs}"
        )
    if _config_value(config_root, "processing", "ignore-route-errors") != "false":
        raise ValueError("ETH route errors are not fail-closed")
    return {
        "package": package,
        "package_manifest_sha256": manifest_sha256,
        "trip_count": trip_count,
        "network": root / network_relative,
        "routes": root / route_relative,
        "taz": root / f"source/{code}/taz.xml",
    }


def duarouter_binary() -> Path:
    sumo_home = Path(os.environ.get("SUMO_HOME", ""))
    binary = sumo_home / "bin/duarouter"
    if not sumo_home.is_dir() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError(f"native duarouter is unavailable under SUMO_HOME: {binary}")
    return binary


def build_duarouter_command(
    *,
    binary: Path,
    network: Path,
    routes: Path,
    taz: Path,
    error_log: Path,
    routing_threads: int,
) -> list[str]:
    if not 1 <= int(routing_threads) <= 20:
        raise ValueError("routing threads must be in [1, 20]")
    return [
        str(binary),
        "--net-file",
        str(network),
        "--route-files",
        str(routes),
        "--additional-files",
        str(taz),
        "--output-file",
        os.devnull,
        "--with-taz",
        "true",
        "--no-internal-links",
        "true",
        "--routing-algorithm",
        "CH",
        "--routing-threads",
        str(int(routing_threads)),
        "--ignore-errors",
        "false",
        "--xml-validation",
        "never",
        "--xml-validation.routes",
        "never",
        "--no-step-log",
        "true",
        "--aggregate-warnings",
        "20",
        "--error-log",
        str(error_log),
    ]


def run_route_admission(
    *,
    package_root: Path,
    expected_package_manifest_sha256: str,
    expected_city_code: str,
    expected_sumo_version: str,
    routing_threads: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    validated = validate_package(
        package_root,
        expected_manifest_sha256=expected_package_manifest_sha256,
        expected_city_code=expected_city_code,
    )
    binary = duarouter_binary()
    version = subprocess.run(
        [str(binary), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    version_line = version.stdout.splitlines()[0] if version.stdout else ""
    if version.returncode != 0 or f"Version {expected_sumo_version}" not in version_line:
        raise RuntimeError(
            f"duarouter version changed: rc={version.returncode}, {version_line}"
        )

    error_log = package_root.resolve().parent / (
        f".{package_root.name}-{expected_city_code.lower()}-duarouter-v2-errors.log"
    )
    if error_log.exists():
        raise FileExistsError(f"refusing to overwrite duarouter error log: {error_log}")
    command = build_duarouter_command(
        binary=binary,
        network=validated["network"],
        routes=validated["routes"],
        taz=validated["taz"],
        error_log=error_log,
        routing_threads=routing_threads,
    )
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    error_text = error_log.read_text(encoding="utf-8") if error_log.is_file() else ""
    passed = completed.returncode == 0 and "Error:" not in error_text
    return {
        "experiment": "traffic_signal_eth_city_route_admission",
        "protocol": PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "decision": (
            "authorize_microscopic_operational_preflight"
            if passed
            else "reject_eth_city_before_microscopic_execution"
        ),
        "package_root": str(package_root.resolve()),
        "package_manifest_sha256": validated["package_manifest_sha256"],
        "city_code": expected_city_code.upper(),
        "trip_count": validated["trip_count"],
        "route_file": str(validated["routes"]),
        "duarouter": {
            "binary": str(binary),
            "version": version_line,
            "routing_threads": int(routing_threads),
            "with_taz": True,
            "bulk_routing": False,
            "ignore_errors": False,
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-12000:],
            "error_log": str(error_log),
            "error_log_tail": error_text[-12000:],
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "claim_boundary": (
            "A passing route admission establishes complete-demand TAZ "
            "routability only; it does not establish microscopic safety or "
            "controller efficacy."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--expected-package-manifest-sha256", required=True)
    parser.add_argument("--expected-city-code", required=True)
    parser.add_argument("--expected-sumo-version", default="1.22.0")
    parser.add_argument("--routing-threads", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite route admission: {args.out}")
    payload = run_route_admission(
        package_root=args.package_root,
        expected_package_manifest_sha256=args.expected_package_manifest_sha256,
        expected_city_code=args.expected_city_code,
        expected_sumo_version=args.expected_sumo_version,
        routing_threads=int(args.routing_threads),
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS" if payload["passed"] else "REJECT",
                "decision": payload["decision"],
                "trip_count": payload["trip_count"],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
