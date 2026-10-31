import json
from pathlib import Path
import subprocess
import sys

from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_multicity_source_compatibility_inventory import (
    CONFIG_RELATIVE,
    MODULE,
    RESOURCE_FAMILY,
    SIGNATURE_PREFIX,
    _validate_config,
)
from scripts.cluster.launch_tsc_multicity_uniform_source_ensemble import (
    ASSIGNMENTS,
    build_specs,
)


def _specs() -> list[dict]:
    return build_specs(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit.json"),
        fit_result_sha256="a" * 64,
        source_cache_root=Path("/source"),
        source_cache_audit_remote=Path("/audit.json"),
        source_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/remote/v144"),
        local_output_root=Path("/local/v144"),
        cache_workers=8,
        fit_workers=8,
        module=MODULE,
        description_prefix="CFCMT V144 source compatibility target",
        signature_prefix=SIGNATURE_PREFIX,
        resource_family=RESOURCE_FAMILY,
    )


def test_v144_config_and_specs_cover_all_cities_without_node004() -> None:
    _validate_config(json.loads(CONFIG_RELATIVE.read_text(encoding="utf-8")))
    specs = _specs()
    assert tuple(city for city, _ in ASSIGNMENTS) == EXPECTED_CITY_GROUPS
    assert len(specs) == 7
    assert all(spec["require_node"] != "node004" for spec in specs)
    assert all(f"-m {MODULE}" in spec["cmd"] for spec in specs)
    assert all(spec["signature"].startswith(SIGNATURE_PREFIX) for spec in specs)
    assert all(spec["resource_family"] == RESOURCE_FAMILY for spec in specs)


def test_v144_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_tsc_multicity_source_compatibility_inventory.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
