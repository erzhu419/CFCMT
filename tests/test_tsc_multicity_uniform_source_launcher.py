import json
from pathlib import Path
import subprocess
import sys

from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_multicity_uniform_source_ensemble import (
    ASSIGNMENTS,
    CONFIG_RELATIVE,
    CPU_CORES,
    RAM_MB,
    _validate_config,
    build_specs,
)


def _specs() -> list[dict]:
    return build_specs(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/inputs/fit.json"),
        fit_result_sha256="a" * 64,
        source_cache_root=Path("/source-cache"),
        source_cache_audit_remote=Path("/inputs/source-audit.json"),
        source_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/remote/v143"),
        local_output_root=Path("/local/v143"),
        cache_workers=8,
        fit_workers=8,
    )


def test_v143_frozen_config_matches_launcher_contract() -> None:
    _validate_config(json.loads(CONFIG_RELATIVE.read_text(encoding="utf-8")))


def test_v143_specs_cover_all_cities_without_node004() -> None:
    specs = _specs()
    assert tuple(city for city, _ in ASSIGNMENTS) == EXPECTED_CITY_GROUPS
    assert len(specs) == len(EXPECTED_CITY_GROUPS)
    assert all(spec["require_node"] != "node004" for spec in specs)
    assert all(spec["cpu"] == CPU_CORES for spec in specs)
    assert all(spec["ram_mb"] == RAM_MB for spec in specs)
    assert len({spec["signature"] for spec in specs}) == len(specs)


def test_v143_specs_hold_out_one_city_and_write_one_json_each() -> None:
    for city, spec in zip(EXPECTED_CITY_GROUPS, _specs()):
        assert f"--target-city {city}" in spec["cmd"]
        assert "--source-cache-root /source-cache" in spec["cmd"]
        assert spec["cmd"].count("result.json") == 1


def test_v143_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_tsc_multicity_uniform_source_ensemble.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
