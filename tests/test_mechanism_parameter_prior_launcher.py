from pathlib import Path
import subprocess
import sys

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_mechanism_parameter_prior_feasibility import (
    NODES,
    build_specs,
)


def test_v150c_specs_cover_seven_cities_with_eight_workers() -> None:
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        source_cache_root=Path("/cache"),
        source_cache_audit_remote=Path("/audit.json"),
        source_cache_audit_sha256="a" * 64,
        remote_output_root=Path("/remote"),
        local_output_root=Path("/local"),
        cache_workers=8,
    )
    assert len(specs) == len(EXPECTED_CITY_GROUPS) == len(NODES)
    assert [spec["require_node"] for spec in specs] == list(NODES)
    assert all(spec["cpu"] == 8 for spec in specs)
    assert all("--cache-workers 8" in spec["cmd"] for spec in specs)
    assert all(spec["result_dir"].endswith(city) for spec, city in zip(specs, EXPECTED_CITY_GROUPS, strict=True))


def test_v150c_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_tsc_mechanism_parameter_prior_feasibility.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
