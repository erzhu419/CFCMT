from pathlib import Path

from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_rigid_mechanism_prior_integration import (
    build_specs,
)


def test_v150d_specs_cover_seven_cities_and_exact_authorization() -> None:
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/evidence/v150c.json"),
        authorization_sha256="a" * 64,
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit/result.json"),
        fit_result_sha256="b" * 64,
        source_cache_root=Path("/source-cache"),
        source_cache_audit_remote=Path("/source-audit.json"),
        source_cache_audit_sha256="c" * 64,
        remote_output_root=Path("/remote/results"),
        local_output_root=Path("/local/results"),
        cache_workers=8,
    )
    assert len(specs) == len(EXPECTED_CITY_GROUPS)
    assert {
        spec["signature"].rsplit("/", 1)[-1] for spec in specs
    } == set(EXPECTED_CITY_GROUPS)
    assert all(spec["cpu"] == 8 and spec["vram"] == 0 for spec in specs)
    assert all("--authorization-result-sha256 " + "a" * 64 in spec["cmd"] for spec in specs)
    assert all("/evidence/v150c.json" in spec["cmd"] for spec in specs)
