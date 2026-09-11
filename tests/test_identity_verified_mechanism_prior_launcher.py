from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_identity_verified_mechanism_prior import (
    MODULE,
    build_specs,
)


def test_v150e_specs_cover_cities_and_bind_rejected_v150d() -> None:
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/evidence/v150c.json"),
        authorization_sha256="a" * 64,
        v150d_rejection_remote=Path("/evidence/v150d.json"),
        v150d_rejection_sha256="d" * 64,
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
    assert all(f"-m {MODULE}" in spec["cmd"] for spec in specs)
    assert all("--v150d-rejection /evidence/v150d.json" in spec["cmd"] for spec in specs)
    assert all("--v150d-rejection-sha256 " + "d" * 64 in spec["cmd"] for spec in specs)
    assert all("/v150e/" in spec["signature"] for spec in specs)
