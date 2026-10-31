from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from cf_h2o.eval.traffic_signal_source_identifiability_budget_curve import (
    TARGET_BUDGETS,
)
from scripts.cluster.launch_tsc_sample_coherent_source_prior import (
    RAM_MB,
    build_specs,
)


def test_launcher_builds_low_memory_city_budget_matrix() -> None:
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/inputs/v150c.json"),
        authorization_sha256="a" * 64,
        v150h_rejection_remote=Path("/inputs/v150h.json"),
        v150h_rejection_sha256="b" * 64,
        conversion_root=Path("/conversions"),
        fit_result_remote=Path("/inputs/fit.json"),
        fit_result_sha256="c" * 64,
        source_cache_root=Path("/cache"),
        source_cache_audit_remote=Path("/inputs/audit.json"),
        source_cache_audit_sha256="d" * 64,
        remote_output_root=Path("/results"),
        local_output_root=Path("/local"),
        cache_workers=8,
    )

    assert len(specs) == len(EXPECTED_CITY_GROUPS) * len(TARGET_BUDGETS)
    assert all(spec["ram_mb"] == RAM_MB for spec in specs)
    assert all("--v150h-rejection" in spec["cmd"] for spec in specs)
