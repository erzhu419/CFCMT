from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import (
    FIT_WORKERS,
    RAM_MB,
    build_specs,
    select_target_specs,
)


def test_launcher_builds_one_nested_low_memory_task_per_city() -> None:
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/inputs/v150c.json"),
        authorization_sha256="a" * 64,
        v150j_rejection_remote=Path("/inputs/v150j.json"),
        v150j_rejection_sha256="b" * 64,
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

    assert len(specs) == len(EXPECTED_CITY_GROUPS)
    assert all(spec["ram_mb"] == RAM_MB for spec in specs)
    assert all("--v150j-rejection" in spec["cmd"] for spec in specs)
    assert all(f"--fit-workers {FIT_WORKERS}" in spec["cmd"] for spec in specs)

    recovered = select_target_specs(specs, ("atlanta",))
    assert len(recovered) == 1
    assert recovered[0]["signature"].endswith("/atlanta")


def test_launcher_keeps_runtime_models_out_of_downloaded_results() -> None:
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/inputs/v150c.json"),
        authorization_sha256="a" * 64,
        v150j_rejection_remote=Path("/inputs/v150j.json"),
        v150j_rejection_sha256="b" * 64,
        conversion_root=Path("/conversions"),
        fit_result_remote=Path("/inputs/fit.json"),
        fit_result_sha256="c" * 64,
        source_cache_root=Path("/cache"),
        source_cache_audit_remote=Path("/inputs/audit.json"),
        source_cache_audit_sha256="d" * 64,
        remote_output_root=Path("/results"),
        local_output_root=Path("/local"),
        cache_workers=8,
        runtime_model_root=Path("/models"),
    )

    for spec in specs:
        city = spec["signature"].rsplit("/", 1)[-1]
        assert f"--runtime-model-out /models/{city}/model.pkl" in spec["cmd"]
        assert spec["result_dir"] == f"/results/{city}"
        assert spec["local_result_dir"] == f"/local/{city}"
