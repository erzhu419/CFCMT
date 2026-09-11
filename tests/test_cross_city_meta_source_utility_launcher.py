from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_cross_city_meta_source_utility import (
    CONFIG_RELATIVE,
    PROJECT_ROOT,
    RAM_MB,
    _validate_config,
    build_specs,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import _read_json


def test_v151a_config_and_launcher_contract() -> None:
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/authorization.json"),
        authorization_sha256="a" * 64,
        v150j_rejection_remote=Path("/v150j.json"),
        v150j_rejection_sha256="b" * 64,
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit.json"),
        fit_result_sha256="c" * 64,
        source_cache_root=Path("/source"),
        source_cache_audit_remote=Path("/source-audit.json"),
        source_cache_audit_sha256="d" * 64,
        remote_output_root=Path("/results"),
        local_output_root=Path("/local-results"),
        cache_workers=8,
        extra_target_args=(
            "--v150o-rejection",
            "/v150o.json",
            "--v150o-rejection-sha256",
            "e" * 64,
        ),
    )

    assert len(specs) == len(EXPECTED_CITY_GROUPS)
    assert all(spec["ram_mb"] == RAM_MB for spec in specs)
    for spec in specs:
        command = spec["cmd"]
        assert command.count("--v150o-rejection ") == 1
        assert command.count("--v150o-rejection-sha256 ") == 1
        assert "traffic_signal_cross_city_meta_source_utility target" in command
        assert "--cache-workers 8" in command
        assert "--fit-workers 6" in command
