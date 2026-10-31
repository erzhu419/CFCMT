from copy import deepcopy
from pathlib import Path

import pytest

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import (
    _read_json,
)
from scripts.cluster.launch_tsc_target_calibrated_hierarchical_prior import (
    CONFIG_RELATIVE,
    FIT_WORKERS,
    PROJECT_ROOT,
    RAM_MB,
    SIGNATURE_PREFIX,
    _validate_config,
    build_specs,
)


def test_v153a_config_and_launcher_contract() -> None:
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/authorization.json"),
        authorization_sha256="a" * 64,
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit.json"),
        fit_result_sha256="b" * 64,
        source_cache_root=Path("/source"),
        source_cache_audit_remote=Path("/source-audit.json"),
        source_cache_audit_sha256="c" * 64,
        remote_output_root=Path("/results"),
        local_output_root=Path("/local-results"),
        cache_workers=8,
        extra_target_args=(
            "--v152a-rejection",
            "/v152a.json",
            "--v152a-rejection-sha256",
            "d" * 64,
        ),
    )

    assert len(specs) == len(EXPECTED_CITY_GROUPS)
    assert all(spec["ram_mb"] == RAM_MB for spec in specs)
    assert all(spec["require_node"] != "node004" for spec in specs)
    assert SIGNATURE_PREFIX.endswith("-v3")
    for spec in specs:
        assert spec["signature"].startswith(SIGNATURE_PREFIX + "/")
        command = spec["cmd"]
        assert command.count("--v152a-rejection ") == 1
        assert command.count("--v152a-rejection-sha256 ") == 1
        assert "traffic_signal_target_calibrated_hierarchical_prior target" in command
        assert "--cache-workers 8" in command
        assert f"--fit-workers {FIT_WORKERS}" in command


def test_v153a_launcher_rejects_non_nested_configuration() -> None:
    config = deepcopy(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    config["selector"]["inner_training_group_count"] = 80

    with pytest.raises(ValueError, match="frozen configuration changed"):
        _validate_config(config)


@pytest.mark.parametrize("binding", [None, "training_column_indices"])
def test_v153a_launcher_requires_named_feature_binding(binding) -> None:
    config = deepcopy(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    config["feature_binding"] = binding

    with pytest.raises(ValueError, match="frozen configuration changed"):
        _validate_config(config)
