import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_crossfitted_source_selector import (
    CONFIG_RELATIVE,
    PROJECT_ROOT,
    _validate_config,
    build_specs,
)


def test_v150g_frozen_config():
    config = json.loads((PROJECT_ROOT / CONFIG_RELATIVE).read_text(encoding="utf-8"))

    _validate_config(config)

    assert tuple(config["target_city_groups"]) == EXPECTED_CITY_GROUPS


def test_v150g_specs_use_crossfitted_module_and_rejection_identity():
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        authorization_result_remote=Path("/remote/v150c.json"),
        authorization_sha256="a" * 64,
        v150e_rejection_remote=Path("/remote/v150e.json"),
        v150e_rejection_sha256="b" * 64,
        conversion_root=Path("/remote/conversion"),
        fit_result_remote=Path("/remote/fit.json"),
        fit_result_sha256="c" * 64,
        source_cache_root=Path("/remote/cache"),
        source_cache_audit_remote=Path("/remote/audit.json"),
        source_cache_audit_sha256="d" * 64,
        remote_output_root=Path("/remote/results"),
        local_output_root=Path("/local/results"),
        cache_workers=8,
    )

    assert len(specs) == 7
    assert all("traffic_signal_crossfitted_source_selector" in spec["cmd"] for spec in specs)
    assert all("--v150e-rejection-sha256" in spec["cmd"] for spec in specs)
    assert len({spec["signature"] for spec in specs}) == 7
