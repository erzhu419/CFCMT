from pathlib import Path

from scripts.cluster.launch_tsc_external_long_horizon_target_veto_cache import (
    NODES,
    build_specs,
)


def test_v65_cache_builds_one_full_horizon_task_per_node(tmp_path: Path) -> None:
    specs, authorization_sha = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        protocol_path=Path(
            "cf_h2o/config/traffic_signal_tsc_v47_external_v9_long_horizon_target_veto.json"
        ),
        authorization_audit_local=Path(
            "cf_h2o/results/cluster/tsc_v65r61_external_v9_long_horizon_target_veto_20260810/cache_authorization_v1.json"
        ),
        authorization_audit_remote=Path("/remote/evidence/authorization.json"),
        conversion_root=Path("/remote/conversion"),
        conversion_manifest_sha256="a" * 64,
        conversion_tree_sha256="b" * 64,
        remote_cache_root=Path("/remote/cache"),
        remote_results_root=Path("/remote/results"),
        local_results_root=tmp_path / "results",
        workers=64,
        cpu_cores=68,
        ram_mb=98304,
    )

    assert len(specs) == 6
    assert tuple(spec["require_node"] for spec in specs) == NODES
    assert len(authorization_sha) == 64
    assert all("--duration-sec 3600" in spec["cmd"] for spec in specs)
    assert all(
        "--counterfactual-horizon-intervals 30" in spec["cmd"] for spec in specs
    )
    assert all("--workers 64" in spec["cmd"] for spec in specs)
    assert all(spec["cpu"] == 68 and spec["ram_mb"] == 98304 for spec in specs)
