from pathlib import Path

from scripts.cluster.launch_eth_boston_route_admission_v5 import (
    ALLOWED_NODES,
    CPU_CORES,
    RAM_MB,
    SIGNATURE,
    build_spec,
)


def test_route_admission_spec_freezes_v7_contract() -> None:
    digest = "a" * 64
    spec = build_spec(
        snapshot_root=Path("/snapshot"),
        package_root=Path("/package-v7"),
        expected_package_manifest_sha256=digest,
        remote_output_root=Path("/remote/result"),
        local_output_root=Path("/local/result"),
    )
    assert spec["signature"] == SIGNATURE
    assert spec["cpu"] == CPU_CORES == 20
    assert spec["ram_mb"] == RAM_MB == 16_384
    assert spec["allowed_nodes"] == list(ALLOWED_NODES)
    assert "node004" not in spec["allowed_nodes"]
    assert "traffic_signal_eth_city_route_admission" in spec["cmd"]
    assert "--routing-threads 20" in spec["cmd"]
    assert "--expected-package-manifest-sha256 " + digest in spec["cmd"]
