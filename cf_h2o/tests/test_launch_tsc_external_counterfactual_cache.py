from pathlib import Path

from scripts.cluster.launch_tsc_external_counterfactual_cache import (
    ASSIGNMENTS,
    build_specs,
)


def test_external_adaptation_cache_specs_cover_frozen_matrix(
    tmp_path: Path,
) -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion"),
        conversion_manifest_sha256="a" * 64,
        conversion_tree_sha256="b" * 64,
        remote_cache_root=Path("/remote/cache"),
        remote_results_root=Path("/remote/results"),
        local_results_root=tmp_path / "results",
        collection_shards=16,
        workers=16,
        cpu_cores=16,
        ram_mb=16384,
    )

    assert len(specs) == 8
    assert len({spec["signature"] for spec in specs}) == 8
    assert [spec["require_node"] for spec in specs] == [
        node for node, _, _ in ASSIGNMENTS
    ]
    assert sum(spec["cpu"] for spec in specs) == 128
    for spec in specs:
        assert "--collection-shards 16" in spec["cmd"]
        assert "--workers 16" in spec["cmd"]
        assert "--expected-input-manifest-sha256" in spec["cmd"]
        assert "--expected-input-tree-sha256" in spec["cmd"]
        assert spec["extra_env"]["CFCMT_EXTERNAL_CONVERSION_ROOT"] == (
            "/remote/conversion"
        )


def test_v9_cache_specs_use_repaired_manifest_and_fresh_identity(
    tmp_path: Path,
) -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot-v9"),
        conversion_root=Path("/remote/conversion-v9"),
        conversion_manifest_sha256="c" * 64,
        conversion_tree_sha256="d" * 64,
        remote_cache_root=Path("/remote/cache-v9"),
        remote_results_root=Path("/remote/results-v9"),
        local_results_root=tmp_path / "results-v9",
        collection_shards=16,
        workers=16,
        cpu_cores=16,
        ram_mb=16384,
        generation="v9",
    )

    assert all("adaptation-cache-v9" in spec["signature"] for spec in specs)
    assert all(
        "traffic_signal_tsc_v39_external_la_jinan_v9_manifest.json"
        in spec["cmd"]
        for spec in specs
    )
    assert all("/remote/conversion-v9" in spec["cmd"] for spec in specs)
