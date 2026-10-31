from pathlib import Path

from scripts.cluster.launch_tsc_external_network_admission_v6 import (
    ASSIGNMENTS,
    build_specs,
)


def test_v6_admission_specs_cover_every_scenario_seed_once(
    tmp_path: Path,
) -> None:
    manifest_hash = "a" * 64
    tree_hash = "b" * 64

    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion"),
        remote_results_root=Path("/remote/results"),
        local_results_root=tmp_path / "results",
        conversion_manifest_sha256=manifest_hash,
        conversion_tree_sha256=tree_hash,
        cpu_cores=2,
        ram_mb=4096,
    )

    assert len(specs) == 8
    assert len({spec["signature"] for spec in specs}) == 8
    assert [spec["require_node"] for spec in specs] == [
        node for node, _, _ in ASSIGNMENTS
    ]
    for spec in specs:
        assert spec["cwd"] == "/remote/snapshot"
        assert spec["extra_env"]["CFCMT_SOURCE_ROOT"] == "/remote/snapshot"
        assert manifest_hash in spec["cmd"]
        assert tree_hash in spec["cmd"]
        assert "--maximum-collision-count 0" in spec["cmd"]
        assert "--maximum-teleport-fraction 0" in spec["cmd"]
        assert "--horizon-sec 3600" in spec["cmd"]


def test_v7_admission_specs_use_distinct_frozen_signatures(
    tmp_path: Path,
) -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion-v7"),
        remote_results_root=Path("/remote/results-v7"),
        local_results_root=tmp_path / "results-v7",
        conversion_manifest_sha256="c" * 64,
        conversion_tree_sha256="d" * 64,
        cpu_cores=2,
        ram_mb=4096,
        generation="v7",
    )

    assert len(specs) == 8
    assert all("/admission-v7/" in spec["signature"] for spec in specs)
    assert all(spec["resource_family"].endswith("-v7") for spec in specs)


def test_v8_admission_specs_do_not_reuse_v7_identity(tmp_path: Path) -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot-v8"),
        conversion_root=Path("/remote/conversion-v8"),
        remote_results_root=Path("/remote/results-v8"),
        local_results_root=tmp_path / "results-v8",
        conversion_manifest_sha256="e" * 64,
        conversion_tree_sha256="f" * 64,
        cpu_cores=2,
        ram_mb=4096,
        generation="v8",
    )

    assert all("/admission-v8/" in spec["signature"] for spec in specs)
    assert all("admission-v7" not in spec["signature"] for spec in specs)


def test_v9_admission_specs_bind_repaired_conversion_hashes(
    tmp_path: Path,
) -> None:
    manifest_hash = "1" * 64
    tree_hash = "2" * 64
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot-v9"),
        conversion_root=Path("/remote/conversion-v9"),
        remote_results_root=Path("/remote/results-v9"),
        local_results_root=tmp_path / "results-v9",
        conversion_manifest_sha256=manifest_hash,
        conversion_tree_sha256=tree_hash,
        cpu_cores=2,
        ram_mb=4096,
        generation="v9",
    )

    assert all("/admission-v9/" in spec["signature"] for spec in specs)
    assert all("/remote/conversion-v9" in spec["cmd"] for spec in specs)
    assert all(manifest_hash in spec["cmd"] for spec in specs)
    assert all(tree_hash in spec["cmd"] for spec in specs)
