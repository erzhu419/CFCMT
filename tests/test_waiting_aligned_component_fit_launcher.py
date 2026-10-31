from __future__ import annotations

from pathlib import Path

from scripts.cluster.launch_tsc_waiting_aligned_component_fit import (
    build_specs,
    remote_cache_count_command,
)


def test_waiting_aligned_component_fit_specs_keep_equal_information_contract() -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion"),
        source_cache_root=Path("/remote/source_cache"),
        source_cache_audit=Path("/remote/source_audit.json"),
        target_cache_root=Path("/remote/target_cache"),
        target_cache_audit=Path("/remote/target_audit.json"),
        remote_output_root=Path("/remote/fit"),
        cache_workers=32,
        fit_workers=7,
    )
    assert len(specs) == 2
    assert {spec["require_node"] for spec in specs} == {"node001", "node002"}
    assert all(spec["cpu"] == 20 for spec in specs)
    assert all("--source-cache-audit /remote/source_audit.json" in spec["cmd"] for spec in specs)
    assert all("--target-cache-audit /remote/target_audit.json" in spec["cmd"] for spec in specs)
    assert all("--fit-workers 7" in spec["cmd"] for spec in specs)
    assert {spec["signature"].rsplit("/", 1)[-1] for spec in specs} == {
        "los_angeles",
        "jinan",
    }


def test_waiting_aligned_component_fit_specs_allow_explicit_free_nodes() -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        conversion_root=Path("/remote/conversion"),
        source_cache_root=Path("/remote/source_cache"),
        source_cache_audit=Path("/remote/source_audit.json"),
        target_cache_root=Path("/remote/target_cache"),
        target_cache_audit=Path("/remote/target_audit.json"),
        remote_output_root=Path("/remote/fit"),
        cache_workers=20,
        fit_workers=7,
        city_nodes={"los_angeles": "node005", "jinan": "node006"},
    )
    assert [spec["require_node"] for spec in specs] == ["node005", "node006"]
    assert all("--cache-workers 20" in spec["cmd"] for spec in specs)


def test_remote_cache_count_command_counts_each_cache_independently() -> None:
    command = remote_cache_count_command(
        [Path("/remote/source cache"), Path("/remote/target")]
    )
    assert command.startswith("printf '%s\\n'")
    assert "find '/remote/source cache' -type f -name '*.npz'" in command
    assert "find /remote/target -type f -name '*.npz'" in command
    assert command.count("wc -l") == 2
