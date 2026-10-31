from pathlib import Path

import pytest

from scripts.cluster.build_tsc_combined_counterfactual_cache import (
    aggregate_cache_sha256,
    build_combined_cache,
)


def _write(root: Path, name: str, payload: bytes) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(payload)
    return path


def test_build_combined_cache_hard_links_disjoint_inputs(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source = _write(source_root, "source.npz", b"source")
    target = _write(target_root, "target.npz", b"target")
    output = tmp_path / "combined"

    result = build_combined_cache(
        source_root=source_root,
        target_root=target_root,
        output_root=output,
        expected_source_count=1,
        expected_target_count=1,
        expected_source_sha256=aggregate_cache_sha256([source]),
        expected_target_sha256=aggregate_cache_sha256([target]),
    )

    assert result["combined_file_count"] == 2
    assert (output / "source.npz").stat().st_ino == source.stat().st_ino
    assert (output / "target.npz").stat().st_ino == target.stat().st_ino
    assert (output / "cache_manifest.json").is_file()


def test_build_combined_cache_refuses_overlapping_names(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    source = _write(source_root, "same.npz", b"source")
    target = _write(target_root, "same.npz", b"target")

    with pytest.raises(ValueError, match="basenames overlap"):
        build_combined_cache(
            source_root=source_root,
            target_root=target_root,
            output_root=tmp_path / "combined",
            expected_source_count=1,
            expected_target_count=1,
            expected_source_sha256=aggregate_cache_sha256([source]),
            expected_target_sha256=aggregate_cache_sha256([target]),
        )


def test_build_combined_cache_refuses_existing_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "target"
    _write(source_root, "source.npz", b"source")
    _write(target_root, "target.npz", b"target")
    output = tmp_path / "combined"
    output.mkdir()

    with pytest.raises(FileExistsError):
        build_combined_cache(
            source_root=source_root,
            target_root=target_root,
            output_root=output,
            expected_source_count=1,
            expected_target_count=1,
            expected_source_sha256="0" * 64,
            expected_target_sha256="0" * 64,
        )
