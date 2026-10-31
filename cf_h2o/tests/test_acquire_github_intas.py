from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from scripts.data.acquire_github_intas import (
    COMMIT,
    EXPECTED_BLOB_BYTES,
    EXPECTED_BLOB_COUNT,
    REQUIRED_PATHS,
    _extraction_command,
    _finalize_command,
    _validate_git_tree,
)


def _valid_tree() -> tuple[dict, dict]:
    required = sorted(REQUIRED_PATHS)
    filler_count = EXPECTED_BLOB_COUNT - len(required)
    paths = [*required, *(f"filler/{index}.txt" for index in range(filler_count))]
    sizes = [1] * len(paths)
    sizes[-1] += EXPECTED_BLOB_BYTES - sum(sizes)
    tree = {
        "truncated": False,
        "tree": [
            {
                "path": path,
                "type": "blob",
                "size": size,
                "sha": f"sha-{index}",
            }
            for index, (path, size) in enumerate(zip(paths, sizes, strict=True))
        ],
    }
    return {"sha": COMMIT}, tree


def test_git_tree_gate_requires_complete_fixed_inventory() -> None:
    commit, tree = _valid_tree()
    blobs = _validate_git_tree(commit_payload=commit, tree_payload=tree)

    assert len(blobs) == EXPECTED_BLOB_COUNT
    assert sum(row["size_bytes"] for row in blobs) == EXPECTED_BLOB_BYTES
    assert REQUIRED_PATHS.issubset({row["path"] for row in blobs})


def test_git_tree_gate_rejects_size_drift() -> None:
    commit, tree = _valid_tree()
    tree["tree"][0]["size"] += 1

    with pytest.raises(ValueError, match="blob bytes changed"):
        _validate_git_tree(commit_payload=commit, tree_payload=tree)


def test_extraction_command_checks_every_blob_without_local_raw_data() -> None:
    blobs = [
        {"path": "scenario/ingolstadt.net.xml", "size_bytes": 123},
        {"path": "README.md", "size_bytes": 456},
    ]
    command = _extraction_command(
        staging=PurePosixPath("/remote/staging"),
        blobs=blobs,
    )

    assert "tar -xzf" in command
    assert "--strip-components=1" in command
    assert "/remote/staging/source/scenario/ingolstadt.net.xml" in command
    assert "-eq 123" in command
    assert f"-eq {EXPECTED_BLOB_COUNT}" in command


def test_finalize_command_is_idempotent_after_transport_disconnect() -> None:
    command = _finalize_command(
        staging=PurePosixPath("/remote/.staging"),
        output_root=PurePosixPath("/remote/acquisition"),
    )

    assert "if test -d /remote/acquisition" in command
    assert "test ! -e /remote/.staging" in command
    assert "/remote/acquisition/acquisition_manifest.json" in command
    assert "mv /remote/.staging /remote/acquisition" in command
