#!/usr/bin/env python3
"""Build an immutable combined TSC cache from audited source and target banks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "hard-linked-disjoint-counterfactual-cache-union-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_cache_sha256(paths: Iterable[Path]) -> str:
    ordered = sorted((Path(path) for path in paths), key=lambda path: path.name)
    names = [path.name for path in ordered]
    if len(names) != len(set(names)):
        raise ValueError("cache basenames are not unique")
    digest = hashlib.sha256()
    for path in ordered:
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _partition_files(root: Path, expected_count: int, label: str) -> list[Path]:
    root = Path(root).resolve()
    files = sorted(root.glob("*.npz"), key=lambda path: path.name)
    non_npz = [path for path in root.iterdir() if path.is_file() and path.suffix != ".npz"]
    if non_npz:
        raise ValueError(f"{label} cache contains non-NPZ files: {non_npz[:3]}")
    if len(files) != int(expected_count):
        raise ValueError(
            f"{label} cache file count mismatch: {len(files)} != {int(expected_count)}"
        )
    return files


def build_combined_cache(
    *,
    source_root: Path,
    target_root: Path,
    output_root: Path,
    expected_source_count: int,
    expected_target_count: int,
    expected_source_sha256: str,
    expected_target_sha256: str,
) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"combined cache already exists: {output_root}")
    if source_root.stat().st_dev != target_root.stat().st_dev:
        raise ValueError("source and target caches must share a filesystem for hard links")
    source_files = _partition_files(source_root, expected_source_count, "source")
    target_files = _partition_files(target_root, expected_target_count, "target")
    all_files = source_files + target_files
    names = [path.name for path in all_files]
    if len(names) != len(set(names)):
        raise ValueError("source and target cache basenames overlap")

    source_digest = aggregate_cache_sha256(source_files)
    target_digest = aggregate_cache_sha256(target_files)
    if source_digest != str(expected_source_sha256):
        raise ValueError(
            f"source cache aggregate mismatch: {source_digest} != {expected_source_sha256}"
        )
    if target_digest != str(expected_target_sha256):
        raise ValueError(
            f"target cache aggregate mismatch: {target_digest} != {expected_target_sha256}"
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        partition_by_name = {
            **{path.name: "source" for path in source_files},
            **{path.name: "target" for path in target_files},
        }
        source_by_name = {path.name: path for path in all_files}
        file_records: list[dict[str, Any]] = []
        for name in sorted(source_by_name):
            original = source_by_name[name]
            linked = staging_root / name
            os.link(original, linked)
            original_stat = original.stat()
            linked_stat = linked.stat()
            if (
                original_stat.st_dev != linked_stat.st_dev
                or original_stat.st_ino != linked_stat.st_ino
            ):
                raise RuntimeError(f"hard-link identity check failed for {name}")
            file_records.append(
                {
                    "name": name,
                    "partition": partition_by_name[name],
                    "source_path": str(original),
                    "sha256": _file_sha256(linked),
                }
            )

        combined_files = sorted(staging_root.glob("*.npz"), key=lambda path: path.name)
        combined_digest = aggregate_cache_sha256(combined_files)
        manifest: dict[str, Any] = {
            "protocol": PROTOCOL,
            "source_root": str(source_root),
            "target_root": str(target_root),
            "output_root": str(output_root),
            "source_file_count": len(source_files),
            "target_file_count": len(target_files),
            "combined_file_count": len(combined_files),
            "source_aggregate_sha256": source_digest,
            "target_aggregate_sha256": target_digest,
            "combined_aggregate_sha256": combined_digest,
            "files": file_records,
        }
        (staging_root / "cache_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging_root, output_root)
        return manifest
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-source-count", type=int, required=True)
    parser.add_argument("--expected-target-count", type=int, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-target-sha256", required=True)
    args = parser.parse_args()
    result = build_combined_cache(
        source_root=args.source_root,
        target_root=args.target_root,
        output_root=args.output_root,
        expected_source_count=args.expected_source_count,
        expected_target_count=args.expected_target_count,
        expected_source_sha256=args.expected_source_sha256,
        expected_target_sha256=args.expected_target_sha256,
    )
    print(
        "combined cache built: "
        f"{result['combined_file_count']} files, "
        f"SHA-256 {result['combined_aggregate_sha256']}"
    )


if __name__ == "__main__":
    main()
