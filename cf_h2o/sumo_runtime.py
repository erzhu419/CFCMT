"""Lightweight SUMO runtime helpers shared by bus and signal experiments."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def load_libsumo() -> Any:
    """Import libsumo without pulling in any experiment-specific dependencies."""

    try:
        module = importlib.import_module("libsumo")
    except ImportError as original_error:
        sumo_home = Path(os.environ.get("SUMO_HOME", "/usr/share/sumo"))
        tools_path = sumo_home / "tools"
        if tools_path.is_dir() and str(tools_path) not in sys.path:
            sys.path.insert(0, str(tools_path))
        try:
            module = importlib.import_module("libsumo")
        except ImportError as fallback_error:
            raise ImportError(
                "libsumo is unavailable; install the libsumo wheel or set SUMO_HOME "
                "to a SUMO installation containing tools/libsumo.py"
            ) from fallback_error
        if module is None:  # pragma: no cover - defensive guard for custom import hooks
            raise original_error

    expected = os.environ.get("CFCMT_EXPECTED_SUMO_VERSION", "").strip()
    if expected:
        actual = libsumo_version(module)
        if actual != expected:
            raise RuntimeError(
                f"libsumo version mismatch: expected {expected!r}, found {actual!r}"
            )
    return module


def libsumo_version(module: Any | None = None) -> str:
    module = module or load_libsumo()
    try:
        raw = module.getVersion()
    except Exception:
        return "unknown"
    text = str(raw[1] if isinstance(raw, (tuple, list)) and len(raw) > 1 else raw)
    return text.removeprefix("SUMO ").strip() or "unknown"


def _usable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)
    except OSError:
        return False


def sumo_scratch_root() -> Path:
    """Return node-local scratch, avoiding WSL's slow Windows-mounted temp path."""

    explicit = os.environ.get("CFCMT_SUMO_SCRATCH_DIR", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not _usable_directory(path):
            raise RuntimeError(f"CFCMT_SUMO_SCRATCH_DIR is not writable: {path}")
        return path

    candidates = [
        os.environ.get("SLURM_TMPDIR", ""),
        "/dev/shm",
        os.environ.get("TMPDIR", ""),
        "/tmp",
        tempfile.gettempdir(),
    ]
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser().resolve()
        if str(path).startswith("/mnt/"):
            continue
        if _usable_directory(path):
            return path
    raise RuntimeError("no writable Linux-local scratch directory is available for SUMO states")


@contextmanager
def sumo_state_directory(*, prefix: str) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=prefix, dir=sumo_scratch_root()) as directory:
        yield Path(directory)


def source_tree_sha256(root: Path | None = None) -> str:
    """Fingerprint executable project Python sources, including uncommitted files."""

    source_root = Path(root or Path(__file__).resolve().parent).resolve()
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py"), key=lambda item: str(item.relative_to(source_root))):
        if "__pycache__" in path.parts or "results" in path.parts:
            continue
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def runtime_metadata() -> dict[str, Any]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    try:
        sumo_binary_version = subprocess.check_output(
            ["sumo", "--version"], stderr=subprocess.DEVNULL, text=True
        ).splitlines()[0]
    except Exception:
        sumo_binary_version = "unknown"
    try:
        loaded_libsumo_version = libsumo_version()
    except Exception as exc:
        loaded_libsumo_version = f"unavailable:{type(exc).__name__}"
    return {
        "git_commit": git_commit,
        "source_tree_sha256": source_tree_sha256(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "sumo_binary": shutil.which("sumo") or "unknown",
        "sumo_version": sumo_binary_version,
        "libsumo_version": loaded_libsumo_version,
        "numpy_version": _package_version("numpy"),
        "scikit_learn_version": _package_version("scikit-learn"),
        "sumo_scratch_root": str(sumo_scratch_root()),
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "determinism_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONHASHSEED",
                "LC_ALL",
                "LANG",
            )
        },
    }


# Backward-compatible private name used by historical experiment modules.
_load_libsumo = load_libsumo


__all__ = [
    "_load_libsumo",
    "libsumo_version",
    "load_libsumo",
    "runtime_metadata",
    "source_tree_sha256",
    "sumo_scratch_root",
    "sumo_state_directory",
]
