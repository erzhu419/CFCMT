import os
import subprocess
import sys
from pathlib import Path

import pytest

from cf_h2o.sumo_runtime import load_libsumo, sumo_scratch_root


def test_explicit_sumo_scratch_directory(monkeypatch, tmp_path):
    scratch = tmp_path / "sumo-scratch"
    monkeypatch.setenv("CFCMT_SUMO_SCRATCH_DIR", str(scratch))

    assert sumo_scratch_root() == scratch.resolve()
    assert scratch.is_dir()


def test_invalid_explicit_sumo_scratch_fails(monkeypatch, tmp_path):
    regular_file = tmp_path / "not-a-directory"
    regular_file.write_text("x", encoding="utf-8")
    monkeypatch.setenv("CFCMT_SUMO_SCRATCH_DIR", str(regular_file))

    with pytest.raises(RuntimeError, match="not writable"):
        sumo_scratch_root()


def test_traffic_signal_import_does_not_require_bus_ml_dependencies():
    code = """
import importlib.abc
import sys

class BlockOptionalBusDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'pandas'}:
            raise ImportError(f'blocked optional dependency: {fullname}')
        return None

sys.meta_path.insert(0, BlockOptionalBusDependencies())
import cf_h2o.eval
import cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite
assert 'torch' not in sys.modules
assert 'pandas' not in sys.modules
"""
    root = Path(__file__).resolve().parents[2]
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
    )


def test_expected_libsumo_version_guard(monkeypatch):
    try:
        module = load_libsumo()
    except ImportError:
        pytest.skip("libsumo is not installed")
    actual = str(module.getVersion()[1]).removeprefix("SUMO ")
    monkeypatch.setenv("CFCMT_EXPECTED_SUMO_VERSION", actual)
    assert load_libsumo() is module
    monkeypatch.setenv("CFCMT_EXPECTED_SUMO_VERSION", "0.0.invalid")
    with pytest.raises(RuntimeError, match="version mismatch"):
        load_libsumo()
