import json
from pathlib import Path
import subprocess
import sys

from scripts.cluster.launch_tsc_total_budget_policy_aligned_selection import (
    ALLOWED_NODES,
    CONFIG_RELATIVE,
    CPU_CORES,
    RAM_MB,
    _validate_config,
    build_spec,
)


def _spec() -> dict:
    return build_spec(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/inputs/fit.json"),
        fit_result_sha256="a" * 64,
        source_cache_root=Path("/source-cache"),
        source_cache_audit_remote=Path("/inputs/source-audit.json"),
        selector_cache_root=Path("/selector-cache"),
        selector_cache_audit_remote=Path("/inputs/selector-audit.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/remote/v142"),
        local_output_root=Path("/local/v142"),
        cache_workers=20,
        fit_workers=20,
    )


def test_v142_frozen_config_matches_launcher_contract() -> None:
    config = json.loads(CONFIG_RELATIVE.read_text(encoding="utf-8"))
    _validate_config(config)


def test_v142_spec_is_one_b25_task_without_node004() -> None:
    spec = _spec()
    assert spec["allowed_nodes"] == list(ALLOWED_NODES)
    assert "node004" not in spec["allowed_nodes"]
    assert spec["cpu"] == CPU_CORES
    assert spec["ram_mb"] == RAM_MB
    assert "--budget 25" in spec["cmd"]
    assert "traffic_signal_total_budget_policy_aligned_selection" in spec["cmd"]
    assert spec["cmd"].count("result.json") == 1


def test_v142_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_tsc_total_budget_policy_aligned_selection.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
