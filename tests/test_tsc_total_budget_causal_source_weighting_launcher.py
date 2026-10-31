import json
from pathlib import Path
import subprocess
import sys

from scripts.cluster.launch_tsc_total_budget_causal_source_weighting import (
    ASSIGNMENTS,
    CONFIG_RELATIVE,
    CPU_CORES,
    RAM_MB,
    TARGET_BUDGETS,
    _validate_config,
    build_specs,
)


def _specs() -> list[dict]:
    return build_specs(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/inputs/fit.json"),
        fit_result_sha256="a" * 64,
        source_cache_root=Path("/source-cache"),
        source_cache_audit_remote=Path("/inputs/source-audit.json"),
        selector_cache_root=Path("/selector-cache"),
        selector_cache_audit_remote=Path("/inputs/selector-audit.json"),
        selector_cache_audit_sha256="b" * 64,
        remote_output_root=Path("/remote/v140"),
        local_output_root=Path("/local/v140"),
        cache_workers=20,
        fit_workers=20,
    )


def test_v140_frozen_config_matches_launcher_contract() -> None:
    config = json.loads(CONFIG_RELATIVE.read_text(encoding="utf-8"))
    _validate_config(config)


def test_v140_specs_cover_every_budget_without_node004() -> None:
    specs = _specs()
    assert len(specs) == len(TARGET_BUDGETS)
    assert tuple(budget for budget, _ in ASSIGNMENTS) == TARGET_BUDGETS
    assert {spec["require_node"] for spec in specs} <= {
        "node001",
        "node002",
        "node003",
        "node005",
        "node006",
    }
    assert all(spec["require_node"] != "node004" for spec in specs)
    assert all(spec["cpu"] == CPU_CORES for spec in specs)
    assert all(spec["ram_mb"] == RAM_MB for spec in specs)
    assert len({spec["signature"] for spec in specs}) == len(specs)


def test_v140_specs_reuse_only_frozen_inputs_and_write_small_results() -> None:
    for budget, spec in zip(TARGET_BUDGETS, _specs()):
        command = spec["cmd"]
        assert f"--budget {budget}" in command
        assert "--fit-result /inputs/fit.json" in command
        assert "--source-cache-root /source-cache" in command
        assert "--selector-cache-root /selector-cache" in command
        assert "--target-cache-root" not in command
        assert "--artifact" not in command
        assert command.count("result.json") == 1


def test_v140_launcher_is_directly_executable() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/cluster/launch_tsc_total_budget_causal_source_weighting.py",
            "--help",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
