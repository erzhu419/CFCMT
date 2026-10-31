from pathlib import Path

from scripts.cluster.launch_tsc_external_hierarchical_guard_freeze import (
    ASSIGNMENTS,
    DIRECT_OUTPUT_LIMIT,
    _bounded_direct_output,
    _direct_remote_command,
)


def test_hierarchical_freeze_uses_distinct_physical_nodes() -> None:
    nodes = [node for node, _ in ASSIGNMENTS]
    cities = [city for _, city in ASSIGNMENTS]

    assert len(nodes) == len(set(nodes)) == 2
    assert set(cities) == {"los_angeles", "jinan"}


def test_hierarchical_freeze_result_names_are_city_scoped() -> None:
    root = Path("/tmp/results")

    assert {str(root / city) for _, city in ASSIGNMENTS} == {
        "/tmp/results/los_angeles",
        "/tmp/results/jinan",
    }


def test_direct_command_changes_to_declared_snapshot_before_spawn() -> None:
    command = _direct_remote_command(
        {
            "cwd": "/shared/immutable snapshot",
            "cmd": "python3 -m package.module",
            "extra_env": {"CFCMT_SOURCE_ROOT": "/shared/immutable snapshot"},
        }
    )

    assert command.startswith("cd '/shared/immutable snapshot' && ")
    assert "export CFCMT_SOURCE_ROOT='/shared/immutable snapshot' && " in command
    assert command.endswith("python3 -m package.module")


def test_direct_command_exports_environment_across_shell_compounds() -> None:
    command = _direct_remote_command(
        {
            "cwd": "/shared/snapshot",
            "cmd": "test -d input && python3 worker.py",
            "extra_env": {"CFCMT_SOURCE_ROOT": "/shared/snapshot"},
        }
    )

    assert command == (
        "cd /shared/snapshot && export CFCMT_SOURCE_ROOT=/shared/snapshot && "
        "test -d input && python3 worker.py"
    )


def test_direct_failure_output_is_bounded_and_hashed() -> None:
    output = _bounded_direct_output("x" * (DIRECT_OUTPUT_LIMIT + 100))

    assert len(output) < DIRECT_OUTPUT_LIMIT + 256
    assert "direct output truncated" in output
    assert "original_characters=" in output
    assert "sha256=" in output
