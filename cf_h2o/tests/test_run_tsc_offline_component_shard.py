import json
from pathlib import Path

import pytest

from scripts.cluster.run_tsc_offline_component_shard import (
    _family_command,
    _validate_result,
)


def _result_payload(*, budget: int, adaptation_groups: int) -> dict:
    family = "causal_antisymmetric_pairwise_advantage"
    excluded = [f"g{index}" for index in range(budget)]
    return {
        "target_group_budget": budget,
        "model_families": [family],
        "targets": [
            {
                "target": "grid4x4",
                "target_group_budget": budget,
                "excluded_group_ids": excluded,
                "model_diagnostics": {
                    "target_adaptation_groups": adaptation_groups,
                },
                "offline_evaluation": {
                    "evaluation_group_count": 17,
                    "families": {
                        family: {
                            "group_count": 17,
                            "mean_normalized_action_regret": 0.25,
                            "optimal_action_rate": 0.5,
                        }
                    },
                },
            }
        ],
    }


def test_family_command_forwards_target_budget(tmp_path: Path) -> None:
    command = _family_command(
        cache_root=tmp_path / "cache",
        baseline_result=tmp_path / "baseline.json",
        manifest=tmp_path / "manifest.json",
        target="grid4x4",
        family="causal_antisymmetric_pairwise_advantage",
        result_path=tmp_path / "result.json",
        cache_workers=8,
        target_group_budget=32,
    )
    index = command.index("--target-group-budget")
    assert command[index + 1] == "32"


@pytest.mark.parametrize(
    ("budget", "adaptation_groups"),
    [(0, 0), (8, 4), (32, 19), (120, 72)],
)
def test_validate_result_accepts_exact_budget_accounting(
    tmp_path: Path,
    budget: int,
    adaptation_groups: int,
) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            _result_payload(
                budget=budget,
                adaptation_groups=adaptation_groups,
            )
        ),
        encoding="utf-8",
    )
    row = _validate_result(
        path,
        target="grid4x4",
        family="causal_antisymmetric_pairwise_advantage",
        target_group_budget=budget,
    )
    assert row["excluded_group_count"] == budget
    assert row["target_adaptation_groups"] == adaptation_groups


def test_validate_result_rejects_budget_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(_result_payload(budget=8, adaptation_groups=4)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="target budget mismatch"):
        _validate_result(
            path,
            target="grid4x4",
            family="causal_antisymmetric_pairwise_advantage",
            target_group_budget=16,
        )


def test_validate_result_rejects_missing_excluded_group(tmp_path: Path) -> None:
    payload = _result_payload(budget=8, adaptation_groups=4)
    payload["targets"][0]["excluded_group_ids"].pop()
    path = tmp_path / "result.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="expected 8 excluded"):
        _validate_result(
            path,
            target="grid4x4",
            family="causal_antisymmetric_pairwise_advantage",
            target_group_budget=8,
        )
