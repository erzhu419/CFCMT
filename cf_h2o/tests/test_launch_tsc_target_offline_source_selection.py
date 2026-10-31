from pathlib import Path

from scripts.cluster.launch_tsc_target_offline_source_selection import _command


def test_target_offline_launch_splits_familywise_error_budget() -> None:
    command = _command(
        snapshot_root=Path("/snapshot"),
        input_root=Path("/inputs"),
        main_root=Path("/main"),
        conversion_root=Path("/conversion"),
        remote_result_root=Path("/results"),
        city="jinan",
        model_sha256="a" * 64,
        result_sha256="b" * 64,
        workers=32,
        fit_workers=20,
        minimum_mean_improvement=0.005,
        maximum_fold_regression=0.01,
        source_familywise_alpha=0.025,
        guard_familywise_alpha=0.025,
        guard_minimum_mean_improvement=0.002,
        guard_maximum_fold_regression=0.01,
        guard_min_context_trust=0.1,
    )

    assert "--source-familywise-alpha 0.025" in command
    assert "--guard-familywise-alpha 0.025" in command
    assert "--guard-minimum-mean-improvement 0.002" in command
    assert "--guard-min-context-trust 0.1" in command
    assert "--confidence-multiplier" not in command
