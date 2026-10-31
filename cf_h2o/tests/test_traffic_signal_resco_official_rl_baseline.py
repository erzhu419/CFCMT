import json
import zipfile

from cf_h2o.eval.traffic_signal_resco_official_rl_baseline import (
    OfficialRunSpec,
    aggregate_rows,
    _ensure_route_override_file,
    _parse_skip_runs,
    cache_path_for_spec,
    load_cached_row,
    run_experiment,
    summarize_official_result,
    write_cached_row,
)


def test_summarize_official_result_uses_testing_tail(tmp_path):
    result_path = tmp_path / "official.json"
    result_path.write_text(
        json.dumps(
            {
                "timeLoss": {"hash": [10.0, 20.0, 30.0]},
                "duration": {"hash": [100.0, 110.0, 120.0]},
                "waitingTime": {"hash": [1.0, 2.0, 3.0]},
                "queue_lengths": {"hash": [5.0, 6.0, 7.0]},
                "max_queues": {"hash": [8.0, 9.0, 10.0]},
                "rewards": {"hash": [-3.0, -2.0, -1.0]},
                "vehicles": {"hash": [50.0, 51.0, 52.0]},
            }
        ),
        encoding="utf-8",
    )
    row = summarize_official_result(
        spec=OfficialRunSpec(scenario="cologne1", algorithm="IDQN", episodes=2, testing=1, seed=131),
        result_path=result_path,
        ok=True,
        returncode=0,
        elapsed_sec=12.0,
        stdout_tail="",
        stderr_tail="",
    )

    assert row["test_mean_timeLoss"] == 30.0
    assert row["test_mean_duration"] == 120.0
    assert row["all_mean_timeLoss"] == 20.0


def test_aggregate_rows_groups_by_algorithm_and_scenario(tmp_path):
    rows = [
        {
            "ok": True,
            "algorithm": "IDQN",
            "scenario": "cologne1",
            "test_mean_timeLoss": 10.0,
            "test_mean_duration": 100.0,
            "test_mean_waitingTime": 2.0,
            "test_mean_queue_lengths": 4.0,
            "test_mean_max_queues": 5.0,
            "elapsed_sec": 6.0,
        },
        {
            "ok": True,
            "algorithm": "IDQN",
            "scenario": "cologne1",
            "test_mean_timeLoss": 14.0,
            "test_mean_duration": 110.0,
            "test_mean_waitingTime": 4.0,
            "test_mean_queue_lengths": 6.0,
            "test_mean_max_queues": 7.0,
            "elapsed_sec": 8.0,
        },
        {"ok": False, "algorithm": "MPLight", "scenario": "cologne1"},
    ]

    agg = aggregate_rows(rows)

    assert len(agg) == 1
    assert agg[0]["runs"] == 2
    assert agg[0]["test_timeLoss"] == 12.0
    assert agg[0]["test_queue_lengths"] == 5.0


def test_ensure_route_override_file_extracts_zipped_route(tmp_path):
    env_dir = tmp_path / "environments" / "grid4x4"
    env_dir.mkdir(parents=True)
    with zipfile.ZipFile(env_dir / "grid4x4.zip", "w") as zf:
        zf.writestr("grid4x4_1.rou.xml", "<routes />")

    _ensure_route_override_file(tmp_path, "grid4x4")

    assert (env_dir / "grid4x4.rou.xml").read_text(encoding="utf-8") == "<routes />"


def test_successful_cache_row_is_reused_but_failed_row_is_not(tmp_path):
    spec = OfficialRunSpec(scenario="cologne1", algorithm="IDQN", episodes=5, testing=2, seed=131)
    path = cache_path_for_spec(tmp_path, spec, extra_args=(), keep_episode_logs=True)

    write_cached_row(path, {"ok": False, "algorithm": "IDQN"})
    assert load_cached_row(path) is None

    write_cached_row(path, {"ok": True, "algorithm": "IDQN"})
    cached = load_cached_row(path)

    assert cached is not None
    assert cached["cached"] is True


def test_parse_skip_runs():
    assert _parse_skip_runs("MPLight:cologne8, IDQN:grid4x4") == (
        ("MPLight", "cologne8"),
        ("IDQN", "grid4x4"),
    )


def test_run_experiment_skips_requested_pairs(tmp_path):
    spec = OfficialRunSpec(scenario="cologne1", algorithm="IDQN", episodes=1, testing=1, seed=None)
    path = cache_path_for_spec(tmp_path / "cache", spec, extra_args=(), keep_episode_logs=True)
    write_cached_row(
        path,
        {
            "ok": True,
            "scenario": "cologne1",
            "algorithm": "IDQN",
            "seed": None,
            "test_mean_timeLoss": 10.0,
            "test_mean_duration": 20.0,
            "test_mean_waitingTime": 3.0,
            "test_mean_queue_lengths": 4.0,
            "test_mean_max_queues": 5.0,
            "elapsed_sec": 6.0,
        },
    )

    result = run_experiment(
        repo_root=tmp_path / "repo",
        log_dir=tmp_path / "logs",
        scenarios=("cologne1", "cologne8"),
        algorithms=("IDQN", "MPLight"),
        episodes=1,
        testing=1,
        seeds=(),
        timeout_sec=1,
        extra_args=(),
        keep_episode_logs=True,
        cache_dir=tmp_path / "cache",
        resume=True,
        skip_runs=(("MPLight", "cologne8"), ("MPLight", "cologne1"), ("IDQN", "cologne8")),
        workers=1,
    )

    assert len(result["runs"]) == 1
    assert result["runs"][0]["algorithm"] == "IDQN"
    assert result["runs"][0]["scenario"] == "cologne1"
    assert result["setting"]["skipped_runs"] == [
        {"algorithm": "MPLight", "scenario": "cologne8"},
        {"algorithm": "MPLight", "scenario": "cologne1"},
        {"algorithm": "IDQN", "scenario": "cologne8"},
    ]
