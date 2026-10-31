from cf_h2o.eval.traffic_signal_libsignal_baseline import (
    LibSignalRunSpec,
    aggregate_rows,
    cache_path_for_spec,
    ensure_sumo_assets,
    load_cached_row,
    parse_final_metrics,
    prepare_isolated_repo,
    write_cached_row,
)


def test_parse_final_metrics_uses_last_final_line():
    stdout = "\n".join(
        [
            "Final Travel Time is 39.4000, mean rewards: -7.4500, queue: 2.5000, delay: 1.5158, throughput: 5",
            "Final Travel Time is 19.0000, mean rewards: -0.7750, queue: 0.8333, delay: 0.1350, throughput: 7",
        ]
    )

    metrics = parse_final_metrics(stdout)

    assert metrics["travel_time"] == 19.0
    assert metrics["reward"] == -0.775
    assert metrics["queue"] == 0.8333
    assert metrics["throughput"] == 7.0


def test_aggregate_rows_ignores_failed_rows():
    rows = [
        {
            "ok": True,
            "agent": "dqn",
            "network": "sumo1x1",
            "travel_time": 10.0,
            "reward": -1.0,
            "queue": 2.0,
            "delay": 3.0,
            "throughput": 4.0,
            "elapsed_sec": 5.0,
        },
        {
            "ok": True,
            "agent": "dqn",
            "network": "sumo1x1",
            "travel_time": 20.0,
            "reward": -3.0,
            "queue": 4.0,
            "delay": 5.0,
            "throughput": 6.0,
            "elapsed_sec": 7.0,
        },
        {"ok": False, "agent": "frap", "network": "sumo1x1"},
    ]

    agg = aggregate_rows(rows)

    assert len(agg) == 1
    assert agg[0]["runs"] == 2
    assert agg[0]["travel_time"] == 15.0
    assert agg[0]["queue"] == 3.0


def test_successful_cache_row_is_reused_but_failed_row_is_not(tmp_path):
    spec = LibSignalRunSpec(agent="dqn", network="sumo1x1", episodes=5, steps=600, test_steps=600, seed=131)
    path = cache_path_for_spec(tmp_path, spec, interface="libsumo", extra_args=())

    write_cached_row(path, {"ok": False, "agent": "dqn"})
    assert load_cached_row(path) is None

    write_cached_row(path, {"ok": True, "agent": "dqn"})
    cached = load_cached_row(path)

    assert cached is not None
    assert cached["cached"] is True


def test_ensure_sumo_assets_copies_missing_grid4x4_assets(tmp_path):
    libsignal_repo = tmp_path / "libsignal" / "repo"
    resco_repo = tmp_path / "resco" / "repo"
    source_dir = resco_repo / "resco_benchmark" / "environments" / "grid4x4"
    source_dir.mkdir(parents=True)
    (source_dir / "grid4x4.net.xml").write_text("<net />", encoding="utf-8")
    (source_dir / "grid4x4.rou.xml").write_text("<routes />", encoding="utf-8")

    ensure_sumo_assets(libsignal_repo, "sumo4x4", resco_repo=resco_repo)

    target_dir = libsignal_repo / "data" / "raw_data" / "grid4x4"
    assert (target_dir / "grid4x4.net.xml").read_text(encoding="utf-8") == "<net />"
    assert (target_dir / "grid4x4.rou.xml").read_text(encoding="utf-8") == "<routes />"


def test_prepare_isolated_repo_copies_only_needed_sumo_raw_data(tmp_path):
    repo = tmp_path / "repo"
    (repo / "configs" / "sim").mkdir(parents=True)
    (repo / "configs" / "sim" / "sumo1x1.cfg").write_text("{}", encoding="utf-8")
    (repo / "run.py").write_text("print('ok')", encoding="utf-8")
    (repo / "agent").mkdir()
    (repo / "agent" / "__init__.py").write_text("", encoding="utf-8")
    raw = repo / "data" / "raw_data"
    (raw / "cologne1").mkdir(parents=True)
    (raw / "cologne1" / "cologne1.net.xml").write_text("<net />", encoding="utf-8")
    (raw / "unused_big_dir").mkdir()
    (raw / "unused_big_dir" / "unused.xml").write_text("<unused />", encoding="utf-8")

    spec = LibSignalRunSpec(agent="dqn", network="sumo1x1", episodes=1, steps=60, test_steps=60, seed=131)
    isolated = prepare_isolated_repo(
        repo_root=repo,
        spec=spec,
        work_root=tmp_path / "work",
        interface="libsumo",
        extra_args=(),
    )

    assert (isolated / "run.py").exists()
    assert (isolated / "configs" / "sim" / "sumo1x1.cfg").exists()
    assert (isolated / "data" / "raw_data" / "cologne1" / "cologne1.net.xml").exists()
    assert not (isolated / "data" / "raw_data" / "unused_big_dir").exists()
    assert (isolated / "data" / "output_data").exists()
