from types import SimpleNamespace

from cf_h2o.eval.traffic_signal_resco_phase_benchmark import (
    _green_phase_candidates,
    _phase_green_count,
    _reload_sumo_state,
    _start_sumo,
)


def test_phase_green_count_includes_minor_green():
    assert _phase_green_count("GGggrrrs") == 4


def test_green_phase_candidates_skip_yellow_and_duplicates():
    phases = [
        SimpleNamespace(state="GGrr", duration=30),
        SimpleNamespace(state="yyrr", duration=3),
        SimpleNamespace(state="GGrr", duration=10),
        SimpleNamespace(state="rrGG", duration=30),
        SimpleNamespace(state="rrrr", duration=5),
    ]

    candidates = _green_phase_candidates(phases, controlled_link_count=4)

    assert [item.state for item in candidates] == ["GGrr", "rrGG"]
    assert [item.phase_index for item in candidates] == [0, 3]


def test_start_sumo_enables_matched_snapshot_protocol_and_clears_tripinfo(tmp_path):
    calls = []
    api = SimpleNamespace(
        close=lambda: None,
        start=lambda arguments: calls.append(arguments),
    )
    tripinfo = tmp_path / "tripinfo.xml"
    tripinfo.write_text("stale", encoding="utf-8")

    _start_sumo(api, tmp_path / "scenario.sumocfg", 17, tripinfo_output=tripinfo)

    arguments = calls[0]
    assert arguments[arguments.index("--save-state.rng") + 1] == "true"
    assert arguments[arguments.index("--save-state.precision") + 1] == "8"
    assert arguments[arguments.index("--thread-rngs") + 1] == "1"
    assert arguments[arguments.index("--time-to-teleport") + 1] == "-1"
    assert arguments[arguments.index("--collision.action") + 1] == "warn"
    assert arguments[arguments.index("--collision.check-junctions") + 1] == "true"
    assert (
        arguments[arguments.index("--tripinfo-output.write-unfinished") + 1]
        == "true"
    )
    assert not tripinfo.exists()


def test_full_state_reload_uses_network_configuration_and_saved_time(tmp_path):
    calls = []
    api = SimpleNamespace(load=lambda arguments: calls.append(arguments))
    state = tmp_path / "state.xml"

    _reload_sumo_state(
        api,
        tmp_path / "scenario.sumocfg",
        17,
        state,
        begin_time=63.5,
    )

    arguments = calls[0]
    assert arguments[0] == "-c"
    assert arguments[arguments.index("--load-state") + 1] == str(state)
    assert arguments[arguments.index("--begin") + 1] == "63.50000000"
    assert "sumo" not in arguments


def test_rng_free_state_reload_uses_new_future_seed(tmp_path):
    calls = []
    api = SimpleNamespace(load=lambda arguments: calls.append(arguments))

    _reload_sumo_state(
        api,
        tmp_path / "scenario.sumocfg",
        9102,
        tmp_path / "state_without_rng.xml",
        begin_time=300.0,
        save_state_rng=False,
    )

    arguments = calls[0]
    assert arguments[arguments.index("--seed") + 1] == "9102"
    assert arguments[arguments.index("--save-state.rng") + 1] == "false"
