import json
import shutil

import numpy as np
import pandas as pd
import pytest

from cf_h2o.eval.sumo_apc_avl_sumo_generation import build_city_sumo
from cf_h2o.eval import sumo_policy_rollout_validation as rollout
from cf_h2o.eval.sumo_policy_rollout_validation import (
    _advance_obs_from_prediction,
    _event_obs,
    _load_libsumo,
    _run_city_policy,
    _stop_index,
    _target_headways,
)


def _write_line(line_dir) -> None:
    line_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "stop_name": ["S1", "S2", "S3", "S4"],
            "latitude": [42.0, 42.001, 42.002, 42.003],
            "longitude": [-71.0, -71.001, -71.002, -71.003],
        }
    ).to_excel(line_dir / "stop_news.xlsx", index=False)
    pd.DataFrame(
        {
            "start_stop": ["S1", "S2", "S3"],
            "end_stop": ["S2", "S3", "S4"],
            "distance": [100.0, 120.0, 140.0],
            "V_max": [8.0, 8.0, 8.0],
            **{f"{hour:02d}:00:00": [8.0, 8.0, 8.0] for hour in range(24)},
        }
    ).to_excel(line_dir / "route_news.xlsx", index=False)
    pd.DataFrame({"launch_time": [0, 120, 240], "direction": [1, 1, 1]}).to_excel(
        line_dir / "time_table.xlsx",
        index=False,
    )


def test_stop_index_parses_generated_bus_stop_id():
    assert _stop_index("bs_city_line_0012") == 12


def test_target_headways_from_line_map():
    assert _target_headways([{"line_uid": "L", "departures": [0, 120, 300]}])["L"] == 150.0


def test_sumo_prediction_uses_absolute_clock_and_current_route_segment(tmp_path):
    segment_map = tmp_path / "segments.jsonl"
    segment_map.write_text("".join(json.dumps(row) + "\n" for row in [
        {"line_uid": "L", "segment_idx": 2, "length_m": 2080, "speed_mps": 8},
        {"line_uid": "L", "segment_idx": 0, "length_m": 240, "speed_mps": 8},
        {"line_uid": "L", "segment_idx": 1, "length_m": 616, "speed_mps": 8},
    ]))
    times = rollout._segment_times_by_line({"segment_map_path": str(segment_map)})["L"]
    obs = _event_obs(line_uid="L", stop_id="bs_L_0001", now=6 * 3600 + 120,
                     target_hw=600, forward_hw=570, backward_hw=630,
                     station_count=3, speed_mps=8, base_dwell_sec=8)
    pred, inputs = rollout._predict_next_y_batch_from_model(
        obs, policy="uncalibrated", model={}, actions=[0, 30],
        line_uid="L", station_count=3, segment_times=times,
    )

    assert inputs[:, 3] == pytest.approx([6 / 23, 6 / 23])
    assert inputs[:, 10] == pytest.approx([21720, 21720])
    assert inputs[:, 2] == pytest.approx([0.5, 0.5])
    # The source baseline has +0.030 seconds of headway per segment travel second.
    assert pred[:, 0] == pytest.approx([599.61, 608.01])


def test_mpc_station_advance_includes_current_dwell_hold_and_travel():
    obs = _event_obs(line_uid="L", stop_id="bs_L_0000", now=6 * 3600 + 3580,
                     target_hw=600, forward_hw=570, backward_hw=630,
                     station_count=3, speed_mps=8, base_dwell_sec=12)
    pred = np.array([580, 620, 12, 99, -40, 580, 620, 7, 0])
    following = _advance_obs_from_prediction(obs, pred, 15, station_count=3, travel_time=70)

    assert following[10] == pytest.approx(7 * 3600 + 77)
    assert following[3] == pytest.approx(7 / 23)
    assert following[2] == 1
    assert following[9] == 99  # Predicted next dwell must not replace this stop's twelve seconds.
    assert obs[10] == 6 * 3600 + 3580


@pytest.mark.parametrize("policy", ["h2oplus_dense_policy", "h2oplus_dense_mpc_policy"])
def test_model_policy_forwards_route_times_and_clock_across_mpc_steps(monkeypatch, policy):
    original = rollout._uncalibrated_transition
    seen = []

    def capture(fwd, bwd, waiting, target, speed, travel_time, action, hour):
        seen.append((float(travel_time[0]), hour))
        return original(fwd, bwd, waiting, target, speed, travel_time, action, hour)

    monkeypatch.setattr(rollout, "_uncalibrated_transition", capture)
    obs = _event_obs(line_uid="L", stop_id="bs_L_0000", now=6 * 3600 + 3590,
                     target_hw=600, forward_hw=570, backward_hw=630,
                     station_count=3, speed_mps=8, base_dwell_sec=8)
    model = {"h2o_beta": np.zeros((rollout.h2o_features(np.zeros((1, 16))).shape[1], 9))}
    action = rollout._choose_extra_hold(
        policy=policy, obs=obs, model=model, actions=[0], line_uid="L", station_count=3,
        segment_times=[80, 600, 35], mpc_horizon=2, mpc_discount=0.85, mpc_action_cost=0.05,
    )

    assert action == 0
    assert seen == ([(80, 6)] if policy == "h2oplus_dense_policy" else [(80, 6), (600, 7)])


def test_mpc_last_stop_has_no_further_holding_decisions(monkeypatch):
    original = rollout._predict_next_y_batch_from_model
    prediction_calls = []

    def capture(obs, **kwargs):
        prediction_calls.append(int(obs[2]))
        return original(obs, **kwargs)

    monkeypatch.setattr(rollout, "_predict_next_y_batch_from_model", capture)
    obs = _event_obs(line_uid="L", stop_id="bs_L_0002", now=22000,
                     target_hw=600, forward_hw=570, backward_hw=630,
                     station_count=3, speed_mps=8, base_dwell_sec=8)
    model = {"h2o_beta": np.zeros((rollout.h2o_features(np.zeros((1, 16))).shape[1], 9))}
    actions = [rollout._predict_mpc_action_from_obs(
        obs, "h2oplus_dense_mpc_policy", model, [0, 30, 60],
        line_uid="L", station_count=3, segment_times=[80, 600, 35],
        horizon=horizon, discount=0.85, action_cost=0.05,
    ) for horizon in [1, 3]]

    assert actions[0] == actions[1]
    assert prediction_calls == [2, 2]  # Exactly one action evaluation per horizon.


def test_libsumo_policy_rollout_changes_dwell(tmp_path):
    if not shutil.which("netconvert") or not shutil.which("sumo"):
        pytest.skip("SUMO binaries are not installed")
    sumo_api = _load_libsumo()
    assert sumo_api.__name__ == "libsumo"
    line_dir = tmp_path / "env" / "data" / "L1"
    _write_line(line_dir)
    manifest = [
        {
            "city_key": "toy",
            "city": "Toy",
            "line_key": "L1",
            "line_dir": str(line_dir),
            "ok": True,
        }
    ]
    city = build_city_sumo(
        city_key="toy",
        line_manifest=manifest,
        out_dir=tmp_path / "sumo",
        max_lines=0,
        max_departures_per_line=3,
        duration_sec=600.0,
        include_bus_stops=True,
        base_dwell_sec=8.0,
    )

    no_hold = _run_city_policy(
        sumo_api=sumo_api,
        city_key="toy",
        city=city,
        policy="no_hold",
        model=None,
        actions=[0.0, 30.0],
        base_dwell_sec=8.0,
        max_steps=1000,
        max_events=6,
        mpc_horizon=2,
        mpc_discount=0.85,
    )
    fixed = _run_city_policy(
        sumo_api=sumo_api,
        city_key="toy",
        city=city,
        policy="fixed_30",
        model=None,
        actions=[0.0, 30.0],
        base_dwell_sec=8.0,
        max_steps=1000,
        max_events=6,
        mpc_horizon=2,
        mpc_discount=0.85,
    )

    assert no_hold["ok"] is True
    assert fixed["ok"] is True
    assert no_hold["mean_hold_seconds"] == 0.0
    assert fixed["mean_hold_seconds"] == 30.0
    assert fixed["mean_dwell_seconds"] > no_hold["mean_dwell_seconds"]
    assert len(no_hold["line_rows"]) == 1
    assert no_hold["line_rows"][0]["events"] == no_hold["events"]
