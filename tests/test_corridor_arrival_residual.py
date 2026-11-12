"""Reachable label, temporal-history and repeated-observation cases."""
import gzip
import json

import numpy as np
import pytest

from experiments.tsc_corridor_arrival_residual_20261109 import diagnose as d


def flight(*, entry=0, eta=8, straight=True, vid="car", age=20):
    return dict(edge="road", vid=vid, entry_s=entry, age_s=age,
                queued_at_entry=False, freeflow_s=50, tail_s=3,
                on_straight_lane=straight, remaining_m=eta * 10,
                speed_limit_mps=10, analytic_eta_s=eta)


def row(t=20, flights=None):
    return dict(t=t, tls="signal", states=["G", "r"], pp=0, gp=0, base=[0, 0],
                protected={"road": [1023, 0]}, spillback=False, unmatched=False,
                flights=flights if flights is not None else [flight(age=t)])


def outcome(f, *, end=30, status="straight"):
    return dict(edge=f["edge"], vid=f["vid"], entry_s=f["entry_s"], end_s=end, status=status)


@pytest.mark.parametrize("straight,eta,status,end,category,reason", [
    (True, 8, "straight", 35, "false_positive", "timing_fp"),
    (True, 12, "straight", 28, "false_negative", "timing_fn"),
    (False, 8, "straight", 28, "false_negative", "lanechange_fn"),
    (True, 8, "other", 28, "false_positive", "competing_fp"),
    (True, 8, "terminal", 28, "false_positive", "competing_fp"),
    (True, 8, "censored", 100, "false_positive", "censored_fp"),
])
def test_decomposes_timing_lanechange_competing_without_dropping_censoring(straight, eta, status, end, category, reason):
    f = flight(straight=straight, eta=eta)
    item = d.observation(row(), f, outcome(f, end=end, status=status), {}, ("city", "dev", 1))
    assert (item["category"], item["reason"]) == (category, reason)
    assert item["residual"] == (end - 20 - eta if straight and status == "straight" else None)


def test_history_conditions_use_previous_release_not_future_or_different_release():
    f = flight(eta=8)
    previous = flight(eta=16)
    cell = d.conditions(row(), f, {d.release_id(f): previous})
    assert cell["past_10s_speed_ratio"] == "(0.5,0.9]"
    different_release = flight(entry=-100, eta=16)
    assert d.conditions(row(), f, {d.release_id(different_release): different_release})["past_10s_speed_ratio"] == "missing"


def test_nearest_observation_is_chosen_by_current_eta_without_final_status_filter():
    panel = d.Panel()
    for t, eta in ((20, 18), (30, 10), (40, 3)):
        f = flight(eta=eta, age=t)
        panel.add(d.observation(row(t), f, outcome(f, end=100, status="censored"), {}, ("city", "dev", 1)))
    result = panel.result()
    assert result["snapshot_weighted"]["overall"]["repeated_snapshots"] == 2
    nearest = result["nearest_eta10_current_straight"]["overall"]
    assert nearest["snapshots"] == nearest["final_censored"] == 1
    assert nearest["residual_n"] == 0
    assert panel.nearest[("city", "dev", 1, "road", "car", 0)]["t"] == 30


def write_gzip(path, records):
    with gzip.open(path, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def test_common_mae_denominator_and_history_include_prior_excluded_rows(tmp_path):
    # t=10 is excluded by the old common-support filter, but its past position
    # must still be available for the independent history baseline at t=20.
    first = row(10, [flight(eta=16, age=10)])
    second = row(20, [flight(eta=8, age=20)])
    third = row(30, [flight(eta=2, age=30)])
    third["spillback"] = True
    write_gzip(tmp_path / "decisions.jsonl.gz", [first, second, third])
    write_gzip(tmp_path / "labels.jsonl.gz", [outcome(first["flights"][0], end=31)])
    (tmp_path / "result.json").write_text(json.dumps(dict(horizon_sec=3600)))

    class Policy:
        def forecast(self, query):
            if query["t"] != 20:
                return None
            prediction = np.zeros((4, 11))
            prediction[3, 10] = 1
            return {"road": prediction}

    record, _ = d.analyze_hour(tmp_path, city="jinan", seed=1, policy=Policy())
    assert record["counts"]["all_decisions"] == 3
    assert record["counts"]["observable_decisions"] == 2
    assert record["counts"]["original_common_decisions"] == 1
    assert record["counts"]["observable_noncommon_decisions"] == 1
    common = record["scopes"]["original_common_scored"]
    assert common["edge_epochs"] == 1
    assert common["analytic_count_mae"] == 1
    # Past speed is 8m/s: 80m takes 10s and is still a prediction at t+10.
    assert common["history_count_mae"] == 1
    assert common["snapshot_weighted"]["past_10s_speed_ratio:(0.5,0.9]"]["snapshots"] == 1
    assert record["scopes"]["all_observable_risk"]["edge_epochs"] == 2
