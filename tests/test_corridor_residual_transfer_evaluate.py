"""Risk-mask, causal-history and fixed-score checks for the existing-hour evaluator."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from experiments.tsc_corridor_arrival_residual_20261109 import history_baseline as historical
from experiments.tsc_corridor_residual_transfer_20261109 import evaluate as e


def test_request_score_uses_shifted_line_columns_and_preserves_first_feasible_tie():
    query = dict(base=[0, 0], protected={"road": [2, 256]}, gp=1)
    mass = np.zeros((5, 11))
    mass[e.ARMS.index("S"), 8:10] = 1
    # A large detector-only prediction must never leak into stop-line credit.
    mass[e.ARMS.index("T"), 10] = 10
    mass[e.ARMS.index("P"), :10] = 1
    chosen = e.choose_requests(query, {"road": mass})
    assert chosen["S"] == 1
    assert chosen["T"] == chosen["P"] == 0
    assert set(e.choose_requests(query, None).values()) == {1}


def flight(t, remaining):
    return dict(edge="road", vid="car", entry_s=0, age_s=t, on_straight_lane=True,
                remaining_m=remaining, speed_limit_mps=10, analytic_eta_s=remaining / 10,
                queued_at_entry=False, freeflow_s=50, tail_s=3)


def write_rows(path, records):
    with gzip.open(path, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


class Predictor:
    def __init__(self):
        self.history = {}
        self.calls = 0

    def forecast(self, row):
        self.calls += 1
        baseline = historical.causal_predictions(row, self.history)
        forecasts = {}
        for edge in baseline["analytic"]:
            mass = np.zeros((5, 11))
            mass[e.ARMS.index("S"), :10] = 1
            mass[e.ARMS.index("S"), 10] = 1
            mass[e.ARMS.index("history"), 10] = baseline["history"][edge][-1]
            mass[e.ARMS.index("analytic"), 10] = baseline["analytic"][edge][-1]
            forecasts[edge] = mass
        return forecasts


class OldPolicy:
    def forecast(self, row):
        return {"road": np.zeros((4, 11))} if row["tls"] == "tls0" and row["t"] == 20 else None


def make_hour(tmp_path, end):
    queries = []
    for t in range(0, 3600, 10):
        for tls in range(12):
            active = tls == 0 and t in (10, 20)
            queries.append(dict(t=t, tls=f"tls{tls}", states=["r", "G"] if active else ["r"],
                pp=0, gp=0, base=[0, 0] if active else [0], protected={"road": [0, 1023] if active else [0]},
                flights=[flight(t, 160 if t == 10 else 100)] if active else [], spillback=False, unmatched=False))
    # These competing/censored releases remain current observations and produce no straight arrival.
    query20 = next(q for q in queries if q["tls"] == "tls0" and q["t"] == 20)
    for vid in ("other", "censored"):
        competitor = flight(20, 100)
        competitor.update(vid=vid, on_straight_lane=False)
        query20["flights"].append(competitor)
    write_rows(tmp_path / "decisions.jsonl.gz", queries)
    labels = [dict(edge="road", vid="car", entry_s=0, end_s=end, status="straight"),
              dict(edge="road", vid="other", entry_s=0, end_s=27, status="other"),
              dict(edge="road", vid="censored", entry_s=0, end_s=3600, status="censored")]
    write_rows(tmp_path / "labels.jsonl.gz", labels)
    (tmp_path / "result.json").write_text(json.dumps(dict(horizon_sec=3600)))


@pytest.mark.parametrize("end,actual", [(30, 1), (31, 0)])
def test_common_mask_keeps_causal_history_and_full_action_denominators(tmp_path, end, actual):
    make_hour(tmp_path, end)
    predictor = Predictor()
    record = e.evaluate_hour("jinan", 1, 0, tmp_path, predictor, OldPolicy())
    assert predictor.calls == record["counts"]["decisions"] == 4320
    assert record["counts"]["unlocked_decisions"] == 2
    common = record["scopes"]["original_common_scored"]
    observable = record["scopes"]["all_observable_risk"]
    assert common["counts"]["scored_edge_epochs"] == 1
    assert observable["counts"]["scored_edge_epochs"] == 2
    assert common["counts"]["scored_flights"] == 3
    assert common["counts"]["scored_observed_arrivals"] == actual
    # The excluded t=10 position gives 6m/s, so the history ETA at t=20 is >10.
    assert common["mae"]["history"] == actual
    assert common["mae"]["analytic"] == 1 - actual
    assert common["action_differences"]["S_vs_T"] == common["action_differences_unlocked"]["S_vs_T"] == 1
    assert observable["action_differences"]["S_vs_T"] == 2
    assert common["action_difference_rates"]["S_vs_T"] == 1 / 4320
    assert common["action_difference_unlocked_rates"]["S_vs_T"] == .5
    assert common["fallback"]["outside_original_common_support"] == 1


def test_summary_uses_equal_seeds_and_requires_both_source_disagreements():
    panel = []
    for index in range(6):
        differences = {f"{a}_vs_{b}": 0 for a, b in e.PAIRS}
        differences["S_vs_T"] = differences["S_vs_P"] = int(index < 2)
        scoped = dict(mae={name: value for name, value in zip(e.ARMS, (.2, .1, .3, .4, .5))},
                      action_differences=differences, action_differences_unlocked=deepcopy(differences))
        panel.append(dict(seed=index, block=index // 2, counts=dict(decisions=100, unlocked_decisions=20),
                          scopes={name: deepcopy(scoped) for name in e.SCOPES}))
    result = e.summarize(panel)["scopes"]["original_common_scored"]
    assert result["passed"]
    assert result["equal_seed_mae"]["S"] == .1
    assert result["disagreement_seeds"]["S_vs_P"] == [0, 1]
    assert result["decision_denominator"] == 600
    assert result["unlocked_decision_denominator"] == 120
    assert result["action_difference_rates"]["S_vs_T"] == 2 / 600
    panel[1]["scopes"]["original_common_scored"]["action_differences"]["S_vs_P"] = 0
    assert not e.summarize(panel)["scopes"]["original_common_scored"]["passed"]
