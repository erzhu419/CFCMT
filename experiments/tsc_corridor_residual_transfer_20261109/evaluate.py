"""Score frozen residual forecasts and requests on the existing PP development panel."""
from collections import Counter
from itertools import combinations
from pathlib import Path
from statistics import mean
import json
import sys

import numpy as np

ORIGINAL = Path(__file__).resolve().parents[1] / "tsc_corridor_arrival_transfer_20261107"
sys.path.insert(0, str(ORIGINAL))
from model_gate import release_id, rows  # noqa: E402

ARMS = ("T", "S", "P", "history", "analytic")
SCOPES = ("original_common_scored", "all_observable_risk")
PAIRS = tuple(combinations(("S", "T", "P", "history", "analytic", "GP"), 2))


def choose_requests(row, forecasts):
    """Use the original score and tie order; forecast columns already include tail delay."""
    gp = row["gp"]
    if not forecasts:
        return {name: gp for name in (*ARMS, "GP")}
    scores = np.broadcast_to(row["base"], (len(ARMS), len(row["base"]))).astype(float).copy()
    for edge, masses in forecasts.items():
        for request, mask in enumerate(row["protected"][edge]):
            for bit in range(10):
                if int(mask) & (1 << bit):
                    scores[:, request] += masses[:, bit]
    chosen = np.argmax(np.round(scores, 12), axis=1)
    return {**{name: int(index) for name, index in zip(ARMS, chosen)}, "GP": gp}


def _scope():
    return dict(counts=Counter(), error={name: 0.0 for name in ARMS},
                action_differences={f"{a}_vs_{b}": 0 for a, b in PAIRS},
                action_differences_unlocked={f"{a}_vs_{b}": 0 for a, b in PAIRS},
                fallback=Counter())


def _finalize(scope, decisions, unlocked):
    denominator = scope["counts"]["scored_edge_epochs"]
    return dict(counts=dict(scope["counts"]),
                mae={name: loss / denominator if denominator else None for name, loss in scope["error"].items()},
                fallback=dict(scope["fallback"]), action_differences=scope["action_differences"],
                action_differences_unlocked=scope["action_differences_unlocked"],
                action_difference_rates={key: value / decisions for key, value in scope["action_differences"].items()},
                action_difference_unlocked_rates={key: value / unlocked if unlocked else None
                                                 for key, value in scope["action_differences_unlocked"].items()})


def evaluate_hour(city, seed, block, directory, predictor, old_policy, disagreement_stream=None):
    """predictor is new/reset for this hour; forecast(row) is called once on every row.

    It returns edge -> (T,S,P,history,analytic) x 11. The first ten columns are
    cumulative stop-line credits at seconds 1..10 after the per-flight continuous
    detector-to-line delay. The final column is detector arrival by t+10.
    """
    records = list(rows(directory / "labels.jsonl.gz"))
    outcomes = {release_id(record): record for record in records}
    if len(records) != len(outcomes) or any(r["status"] == "unknown" for r in records):
        raise ValueError(f"Unresolved or duplicated development labels: {city}/{seed}")
    horizon = json.loads((directory / "result.json").read_text())["horizon_sec"]
    counts, timestamps = Counter(), {}
    scopes = {name: _scope() for name in SCOPES}
    for row in rows(directory / "decisions.jsonl.gz"):
        t, gp = row["t"], row["gp"]
        if not 0 <= gp < len(row["states"]) or not 0 <= row["pp"] < len(row["states"]):
            raise ValueError("A stored request is outside its feasible ordering")
        timestamps.setdefault(row["tls"], []).append(t)
        unlocked = len(row["states"]) > 1
        counts["decisions"] += 1
        counts["unlocked_decisions"] += unlocked
        counts["locked_decisions"] += not unlocked
        counts["inflight_decisions"] += bool(row["flights"])
        counts["spillback_decisions"] += row["spillback"]
        counts["unmatched_decisions"] += row["unmatched"]
        # Update causal position history even when either scoring mask excludes this row.
        forecasts = predictor.forecast(row)
        old_forecasts = old_policy.forecast(row)
        observable = bool(row["flights"] and not row["spillback"] and not row["unmatched"] and t + 10 <= horizon)
        common = bool(old_forecasts) and observable
        enabled = dict(original_common_scored=common, all_observable_risk=observable)
        if observable and set(forecasts) != {flight["edge"] for flight in row["flights"]}:
            raise ValueError("Residual forecasts changed the observable edge denominator")
        chosen = {scope: choose_requests(row, forecasts if enabled[scope] else None) for scope in SCOPES}
        # Only now read this row's future arrival labels, after forecasts and requests are fixed.
        arrived = Counter()
        if observable:
            for flight in row["flights"]:
                outcome = outcomes[release_id(flight)]
                if outcome["end_s"] <= t or t - outcome["entry_s"] != flight["age_s"]:
                    raise ValueError("Future label does not match its current risk-set release")
                if outcome["status"] == "censored" and outcome["end_s"] < t + 10:
                    raise ValueError("Incomplete next-ten-second label window")
                if outcome["status"] == "straight" and outcome["end_s"] <= t + 10:
                    arrived[flight["edge"]] += 1
        for name, scope in scopes.items():
            actions = chosen[name]
            if enabled[name]:
                scope["counts"]["scored_decisions"] += 1
                scope["counts"]["scored_unlocked_decisions"] += unlocked
                scope["counts"]["scored_flights"] += len(row["flights"])
                scope["counts"]["scored_observed_arrivals"] += sum(arrived.values())
                for edge, masses in forecasts.items():
                    scope["counts"]["scored_edge_epochs"] += 1
                    for index, method in enumerate(ARMS):
                        scope["error"][method] += abs(float(masses[index, 10]) - arrived[edge])
            else:
                reason = ("spillback" if row["spillback"] else "unmatched" if row["unmatched"]
                          else "no_eligible_flight" if not row["flights"] else "incomplete_window" if t + 10 > horizon
                          else "outside_original_common_support")
                scope["fallback"][reason] += 1
            for a, b in PAIRS:
                key = f"{a}_vs_{b}"
                different = actions[a] != actions[b]
                scope["action_differences"][key] += different
                scope["action_differences_unlocked"][key] += different and unlocked
            different_learning = len({actions[arm] for arm in ("T", "S", "P")}) > 1
            scope["counts"]["learning_disagreement_epochs"] += different_learning
            scope["counts"]["eligible_learning_disagreement_epochs"] += different_learning and 60 <= t <= 3150
            if different_learning and disagreement_stream is not None:
                disagreement_stream.write(json.dumps(dict(city=city, seed=seed, block=block, scope=name,
                    t=t, tls=row["tls"], states=row["states"], pp=row["pp"], actions=actions), separators=(",", ":")) + "\n")
    expected_tls = 16 if city == "hangzhou" else 12
    if (horizon != 3600 or len(timestamps) != expected_tls
            or any(times != list(range(0, 3600, 10)) for times in timestamps.values())):
        raise ValueError(f"Incomplete frozen development hour: {city}/{seed}")
    return dict(city=city, seed=seed, block=block, counts=dict(counts),
                scopes={name: _finalize(scope, counts["decisions"], counts["unlocked_decisions"])
                        for name, scope in scopes.items()})


def summarize(panel):
    """Equal complete seeds, retain three fixed adaptation blocks and both scoring masks."""
    if len(panel) != 6 or any(sum(r["block"] == b for r in panel) != 2 for b in range(3)):
        raise ValueError("The city panel must contain exactly two development seeds per fixed block")
    result = dict(by_seed=panel, scopes={})
    for scope in SCOPES:
        losses = {name: mean(r["scopes"][scope]["mae"][name] for r in panel) for name in ARMS}
        by_block = [dict(block=b, mae={name: mean(r["scopes"][scope]["mae"][name]
                                                for r in panel if r["block"] == b) for name in ARMS}) for b in range(3)]
        differences = {f"{a}_vs_{b}": sum(r["scopes"][scope]["action_differences"][f"{a}_vs_{b}"] for r in panel)
                       for a, b in PAIRS}
        differences_unlocked = {f"{a}_vs_{b}": sum(r["scopes"][scope]["action_differences_unlocked"][f"{a}_vs_{b}"] for r in panel)
                                for a, b in PAIRS}
        disagreement_seeds = {key: [r["seed"] for r in panel if r["scopes"][scope]["action_differences"][key]]
                              for key in ("S_vs_T", "S_vs_P")}
        total_decisions = sum(r["counts"]["decisions"] for r in panel)
        total_unlocked = sum(r["counts"]["unlocked_decisions"] for r in panel)
        forecast_passed = all(losses["S"] < losses[other] for other in ("T", "P", "history"))
        action_passed = all(len(seeds) >= 2 for seeds in disagreement_seeds.values())
        result["scopes"][scope] = dict(equal_seed_mae=losses, by_block=by_block,
            action_differences=differences, action_differences_unlocked=differences_unlocked,
            decision_denominator=total_decisions, unlocked_decision_denominator=total_unlocked,
            action_difference_rates={key: count / total_decisions for key, count in differences.items()},
            action_difference_unlocked_rates={key: count / total_unlocked if total_unlocked else None
                                             for key, count in differences_unlocked.items()},
            disagreement_seeds=disagreement_seeds, better_forecast=forecast_passed,
            distinct_requests=action_passed, passed=forecast_passed and action_passed)
    return result
