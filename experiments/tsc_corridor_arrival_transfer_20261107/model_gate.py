"""Fit frozen corridor arrival distributions and score held-out development hours."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import random
from statistics import mean

from arrival_policy import ArrivalPolicy, PREDICTORS, fit_hours, select


CITIES = ("hangzhou", "jinan")
BRANCH_SAMPLE_SEED = 11072027
PAIRS = (("S", "T"), ("S", "P"), ("S", "GP"), ("T", "GP"), ("P", "GP"), ("S", "analytic"))


def read(path):
    return json.loads(path.read_text())


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def rows(path):
    with gzip.open(path, "rt") as stream:
        for line in stream:
            yield json.loads(line)


def hour(root, part, city, seed):
    directory = root / part / city / str(seed)
    receipt = read(directory / "result.json")
    if receipt["status"] != "PASS_HOUR" or receipt["city"] != city or receipt["seed"] != seed:
        raise ValueError(f"Invalid {part} input: {directory}")
    return directory


def load_training(root, part, city, seeds):
    return {s: list(rows(hour(root, part, city, s) / "labels.jsonl.gz")) for s in seeds}


def release_id(record):
    return record["edge"], record["vid"], record["entry_s"]


def verify_protocol(cities):
    all_seeds = []
    for city in CITIES:
        spec = cities[city]
        if (len(spec["donor"]) != 8 or len(spec["adapt"]) != 3
                or any(len(block) != 2 for block in spec["adapt"]) or len(spec["dev"]) != 6):
            raise ValueError(f"Unexpected fit/development budgets in {city}")
        all_seeds.extend([*spec["pilot"], *spec["donor"], *sum(spec["adapt"], []), *spec["dev"]])
    if len(all_seeds) != len(set(all_seeds)):
        raise ValueError("Frozen pilot/donor/adaptation/development seeds must be disjoint")


def evaluate(city, seed, block, directory, policy, disagreement_stream):
    records = list(rows(directory / "labels.jsonl.gz"))
    outcomes = {release_id(r): r for r in records}
    if len(records) != len(outcomes) or any(r["status"] == "unknown" for r in records):
        raise ValueError(f"Incomplete or duplicated learning targets in {city} seed {seed}")
    error = {name: 0.0 for name in PREDICTORS}
    differences = {f"{a}_vs_{b}": 0 for a, b in PAIRS}
    counts, fallback = Counter(), Counter()
    timestamps = {}
    branch_pool = []
    target = policy.curves[0]
    independent_target = ArrivalPolicy(target, target, target)
    for row in rows(directory / "decisions.jsonl.gz"):
        t, tls, gp = row["t"], row["tls"], row["gp"]
        states = row["states"]
        if not 0 <= gp < len(states) or not 0 <= row["pp"] < len(states):
            raise ValueError("A stored rule request is outside its feasible state list")
        if tls not in timestamps:
            timestamps[tls] = []
        timestamps[tls].append(t)
        counts["decisions"] += 1
        counts["locked"] += len(states) == 1
        counts["inflight"] += bool(row["flights"])
        counts["spillback"] += row["spillback"]
        counts["unmatched"] += row["unmatched"]
        if t >= 60 and t <= 3150:
            counts["branch_population"] += 1

        forecasts = policy.forecast(row)
        chosen = select(row["base"], row["protected"], forecasts, gp)
        chosen["GP"] = gp
        if not forecasts:
            reason = ("spillback" if row["spillback"] else "unmatched" if row["unmatched"]
                      else "no_eligible_flight" if not row["flights"] else "insufficient_common_age_support")
            fallback[reason] += 1
            if reason == "insufficient_common_age_support":
                local = independent_target.forecast(row)
                if local:
                    counts["local_support_only"] += 1
                    local_choice = select(row["base"], row["protected"], local, gp)["T"]
                    counts["source_support_changes_T_action"] += local_choice != gp
        else:
            # Labels are first opened here for this epoch, after current predictions and actions are fixed.
            arrived = Counter()
            for flight in row["flights"]:
                outcome = outcomes[release_id(flight)]
                if outcome["end_s"] <= t or t - outcome["entry_s"] != flight["age_s"]:
                    raise ValueError("Future outcome identity does not match the prerequest risk set")
                if outcome["status"] == "censored" and outcome["end_s"] < t+10:
                    raise ValueError("Held-out forecast window is incomplete")
                if outcome["status"] == "straight" and outcome["end_s"] <= t+10:
                    arrived[flight["edge"]] += 1
            counts["scored_decisions"] += 1
            counts["scored_flights"] += len(row["flights"])
            counts["scored_observed_arrivals"] += sum(arrived.values())
            for edge, mass in forecasts.items():
                counts["scored_edge_epochs"] += 1
                for i, name in enumerate(PREDICTORS):
                    error[name] += abs(float(mass[i, 10]) - arrived[edge])

        for a, b in PAIRS:
            differences[f"{a}_vs_{b}"] += chosen[a] != chosen[b]
        actions = {arm: chosen[arm] for arm in ("S", "T", "P", "GP")}
        if len(set(actions.values())) > 1:
            record = dict(city=city, seed=seed, block=block, t=t, tls=tls, states=states,
                          pp=row["pp"], gp=gp, actions=actions)
            disagreement_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            if 60 <= t <= 3150:
                branch_pool.append(record)

    expected_tls = 16 if city == "hangzhou" else 12
    if len(timestamps) != expected_tls or any(ts != list(range(0, 3600, 10)) for ts in timestamps.values()):
        raise ValueError("Expected every junction and decision in the held-out PP hour")
    denominator = counts["scored_edge_epochs"]
    mae = {name: value / denominator if denominator else None for name, value in error.items()}
    record = dict(city=city, seed=seed, block=block, counts=dict(counts), fallback=dict(fallback),
                  mae=mae, action_differences=differences, eligible_disagreements=len(branch_pool))
    return record, branch_pool


def summarize(panel):
    valid = all(p["mae"]["S"] is not None for p in panel)
    loss = {name: mean(p["mae"][name] for p in panel) if valid else None for name in PREDICTORS}
    seeds = {pair: [r["seed"] for r in panel if r["action_differences"][pair] > 0]
             for pair in ("S_vs_T", "S_vs_P")}
    forecast = valid and loss["S"] < loss["T"] and loss["S"] < loss["analytic"]
    actions = all(len(values) >= 2 for values in seeds.values())
    blocks = [dict(block=b, mae={name: mean(r["mae"][name] for r in panel if r["block"] == b)
                                 if valid else None for name in PREDICTORS}) for b in range(3)]
    return dict(by_seed=panel, by_block=blocks, equal_seed_mae=loss, disagreement_seeds=seeds,
                better_forecast=forecast, distinct_requests=actions, passed=forecast and actions)


def development(root, protocol, output):
    cities = protocol["cities"]
    verify_protocol(cities)
    output.mkdir(exist_ok=False, parents=True)
    summary, possible = {}, {}
    with gzip.open(output / "disagreements_server_only.jsonl.gz", "wt") as stream:
        for city, donor_city in (("hangzhou", "jinan"), ("jinan", "hangzhou")):
            donor = load_training(root, "donor", donor_city, cities[donor_city]["donor"])
            source, placebo = fit_hours(donor), fit_hours(donor, permute=True)
            panel = []
            for b, block in enumerate(cities[city]["adapt"]):
                target = fit_hours(load_training(root, "adapt", city, block))
                policy = ArrivalPolicy(target, source, placebo)
                for seed in cities[city]["dev"][2*b:2*b+2]:
                    result, locus = evaluate(city, seed, b, hour(root, "dev", city, seed), policy, stream)
                    panel.append(result)
                    possible[city, seed] = locus
            summary[city] = summarize(panel)
    passed = all(s["passed"] for s in summary.values())
    receipt = dict(status="PASS_DEVELOPMENT" if passed else "HOLD_DEVELOPMENT", cities=summary,
                   prediction_unit="MAE for active edge-by-epoch, then equally weighted held-out seeds",
                   branch_roster_frozen=passed)
    if passed:
        roster = []
        rng = random.Random(BRANCH_SAMPLE_SEED)
        for (city, seed), locus in sorted(possible.items()):
            selected = rng.sample(sorted(locus, key=lambda r: (r["t"], r["tls"])), k=min(4, len(locus)))
            n = next(r["counts"]["branch_population"] for r in summary[city]["by_seed"] if r["seed"] == seed)
            roster.append(dict(city=city, seed=seed, total_decisions=n, different_decisions=len(locus),
                               inclusion_probability=len(selected)/len(locus) if locus else None,
                               cases=sorted(selected, key=lambda r: (r["t"], r["tls"]))))
        save(output / "branch_roster.json", roster)
    save(output / "result.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("root", "protocol", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    args = parser.parse_args()
    print(development(args.root, read(args.protocol), args.output)["status"])
