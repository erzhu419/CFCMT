"""Score the frozen Hangzhou and Jinan lookup predictions on fresh paired responses."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import permutations
import json
from pathlib import Path
from statistics import mean
import sys


EXPERIMENTS = Path(__file__).resolve().parents[1]
HANGZHOU = EXPERIMENTS / "tsc_hangzhou_matched_response_20261106"
SELECTION = EXPERIMENTS / "tsc_jinan_fresh_opportunity_20261106" / "selection.json"
sys.path.insert(0, str(HANGZHOU))
from conditional_readout import errors, labels, vector_mean
from summarize_response import IDENTITY, measured_pair, profile


def prediction_tables(plan):
    models = plan["source_models"]
    points = [tuple(p["point"]) for p in models["points"]]
    tables = {name: {tuple(p["point"]): tuple(Fraction(str(x)) for x in p[name]) for p in models["points"]}
              for name in ("hangzhou_matched", "jinan_target_only")}
    tables["hangzhou_constant"] = dict.fromkeys(points, tuple(Fraction(str(v)) for v in models["hangzhou_constant"]))
    return points, tables


def losses(points, tables, observations):
    """Average the two errors on each observation, the observations within point, then points."""
    vectors = {point: [(r["delta_right"], r["delta_straight"]) for r in rows]
               for point, rows in observations.items()}
    by_method = {name: vector_mean([errors(table[p], vectors[p]) for p in points]) for name, table in tables.items()}
    shuffled = [[errors(v, vectors[p]) for v in tables["hangzhou_matched"].values()] for p in points]
    by_method["permutation_mean"] = vector_mean([vector_mean(row) for row in shuffled])
    comparisons = {name: labels(tuple(m - b for m, b in zip(by_method["hangzhou_matched"], by_method[name])))
                   for name in ("hangzhou_constant", "jinan_target_only", "permutation_mean")}
    return dict(mae={name: labels(v) for name, v in by_method.items()}, matched_minus=comparisons)


def distribution(points, table, observations):
    # Pair the two movement outputs throughout the relabeling. Fractions preserve exact ties.
    costs = [[mean(errors(table[q], [(r["delta_right"], r["delta_straight"]) for r in observations[p]]))
              for q in points] for p in points]
    counts = Counter(sum(costs[i][j] for i, j in enumerate(perm)) / len(points)
                     for perm in permutations(range(len(points))))
    matched = mean(costs[i][i] for i in range(len(points)))
    below = sum(n for v, n in counts.items() if v < matched)
    tied = counts[matched]
    return dict(assignments=sum(counts.values()), lower_than_matched=below, equal_matched=tied,
                at_most_matched=below + tied, matched_rank_inclusive=[below + 1, below + tied],
                histogram=[dict(exact=str(v), joint_mae=float(v), assignments=n) for v, n in sorted(counts.items())])


def score(rows, selection, plan):
    points, tables = prediction_tables(plan)
    roster = {(c["seed"], c["time_sec"]): c for c in selection["cases"]}
    if (selection["status"] != "PASS_FRESH_TARGET_OPPORTUNITY" or selection["seeds"] != plan["seeds"]
            or list(map(tuple, selection["count_pairs"])) != points or not roster
            or len(roster) != len(selection["cases"])):
        raise ValueError("Frozen predictions and approved census roster do not agree")
    seen, cases = set(), []
    safety = Counter()
    for row in rows:
        key = row["seed"], row["time_sec"]
        if key in seen or key not in roster:
            raise ValueError(f"Unexpected or duplicate fresh case {key}")
        fixed = roster[key]
        if any(row[k] != fixed[k] for k in IDENTITY):
            raise ValueError(f"Response differs from the approved roster at {key}")
        seen.add(key)
        for arm in ("high", "low"):
            a = row[arm]
            if (a["arm"] != arm or a["candidate_index"] != fixed[arm + "_candidate"]
                    or (a["seed"], a["time_sec"]) != key or a["status"] != "COMPLETE"
                    or a["sample_count"] != 10 or a["unresolved_events"] or a["unobserved_first_receiving"]
                    or a["right_nominal"] != fixed["right_sequence"]
                    or a["straight_nominal"] != fixed[arm + "_straight_sequence"]):
                raise ValueError(f"Arm differs from the frozen checkpoint or measurement is incomplete: {key} {arm}")
            for phase in ("prefix_safety", "safety"):
                for event in ("raw_collision_events", "starting_teleports", "ending_teleports"):
                    safety[event] += a[phase][event]
        cases.append(dict(measured_pair(row), time_sec=row["time_sec"], tls_id=row["tls_id"]))
    result = dict(planned_pairs=len(roster), measured_pairs=len(cases),
                  missing_checkpoints=[dict(seed=s, time_sec=t) for s, t in sorted(roster.keys() - seen)],
                  safety=dict(safety), movement_response=profile(cases))
    if result["missing_checkpoints"] or any(safety.values()):
        return dict(status="INCOMPLETE_FRESH_RESPONSE_READOUT", **result)
    by_point, by_seed = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for case in cases:
        p, seed = tuple(case["near_counts"]), case["seed"]
        by_point[p].append(case)
        if p in by_seed[seed]:
            raise ValueError(f"Duplicate seed at point {p}")
        by_seed[seed][p].append(case)
    if any(len(by_point[p]) < 2 for p in points):
        raise ValueError("Response roster fails its two-seed-per-point gate")
    return dict(status="COMPLETE_FRESH_RESPONSE_READOUT", **result,
                unit="Absolute error of high-minus-low counts, vehicles per movement per 10 seconds.",
                scoring="Each movement receives half weight, seeds are averaged within each point and all seven points receive equal weight. Every realized exposure is retained. Each seed-specific loss averages its own represented points.",
                primary=losses(points, tables, by_point),
                per_point=[dict(point=p, seeds=[r["seed"] for r in by_point[p]], **losses([p], tables, by_point))
                           for p in points],
                per_seed=[dict(seed=seed, n_points=len(obs), **losses(sorted(obs), tables, obs))
                          for seed, obs in sorted(by_seed.items())],
                permutation=distribution(points, tables["hangzhou_matched"], by_point),
                rank_semantics="Rank among all paired label assignments; a descriptive placement control.",
                cases=cases)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    parser.add_argument("--plan", type=Path, default=HANGZHOU / "next_plan.json")
    args = parser.parse_args()
    data = score(json.loads(args.pairs.read_text()), json.loads(args.selection.read_text()),
                 json.loads(args.plan.read_text()))
    print(json.dumps(data, indent=2))
