"""Score frozen probability forecasts for a complete four-seed, six-point Jinan panel."""
from fractions import Fraction
from statistics import mean


CANDIDATE = "(T+A)/2"
MATCHED_TARGET = "(T+T_pool)/2"


def predictions(plan):
    rows = plan["development_counts"]["rows"]
    target_pool = Fraction(sum(r[3] for r in rows), sum(r[2] for r in rows))
    aligned_pool = Fraction(sum(r[5] for r in rows), sum(r[4] for r in rows))
    result = {}
    for right, straight, target_n, target_k, source_n, aligned_k, native_k in rows:
        target = Fraction(target_k, target_n)
        aligned = Fraction(aligned_k, source_n)
        native = Fraction(native_k, source_n)
        result[right, straight] = {
            CANDIDATE: (target + aligned) / 2,
            "T": target,
            "T_pool": target_pool,
            MATCHED_TARGET: (target + target_pool) / 2,
            "(T+A_pool)/2": (target + aligned_pool) / 2,
            "(T+H)/2": (target + native) / 2,
        }
    return result


def score(plan, cases):
    forecast = predictions(plan)
    controls = plan["frozen_predictions"]["controls"]
    names = [CANDIDATE, *controls]
    seeds = plan["target_seeds"]
    observed = {(r["seed"], tuple(r["point"])): r["crossed"] for r in cases}
    expected = {(s, p) for s in seeds for p in forecast}
    if (len(cases) != len(expected) or len(expected) != 24 or set(observed) != expected
            or any(type(value) is not bool for value in observed.values())):
        raise ValueError("Brier loss requires one observed second-car outcome in each frozen cell")
    per_seed = {
        s: {name: mean((f[name] - observed[s, point]) ** 2 for point, f in forecast.items())
            for name in names}
        for s in seeds
    }
    overall = {name: mean(per_seed[s][name] for s in seeds) for name in names}

    def readout(losses):
        return {"brier": {name: float(losses[name]) for name in names},
                "candidate_minus_control": {
                    name: float(losses[CANDIDATE] - losses[name]) for name in controls}}

    # Keep exact fractions through the strict comparisons, particularly when losses tie.
    best = all(overall[CANDIDATE] < overall[name] for name in controls)
    improved_seeds = [s for s in seeds if per_seed[s][CANDIDATE] < per_seed[s][MATCHED_TARGET]]
    return {
        "status": "PASS_PREDICTION" if best and len(improved_seeds) >= 3 else "HOLD_PREDICTION",
        "cases": len(cases), "crossed": sum(observed.values()),
        "overall": readout(overall),
        "per_seed": [dict(seed=s, crossed=sum(observed[s, p] for p in forecast),
                          n=len(forecast), **readout(per_seed[s])) for s in seeds],
        "per_point": [
            dict(point=list(point), crossed=sum(observed[s, point] for s in seeds), n=len(seeds),
                 predictions={name: float(f[name]) for name in names})
            for point, f in forecast.items()
        ],
        "gate": {"strictly_better_than_all_controls": best,
                 "seeds_improved_over_matched_target": improved_seeds,
                 "minimum_seeds_improved": 3}
    }
