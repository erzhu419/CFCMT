"""Compare the two frozen right-yield action rules on the compact development rows."""
import argparse
from fractions import Fraction
import json
from pathlib import Path


SEEDS = {"jinan": (33061101, 33061102, 33061103, 33061104),
         "hangzhou": (32061101, 32061102)}
POINTS = ((1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (3, 2))


def training(rows):
    n = sum(r["n"] for r in rows)
    k = sum(r["k"] for r in rows)
    rate = Fraction(k, n) if n else None
    return rate, dict(starts=len(rows), starts_with_halted_near_right=sum(r["n"] > 0 for r in rows),
                      entrants=k, halted_near_right=n, rate=str(rate) if rate is not None else None)


def choice(row, rate):
    high = row["b_hi"] - (1 - rate) * row["n"]
    low = row["b_lo"]
    if high == low:
        return min(row["hi"], row["lo"]), high
    return (row["hi"] if high > low else row["lo"]), high


def decision(row, target_rate, mixed_rate):
    zero, high_zero = choice(row, Fraction(0))
    one, high_one = choice(row, Fraction(1))
    pair = row["pp"] in (row["hi"], row["lo"])
    supported = mixed_rate is not None
    target = mixed = row["pp"]
    target_score = mixed_score = None
    reason = None
    if not pair:
        reason = "pp_other"
    elif not supported:
        reason = "no_training_support"
    else:
        target, high = choice(row, target_rate)
        target_score = str(high)
        mixed, high = choice(row, mixed_rate)
        mixed_score = str(high)
    boundary = Fraction(row["n"] + row["b_lo"] - row["b_hi"], row["n"]) if row["n"] else None
    return {
        **{k: row[k] for k in ("arm_index", "seed", "point", "time_sec", "tls_id", "right_id", "straight_id",
                                "hi", "lo", "pp", "b_hi", "b_lo", "n")},
        "pp_on_pair": pair, "both_rates_supported": supported, "fallback": reason,
        "target_choice": target, "mixed_choice": mixed, "target_high_score": target_score,
        "mixed_high_score": mixed_score, "low_score": row["b_lo"],
        "pair_only_at_r0": zero, "pair_only_at_r1": one, "high_score_r0": str(high_zero),
        "high_score_r1": str(high_one), "tie_rate_in_unit_interval":
            str(boundary) if boundary is not None and 0 <= boundary <= 1 else None,
        "pair_scores_tie_at_all_rates": row["n"] == 0 and row["b_hi"] == row["b_lo"],
    }


def counts(cases):
    on_pair = [c for c in cases if c["pp_on_pair"]]
    sensitive = [c for c in cases if c["pair_only_at_r0"] != c["pair_only_at_r1"]]
    return {
        "n": len(cases), "pp_on_pair": len(on_pair),
        "pp_other": len(cases) - len(on_pair),
        "on_pair_without_training_support": sum(not c["both_rates_supported"] for c in on_pair),
        "on_pair_with_training_support": sum(c["both_rates_supported"] for c in on_pair),
        "positive_current_n_on_pair": sum(c["n"] > 0 for c in on_pair),
        "sensitive_pair_only_ignore_pp_and_support": len(sensitive),
        "sensitive_with_pp_gate_ignore_support": sum(c["pp_on_pair"] for c in sensitive),
        "sensitive_with_pp_and_support_gates": sum(c["pp_on_pair"] and c["both_rates_supported"] for c in sensitive),
        "target_changes_pp": sum(c["target_choice"] != c["pp"] for c in cases),
        "mixed_changes_pp": sum(c["mixed_choice"] != c["pp"] for c in cases),
        "between_rule_disagreements": sum(c["target_choice"] != c["mixed_choice"] for c in cases),
    }


def screen(rows):
    expected = {(city, seed, point) for city, seeds in SEEDS.items() for seed in seeds for point in POINTS}
    actual = {(r["city"], r["seed"], tuple(r["point"])) for r in rows}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Requires each frozen development city/seed/point exactly once")
    if any(r["hi"] == r["lo"] or not 0 <= r["k"] <= r["n"] for r in rows):
        raise ValueError("Invalid request pair or right-vehicle service count")
    target = [r for r in rows if r["city"] == "jinan"]
    h, source = training([r for r in rows if r["city"] == "hangzhou"])
    folds, cases, per_seed = [], [], []
    for seed in SEEDS["jinan"]:
        t, fold = training([r for r in target if r["seed"] != seed])
        mixed = (t + h) / 2 if t is not None and h is not None else None
        folds.append(dict(held_out_seed=seed, **fold, mixed_rate=str(mixed) if mixed is not None else None))
        held_out = sorted((r for r in target if r["seed"] == seed), key=lambda r: tuple(r["point"]))
        batch = [decision(r, t, mixed) for r in held_out]
        cases.extend(batch)
        per_seed.append(dict(seed=seed, **counts(batch)))
    distinct = [{k: c[k] for k in ("arm_index", "seed", "point", "time_sec", "tls_id", "right_id",
                                    "straight_id", "target_choice", "mixed_choice", "pp")}
                for c in cases if c["target_choice"] != c["mixed_choice"]]
    return dict(status="FREEZE_DISAGREEING_PAIRS" if distinct else "CLOSE_NO_DECISION_DISAGREEMENT",
                training={"hangzhou": source, "jinan_folds": folds}, overall=counts(cases),
                per_seed=per_seed, cases=cases, distinct_pairs=distinct)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    args = parser.parse_args()
    print(json.dumps(screen(json.loads(args.inputs.read_text())), indent=2))
