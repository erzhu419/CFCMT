"""Recompute the three-city comparisons for Table III and Figure 3(b)."""

from collections import defaultdict
from itertools import product
import argparse
import json
from pathlib import Path

import numpy as np


EXP_DIR = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "tsc_partial_cv_pilot_20261118"
)
METRICS = ("mean_waiting_sec", "system_vehicle_hours")
RL_RESULT_FILE = "phase3_rl_baselines_repaired_result.json"
COMPARATORS = {
    "conservation_gp_10": "Cons-GP",
    "maxred_conservation_gp_10": "MaxRed-Cons-GP",
    "fixed_program": "Fixed-Time",
    "observed_mp_10": "Observed-MP",
    "mplight_cv_ep2": "MPLight-CV (ep2)",
    "mplight_cv_ep16": "MPLight-CV (ep16)",
    "transfer_mplight_cv_ep2": "Trans-MPLight-CV",
    "target_only_b6": "Target-Only ($B=6$)",
    "source_placebo_b6": "Source-Placebo ($B=6$)",
    "target_only_b36": "Target-Only ($B=36$)",
    "source_placebo_b36": "Source-Placebo ($B=36$)",
    "pool_all_b36": "Pool-All ($B=36$)",
}


def holm(p_values):
    """Return step-down adjusted p-values in the original order."""
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    adjusted[order] = np.minimum(
        1, np.maximum.accumulate(p_values[order] * np.arange(len(order), 0, -1))
    )
    return adjusted


def _resampling(panels):
    signs = [np.array(list(product((-1, 1), repeat=len(p)))) for p in panels]
    rng = np.random.default_rng(20261119)
    bootstrap = [rng.integers(len(p), size=(10_000, len(p))) for p in panels]
    return bootstrap, signs


def _paired_statistics(first, second, bootstrap, sign_matrices):
    means = first.mean(axis=0) + second.mean(axis=0)
    draws = first[bootstrap[0]].mean(axis=1) + second[bootstrap[1]].mean(axis=1)
    intervals = np.quantile(draws, [0.025, 0.975], axis=0).T

    # Exhaustively flip the paired differences independently across the two panels.
    first_flips, second_flips = sign_matrices[0] @ first, sign_matrices[1] @ second
    totals = np.abs(first_flips[:, None] + second_flips[None, :])
    observed = np.abs(first.sum(axis=0) + second.sum(axis=0))
    raw_p = np.mean(totals >= observed - 1e-8, axis=(0, 1))
    return means, intervals, raw_p


def summarize(exp_dir=EXP_DIR):
    """Compare B=6 with each control, retaining the two original seed panels."""
    p2 = json.loads((exp_dir / "phase2_eval_fresh_result.json").read_text())
    sl = json.loads((exp_dir / "phase3_saltlake_eval_fresh_result.json").read_text())
    rl = json.loads((exp_dir / RL_RESULT_FILE).read_text())
    by_seed = defaultdict(list)
    for row in p2["rows"] + sl["rows"] + rl["rows"]:
        key = (row["family"], row["policy"], row["seed"])
        by_seed[key].append([row[metric] for metric in METRICS])
    city_means = {key: np.mean(rows, axis=0) for key, rows in by_seed.items()}

    def by_panel(policy):
        la_jinan = np.array(
            [
                city_means["la", policy, seed] + city_means["jinan", policy, seed]
                for seed in p2["seeds"]
            ]
        )
        saltlake = np.array(
            [city_means["saltlake", policy, seed] for seed in sl["seeds"]]
        )
        return la_jinan / 3, saltlake / 3

    candidate = by_panel("cfcmt_b6")
    resampling = _resampling(candidate)
    results = {}
    for policy in COMPARATORS:
        first, second = (c - b for c, b in zip(candidate, by_panel(policy)))
        means, intervals, raw_p = _paired_statistics(first, second, *resampling)
        results[policy] = {
            metric: {"mean": mean, "interval": interval, "raw_p": p}
            for metric, mean, interval, p in zip(METRICS, means, intervals, raw_p)
        }
    for metric in METRICS:
        raw_p = np.array([result[metric]["raw_p"] for result in results.values()])
        for result, adjusted in zip(results.values(), holm(raw_p)):
            result[metric]["holm_p"] = adjusted
    return results


def summarize_source_only(exp_dir=EXP_DIR):
    """Pair the two zero-shot policies on common scenarios and seeds."""
    p2 = json.loads((exp_dir / "phase2_eval_fresh_result.json").read_text())
    sl = json.loads((exp_dir / "phase3_saltlake_eval_fresh_result.json").read_text())
    rl = json.loads((exp_dir / "phase3_rl_source_only_result.json").read_text())
    trained_rl = json.loads((exp_dir / RL_RESULT_FILE).read_text())
    pretrain = rl["training_audit"]["source_pretrain"]
    if pretrain != trained_rl["training_audit"]["source_pretrain"]:
        raise ValueError("The source-only and adapted RL arms use different source pretraining.")

    metrics = (*METRICS, "completion_ratio")

    def index(rows):
        cells = {
            (r["family"], r["scenario"], r["seed"]): [r[m] for m in metrics]
            for r in rows
        }
        if len(cells) != len(rows):
            raise ValueError("Duplicate source-only comparison cell.")
        return cells

    candidate = index([r for r in p2["rows"] + sl["rows"] if r["policy"] == "zero_shot_cfcmt_b0"])
    reference = index(rl["rows"])
    if candidate.keys() != reference.keys():
        raise ValueError("The zero-shot policies have different scenario/seed cells.")

    paired = defaultdict(list)
    for key, values in candidate.items():
        city, _, seed = key
        paired[city, seed].append([values, reference[key]])

    cities = {}
    differences = {}
    for city, seeds in (("la", p2["seeds"]), ("jinan", p2["seeds"]), ("saltlake", sl["seeds"])):
        candidate_by_seed, rl_by_seed = np.array(
            [np.mean(paired[city, seed], axis=0) for seed in seeds]
        ).transpose(1, 0, 2)
        diff = candidate_by_seed - rl_by_seed
        differences[city] = diff
        cities[city] = {
            "cfcmt_b0": candidate_by_seed.mean(axis=0).tolist(),
            "source_only_rl": rl_by_seed.mean(axis=0).tolist(),
            "cfcmt_minus_rl": diff.mean(axis=0).tolist(),
            "cfcmt_better_seeds": np.sum(diff * [-1, -1, 1] > 0.0, axis=0).tolist(),
        }

    blocks = ((differences["la"] + differences["jinan"]) / 3, differences["saltlake"] / 3)
    means, intervals, raw_p = _paired_statistics(*blocks, *_resampling(blocks))
    return {
        "metrics_in_order": list(metrics),
        "source_pretrain_matches_repaired_rl": pretrain,
        "pair_count": len(candidate),
        "cities": cities,
        "equal_city": {
            "cfcmt_b0": np.mean([r["cfcmt_b0"] for r in cities.values()], axis=0).tolist(),
            "source_only_rl": np.mean([r["source_only_rl"] for r in cities.values()], axis=0).tolist(),
            "cfcmt_minus_rl": means.tolist(),
            "intervals": intervals.tolist(),
            "time_outcome_raw_p": raw_p[:2].tolist(),
            "time_outcome_holm_p": holm(raw_p[:2]).tolist(),
        },
    }


def latex_p(value):
    if value >= 0.001:
        return f"{value:.4f}"
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"{coefficient}\times 10^{{{int(exponent)}}}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    if args.source_only:
        print(json.dumps(summarize_source_only(), indent=2))
        raise SystemExit(0)

    for policy, result in summarize().items():
        cells = [COMPARATORS[policy]]
        for metric in METRICS:
            stats = result[metric]
            low, high = stats["interval"]
            cells.extend(
                [
                    f"${stats['mean']:.1f}$",
                    f"$[{low:.1f},{high:.1f}]$",
                    f"${latex_p(stats['holm_p'])}$",
                ]
            )
        print(" & ".join(cells) + r" \\")
