"""Read the two completed Hangzhou supply summaries and compare exact inlet counts with Jinan."""
import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent
TARGET = DIRECTORY.parent / "tsc_jinan_phase_pair_readout_20261106" / "pairs.json"
SUPPORT = {
    "physical": "all_pairs",
    "switchable": "switchable_pairs",
    "high4_low0": "target_exposure_pairs",
}


def observed_tradeoff(row):
    high, low = row["high"], row["low"]
    return (high["right_actual"] == low["right_actual"]
            and high["exposure_s"] == row["priority_exposure_s_high"]
            and low["exposure_s"] == row["priority_exposure_s_low"]
            and row["delta_right_entries"] < 0 and row["delta_straight_entries"] > 0)


def coverage(rows, seeds):
    counts = {"target_rows": len(rows),
              "distinct_near_count_points": len({tuple(row["point"]) for row in rows})}
    for level in SUPPORT:
        by_seed = [{i for i, row in enumerate(rows) if row["source_pair_ticks"][level][seed] > 0}
                   for seed in seeds]
        covered = dict(zip(seeds, by_seed))
        covered.update(both_seeds=by_seed[0] & by_seed[1], either_seed=by_seed[0] | by_seed[1])
        counts[level] = {
            group: {"rows": len(indices), "distinct_points": len({tuple(rows[i]["point"]) for i in indices})}
            for group, indices in covered.items()
        }
    return counts


def compare(summaries, target):
    seeds = [summary["seed"] for summary in summaries]
    if len(set(seeds)) != 2 or len(seeds) != 2:
        raise ValueError("Expected a completed summary for each of the two different source seeds")
    histograms = {
        level: {summary["seed"]: {(r, s): n for r, s, n in summary[key]["joint_histogram"]}
                for summary in summaries}
        for level, key in SUPPORT.items()
    }
    rows = []
    for index, row in enumerate(target, 1):
        point = row["own_near"], row["priority_near"]
        rows.append(dict(target_row=index, case_id=row["case_id"], right_movement=row["own_movement"],
                         point=point, previously_observed_tradeoff=observed_tradeoff(row),
                         source_pair_ticks={level: {seed: hist.get(point, 0) for seed, hist in per_seed.items()}
                                            for level, per_seed in histograms.items()}))
    return dict(source_seeds=seeds, point_fields=["right_near", "straight_near"],
                all_fixed_target_rows=coverage(rows, seeds),
                previously_observed_tradeoff_subset=coverage([r for r in rows if r["previously_observed_tradeoff"]], seeds),
                target_rows=rows,
                interpretation="A row is covered when its two near counts occur together on a source pair. "
                "High4_low0 requires a switchable pair with planned high and low exposures of 4 and 0 s. "
                "The tradeoff subset uses prior Jinan outcomes. The source counts describe supply opportunities.")


if __name__ == "__main__":
    seeds = json.loads((DIRECTORY / "protocol.json").read_text())["source_seeds"]
    summaries = [json.loads((DIRECTORY / f"summary_{seed}.json").read_text()) for seed in seeds]
    if [s["seed"] for s in summaries] != seeds:
        raise ValueError("Source summary seed differs from its protocol slot")
    print(json.dumps(compare(summaries, json.loads(TARGET.read_text())), indent=2))
