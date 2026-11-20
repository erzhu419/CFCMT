"""Budget-matched target-only MPLight-CV with a training-only simulator reward."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
from statistics import mean

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_transfer import (
    ADAPT_SEEDS,
    FRESH_PHASE2_SEEDS,
    TARGET_FAMILIES,
    _dict_to_rec,
    read_paper_label_archive,
    resolve_phase2_scenarios,
    subset_residual_budget,
)
from experiments.tsc_partial_cv_pilot_20261118.phase3_tits_suite import (
    FRESH_SALTLAKE_SEEDS,
    TRUTH_REWARD_RL_POLICY,
    eval_custom_policy_worker,
    saltlake_scenario_configs,
    train_mplight_agent,
)

TRAINING_SEEDS = (2026112001, 2026112002, 2026112003)
REPEAT_TEST_SEEDS = (971201, 971202, 971203, 971204)
SCAN_SECONDS = 3430
BRANCH_SECONDS = 60
METRICS = ("mean_waiting_sec", "system_vehicle_hours", "completion_ratio")


def acquisition_cost(groups, scans: int) -> dict:
    """Count simulator seconds for the archived candidate scans and native branches."""
    prefix = sum(int(g.time_sec) * len(g.candidate_states) for g in groups)
    branches = sum(len(g.candidate_states) for g in groups)
    return {
        "groups": len(groups),
        "scan_runs": scans,
        "scan_sec": scans * SCAN_SECONDS,
        "branch_runs": branches,
        "prefix_sec": prefix,
        "branch_sec": branches * BRANCH_SECONDS,
        "total_sec": scans * SCAN_SECONDS + prefix + branches * BRANCH_SECONDS,
    }


def fit_truth_policy(task: dict) -> dict:
    family, seed, cfgs, duration, work_dir = (
        task["family"], task["seed"], task["cfgs"], task["duration"], task["work_dir"],
    )
    hours, remaining = divmod(duration, 3600)
    episode_lengths = [3600] * hours + ([remaining] if remaining else [])
    net, audit = train_mplight_agent(
        sumocfgs=tuple(cfgs.values()), seeds=ADAPT_SEEDS, n_episodes=len(episode_lengths),
        init_net=None, rng_seed=seed, workers=2, episode_durations_sec=episode_lengths,
        ground_truth_reward=True,
    )
    model_file = work_dir / f"{family}_{seed}.pkl"
    with model_file.open("wb") as file:
        pickle.dump({family: {TRUTH_REWARD_RL_POLICY: net}}, file)
    audit.update({
        "family": family, "seed": seed, "simulator_sec": sum(episode_lengths),
        "final_episode_sec": episode_lengths[-1], "models_pkl": str(model_file),
    })
    print(f"FIT {family} {seed} episodes={audit['episodes']} sim_sec={duration}", flush=True)
    return audit


def evaluate_truth_policy(task: dict) -> dict:
    row = eval_custom_policy_worker(task)
    row["training_seed"] = task["training_seed"]
    return row


def summarize_evaluations(rows: list[dict], families: list[str]) -> dict:
    summaries = {}
    for city in families:
        original = FRESH_SALTLAKE_SEEDS if city == "saltlake" else FRESH_PHASE2_SEEDS
        summaries[city] = {}
        for panel, seeds in (("archived", original), ("new_seeds", REPEAT_TEST_SEEDS)):
            rows_by_fit = [
                [r for r in rows if r["family"] == city and r["seed"] in seeds
                 and r["training_seed"] == training_seed]
                for training_seed in TRAINING_SEEDS
            ]
            summary = {}
            for metric in METRICS:
                means = [mean(r[metric] for r in fit_rows) for fit_rows in rows_by_fit]
                summary[metric] = {"means": means, "average": mean(means), "range": [min(means), max(means)]}
            summary["severe_counts"] = [
                sum(r["completion_ratio"] < 0.5 for r in fit_rows) for fit_rows in rows_by_fit
            ]
            summaries[city][panel] = summary
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-labels", type=Path, required=True)
    parser.add_argument("--saltlake-labels", type=Path)
    parser.add_argument("--staged-scenarios-dir", type=Path, required=True)
    parser.add_argument("--source-scenarios-dir", type=Path, required=True)
    parser.add_argument("--ext-targets-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=("la", "jinan", "saltlake"), required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sources, target, _ = read_paper_label_archive(args.archive_labels)
    scenarios = resolve_phase2_scenarios(args.staged_scenarios_dir, args.source_scenarios_dir)
    cfgs = {city: {name: scenarios[name] for name in names} for city, names in TARGET_FAMILIES.items()}
    if "saltlake" in args.families:
        if args.saltlake_labels is None:
            parser.error("--saltlake-labels is required for Salt Lake City")
        target["saltlake"] = [_dict_to_rec(row) for row in json.loads(args.saltlake_labels.read_text())]
        cfgs["saltlake"] = saltlake_scenario_configs(args.ext_targets_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    source_scans = len({(group.scenario, group.seed) for group in sources})
    report = {"source_library": acquisition_cost(sources, source_scans), "target": {}}
    fit_tasks = []
    for city in args.families:
        # Original B=6 samples three groups per adaptation seed.
        selected = subset_residual_budget(target[city], 3)
        if len(selected) != 6:
            raise ValueError(f"Expected 6 adaptation groups in {city}.")
        scans = len(cfgs[city]) * len(ADAPT_SEEDS)
        used = acquisition_cost(selected, scans)
        collected = acquisition_cost(target[city], scans)
        report["target"][city] = {"selected_only": used, "archived_pool": collected}
        for seed in TRAINING_SEEDS:
            fit_tasks.append({
                "family": city, "seed": seed, "cfgs": cfgs[city],
                "duration": used["total_sec"], "work_dir": args.work_dir,
            })
    print(json.dumps(report), flush=True)

    with ProcessPoolExecutor(max_workers=max(1, args.workers // 2)) as pool:
        fits = list(pool.map(fit_truth_policy, fit_tasks))

    eval_tasks = [
        {
            "family": fit["family"], "scenario": scenario, "sumocfg": str(config),
            "seed": seed, "p_cv": 0.1, "policy": TRUTH_REWARD_RL_POLICY,
            "training_seed": fit["seed"], "models_pkl": fit["models_pkl"],
        }
        for fit in fits
        for scenario, config in cfgs[fit["family"]].items()
        for seed in (
            FRESH_SALTLAKE_SEEDS if fit["family"] == "saltlake" else FRESH_PHASE2_SEEDS
        ) + REPEAT_TEST_SEEDS
    ]
    print(f"EVAL {len(eval_tasks)} runs", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(evaluate_truth_policy, eval_tasks), 1):
            rows.append(row)
            if i % 30 == 0 or i == len(eval_tasks):
                print(f"EVAL {i}/{len(eval_tasks)}", flush=True)
    if not np.isfinite([row["system_vehicle_hours"] for row in rows]).all():
        raise ValueError("Non-finite vehicle-hours in truth-reward evaluation.")
    for fit in fits:
        fit.pop("models_pkl")
    result = {
        "acquisition": report, "fits": fits, "rows": rows,
        "repeat_eval_seeds": REPEAT_TEST_SEEDS,
        "target_training_seeds": ADAPT_SEEDS,
        "city_metrics": summarize_evaluations(rows, args.families),
    }
    (args.work_dir / "truth_reward_rl_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
