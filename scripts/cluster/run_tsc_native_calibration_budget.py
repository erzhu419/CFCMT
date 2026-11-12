"""Fixed native calibration budget curve; reuse the frozen 80-group endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


METRICS = ("cost", "pp_cost", "harm", "benefit", "overrides", "harmful")


def nested_cells(seeds, folds):
    """Eight balanced cyclic subsets per outer fold, never touching its test seeds."""
    if len(seeds) != 10 or len(set(seeds)) != 10 or sorted(s for f in folds for s in f) != sorted(seeds):
        raise ValueError("expected ten unique seeds partitioned into five folds")
    for fold_id, heldout in enumerate(folds):
        if len(heldout) != 2:
            raise ValueError("each outer fold must hold out two simulator seeds")
        training = [s for s in seeds if s not in heldout]
        for rotation in range(8):
            order = training[rotation:] + training[:rotation]
            for size in (2, 4):
                yield dict(fold=fold_id, rotation=rotation, budget=10 * size,
                           training_seeds=order[:size], heldout_seeds=list(heldout))


def outcome_metrics(rows):
    gains = np.asarray([r["actual_advantage"] for r in rows], dtype=float)
    return dict(cost=float(np.mean([r["selected_cost"] for r in rows])),
        pp_cost=float(np.mean([r["phase_pressure_cost"] for r in rows])),
        harm=float(np.maximum(-gains, 0).mean()), benefit=float(np.maximum(gains, 0).mean()),
        overrides=float(sum(r["different_action"] for r in rows)),
        harmful=float(np.sum(gains < -1e-12)))


def aggregate_cells(cells, seeds, scenario=None):
    budgets = {cell["budget"] for cell in cells}
    if len(budgets) != 1 or not budgets <= {20, 40, 80}:
        raise ValueError("aggregate one declared budget at a time")
    budget = next(iter(budgets))
    per_seed, variation = {}, {}
    for seed in seeds:
        entries = [cell for cell in cells if seed in cell["heldout_seeds"]]
        if len(entries) != (1 if budget == 80 else 8):
            raise ValueError("each held-out seed needs all eight rotations or one reused endpoint")
        if budget == 80 and (entries[0].get("reused") is not True or entries[0]["rotation"] is not None):
            raise ValueError("80-group endpoint must be explicitly reused")
        if len(entries) == 8 and sorted(c["rotation"] for c in entries) != list(range(8)):
            raise ValueError("missing or duplicated rotation")
        values = []
        expected_groups = None
        for cell in entries:
            rows = [r for r in cell["records"] if r["seed"] == seed and (scenario is None or r["scenario"] == scenario)]
            groups = {r["group_id"] for r in rows}
            if not rows or len(groups) != len(rows) or (expected_groups is not None and groups != expected_groups):
                raise ValueError("rotations must evaluate the same complete group roster")
            if scenario is None and len(rows) != 10:
                raise ValueError("each validation seed must retain all ten groups")
            expected_groups = groups
            values.append(outcome_metrics(rows))
        per_seed[int(seed)] = {key: float(np.mean([v[key] for v in values])) for key in METRICS}
        variation[int(seed)] = {key: dict(min=float(min(v[key] for v in values)),
            max=float(max(v[key] for v in values)), std=float(np.std([v[key] for v in values])))
            for key in ("cost", "harm", "harmful")}
    mean = {key: float(np.mean([v[key] for v in per_seed.values()])) for key in METRICS}
    return dict(per_seed=per_seed, mean=mean, variation_across_fixed_subsets=variation,
        harmful_fraction=(mean["harmful"] / mean["overrides"] if mean["overrides"] else None),
        independent_seed_count=len(seeds), subset_count_per_seed=len(entries))


def paired(v158, left, right, metric):
    return v158._paired_summary(
        {s: r[metric] for s, r in left["per_seed"].items()},
        {s: r[metric] for s, r in right["per_seed"].items()}, draws=10000, seed=20261106)


def run(args):
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    sys.path.insert(0, str(args.parent))
    import fit_tsc_predecision_history as history
    v158 = history.v158
    spec = json.loads(args.spec.read_text())
    if (spec["model_config"] != history.MODEL_CONFIG or spec["development_seeds"] != list(v158.DEVELOPMENT_SEEDS)
            or spec["outer_folds"] != [list(f) for f in v158.CROSSFIT_FOLDS]
            or spec["native_group_budgets"] != [20, 40, 80] or spec["new_fits"] != 160
            or spec["arms"] != ["static", "history"]
            or spec["execution"] != dict(threshold=0, no_stay_override_of_pp_switch=True, reference="phase_pressure")
            or spec["history_definition"]["feature_names"] != list(history.HISTORY_NAMES)):
        raise ValueError("budget diagnostic differs from the prepared specification")
    seeds, folds = spec["development_seeds"], spec["outer_folds"]
    data = history.load_mechanism_dataset(args.bank)
    packets = [json.loads((args.parent / "history" / f"seed_{s}" / "history.json").read_text()) for s in seeds]
    data = history.join_history(data, [r for p in packets for r in p["groups"]])
    history.seed_folds(data, folds)
    old = json.loads((args.parent / "analysis/oof_server_only.json").read_text())
    old_result = json.loads((args.parent / "analysis/result.json").read_text())
    group_ids = np.asarray(data.metadata["action_group_ids"])
    row_seeds = np.asarray(data.metadata["row_seeds"])
    all_groups = set(group_ids.tolist())
    results = {arm: {budget: [] for budget in (20, 40, 80)} for arm in spec["arms"]}
    for arm in spec["arms"]:
        original = old["records"][arm]
        if len(original) != 100 or {r["group_id"] for r in original} != all_groups:
            raise ValueError("frozen endpoint is incomplete")
        observed = v158._seed_costs(original)
        for seed in seeds:
            if abs(observed[seed] - old_result["per_seed_costs"][arm][str(seed)]) > 1e-12:
                raise ValueError("frozen endpoint no longer matches its reported cost")
        for fold_id, heldout in enumerate(folds):
            results[arm][80].append(dict(fold=fold_id, rotation=None, budget=80,
                heldout_seeds=heldout, training_seeds=[s for s in seeds if s not in heldout],
                reused=True, records=[r for r in original if r["seed"] in heldout]))
    started, count = time.monotonic(), 0
    args.out_dir.mkdir(parents=True, exist_ok=False)
    record_path = args.out_dir / "cells_server_only.jsonl"
    with record_path.open("x") as handle:
        for cell in nested_cells(seeds, folds):
            train_groups = set(group_ids[np.isin(row_seeds, cell["training_seeds"])].tolist())
            test_groups = set(group_ids[np.isin(row_seeds, cell["heldout_seeds"])].tolist())
            if len(train_groups) != cell["budget"] or len(test_groups) != 20 or train_groups & test_groups:
                raise ValueError("native calibration subset or holdout has changed")
            train = v158._subset_groups(data, train_groups)
            test = v158._subset_groups(data, test_groups)
            for arm in spec["arms"]:
                model, diagnostics = history.fit_model(train, arm == "history")
                records = history.policy_records(model, test, arm == "history")
                if {r["group_id"] for r in records} != test_groups or len(records) != 20:
                    raise ValueError("prediction dropped or duplicated a held-out group")
                result = dict(cell, arm=arm, records=records,
                    training_groups=len(train_groups), training_rows=train.size,
                    state_feature_count=diagnostics["state_feature_count"],
                    action_feature_count=diagnostics["action_feature_count"])
                results[arm][cell["budget"]].append(result)
                handle.write(json.dumps(result, allow_nan=False) + "\n")
                handle.flush()
                count += 1
            if cell["budget"] == 40:
                print(json.dumps(dict(completed_fits=count, total_fits=160, fold=cell["fold"],
                                      rotation=cell["rotation"])), flush=True)
    if count != 160:
        raise ValueError("incomplete fit matrix")
    summaries, comparisons, demands = {}, {}, {}
    scenarios = sorted(set(data.metadata["row_scenarios"]))
    for arm, budgets in results.items():
        summaries[arm] = {b: aggregate_cells(cells, seeds) for b, cells in budgets.items()}
        comparisons[arm] = {}
        for left, right in ((40, 20), (80, 40), (80, 20)):
            comparisons[arm][f"{left}_minus_{right}"] = {
                metric: paired(v158, summaries[arm][left], summaries[arm][right], metric)
                for metric in ("cost", "harm", "harmful")}
        for budget in budgets:
            sample = summaries[arm][budget]
            pp = {"per_seed": {s: dict(cost=r["pp_cost"]) for s, r in sample["per_seed"].items()}}
            comparisons[arm][f"{budget}_minus_pp"] = paired(v158, sample, pp, "cost")
        demands[arm] = {scenario: {b: aggregate_cells(cells, seeds, scenario=scenario)
                                  for b, cells in budgets.items()} for scenario in scenarios}
    result = dict(protocol=spec["protocol"], status="DIAGNOSTIC_COMPLETE", new_fits=count,
        reused_endpoint_fits=10, elapsed_sec=time.monotonic() - started,
        native_labels_added=0, simulator_runs=0, development_seed_count=10,
        summaries=summaries, comparisons=comparisons, per_demand=demands,
        original_endpoint_status=old_result["status"], closed_loop_authorized=False,
        estimand="average outcomes over eight fixed cyclic training subsets within held-out seed; then equal mean over ten development seeds; not an ensemble",
        scope="Additional native calibration budget with legacy B100 base predictions fixed; reused development evidence, not a fresh confirmation or policy selection.")
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], new_fits=count,
        means={a: {b: r["mean"] for b, r in v.items()} for a, v in summaries.items()}), allow_nan=False), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "bank", "spec", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())
