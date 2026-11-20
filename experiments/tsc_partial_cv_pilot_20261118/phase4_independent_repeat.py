"""Independent, nested target-label repeats for the Partial-CV revision."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pickle
from statistics import mean, median

import numpy as np

from experiments.tsc_partial_cv_pilot_20261118.phase1_transfer_pilot import (
    evaluate_single_branch_worker,
    scan_decision_candidates_worker,
)
from experiments.tsc_partial_cv_pilot_20261118.phase2_residual_transfer import (
    OMEGA_POS_GRID,
    _dict_to_rec,
    _rec_to_dict,
    eval_phase2_closed_loop_worker,
    fit_phase2_residual_suite,
    read_paper_label_archive,
    subset_residual_budget,
)

ADAPTATION_PAIRS = ((771201, 771202), (771203, 771204), (771205, 771206))
TEST_SEEDS = (971201, 971202, 971203, 971204)
OMEGA_GRID = (0.0, *OMEGA_POS_GRID)
POLICIES = ("cfcmt_b6", "target_only_b6", "cfcmt_b36", "target_only_b36")
METRICS = ("mean_waiting_sec", "system_vehicle_hours", "completion_ratio")
SEVERE_COMPLETION = 0.50
SCAN_SECONDS = 3430
BRANCH_SECONDS = 60

FAMILIES = {
    "la": ("la_1x4",),
    "jinan": ("jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"),
    "saltlake": ("saltlake_400s_peak", "saltlake_state_peak"),
}
SCAN_QUOTA = {"la": 20, "jinan": 8, "saltlake": 10}


def scenario_paths(staged: Path, external: Path) -> dict[str, Path]:
    paths = {
        s: staged / s / f"{s}.sumocfg"
        for city in ("la", "jinan")
        for s in FAMILIES[city]
    }
    for short, long in (
        ("saltlake_400s_peak", "saltlake_400s_200w_q1_weekday_peak"),
        ("saltlake_state_peak", "saltlake_state_university_q1_weekday_peak"),
    ):
        paths[short] = external / long / f"{long}.sumocfg"
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def collect_labels(
    paths: dict[str, Path], repeat: int, seeds: tuple[int, int], workers: int
) -> dict:
    scan_tasks = [
        {
            "scenario": scenario, "sumocfg": str(paths[scenario]), "seed": seed,
            "domain": f"target_{city}", "max_groups": SCAN_QUOTA[city],
        }
        for city, scenarios in FAMILIES.items()
        for seed in seeds
        for scenario in scenarios
    ]
    candidates = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, rows in zip(scan_tasks, pool.map(scan_decision_candidates_worker, scan_tasks)):
            candidates[task["scenario"], task["seed"]] = rows

    specs = {}
    for city, scenarios in FAMILIES.items():
        for seed in seeds:
            rows = sorted(
                (row for scenario in scenarios for row in candidates[scenario, seed]),
                key=lambda r: (r["elapsed_sec"], r["scenario"], r["tls_id"]),
            )
            if len(rows) < 18:
                raise RuntimeError(f"{city}/{seed}: only {len(rows)} eligible decision groups")
            for idx in np.linspace(0, len(rows) - 1, 18, dtype=int):
                row = rows[int(idx)]
                name = f"repeat{repeat}_{city}_{seed}_{row['scenario']}_{row['elapsed_sec']}_{row['tls_id']}"
                specs[name] = {**row, "group_id": name, "sumocfg": str(paths[row["scenario"]])}

    branch_tasks = [
        {**spec, "branch_idx": idx, "cand_state": state}
        for spec in specs.values()
        for idx, state in enumerate(spec["candidate_states"])
    ]
    print(f"Repeat {repeat}: {len(scan_tasks)} scans, {len(specs)} groups, {len(branch_tasks)} branches", flush=True)
    branches = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(evaluate_single_branch_worker, task) for task in branch_tasks]
        for i, fut in enumerate(as_completed(pending), 1):
            row = fut.result()
            branches.setdefault(row["group_id"], []).append(row)
            if i % 100 == 0 or i == len(pending):
                print(f"Repeat {repeat}: completed {i}/{len(pending)} branches", flush=True)

    groups = {city: [] for city in FAMILIES}
    costs = {city: {} for city in FAMILIES}
    for name, spec in specs.items():
        rows = sorted(branches[name], key=lambda r: r["branch_idx"])
        if (
            list(r["branch_idx"] for r in rows) != list(range(len(spec["candidate_states"])))
            or len({r["fingerprint"] for r in rows}) != 1
            or tuple(rows[0]["candidate_states"]) != tuple(spec["candidate_states"])
        ):
            raise RuntimeError(f"Native branches disagree before treatment: {name}")
        head = rows[0]
        action_mat = np.asarray(head["action_mat"], dtype=float)
        raw = action_mat[:, 0]
        baseline = -0.65 * (raw - raw.mean()) + 1e-9 * np.arange(len(raw))
        city = spec["domain"].removeprefix("target_")
        groups[city].append(_rec_to_dict(_dict_to_rec({
            **head,
            "base_scores": baseline.tolist(),
            "branch_costs": [r["branch_cost"] for r in rows],
        })))
        costs[city][name] = {"branches": len(rows), "prefix_seconds_per_branch": spec["elapsed_sec"]}

    ledger = {}
    for city, records in groups.items():
        groups6 = subset_residual_budget([_dict_to_rec(g) for g in records], budget_per_seed=3)
        if len(records) != 36 or len(groups6) != 6:
            raise RuntimeError(f"Incorrect nested label counts in {city}")
        b6 = {g.group_id for g in groups6}
        ledger[city] = {
            "scan_rollouts": len(FAMILIES[city]) * len(seeds),
            "scan_seconds": len(FAMILIES[city]) * len(seeds) * SCAN_SECONDS,
        }
        for budget, entries in (
            ("b6", [c for name, c in costs[city].items() if name in b6]),
            ("b36", list(costs[city].values())),
        ):
            ledger[city][budget] = {
                "groups": len(entries),
                "branches": sum(e["branches"] for e in entries),
                "native_prefix_seconds": sum(e["branches"] * e["prefix_seconds_per_branch"] for e in entries),
                "branch_seconds": sum(e["branches"] for e in entries) * BRANCH_SECONDS,
            }
    return {"seeds": seeds, "ledger": ledger, "groups": groups}


def evaluate_tasks(tasks: list[dict], workers: int) -> list[dict]:
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {
            pool.submit(eval_phase2_closed_loop_worker, task): task["adaptation_repeat"] for task in tasks
        }
        for i, fut in enumerate(as_completed(pending), 1):
            rows.append({**fut.result(), "adaptation_repeat": pending[fut]})
            if i % 36 == 0 or i == len(pending):
                print(f"Completed {i}/{len(pending)} one-hour held-out runs", flush=True)
    return sorted(rows, key=lambda r: (r["adaptation_repeat"], r["family"], r["scenario"], r["policy"], r["seed"]))


def stored_pressure_regrets(groups: dict) -> dict:
    """Score the exact pre-shield Cons-GP action stored by the branch worker."""
    output = {}
    for city, records in groups.items():
        b36 = subset_residual_budget(records, budget_per_seed=18)
        output[city] = {}
        for name, selected in (
            ("b6", subset_residual_budget(b36, budget_per_seed=3)),
            ("b36", b36),
        ):
            output[city][name] = mean(
                (g.branch_costs[g.reference_index] - g.branch_costs.min())
                / max(float(np.ptp(g.branch_costs)), 0.15)
                for g in selected
            )
    return output


def summarize(rows: list[dict], audits: dict) -> dict:
    by_repeat = {}
    for repeat in range(1, len(ADAPTATION_PAIRS) + 1):
        cities = {}
        for city, scenarios in FAMILIES.items():
            family_rows = [r for r in rows if r["family"] == city and r["adaptation_repeat"] in (0, repeat)]
            city_out = {}
            for budget in ("b6", "b36"):
                candidate = [r for r in family_rows if r["policy"] == f"cfcmt_{budget}"]
                city_out[budget] = {
                    "source_weight": audits[str(repeat)][city]["budgets"][budget]["cfcmt_omega"],
                    "source_pool": audits[str(repeat)][city]["budgets"][budget]["selected_source_pool"],
                    "target_scale": audits[str(repeat)][city]["budgets"][budget]["fold_to_full_target_scale"],
                    "candidate_means": {name: mean(r[name] for r in candidate) for name in METRICS},
                    "severe_runs": sum(r["completion_ratio"] < SEVERE_COMPLETION for r in candidate),
                    "comparisons": {},
                }
                for baseline in (f"target_only_{budget}", "conservation_gp_10"):
                    reference = {(r["scenario"], r["seed"]): r for r in family_rows if r["policy"] == baseline}
                    delta = {
                        name: [
                            mean(
                                r[name] - reference[r["scenario"], r["seed"]][name]
                                for r in candidate if r["seed"] == seed
                            )
                            for seed in TEST_SEEDS
                        ]
                        for name in METRICS
                    }
                    city_out[budget]["comparisons"][baseline] = {
                        "mean_delta": {name: mean(values) for name, values in delta.items()},
                        "median_delta": {name: median(values) for name, values in delta.items()},
                        "seed_deltas": delta,
                        "baseline_severe_runs": sum(r["completion_ratio"] < SEVERE_COMPLETION for r in reference.values()),
                    }
            cities[city] = city_out
        by_repeat[str(repeat)] = cities
    return by_repeat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--archive-labels", type=Path, required=True)
    parser.add_argument("--staged-scenarios-dir", type=Path, required=True)
    parser.add_argument("--ext-targets-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    paths = scenario_paths(args.staged_scenarios_dir, args.ext_targets_dir)
    source_groups, _, _ = read_paper_label_archive(args.archive_labels)
    audits = {}
    ledgers = {}
    pressure_regrets = {}
    model_paths = {}
    for repeat, seeds in enumerate(ADAPTATION_PAIRS, 1):
        labels_path = args.work_dir / f"adaptation_labels_{repeat}.json"
        if not labels_path.exists():
            bundle = collect_labels(paths, repeat, seeds, args.workers)
            labels_path.write_text(json.dumps(bundle), encoding="utf-8")
        else:
            bundle = json.loads(labels_path.read_text())
        if tuple(bundle["seeds"]) != seeds:
            raise RuntimeError(f"Incorrect adaptation seeds in {labels_path}")
        ledgers[str(repeat)] = bundle["ledger"]
        groups = {city: [_dict_to_rec(g) for g in rows] for city, rows in bundle["groups"].items()}
        pressure_regrets[str(repeat)] = stored_pressure_regrets(groups)
        models, audits[str(repeat)] = fit_phase2_residual_suite(
            source_groups, groups, adaptation_seeds=seeds, omega_grid=OMEGA_GRID, nested_budgets=True,
        )
        model_paths[repeat] = args.work_dir / f"independent_repeat_{repeat}_models.pkl"
        with model_paths[repeat].open("wb") as fd:
            pickle.dump(models, fd)
        print(f"Repeat {repeat}: model frozen after two-fold adaptation selection", flush=True)

    all_tasks = [
        {
            "family": city, "scenario": scenario, "sumocfg": str(paths[scenario]),
            "seed": seed, "policy": policy, "models_pkl": str(model_paths[repeat or 1]),
            "adaptation_repeat": repeat,
        }
        for city, scenarios in FAMILIES.items()
        for scenario in scenarios
        for seed in TEST_SEEDS
        for repeat in range(len(ADAPTATION_PAIRS) + 1)
        for policy in (POLICIES if repeat else ("conservation_gp_10",))
    ]
    rows = evaluate_tasks(all_tasks, args.workers)
    result = {
        "protocol": {
            "adaptation_pairs": ADAPTATION_PAIRS,
            "held_out_seeds": TEST_SEEDS,
            "source_label_count": len(source_groups),
            "nested_budgets": [6, 36],
            "omega_grid": OMEGA_GRID,
            "completion_threshold_for_severe_runs": SEVERE_COMPLETION,
            "evaluation_duration_seconds": 3600,
        },
        "adaptation_costs": ledgers,
        "reference_action_regrets": pressure_regrets,
        "fit_audit": audits,
        "summary": summarize(rows, audits),
        "rows": rows,
    }
    result_path = args.work_dir / "phase4_independent_repeat_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
