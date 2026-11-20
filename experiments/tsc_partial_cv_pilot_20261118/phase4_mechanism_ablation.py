"""Compare the original six-label fit with mechanism ablations on its test panel."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import pickle

import numpy as np

from .cfcmt_cv_model import (
    BlendedActionRanker,
    CapacityMatchedActionCostRegressor,
    DifferenceOnlyActionRanker,
    DirectActionCostRegressor,
    EndpointAntisymmetricActionRanker,
)
from .phase2_residual_transfer import (
    ADAPT_SEEDS,
    FRESH_PHASE2_SEEDS,
    OMEGA_POS_GRID,
    STRUCTURAL_SOURCE_POOLS,
    TARGET_FAMILIES,
    _dict_to_rec,
    _eval_precomputed_scores,
    _rec_to_dict,
    _to_action_group,
    eval_phase2_closed_loop_worker,
    fit_phase2_residual_suite,
    read_paper_label_archive,
    resolve_phase2_scenarios,
    subset_residual_budget,
)
from .phase3_tits_suite import (
    FRESH_SALTLAKE_SEEDS,
    collect_saltlake_adaptation_groups,
    saltlake_scenario_configs,
)

ABLATIONS = (
    "routed_source_only",
    "routed_allow_zero",
    "all_source_allow_zero",
    "no_scale_calibration",
    "difference_only",
    "action_cost_regression",
    "capacity_matched_action_cost",
)
REFERENCES = (
    "cfcmt_b6",
    "target_only_b6",
    "zero_shot_cfcmt_b0",
    "conservation_gp_10",
)
REPRESENTATIONS = (
    ("difference_only", DifferenceOnlyActionRanker),
    ("action_cost_regression", DirectActionCostRegressor),
    ("capacity_matched_action_cost", CapacityMatchedActionCostRegressor),
)


def fit_source_ranker(source_groups, pool_name, ranker_type):
    domains = set(STRUCTURAL_SOURCE_POOLS[pool_name])
    groups = [g for g in source_groups if g.domain in domains]
    weights = dict.fromkeys({g.domain for g in groups}, 1.0 / len(domains))
    return ranker_type(l2_regularization=15.0, random_state=42).fit(
        groups, domain_weights=weights
    )


def fit_blend_options(groups, source_pools, ranker_type, full_target=None):
    """Apply the original two-fold rule with zero as an eligible source weight."""
    first, second = ([g for g in groups if g.seed == seed] for seed in ADAPT_SEEDS)
    if len(first) != 3 or len(second) != 3:
        raise ValueError("Mechanism ablation requires the original 3+3 label split")
    validation = second + first
    oof = []
    for train, held in ((first, second), (second, first)):
        ranker = ranker_type(l2_regularization=15.0, random_state=42).fit(train)
        oof.extend(ranker.score_group(group.state_vec, group.action_mat) for group in held)
    if full_target is None:
        full_target = ranker_type(l2_regularization=15.0, random_state=42).fit(groups)
    full = [full_target.score_group(g.state_vec, g.action_mat) for g in validation]
    rms_oof = float(np.sqrt(np.mean(np.concatenate(oof) ** 2)))
    rms_full = float(np.sqrt(np.mean(np.concatenate(full) ** 2)))
    scale = min(1.0, rms_oof / max(rms_full, 1e-9))

    result = {}
    for name, source in source_pools.items():
        source_scores = [source.score_group(g.state_vec, g.action_mat) for g in validation]
        candidates = [
            (
                omega,
                _eval_precomputed_scores(
                    validation,
                    [(1.0 - omega) * t + omega * s for t, s in zip(oof, source_scores)],
                ),
            )
            for omega in (0.0, *OMEGA_POS_GRID)
        ]
        omega, metrics = min(
            candidates,
            key=lambda item: (
                round(item[1]["mean_normalized_regret"], 6),
                round(item[1]["smooth_boltzmann_regret"], 6),
            ),
        )
        model = BlendedActionRanker(
            target_ranker=full_target, source_ranker=source, omega=omega, target_scale=scale
        )
        result[name] = (model, metrics)
    return result


def score_identity(model):
    if model.omega <= 0.0 or model.source_ranker is None:
        return ("target", id(model.target_ranker))
    if model.omega >= 1.0 or model.target_ranker is None:
        return ("source", id(model.source_ranker))
    return (id(model.target_ranker), id(model.source_ranker), model.omega, model.target_scale)


def evaluate_ablation_cell(task):
    # Reuse the standard B=6 scoring and no-state-restore execution path.
    row = eval_phase2_closed_loop_worker({**task, "policy": "cfcmt_b6"})
    row["policy"] = task["policy"]
    return row


def read_or_reconstruct_saltlake_labels(path, archive_root, source_groups, workers):
    if path.is_file():
        return [_dict_to_rec(row) for row in json.loads(path.read_text())]
    labels = collect_saltlake_adaptation_groups(
        saltlake_scenario_configs(archive_root / "ext_targets"), workers
    )
    _, fit = fit_phase2_residual_suite(source_groups, {"saltlake": labels})
    archived = json.loads(
        (archive_root / "phase3_tits/phase3_saltlake_fit_audit.json").read_text()
    )
    if len(labels) != archived["saltlake_adapt_groups"] or fit != archived["fit_audit"]:
        raise RuntimeError("Reconstructed labels do not reproduce the archived Salt Lake fit")
    path.write_text(json.dumps([_rec_to_dict(label) for label in labels]), encoding="utf-8")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--saltlake-labels", type=Path)
    parser.add_argument("--saltlake-run-dir", type=Path, help="Salt Lake fit and matching evaluator output")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--arm", action="append", choices=ABLATIONS)
    parser.add_argument("--family", action="append", choices=(*TARGET_FAMILIES, "saltlake"))
    args = parser.parse_args()
    selected_arms = tuple(dict.fromkeys(args.arm or ABLATIONS))

    base = args.archive_root
    phase2 = base / "phase2_frozen_prereg"
    phase3 = args.saltlake_run_dir or base / "phase3_tits"
    work_dir = args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    source_residual, target, _ = read_paper_label_archive(
        phase2 / "phase2_counterfactual_dataset.json"
    )
    source = [_to_action_group(g) for g in source_residual]
    with (phase2 / "phase2_frozen_models.pkl").open("rb") as fh:
        frozen = pickle.load(fh)

    configs = resolve_phase2_scenarios(base / "phase0/staged_scenarios", base / "source_scenarios")
    configs.update(saltlake_scenario_configs(base / "ext_targets"))
    targets = {
        **TARGET_FAMILIES,
        "saltlake": ("saltlake_400s_peak", "saltlake_state_peak"),
    }
    if args.family:
        targets = {city: scenarios for city, scenarios in targets.items() if city in args.family}
    archive_paths = []
    if any(city != "saltlake" for city in targets):
        archive_paths.append(phase2 / "phase2_eval_fresh_result.json")
    if "saltlake" in targets:
        label_path = args.saltlake_labels or (
            phase3 / "phase3_saltlake_labels.json"
            if args.saltlake_run_dir else work_dir / "saltlake_original_labels.json"
        )
        if args.saltlake_run_dir and not label_path.is_file():
            raise FileNotFoundError(f"Salt Lake label file missing for the selected run: {label_path}")
        target["saltlake"] = read_or_reconstruct_saltlake_labels(
            label_path, base, source_residual, args.workers
        )
        with (phase3 / "phase3_saltlake_models.pkl").open("rb") as fh:
            frozen.update(pickle.load(fh))
        archive_paths.append(phase3 / "phase3_saltlake_eval_fresh_result.json")
    archived = [
        row
        for path in archive_paths
        for row in json.loads(path.read_text())["rows"]
        if row["family"] in targets and row["policy"] in REFERENCES
    ]
    reference = {
        (row["family"], row["scenario"], row["seed"], row["policy"]): row for row in archived
    }
    panel_cells = sum(
        len(scenarios) * len(FRESH_SALTLAKE_SEEDS if city == "saltlake" else FRESH_PHASE2_SEEDS)
        for city, scenarios in targets.items()
    )
    if len(reference) != panel_cells * len(REFERENCES):
        raise ValueError("The archive does not contain the complete reference panel")

    models = {}
    audit = {}
    reused = []
    scheduled = []
    aliases = {}
    represented_sources = {}
    for city, scenarios in targets.items():
        selected = subset_residual_budget(target[city], budget_per_seed=3)
        groups = [_to_action_group(g) for g in selected]
        original = frozen[city]["cfcmt_b6"]
        source_ranker = original.source_ranker
        target_ranker = original.target_ranker
        route = "real_cities" if min(len(g.candidate_states) for g in groups) < 6 else "multi_phase_grids"
        all_source = frozen[city]["pool_all_b36"].source_ranker
        fitted = fit_blend_options(
            groups,
            {"routed_allow_zero": source_ranker, "all_source_allow_zero": all_source},
            EndpointAntisymmetricActionRanker,
            full_target=target_ranker,
        )
        fitted["routed_source_only"] = (
            BlendedActionRanker(target_ranker=None, source_ranker=source_ranker, omega=1.0),
            None,
        )
        fitted["no_scale_calibration"] = (
            BlendedActionRanker(
                target_ranker=target_ranker, source_ranker=source_ranker, omega=original.omega
            ),
            None,
        )
        for name, ranker_type in REPRESENTATIONS:
            if name not in selected_arms:
                continue
            if (name, route) not in represented_sources:
                represented_sources[name, route] = fit_source_ranker(source, route, ranker_type)
            fitted.update(
                fit_blend_options(
                    groups, {name: represented_sources[name, route]}, ranker_type
                )
            )

        audit[city] = {
            "selected_label_ids": [g.group_id for g in selected],
            "structural_source_pool": route,
            "original": {"omega": original.omega, "target_scale": original.target_scale},
            "arms": {},
        }
        seeds = FRESH_SALTLAKE_SEEDS if city == "saltlake" else FRESH_PHASE2_SEEDS
        available = {
            score_identity(frozen[city][policy]): policy
            for policy in ("zero_shot_cfcmt_b0", "target_only_b6", "cfcmt_b6")
        }
        for policy in selected_arms:
            model, offline = fitted[policy]
            same_as = available.get(score_identity(model))
            audit[city]["arms"][policy] = {
                "omega": model.omega,
                "target_scale": model.target_scale,
                "oof": offline,
                "identical_to": same_as,
            }
            if same_as in REFERENCES:
                reused.extend(
                    {**reference[city, scenario, seed, same_as], "policy": policy}
                    for scenario in scenarios for seed in seeds
                )
                continue
            if same_as is not None:
                aliases.setdefault((city, same_as), []).append(policy)
                continue
            available[score_identity(model)] = policy
            models.setdefault(policy, {})[city] = {"cfcmt_b6": model}
            scheduled.extend(
                {
                    "family": city,
                    "scenario": scenario,
                    "sumocfg": str(configs[scenario]),
                    "seed": seed,
                    "policy": policy,
                    "models_pkl": str(work_dir / f"{policy}.pkl"),
                }
                for scenario in scenarios for seed in seeds
            )

    for policy, model in models.items():
        with (work_dir / f"{policy}.pkl").open("wb") as fh:
            pickle.dump(model, fh)
    (work_dir / "fit_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Evaluating {len(scheduled)} unique test cells; reused {len(reused)} archived cells", flush=True)
    rows = archived + reused
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(evaluate_ablation_cell, task) for task in scheduled]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            rows.extend(
                {**row, "policy": policy}
                for policy in aliases.get((row["family"], row["policy"]), ())
            )
            if completed % 30 == 0 or completed == len(futures):
                print(f"Ablation evaluation {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["family"], row["scenario"], row["seed"], row["policy"]))
    evaluated = {(r["scenario"], r["seed"], r["policy"]) for r in rows}
    if len(evaluated) != panel_cells * (len(REFERENCES) + len(selected_arms)):
        raise RuntimeError("The ablation panel is incomplete")
    (work_dir / "phase4_mechanism_ablation_result.json").write_text(
        json.dumps(
            {"new_evaluations": len(scheduled), "fit_audit": audit, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
