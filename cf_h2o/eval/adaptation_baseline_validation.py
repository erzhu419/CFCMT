"""Few-shot adaptation baselines for cross-city residual transfer.

The AdaRL and FANsRL reference codebases are environment-specific RL systems
(CartPole/Atari and MuJoCo-style continuous-control training loops).  This
script implements same-information bus residual baselines that match their
adaptation ideas:

* AdaRL-style selected-mechanism refit: use target calibration labels only to
  decide which mechanism groups changed, then refit only those groups.
* FANsRL-style context bias: adapt a low-dimensional per-output context bias
  on top of the source mechanism model.

Both baselines use the same target route budget as the CFCMT few-shot
adaptation sweep and evaluate on held-out target routes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cf_h2o.eval.paper_experiment_suite import (
    OUTPUT_NAMES,
    LinearStats,
    ResidualStats,
    _family_sse,
    _fit_cfcmt,
    _fit_h2o,
    _fit_safe_cfcmt,
    _fit_safe_h2o,
    _merge_stats,
    _merge_stats_weighted,
    _method_metrics_from_sse,
    _output_group,
    _parse_float_list,
    _read_json,
    _repo_root,
    _resolve_path,
    _source_similarity_weights,
    _split_target_lines,
    _strict_leave_one_city_out_splits,
    build_all_city_stats,
)


def _clone_family_beta(beta: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value, dtype=np.float64).reshape(-1).copy() for name, value in beta.items()}


def _intercept_bias(stat: LinearStats, beta: np.ndarray) -> float:
    if stat.n <= 0:
        return 0.0
    coef = np.asarray(beta, dtype=np.float64).reshape(-1)
    residual_sum = float(stat.xty[0, 0] - stat.xtx[0, :] @ coef)
    return residual_sum / float(stat.n)


def _family_context_bias_beta(
    source_beta: dict[str, np.ndarray],
    calibration_family: dict[str, LinearStats],
    *,
    allowed_groups: set[str] | None = None,
) -> dict[str, np.ndarray]:
    out = _clone_family_beta(source_beta)
    for output_name in OUTPUT_NAMES:
        group = _output_group(output_name)
        if allowed_groups is not None and group not in allowed_groups:
            continue
        out[output_name][0] += _intercept_bias(calibration_family[output_name], out[output_name])
    return out


def _select_changed_mechanism_groups(
    source_beta: dict[str, np.ndarray],
    calibration_family: dict[str, LinearStats],
    *,
    min_relative_improvement: float,
) -> dict[str, Any]:
    selected: list[str] = []
    diagnostics: dict[str, dict[str, float | bool]] = {}
    for group in sorted({_output_group(name) for name in OUTPUT_NAMES}):
        outputs = [name for name in OUTPUT_NAMES if _output_group(name) == group]
        source_sse = 0.0
        bias_sse = 0.0
        bias_beta = _family_context_bias_beta(source_beta, calibration_family, allowed_groups={group})
        for output_name in outputs:
            stat = calibration_family[output_name]
            source_sse += float(stat.sse(source_beta[output_name])[0])
            bias_sse += float(stat.sse(bias_beta[output_name])[0])
        if source_sse <= 1e-12:
            ratio = 1.0
            changed = False
        else:
            ratio = bias_sse / source_sse
            changed = ratio < (1.0 - float(min_relative_improvement))
        if changed:
            selected.append(group)
        diagnostics[group] = {
            "source_sse": source_sse,
            "context_bias_sse": bias_sse,
            "context_bias_vs_source_ratio": ratio,
            "selected": bool(changed),
        }
    return {"selected_groups": selected, "diagnostics": diagnostics}


def _selected_refit_beta(
    source_beta: dict[str, np.ndarray],
    refit_beta: dict[str, np.ndarray],
    selected_groups: set[str],
) -> dict[str, np.ndarray]:
    out = _clone_family_beta(source_beta)
    for output_name in OUTPUT_NAMES:
        if _output_group(output_name) in selected_groups:
            out[output_name] = np.asarray(refit_beta[output_name], dtype=np.float64).reshape(-1).copy()
    return out


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or abs(float(den)) <= 1e-12:
        return None
    return float(num) / float(den)


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(valid)) if valid else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    config = _read_json(_resolve_path(root, args.config))
    t0 = time.time()
    city_stats, city_line_stats, sanity = build_all_city_stats(config, root, args)

    rows: list[dict[str, Any]] = []
    for split in _strict_leave_one_city_out_splits(config):
        sources = list(split["source_envs"])
        target = split["target_env"]
        target_city = config["generated_envs"][target].get("city", target)
        source_stats = _merge_stats("source_unweighted", "source", [city_stats[key] for key in sources])
        source_weights = _source_similarity_weights(
            sanity,
            target,
            sources,
            temperature=args.source_weight_temperature,
            floor=args.source_weight_floor,
        )
        weighted_source_stats = _merge_stats_weighted("source_weighted", "source", city_stats, source_weights)
        source_h2o_beta = _fit_h2o(source_stats, args.ridge)
        weighted_source_cfcmt_beta = _fit_cfcmt(weighted_source_stats, args.ridge)

        for budget in args.calibration_budgets:
            calibration_lines, evaluation_lines, in_sample_oracle = _split_target_lines(
                city_line_stats[target],
                fraction=float(budget),
                seed=args.seed,
            )
            calibration_stats = _merge_stats(
                f"{target}::calibration_{budget:g}",
                target,
                calibration_lines,
            )
            evaluation_stats = (
                city_stats[target]
                if in_sample_oracle
                else _merge_stats(f"{target}::evaluation_{budget:g}", target, evaluation_lines)
            )
            if evaluation_stats.n <= 0:
                continue

            source_plus_target = _merge_stats(
                f"{target}::source_plus_calibration_{budget:g}",
                target,
                [source_stats, calibration_stats],
            )
            weighted_source_plus_target = _merge_stats(
                f"{target}::weighted_source_plus_calibration_{budget:g}",
                target,
                [weighted_source_stats, calibration_stats],
            )

            h2o_adapt_beta = _fit_safe_h2o(source_plus_target, args.ridge)
            cfcmt_weighted_adapt_beta = _fit_safe_cfcmt(weighted_source_plus_target, args.ridge)
            target_only_cfcmt_beta = _fit_safe_cfcmt(calibration_stats, args.ridge)

            selected = _select_changed_mechanism_groups(
                weighted_source_cfcmt_beta,
                calibration_stats.cfcmt,
                min_relative_improvement=args.adarl_min_relative_improvement,
            )
            selected_groups = set(selected["selected_groups"])
            adarl_refit_beta = _selected_refit_beta(
                weighted_source_cfcmt_beta,
                cfcmt_weighted_adapt_beta or weighted_source_cfcmt_beta,
                selected_groups,
            )
            fansrl_bias_beta = _family_context_bias_beta(
                weighted_source_cfcmt_beta,
                calibration_stats.cfcmt,
            )
            adarl_bias_beta = _family_context_bias_beta(
                weighted_source_cfcmt_beta,
                calibration_stats.cfcmt,
                allowed_groups=selected_groups,
            )

            uncal_sse = evaluation_stats.h2o.sse(None)
            h2o_source_sse = evaluation_stats.h2o.sse(source_h2o_beta)
            h2o_adapt_sse = evaluation_stats.h2o.sse(h2o_adapt_beta) if h2o_adapt_beta is not None else None
            cfcmt_source_sse = _family_sse(evaluation_stats.cfcmt, weighted_source_cfcmt_beta)
            cfcmt_adapt_sse = (
                _family_sse(evaluation_stats.cfcmt, cfcmt_weighted_adapt_beta)
                if cfcmt_weighted_adapt_beta is not None
                else None
            )
            target_only_sse = (
                _family_sse(evaluation_stats.cfcmt, target_only_cfcmt_beta)
                if target_only_cfcmt_beta is not None
                else None
            )
            adarl_refit_sse = _family_sse(evaluation_stats.cfcmt, adarl_refit_beta)
            adarl_bias_sse = _family_sse(evaluation_stats.cfcmt, adarl_bias_beta)
            fansrl_bias_sse = _family_sse(evaluation_stats.cfcmt, fansrl_bias_beta)

            metrics = {
                "uncalibrated": _method_metrics_from_sse(uncal_sse, evaluation_stats.n),
                "h2oplus_source_only": _method_metrics_from_sse(h2o_source_sse, evaluation_stats.n),
                "h2oplus_source_plus_target_budget": (
                    _method_metrics_from_sse(h2o_adapt_sse, evaluation_stats.n)
                    if h2o_adapt_sse is not None
                    else None
                ),
                "cfcmt_weighted_source_only": _method_metrics_from_sse(cfcmt_source_sse, evaluation_stats.n),
                "cfcmt_weighted_source_plus_target_budget": (
                    _method_metrics_from_sse(cfcmt_adapt_sse, evaluation_stats.n)
                    if cfcmt_adapt_sse is not None
                    else None
                ),
                "cfcmt_target_only_budget": (
                    _method_metrics_from_sse(target_only_sse, evaluation_stats.n)
                    if target_only_sse is not None
                    else None
                ),
                "adarl_style_selected_refit": _method_metrics_from_sse(adarl_refit_sse, evaluation_stats.n),
                "adarl_style_selected_context_bias": _method_metrics_from_sse(adarl_bias_sse, evaluation_stats.n),
                "fansrl_style_context_bias": _method_metrics_from_sse(fansrl_bias_sse, evaluation_stats.n),
            }
            h2o_adapt_total = metrics["h2oplus_source_plus_target_budget"]["total_mse"]
            ours_total = metrics["cfcmt_weighted_source_plus_target_budget"]["total_mse"]
            adarl_total = metrics["adarl_style_selected_refit"]["total_mse"]
            fansrl_total = metrics["fansrl_style_context_bias"]["total_mse"]
            rows.append(
                {
                    **split,
                    "target_city": target_city,
                    "source_weights": source_weights,
                    "target_line_budget_fraction": float(budget),
                    "calibration_lines": len(calibration_lines),
                    "evaluation_lines": len(evaluation_lines) if not in_sample_oracle else len(city_line_stats[target]),
                    "calibration_transitions": calibration_stats.n,
                    "evaluation_transitions": evaluation_stats.n,
                    "in_sample_oracle": bool(in_sample_oracle),
                    "selected_mechanism_groups": sorted(selected_groups),
                    "selection_diagnostics": selected["diagnostics"],
                    "metrics": metrics,
                    "comparisons": {
                        "ours_vs_h2oplus_adapted_ratio": _safe_ratio(ours_total, h2o_adapt_total),
                        "adarl_selected_refit_vs_h2oplus_adapted_ratio": _safe_ratio(adarl_total, h2o_adapt_total),
                        "fansrl_context_bias_vs_h2oplus_adapted_ratio": _safe_ratio(fansrl_total, h2o_adapt_total),
                        "ours_vs_adarl_selected_refit_ratio": _safe_ratio(ours_total, adarl_total),
                        "ours_vs_fansrl_context_bias_ratio": _safe_ratio(ours_total, fansrl_total),
                        "ours_beats_h2oplus_adapted": bool(ours_total < h2o_adapt_total),
                        "adarl_selected_refit_beats_h2oplus_adapted": bool(adarl_total < h2o_adapt_total),
                        "fansrl_context_bias_beats_h2oplus_adapted": bool(fansrl_total < h2o_adapt_total),
                        "ours_beats_adarl_selected_refit": bool(ours_total < adarl_total),
                        "ours_beats_fansrl_context_bias": bool(ours_total < fansrl_total),
                    },
                }
            )

    summary_by_budget: dict[str, Any] = {}
    for budget in sorted({row["target_line_budget_fraction"] for row in rows}):
        group = [row for row in rows if math.isclose(row["target_line_budget_fraction"], budget)]
        summary_by_budget[f"{budget:g}"] = {
            "splits": len(group),
            "mean_calibration_lines": _mean([row["calibration_lines"] for row in group]),
            "mean_calibration_transitions": _mean([row["calibration_transitions"] for row in group]),
            "mean_ours_vs_h2oplus_adapted_ratio": _mean(
                [row["comparisons"]["ours_vs_h2oplus_adapted_ratio"] for row in group]
            ),
            "mean_adarl_selected_refit_vs_h2oplus_adapted_ratio": _mean(
                [row["comparisons"]["adarl_selected_refit_vs_h2oplus_adapted_ratio"] for row in group]
            ),
            "mean_fansrl_context_bias_vs_h2oplus_adapted_ratio": _mean(
                [row["comparisons"]["fansrl_context_bias_vs_h2oplus_adapted_ratio"] for row in group]
            ),
            "mean_ours_vs_adarl_selected_refit_ratio": _mean(
                [row["comparisons"]["ours_vs_adarl_selected_refit_ratio"] for row in group]
            ),
            "mean_ours_vs_fansrl_context_bias_ratio": _mean(
                [row["comparisons"]["ours_vs_fansrl_context_bias_ratio"] for row in group]
            ),
            "ours_wins_vs_h2oplus_adapted": sum(
                1 for row in group if row["comparisons"]["ours_beats_h2oplus_adapted"]
            ),
            "adarl_selected_refit_wins_vs_h2oplus_adapted": sum(
                1 for row in group if row["comparisons"]["adarl_selected_refit_beats_h2oplus_adapted"]
            ),
            "fansrl_context_bias_wins_vs_h2oplus_adapted": sum(
                1 for row in group if row["comparisons"]["fansrl_context_bias_beats_h2oplus_adapted"]
            ),
            "ours_wins_vs_adarl_selected_refit": sum(
                1 for row in group if row["comparisons"]["ours_beats_adarl_selected_refit"]
            ),
            "ours_wins_vs_fansrl_context_bias": sum(
                1 for row in group if row["comparisons"]["ours_beats_fansrl_context_bias"]
            ),
            "contains_in_sample_oracle": any(row["in_sample_oracle"] for row in group),
        }

    return {
        "ok": True,
        "experiment": "few_shot_adaptation_baseline_validation",
        "definition": (
            "Same-information target-route budget comparison against AdaRL-style "
            "selected-mechanism refit and FANsRL-style context-bias adaptation."
        ),
        "elapsed_sec": time.time() - t0,
        "config": str(_resolve_path(root, args.config)),
        "ridge": args.ridge,
        "max_lines_per_city": args.max_lines_per_city,
        "budget_fractions": [float(value) for value in args.calibration_budgets],
        "adarl_min_relative_improvement": args.adarl_min_relative_improvement,
        "rows": rows,
        "summary_by_budget": summary_by_budget,
        "summary": {
            "strict_splits": len(_strict_leave_one_city_out_splits(config)),
            "budgets": [float(value) for value in args.calibration_budgets],
            "non_oracle_budget_count": sum(1 for value in args.calibration_budgets if float(value) < 1.0),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("cf_h2o/config/cross_city_open_transit.json"))
    parser.add_argument("--out", type=Path, default=Path("cf_h2o/results/adaptation_baseline_validation.json"))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--actions", type=_parse_float_list, default=[0.0, 30.0])
    parser.add_argument("--max-lines-per-city", type=int, default=0, help="0 means all lines; >0 for smoke")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--calibration-budgets", type=_parse_float_list, default=[0.0, 0.01, 0.05, 0.10, 0.25, 1.0])
    parser.add_argument("--source-weight-temperature", type=float, default=1.0)
    parser.add_argument("--source-weight-floor", type=float, default=0.05)
    parser.add_argument("--adarl-min-relative-improvement", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Reuse the paper-suite worker contract exactly.
    args = SimpleNamespace(**vars(args))
    result = run(args)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.quiet:
        compact = {
            "ok": result["ok"],
            "elapsed_sec": result["elapsed_sec"],
            "summary_by_budget": result["summary_by_budget"],
        }
        print(json.dumps(compact, indent=2))
    else:
        print(text)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
