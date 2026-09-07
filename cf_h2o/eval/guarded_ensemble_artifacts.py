"""Generate guarded source-ensemble paper artifacts from validation JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cf_h2o.eval.full_module_pipeline_validation import _select_rigid_first_delta_candidate
from cf_h2o.eval.paper_artifacts import _write_table


STRICT_PREFIX = "leave_one_city_out_all::"
BASELINE_METHODS = (
    "h2oplus_dense_ridge",
    "h2oplus_source_weighted_dense_ridge",
    "cfcmt_ridge_template",
    "rigid_first_source_gated_delta",
    "rigid_first_mechanism_source_gated_delta",
)
FEATURE_TIE_BREAK = {
    "sim_only": 0,
    "obs_sim": 1,
    "obs_only": 2,
    "full": 3,
    "local_only": 4,
    "uniform": 5,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _city_name(env_key: str) -> str:
    return env_key.replace("_all", "").replace("_", " ")


def _method_summary(result: dict[str, Any], method: str) -> dict[str, Any]:
    return result.get("summary", {}).get("methods", {}).get(method, {})


def _method_mse(split: dict[str, Any], method: str) -> float | None:
    metric = split.get("methods", {}).get(method)
    return None if metric is None else float(metric["total_mse"])


def _candidate_to_method(candidate: str) -> str:
    return "cfcmt_ridge_template" if candidate == "rigid_cfcmt" else f"rigid_first_{candidate}"


def _selected_candidate(split: dict[str, Any]) -> str:
    selector = split["module_diagnostics"]["source_leave_one_selector"]["rigid_first_delta_selector"]
    return str(selector.get("selected_candidate", "rigid_cfcmt"))


def _source_selector(split: dict[str, Any]) -> dict[str, Any]:
    return split["module_diagnostics"]["source_leave_one_selector"]


def _target_similarity(split: dict[str, Any]) -> dict[str, Any]:
    return _source_selector(split).get("target_similarity", {})


def _summary_row(result: dict[str, Any], method: str, label: str, *, source: str) -> dict[str, Any]:
    summary = _method_summary(result, method)
    return {
        "method": label,
        "result_source": source,
        "splits": int(result["summary"]["splits"]) if summary else 0,
        "mean_total_mse": summary.get("mean_total_mse"),
        "mean_ratio_vs_h2oplus": summary.get("mean_vs_h2oplus_dense_ridge"),
        "wins_vs_h2oplus": summary.get("wins_vs_h2oplus_dense_ridge"),
    }


def build_guarded_ensemble_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split in result["splits"]:
        selected = _selected_candidate(split)
        target_similarity = _target_similarity(split)
        h2o = _method_mse(split, "h2oplus_dense_ridge")
        rigid = _method_mse(split, "cfcmt_ridge_template")
        selected_mse = _method_mse(split, "rigid_first_source_gated_delta")
        rows.append(
            {
                "split": split["name"],
                "target_city": _city_name(split["target_env"]),
                "selected_candidate": selected,
                "selected_is_ensemble": selected.startswith("ensemble_"),
                "max_source_weight": target_similarity.get("ensemble_max_source_weight"),
                "h2oplus_mse": h2o,
                "rigid_cfcmt_mse": rigid,
                "guarded_mse": selected_mse,
                "guarded_ratio_vs_h2oplus": selected_mse / h2o if h2o else None,
                "guarded_ratio_vs_rigid": selected_mse / rigid if rigid else None,
            }
        )
    return pd.DataFrame(rows)


def build_selector_ablation_table(
    *,
    conservative: dict[str, Any] | None,
    unguarded: dict[str, Any] | None,
    guarded: dict[str, Any],
) -> pd.DataFrame:
    rows = [
        _summary_row(guarded, "h2oplus_dense_ridge", "H2O+ dense ridge", source="guarded run"),
        _summary_row(
            guarded,
            "h2oplus_source_weighted_dense_ridge",
            "H2O+ dense source ensemble, same unlabeled target summary",
            source="guarded run",
        ),
        _summary_row(guarded, "cfcmt_ridge_template", "Rigid CFCMT", source="guarded run"),
    ]
    if conservative is not None:
        rows.append(
            _summary_row(
                conservative,
                "rigid_first_source_gated_delta",
                "Rigid-first pooled delta selector",
                source="non-ensemble selector run",
            )
        )
    if unguarded is not None:
        rows.append(
            _summary_row(
                unguarded,
                "rigid_first_source_gated_delta",
                "Source-city ensemble selector, unguarded",
                source="unguarded ensemble run",
            )
        )
    rows.append(
        _summary_row(
            guarded,
            "rigid_first_source_gated_delta",
            "Source-city ensemble selector, guarded",
            source="guarded ensemble run",
        )
    )
    rows.append(
        _summary_row(
            guarded,
            "rigid_first_mechanism_source_gated_delta",
            "Per-mechanism splice selector",
            source="guarded ensemble run",
        )
    )
    return pd.DataFrame([row for row in rows if row["splits"]])


def build_same_information_baseline_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    labels = {
        "h2oplus_dense_ridge": "H2O+ pooled dense residual",
        "h2oplus_source_weighted_dense_ridge": "H2O+ source-weighted dense residual",
        "cfcmt_ridge_template": "Rigid CFCMT",
        "rigid_first_source_gated_delta": "Guarded CFCMT source ensemble",
    }
    for method, label in labels.items():
        summary = _method_summary(result, method)
        if not summary:
            continue
        rows.append(
            {
                "method": label,
                "uses_unlabeled_target_summary": method in {
                    "h2oplus_source_weighted_dense_ridge",
                    "rigid_first_source_gated_delta",
                },
                "uses_target_transition_labels_for_selection": False,
                "mean_total_mse": summary["mean_total_mse"],
                "mean_ratio_vs_h2oplus": summary["mean_vs_h2oplus_dense_ridge"],
                "wins_vs_h2oplus": summary["wins_vs_h2oplus_dense_ridge"],
            }
        )
    return pd.DataFrame(rows)


def build_strict_loo_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split in result["splits"]:
        if not str(split["name"]).startswith(STRICT_PREFIX):
            continue
        h2o = _method_mse(split, "h2oplus_dense_ridge")
        rigid = _method_mse(split, "cfcmt_ridge_template")
        selected = _method_mse(split, "rigid_first_source_gated_delta")
        rows.append(
            {
                "target_city": _city_name(split["target_env"]),
                "source_cities": ", ".join(_city_name(key) for key in split["source_envs"]),
                "selected_candidate": _selected_candidate(split),
                "h2oplus_mse": h2o,
                "rigid_cfcmt_mse": rigid,
                "guarded_mse": selected,
                "guarded_ratio_vs_h2oplus": selected / h2o if h2o else None,
                "guarded_ratio_vs_rigid": selected / rigid if rigid else None,
                "max_source_weight": _target_similarity(split).get("ensemble_max_source_weight"),
            }
        )
    if rows:
        h2o_values = [float(row["h2oplus_mse"]) for row in rows]
        rigid_values = [float(row["rigid_cfcmt_mse"]) for row in rows]
        selected_values = [float(row["guarded_mse"]) for row in rows]
        rows.append(
            {
                "target_city": "mean",
                "source_cities": "",
                "selected_candidate": "",
                "h2oplus_mse": float(np.mean(h2o_values)),
                "rigid_cfcmt_mse": float(np.mean(rigid_values)),
                "guarded_mse": float(np.mean(selected_values)),
                "guarded_ratio_vs_h2oplus": float(np.mean([row["guarded_ratio_vs_h2oplus"] for row in rows])),
                "guarded_ratio_vs_rigid": float(np.mean([row["guarded_ratio_vs_rigid"] for row in rows])),
                "max_source_weight": "",
            }
        )
    return pd.DataFrame(rows)


def _reselect_split(
    split: dict[str, Any],
    *,
    min_win_rate: float,
    min_unweighted_win_rate: float,
    min_improvement: float,
    ensemble_min_win_rate: float,
    ensemble_min_unweighted_win_rate: float,
    ensemble_min_max_source_weight: float,
    weighting: str,
) -> dict[str, Any]:
    selector = _source_selector(split)
    max_source_weight = float(selector.get("target_similarity", {}).get("ensemble_max_source_weight", 0.0))
    selected = _select_rigid_first_delta_candidate(
        selector["folds"],
        min_win_rate=min_win_rate,
        min_improvement=min_improvement,
        min_unweighted_win_rate=min_unweighted_win_rate,
        weighting=weighting,
        ensemble_min_win_rate=ensemble_min_win_rate,
        ensemble_min_unweighted_win_rate=ensemble_min_unweighted_win_rate,
        ensemble_min_max_source_weight=ensemble_min_max_source_weight,
        ensemble_max_source_weight=max_source_weight,
    )
    method = _candidate_to_method(str(selected["selected_candidate"]))
    return {
        "selected": selected,
        "method": method,
        "mse": _method_mse(split, method),
        "h2o": _method_mse(split, "h2oplus_dense_ridge"),
    }


def _aggregate_selected(result: dict[str, Any], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mses = [float(row["mse"]) for row in selected_rows if row["mse"] is not None]
    ratios = [
        float(row["mse"]) / float(row["h2o"])
        for row in selected_rows
        if row["mse"] is not None and row["h2o"] not in {None, 0}
    ]
    selected_candidates = [str(row["selected"]["selected_candidate"]) for row in selected_rows]
    return {
        "splits": len(selected_rows),
        "mean_total_mse": float(np.mean(mses)) if mses else None,
        "mean_ratio_vs_h2oplus": float(np.mean(ratios)) if ratios else None,
        "wins_vs_h2oplus": int(sum(value < 1.0 for value in ratios)),
        "ensemble_selected_splits": int(sum(name.startswith("ensemble_") for name in selected_candidates)),
        "rigid_fallback_splits": int(sum(name == "rigid_cfcmt" for name in selected_candidates)),
    }


def build_threshold_sensitivity_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for min_max_weight in (0.0, 0.4, 0.5, 0.6):
        for ensemble_win_rate in (2.0 / 3.0, 0.8, 1.0):
            selected_rows = [
                _reselect_split(
                    split,
                    min_win_rate=1.0,
                    min_unweighted_win_rate=1.0,
                    min_improvement=0.005,
                    ensemble_min_win_rate=ensemble_win_rate,
                    ensemble_min_unweighted_win_rate=2.0 / 3.0,
                    ensemble_min_max_source_weight=min_max_weight,
                    weighting="similarity",
                )
                for split in result["splits"]
            ]
            summary = _aggregate_selected(result, selected_rows)
            rows.append(
                {
                    "ensemble_min_win_rate": ensemble_win_rate,
                    "ensemble_min_max_source_weight": min_max_weight,
                    **summary,
                }
            )
    return pd.DataFrame(rows).sort_values(["mean_total_mse", "ensemble_min_win_rate"])


def _candidate_method_names(split: dict[str, Any], *, include_ensemble: bool) -> list[str]:
    names = ["cfcmt_ridge_template"]
    for name in split["methods"]:
        if not name.startswith("rigid_first_"):
            continue
        if name in {"rigid_first_source_gated_delta", "rigid_first_mechanism_source_gated_delta"}:
            continue
        if not include_ensemble and name.startswith("rigid_first_ensemble_"):
            continue
        names.append(name)
    return names


def _oracle(split: dict[str, Any], *, include_ensemble: bool) -> tuple[str, float]:
    names = _candidate_method_names(split, include_ensemble=include_ensemble)
    scored = [(name, float(split["methods"][name]["total_mse"])) for name in names if name in split["methods"]]
    return min(scored, key=lambda item: item[1])


def build_oracle_gap_table(result: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for split in result["splits"]:
        selected_mse = _method_mse(split, "rigid_first_source_gated_delta")
        current_oracle_method, current_oracle_mse = _oracle(split, include_ensemble=True)
        non_ensemble_oracle_method, non_ensemble_oracle_mse = _oracle(split, include_ensemble=False)
        rows.append(
            {
                "split": split["name"],
                "selected_candidate": _selected_candidate(split),
                "selected_mse": selected_mse,
                "candidate_oracle_method": current_oracle_method,
                "candidate_oracle_mse": current_oracle_mse,
                "selected_over_oracle": selected_mse / current_oracle_mse if current_oracle_mse else None,
                "non_ensemble_oracle_method": non_ensemble_oracle_method,
                "non_ensemble_oracle_mse": non_ensemble_oracle_mse,
                "selected_over_non_ensemble_oracle": selected_mse / non_ensemble_oracle_mse
                if non_ensemble_oracle_mse
                else None,
            }
        )
    rows.append(
        {
            "split": "mean",
            "selected_candidate": "",
            "selected_mse": float(np.mean([row["selected_mse"] for row in rows])),
            "candidate_oracle_method": "",
            "candidate_oracle_mse": float(np.mean([row["candidate_oracle_mse"] for row in rows])),
            "selected_over_oracle": float(np.mean([row["selected_over_oracle"] for row in rows])),
            "non_ensemble_oracle_method": "",
            "non_ensemble_oracle_mse": float(np.mean([row["non_ensemble_oracle_mse"] for row in rows])),
            "selected_over_non_ensemble_oracle": float(
                np.mean([row["selected_over_non_ensemble_oracle"] for row in rows])
            ),
        }
    )
    return pd.DataFrame(rows)


def _family_from_method(method: str) -> str | None:
    if not method.startswith("rigid_first_"):
        return None
    if method in {"rigid_first_source_gated_delta", "rigid_first_mechanism_source_gated_delta"}:
        return None
    stem = method.removeprefix("rigid_first_")
    if "_alpha_" not in stem:
        return None
    return stem.split("_alpha_", 1)[0]


def build_candidate_family_table(result: dict[str, Any]) -> pd.DataFrame:
    families = sorted(
        {
            family
            for split in result["splits"]
            for method in split["methods"]
            for family in [_family_from_method(method)]
            if family is not None
        }
    )
    rows = []
    for family in families:
        methods = sorted(
            {
                method
                for split in result["splits"]
                for method in split["methods"]
                if _family_from_method(method) == family
            }
        )
        best_method = None
        best_mean = math.inf
        best_ratios: list[float] = []
        for method in methods:
            mses = [_method_mse(split, method) for split in result["splits"]]
            h2os = [_method_mse(split, "h2oplus_dense_ridge") for split in result["splits"]]
            if any(value is None for value in mses):
                continue
            ratios = [float(mse) / float(h2o) for mse, h2o in zip(mses, h2os) if h2o]
            mean_mse = float(np.mean([float(value) for value in mses]))
            if mean_mse < best_mean:
                best_mean = mean_mse
                best_method = method
                best_ratios = ratios
        rows.append(
            {
                "candidate_family": family,
                "target_oracle_best_method": best_method,
                "mean_total_mse": best_mean,
                "mean_ratio_vs_h2oplus": float(np.mean(best_ratios)) if best_ratios else None,
                "wins_vs_h2oplus": int(sum(value < 1.0 for value in best_ratios)),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_total_mse")


def build_label_audit_table(result: dict[str, Any]) -> pd.DataFrame:
    replay_ok = True
    for split in result["splits"]:
        selector = _source_selector(split)["rigid_first_delta_selector"]
        replay = _reselect_split(
            split,
            min_win_rate=float(selector["min_win_rate"]),
            min_unweighted_win_rate=float(selector["min_unweighted_win_rate"]),
            min_improvement=float(selector["min_improvement"]),
            ensemble_min_win_rate=float(selector["ensemble_min_win_rate"]),
            ensemble_min_unweighted_win_rate=float(selector["ensemble_min_unweighted_win_rate"]),
            ensemble_min_max_source_weight=float(selector["ensemble_min_max_source_weight"]),
            weighting=str(selector["weighting"]),
        )
        replay_ok = replay_ok and replay["selected"]["selected_candidate"] == selector["selected_candidate"]
    rows = [
        {
            "component": "source residual fitting",
            "uses_source_labels": "yes",
            "uses_unlabeled_target": "no",
            "uses_target_transition_labels": "no",
            "audit_status": "source-city supervised residual only",
        },
        {
            "component": "target similarity weights",
            "uses_source_labels": "no",
            "uses_unlabeled_target": "yes",
            "uses_target_transition_labels": "no",
            "audit_status": "obs/action, simulator output, static local descriptors",
        },
        {
            "component": "feature-set selector",
            "uses_source_labels": "yes",
            "uses_unlabeled_target": "no",
            "uses_target_transition_labels": "no",
            "audit_status": "leave-one-source score before target evaluation",
        },
        {
            "component": "leave-one-source selector",
            "uses_source_labels": "yes",
            "uses_unlabeled_target": "yes",
            "uses_target_transition_labels": "no",
            "audit_status": "replay matched" if replay_ok else "replay mismatch",
        },
        {
            "component": "ensemble dominance guard",
            "uses_source_labels": "no",
            "uses_unlabeled_target": "yes",
            "uses_target_transition_labels": "no",
            "audit_status": "max source weight threshold only",
        },
        {
            "component": "reported target metrics",
            "uses_source_labels": "no",
            "uses_unlabeled_target": "no",
            "uses_target_transition_labels": "yes",
            "audit_status": "evaluation only; not fed back to selector",
        },
    ]
    return pd.DataFrame(rows)


def _source_feature_score(split: dict[str, Any]) -> tuple[float, str]:
    selector = _source_selector(split)["rigid_first_delta_selector"]
    candidate = str(selector.get("selected_candidate", "rigid_cfcmt"))
    if candidate == "rigid_cfcmt":
        return 1.0, candidate
    summary = selector.get("candidate_summaries", {}).get(candidate, {})
    return float(summary.get("mean_vs_rigid_cfcmt", 1.0)), candidate


def _feature_row(label: str, result: dict[str, Any], *, source: str) -> dict[str, Any]:
    args = result.get("args", {})
    summary = _method_summary(result, "rigid_first_source_gated_delta")
    selected = [_selected_candidate(split) for split in result["splits"]]
    return {
        "configuration": label,
        "result_source": source,
        "selector_weighting": args.get("rigid_first_selector_weighting", "similarity"),
        "ensemble_weighting": args.get("rigid_first_ensemble_weighting", "similarity"),
        "similarity_feature_set": args.get("selector_similarity_feature_set", "full"),
        "mean_total_mse": summary.get("mean_total_mse"),
        "mean_ratio_vs_h2oplus": summary.get("mean_vs_h2oplus_dense_ridge"),
        "wins_vs_h2oplus": summary.get("wins_vs_h2oplus_dense_ridge"),
        "ensemble_selected_splits": int(sum(name.startswith("ensemble_") for name in selected)),
        "rigid_fallback_splits": int(sum(name == "rigid_cfcmt" for name in selected)),
    }


def _parse_labeled_paths(values: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected LABEL=PATH, got {value!r}")
        label, path = value.split("=", 1)
        out[label] = Path(path)
    return out


def build_feature_ablation_table(default_result: dict[str, Any], feature_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    default_feature_set = default_result.get("args", {}).get("selector_similarity_feature_set", "sim_only")
    rows = [_feature_row(f"reported guarded ({default_feature_set})", default_result, source="reported guarded")]
    for label, result in feature_results.items():
        rows.append(_feature_row(label, result, source="feature ablation rerun"))
    return pd.DataFrame(rows).sort_values("mean_total_mse")


def build_source_selected_feature_table(default_result: dict[str, Any], feature_results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    default_label = str(default_result.get("args", {}).get("selector_similarity_feature_set", "sim_only"))
    all_results = {default_label: default_result, **feature_results}
    split_names = [split["name"] for split in default_result["splits"]]
    rows = []
    for split_name in split_names:
        candidates = []
        for label, result in all_results.items():
            split = next(item for item in result["splits"] if item["name"] == split_name)
            score, selected_candidate = _source_feature_score(split)
            candidates.append((score, FEATURE_TIE_BREAK.get(label, 99), label, split, selected_candidate))
        score, _rank, label, split, selected_candidate = min(candidates, key=lambda item: (item[0], item[1], item[2]))
        h2o = _method_mse(split, "h2oplus_dense_ridge")
        guarded = _method_mse(split, "rigid_first_source_gated_delta")
        rows.append(
            {
                "split": split_name,
                "selected_feature_set": label,
                "source_fold_score": score,
                "selected_candidate": selected_candidate,
                "h2oplus_mse": h2o,
                "guarded_mse": guarded,
                "guarded_ratio_vs_h2oplus": guarded / h2o if h2o else None,
            }
        )
    rows.append(
        {
            "split": "mean",
            "selected_feature_set": "",
            "source_fold_score": float(np.mean([row["source_fold_score"] for row in rows])),
            "selected_candidate": "",
            "h2oplus_mse": float(np.mean([row["h2oplus_mse"] for row in rows])),
            "guarded_mse": float(np.mean([row["guarded_mse"] for row in rows])),
            "guarded_ratio_vs_h2oplus": float(np.mean([row["guarded_ratio_vs_h2oplus"] for row in rows])),
        }
    )
    return pd.DataFrame(rows)


def generate(args: argparse.Namespace) -> dict[str, Any]:
    guarded = _read_json(args.guarded)
    conservative = _read_json(args.conservative) if args.conservative.exists() else None
    unguarded = _read_json(args.unguarded) if args.unguarded.exists() else None
    feature_results = {}
    for label, path in _parse_labeled_paths(args.feature_result).items():
        if path.exists():
            result = _read_json(path)
            result["_path"] = str(path)
            feature_results[label] = result

    tables_dir = args.out_dir / "tables"
    tables = {
        "guarded_ensemble": _write_table(build_guarded_ensemble_table(guarded), tables_dir, "guarded_ensemble_table"),
        "guarded_ensemble_same_information_baseline": _write_table(
            build_same_information_baseline_table(guarded),
            tables_dir,
            "guarded_ensemble_same_information_baseline_table",
        ),
        "guarded_ensemble_selector_ablation": _write_table(
            build_selector_ablation_table(conservative=conservative, unguarded=unguarded, guarded=guarded),
            tables_dir,
            "guarded_ensemble_selector_ablation_table",
        ),
        "guarded_ensemble_strict_loo": _write_table(
            build_strict_loo_table(guarded),
            tables_dir,
            "guarded_ensemble_strict_loo_table",
        ),
        "guarded_ensemble_threshold_sensitivity": _write_table(
            build_threshold_sensitivity_table(guarded),
            tables_dir,
            "guarded_ensemble_threshold_sensitivity_table",
        ),
        "guarded_ensemble_feature_ablation": _write_table(
            build_feature_ablation_table(guarded, feature_results),
            tables_dir,
            "guarded_ensemble_feature_ablation_table",
        ),
        "guarded_ensemble_source_selected_feature": _write_table(
            build_source_selected_feature_table(guarded, feature_results),
            tables_dir,
            "guarded_ensemble_source_selected_feature_table",
        ),
        "guarded_ensemble_label_audit": _write_table(
            build_label_audit_table(guarded),
            tables_dir,
            "guarded_ensemble_label_audit_table",
        ),
        "guarded_ensemble_oracle_gap": _write_table(
            build_oracle_gap_table(guarded),
            tables_dir,
            "guarded_ensemble_oracle_gap_table",
        ),
        "guarded_ensemble_candidate_family": _write_table(
            build_candidate_family_table(guarded),
            tables_dir,
            "guarded_ensemble_candidate_family_table",
        ),
    }
    manifest = {
        "ok": True,
        "sources": {
            "guarded": str(args.guarded),
            "conservative": str(args.conservative) if args.conservative.exists() else None,
            "unguarded": str(args.unguarded) if args.unguarded.exists() else None,
            "feature_results": {label: str(path) for label, path in _parse_labeled_paths(args.feature_result).items()},
        },
        "tables": tables,
        "summary": {
            "guarded": guarded["summary"],
            "feature_ablation_results": sorted(feature_results),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "guarded_ensemble_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guarded", type=Path, default=Path("cf_h2o/results/rigid_first_ensemble_sim_only_validation.json"))
    parser.add_argument(
        "--conservative",
        type=Path,
        default=Path("cf_h2o/results/rigid_first_similarity_conservative_validation.json"),
    )
    parser.add_argument("--unguarded", type=Path, default=Path("cf_h2o/results/rigid_first_ensemble_validation.json"))
    parser.add_argument(
        "--feature-result",
        action="append",
        default=[
            "full=cf_h2o/results/rigid_first_ensemble_guarded_validation.json",
            "obs_sim=cf_h2o/results/rigid_first_ensemble_obs_sim_validation.json",
            "obs_only=cf_h2o/results/rigid_first_ensemble_obs_only_validation.json",
            "local_only=cf_h2o/results/rigid_first_ensemble_local_only_validation.json",
            "uniform=cf_h2o/results/rigid_first_ensemble_uniform_validation.json",
        ],
        help="Additional feature ablation result as LABEL=PATH.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("cf_h2o/results/paper_artifacts"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    manifest = generate(parse_args(argv))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
