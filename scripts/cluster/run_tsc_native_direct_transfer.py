#!/usr/bin/env python3
"""One fixed native Cologne-to-Jinan transfer screen; no new simulation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158


PROTOCOL = "tsc-native-direct-transfer-development-v1"
ARMS = ("target_only", "source_target", "source_placebo")
MODEL_CONFIG = dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=20, l2_regularization=10.0,
                    random_state=20260803, loss="squared_error", early_stopping=False)
DOMAIN_GROUPS = {"jinan_3x4_real": 32, "jinan_3x4_real_2000": 24,
                 "jinan_3x4_real_2500": 24}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def group_rows(data):
    ids = np.asarray(data.metadata["action_group_ids"], dtype=str)
    return {group: np.flatnonzero(ids == group) for group in dict.fromkeys(ids.tolist())}


def build_design(data, state_names, action_names):
    """Keep native physical labels; represent actions relative to their PP row."""
    columns = {name: i for i, name in enumerate(data.feature_names)}
    state = data.features[:, [columns[name] for name in state_names]]
    action = data.features[:, [columns[name] for name in action_names]]
    reference = np.asarray(data.metadata["is_reference"], dtype=bool)
    y = np.asarray(data.targets["interval_cost"], dtype=float)
    X = np.empty((data.size, len(state_names) + len(action_names)))
    domains = np.asarray(data.domains, dtype=str)
    for rows in group_rows(data).values():
        refs = rows[reference[rows]]
        if len(rows) != 8 or len(refs) != 1 or len(np.unique(domains[rows])) != 1:
            raise ValueError("native group requires eight actions, one PP row and one domain")
        if not np.allclose(state[rows], state[rows[0]], rtol=0, atol=1e-10):
            raise ValueError("pre-action state parents vary across the action group")
        if abs(y[refs[0]]) > 1e-10:
            raise ValueError("physical interval_cost must be a PP-relative residual")
        X[rows] = np.column_stack((state[rows], action[rows] - action[refs[0]]))
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("native model input and target must be finite")
    return {"X": X, "y": y, "reference": reference, "candidate_rows": np.flatnonzero(~reference),
            "group_ids": np.asarray(data.metadata["action_group_ids"], dtype=str), "domains": domains}


def sample_weights(target, source=None, *, source_fraction=0.5, total_weight=560.0):
    """Balance target demands; source mass is uniform across native source groups."""
    if source_fraction != 0.5 or total_weight != 560.0:
        raise ValueError("fixed pilot uses source fraction 0.5 and total mass 560")
    rows = target["candidate_rows"]
    domains, groups = target["domains"][rows], target["group_ids"][rows]
    domain_groups = {domain: len(set(groups[domains == domain])) for domain in np.unique(domains)}
    target_mass = total_weight if source is None else total_weight * (1 - source_fraction)
    target_weights = np.asarray([target_mass / (len(domain_groups) * domain_groups[domain] * 7)
                                 for domain in domains])
    source_weights = np.empty(0)
    if source is not None:
        source_rows = source["candidate_rows"]
        source_groups = source["group_ids"][source_rows]
        source_weights = np.full(len(source_rows), total_weight * source_fraction / (len(set(source_groups)) * 7))
    return {"target": target_weights, "source": source_weights,
            "diagnostics": {"target_rows": len(target_weights), "source_rows": len(source_weights),
                "target_total_weight": float(target_weights.sum()), "source_total_weight": float(source_weights.sum()),
                "total_weight": float(target_weights.sum() + source_weights.sum()),
                "target_domain_groups": domain_groups,
                "target_per_domain_weight": {domain: float(target_weights[domains == domain].sum())
                                             for domain in domain_groups}}}


def permute_source_labels(source, *, seed=20261106):
    rng = np.random.default_rng(seed)
    labels = source["y"].copy()
    for group in dict.fromkeys(source["group_ids"].tolist()):
        rows = np.flatnonzero((source["group_ids"] == group) & ~source["reference"])
        labels[rows] = rng.permutation(labels[rows])
    return labels


def target_subset(data, selected_groups):
    rows = np.flatnonzero(np.isin(np.asarray(data.metadata["action_group_ids"]), list(selected_groups)))
    keys = ("action_group_ids", "is_reference", "row_tls", "row_times", "candidate_states", "row_seeds", "row_scenarios")
    metadata = {**data.metadata, **{key: np.asarray(data.metadata[key], dtype=object)[rows].tolist() for key in keys}}
    metadata.pop("reference_rows", None)
    return replace(data, features=data.features[rows], context=data.context[rows], domains=data.domains[rows],
                   targets={key: np.asarray(value)[rows] for key, value in data.targets.items()},
                   priors={key: np.asarray(value)[rows] for key, value in data.priors.items()}, metadata=metadata)


def target_folds(data, folds):
    if folds != [list(fold) for fold in v158.CROSSFIT_FOLDS]:
        raise ValueError("target whole-seed folds differ from V158")
    mapping = {}
    seeds = np.asarray(data.metadata["row_seeds"], dtype=int)
    for group, rows in group_rows(data).items():
        unique = np.unique(seeds[rows])
        if len(unique) != 1:
            raise ValueError("target action group crosses seeds")
        mapping[group] = int(unique[0])
    if set(mapping.values()) != set(v158.DEVELOPMENT_SEEDS):
        raise ValueError("target seed roster differs from V158")
    result = []
    for fold in folds:
        heldout = {group for group, seed in mapping.items() if seed in fold}
        training = set(mapping) - heldout
        if len(training) != 80 or len(heldout) != 20:
            raise ValueError("each fixed fold needs 80 training and 20 heldout groups")
        result.append((training, heldout))
    return result


def policy_records(data, physical_scores, *, target_cost_factor=64800.0):
    """Use the original selector in its original cost units and tolerance."""
    scores = np.asarray(physical_scores, dtype=float).copy() / target_cost_factor
    reference = np.asarray(data.metadata["is_reference"], dtype=bool)
    scores[reference] = 0.0
    if not np.isfinite(scores).all():
        raise ValueError("prediction must be finite")
    absolute = np.asarray(data.targets["absolute_cost"], dtype=float)
    records = []
    for group, rows in group_rows(data).items():
        ref = int(rows[reference[rows]][0])
        selected = v158._select_row(data, scores, rows, prohibit_stay_override=True)
        records.append({"group_id": group, "seed": int(data.metadata["row_seeds"][ref]),
            "scenario": str(data.metadata["row_scenarios"][ref]),
            "selected_state": str(data.metadata["candidate_states"][selected]),
            "reference_state": str(data.metadata["candidate_states"][ref]),
            "selected_row": int(selected), "reference_row": ref, "different_action": bool(selected != ref),
            "predicted_advantage": max(float(-scores[selected]), 0.0),
            "actual_advantage": float(absolute[ref] - absolute[selected]),
            "selected_cost": float(absolute[selected]), "phase_pressure_cost": float(absolute[ref])})
    return records, scores


def behavior(records):
    gains = np.asarray([row["actual_advantage"] for row in records])
    overrides = sum(row["different_action"] for row in records)
    harmful = int(np.count_nonzero(gains < -v158.TOLERANCE))
    return {"groups": len(records), "overrides": int(overrides), "harmful_overrides": harmful,
            "harmful_fraction": harmful / overrides if overrides else None,
            "raw_benefit_sum": float(np.maximum(gains, 0).sum()),
            "raw_harm_sum": float(np.maximum(-gains, 0).sum()), "raw_net_gain_mean": float(gains.mean())}


def validate_spec(spec):
    if (spec["protocol"] != PROTOCOL or spec["model"]["config"] != MODEL_CONFIG
            or spec["model"]["sklearn_version"] != "1.7.2" or sklearn.__version__ != "1.7.2"
            or spec["development"]["seeds"] != list(v158.DEVELOPMENT_SEEDS)
            or spec["development"]["max_new_fits"] != 15
            or spec["development"]["new_simulator_trajectories"] != 0
            or spec["arms"] != list(ARMS)
            or spec["weighting"]["total_weight"] != 560.0 or spec["weighting"]["source_fraction"] != 0.5
            or spec["placebo_seed"] != 20261106
            or spec["prediction"]["target_cost_factor"] != 64800.0 or spec["prediction"]["threshold"] != 0
            or spec["prediction"]["no_stay_override_of_pp_switch"] is not True
            or spec["target"]["source_cost_factor"] != 3600 or spec["target"]["target_cost_factor"] != 64800
            or spec["analysis"]["bootstrap_draws"] != 10000 or spec["analysis"]["bootstrap_seed"] != 20261106):
        raise ValueError("run differs from frozen direct native-transfer protocol")


def run(args):
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    spec = json.loads(args.spec.read_text())
    try:
        from tsc_native_transfer_data import load_native_inputs
        validate_spec(spec)
        inputs = json.loads(args.source_inputs.read_text())
        source, target, audit = load_native_inputs(inputs, spec["parent_snapshot"])
        state_names, action_names = tuple(audit["state_feature_names"]), tuple(audit["action_feature_names"])
        if (len(state_names), len(action_names)) != (spec["features"]["state_feature_count"], spec["features"]["action_feature_count"]):
            raise ValueError("adapter feature roster differs from frozen protocol")
        if source.size != 1560 or target.size != 800:
            raise ValueError("expected source195 and original target100 eight-action groups")
        if set(source.metadata["row_seeds"]) & set(target.metadata["row_seeds"]):
            raise ValueError("native source and target seed rosters overlap")
        source_design = build_design(source, state_names, action_names)
        placebo_y = permute_source_labels(source_design, seed=spec["placebo_seed"])
        historical_context = None
        if args.history_result is not None:
            history = json.loads(args.history_result.read_text())
            historical_context = {"history80_mean": history["mean_costs"]["history"],
                                  "source": str(args.history_result), "refitted": False}
        prepared, fold_inputs = [], []
        for index, (training, heldout) in enumerate(target_folds(target, spec["crossfit"]["folds"])):
            train, test = target_subset(target, training), target_subset(target, heldout)
            design = build_design(train, state_names, action_names)
            test_design = build_design(test, state_names, action_names)
            weights = {arm: sample_weights(design, None if arm == "target_only" else source_design,
                total_weight=spec["weighting"]["total_weight"], source_fraction=spec["weighting"]["source_fraction"])
                for arm in ARMS}
            for arm, item in weights.items():
                info = item["diagnostics"]
                if (info["target_domain_groups"] != DOMAIN_GROUPS or info["target_rows"] != 560
                        or info["source_rows"] != (0 if arm == "target_only" else 1365)
                        or not np.isclose(info["total_weight"], 560, rtol=0, atol=1e-9)):
                    raise ValueError("fold group counts or sample-weight mass differ from protocol")
            prepared.append((design, test, test_design, weights))
            fold_inputs.append({"fold": index, "heldout_seeds": spec["crossfit"]["folds"][index],
                                "weights": {arm: value["diagnostics"] for arm, value in weights.items()}})
    except (ValueError, KeyError, FileNotFoundError) as exc:
        result = {"protocol": spec.get("protocol"), "status": "HOLD_INPUT", "error": f"{type(exc).__name__}: {exc}",
                  "new_fits": 0, "new_simulator_trajectories": 0}
        write_json(args.out_dir / "result.json", result)
        return result
    validation = {"adapter": audit, "folds": fold_inputs, "state_feature_names": list(state_names),
                  "action_feature_names": list(action_names), "sklearn_version": sklearn.__version__,
                  "placebo_changed_source_rows": int(np.count_nonzero(placebo_y != source_design["y"]))}
    write_json(args.out_dir / "input_validation.json", validation)
    records, fits, predictions = {arm: [] for arm in ARMS}, [], []
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 0, "total": 15, "unit": "fit",
                                              "elapsed_seconds": time.monotonic() - started}), flush=True)
    sr = source_design["candidate_rows"]
    for index, (design, test, test_design, weights) in enumerate(prepared):
        tr = design["candidate_rows"]
        combined_X = np.vstack((design["X"][tr], source_design["X"][sr]))
        for arm in ARMS:
            X = design["X"][tr] if arm == "target_only" else combined_X
            y = design["y"][tr] if arm == "target_only" else np.concatenate((design["y"][tr],
                 (placebo_y if arm == "source_placebo" else source_design["y"])[sr]))
            w = np.concatenate((weights[arm]["target"], weights[arm]["source"]))
            model = HistGradientBoostingRegressor(**MODEL_CONFIG)
            model.fit(X, y, sample_weight=w)
            selected, scores = policy_records(test, model.predict(test_design["X"]),
                                               target_cost_factor=spec["prediction"]["target_cost_factor"])
            records[arm].extend(selected)
            fits.append({"fold": index, "arm": arm, "heldout_seeds": spec["crossfit"]["folds"][index],
                         "weighted_training_mae_vehicle_seconds": float(np.average(np.abs(y - model.predict(X)), weights=w)),
                         **weights[arm]["diagnostics"]})
            predictions.append({"fold": index, "arm": arm,
                                "group_ids": list(test.metadata["action_group_ids"]),
                                "candidate_states": list(test.metadata["candidate_states"]),
                                "scores_native_cost_units": scores.tolist()})
            write_json(args.out_dir / "oof_server_only.json", {"records": records, "fits": fits, "predictions": predictions})
            print("SCHEDULEURM_PROGRESS " + json.dumps({"current": len(fits), "total": 15, "unit": "fit",
                "fold": index, "arm": arm, "elapsed_seconds": time.monotonic() - started}), flush=True)
    expected = set(group_rows(target))
    if any(len(rows) != 100 or {row["group_id"] for row in rows} != expected for rows in records.values()):
        raise ValueError("incomplete or duplicated target OOF outcomes")
    costs = {arm: v158._seed_costs(rows) for arm, rows in records.items()}
    costs["phase_pressure"] = v158._pp_seed_costs(records["target_only"])
    comparisons = {f"source_target_minus_{arm}": v158._paired_summary(costs["source_target"], costs[arm],
        draws=spec["analysis"]["bootstrap_draws"], seed=spec["analysis"]["bootstrap_seed"])
        for arm in ("target_only", "source_placebo", "phase_pressure")}
    behaviors = {arm: behavior(rows) for arm, rows in records.items()}
    per_demand = {}
    for scenario in sorted(set(target.metadata["row_scenarios"])):
        per_demand[scenario] = {}
        for arm, rows in records.items():
            selected = [row for row in rows if row["scenario"] == scenario]
            values = v158._seed_costs(selected)
            per_demand[scenario][arm] = {"mean_cost": float(np.mean(list(values.values()))),
                "per_seed_cost": values, "behavior": behavior(selected)}
        pp = v158._pp_seed_costs([row for row in records["target_only"] if row["scenario"] == scenario])
        per_demand[scenario]["phase_pressure"] = {"mean_cost": float(np.mean(list(pp.values()))), "per_seed_cost": pp}
    gate = {"all_three_paired_upper_bounds_negative": all(row["paired_bootstrap_95"][1] < 0 for row in comparisons.values()),
            "harmful_cost_not_increased_vs_target": behaviors["source_target"]["raw_harm_sum"] <= behaviors["target_only"]["raw_harm_sum"]}
    result = {"protocol": PROTOCOL, "status": "PASS_DEVELOPMENT" if all(gate.values()) else "HOLD_DEVELOPMENT",
        "target_groups": 100, "source_groups": 195, "new_fits": len(fits), "new_simulator_trajectories": 0,
        "mean_costs": {arm: float(np.mean(list(values.values()))) for arm, values in costs.items()},
        "per_seed_costs": costs, "comparisons": comparisons, "behavior": behaviors, "per_demand": per_demand,
        "gate": gate, "fit_diagnostics": fits, "input_validation": validation,
        "elapsed_sec": time.monotonic() - started, "scope": "Exploratory reused Jinan challenge bank; native source-label pilot, not independent or closed-loop confirmation.",
        "independent_validation_preparation_supported": all(gate.values()), "closed_loop_authorized": False}
    if historical_context is not None:
        result["historical_context_only"] = historical_context
    write_json(args.out_dir / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "source-inputs", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--history-result", type=Path)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "new_fits", "mean_costs", "gate", "error") if key in result}), flush=True)
