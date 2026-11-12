#!/usr/bin/env python3
"""Fixed whole-seed physical-response transfer screen: thirty fits, no simulation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsc_physical_response_data import FEATURE_NAMES, LABEL_NAMES, analytic_baseline


PROTOCOL = "tsc-physical-response-development-v1"
ARMS = ("target_only", "source_target", "source_placebo")
MODEL_CONFIG = dict(loss="squared_error", learning_rate=0.05, max_iter=100,
                    max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=10.0,
                    random_state=20261106, early_stopping=False)


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def validate_spec(spec):
    if (spec["protocol"] != PROTOCOL or spec["response_horizon_sec"] != 10
            or spec["cost_horizon_sec"] != 450 or spec["max_new_fits"] != 30
            or spec["model"]["config"] != MODEL_CONFIG
            or spec["model"]["sklearn_version"] != "1.7.2" or sklearn.__version__ != "1.7.2"
            or spec["weighting"]["total_weight"] != 1536
            or spec["weighting"]["source_fraction"] != 0.5
            or spec["placebo_seed"] != 20261106
            or spec["analysis"]["bootstrap_draws"] != 10000
            or spec["analysis"]["bootstrap_seed"] != 20261106):
        raise ValueError("Run differs from the frozen thirty-fit response protocol")
    source, target, cases = spec["source_seeds"], spec["target_seeds"], spec["cases"]
    if (len(source) != 4 or len(set(source)) != 4 or len(target) != 10
            or len(set(target)) != 10 or set(source) & set(target)
            or len(cases) != 28 or len({case["case_id"] for case in cases}) != 28):
        raise ValueError("Expected disjoint four-source/ten-target seeds and 28 groups")
    for city, seeds in (("cologne", source), ("jinan", target)):
        counts = Counter(case["seed"] for case in cases if case["city"] == city)
        if counts != Counter({seed: 2 for seed in seeds}):
            raise ValueError("Every declared seed requires exactly two native groups")


def load_groups(spec, root):
    groups = []
    for case in spec["cases"]:
        directory = root / f"seed_{case['seed']}" / case["case_id"]
        receipt = json.loads((directory / "result.json").read_text())
        data = json.loads((directory / "group_data_server_only.json").read_text())
        if (receipt["status"] != "COMPLETE" or len(receipt["branches"]) != 8
                or {row["candidate_index"] for row in receipt["branches"]} != set(range(8))
                or any(row["status"] != "COMPLETE" or not all(row["checks"].values())
                       for row in receipt["branches"])):
            raise ValueError(f"{case['case_id']}: incomplete native action group")
        if any(data[key] != case[key] or receipt[key] != case[key]
               for key in ("case_id", "city", "seed")):
            raise ValueError("Collection identity differs from the frozen case")
        movements = data["movement_ids"]
        count = 16 if case["city"] == "cologne" else 12
        if (tuple(data["feature_names"]) != FEATURE_NAMES or tuple(data["label_names"]) != LABEL_NAMES
                or len(movements) != count or len(set(movements)) != count
                or len(data["candidate_states"]) != 8 or len(set(data["candidate_states"])) != 8
                or not 0 <= data["reference_index"] < 8 or len(data["branches"]) != 8):
            raise ValueError("Physical feature, response, movement or action roster differs")
        for index, row in enumerate(data["branches"]):
            if row["candidate_index"] != index or row["preaction"]["movement_ids"] != movements:
                raise ValueError("Every candidate must retain the same ordered focal movements")
        X, Y = np.asarray(data["X"], dtype=float), np.asarray(data["Y"], dtype=float)
        if (X.shape != (8, count, len(FEATURE_NAMES)) or Y.shape != (8, count, 2)
                or not np.isfinite(X).all() or not np.isfinite(Y).all()
                or np.any(Y < 0) or not np.array_equal(Y, np.floor(Y))):
            raise ValueError("Invalid pre-action inputs or measured response counts")
        groups.append({"case_id": case["case_id"], "city": case["city"], "seed": case["seed"],
                       "reference_index": data["reference_index"], "movement_ids": movements,
                       "candidate_states": data["candidate_states"], "X": X, "Y": Y})
    return groups


def folds(groups, seeds):
    target = [group for group in groups if group["city"] == "jinan"]
    result = []
    for offset in range(0, 10, 2):
        heldout = seeds[offset:offset + 2]
        train = [group for group in target if group["seed"] not in heldout]
        test = [group for group in target if group["seed"] in heldout]
        if len(train) != 16 or len(test) != 4 or set(g["seed"] for g in train) & set(heldout):
            raise ValueError("Each fold must hold out both groups from two complete seeds")
        result.append((train, test, heldout))
    return result


def placebo_labels(source, seed=20261106):
    """Permute complete seven-action response vectors once; PP remains unchanged."""
    rng, labels, permutations = np.random.default_rng(seed), {}, {}
    for group in source:
        actions = np.asarray([a for a in range(8) if a != group["reference_index"]])
        donors = rng.permutation(actions)
        value = group["Y"].copy()
        value[actions] = group["Y"][donors]
        labels[group["case_id"]] = value
        permutations[group["case_id"]] = {str(int(a)): int(b) for a, b in zip(actions, donors)}
    return labels, permutations


def training_design(target, source, arm, placebo):
    mixed = arm != "target_only"
    X, Y, weights, audit = [], [], [], {}
    for name, selected, mass in (("target", target, 768.0 if mixed else 1536.0),
                                  ("source", source if mixed else [], 768.0 if mixed else 0.0)):
        count = 0
        for group in selected:
            features = group["X"].reshape(-1, len(FEATURE_NAMES))
            labels = placebo[group["case_id"]] if arm == "source_placebo" and name == "source" else group["Y"]
            X.append(features)
            Y.append(labels.reshape(-1, 2))
            weights.append(np.full(len(features), mass / len(selected) / len(features)))
            count += len(features)
        audit[name + "_groups"] = len(selected)
        audit[name + "_rows"] = count
        audit[name + "_weight"] = mass
    features, labels, w = np.vstack(X), np.vstack(Y), np.concatenate(weights)
    if (audit["target_rows"] != 1536 or audit["source_rows"] != (1024 if mixed else 0)
            or not np.isclose(w.sum(), 1536.0, rtol=0, atol=1e-9)):
        raise ValueError("Training rows or total weight differ from the frozen protocol")
    audit["total_weight"] = float(w.sum())
    return features, labels, w, audit


def response_error(group, prediction):
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    if prediction.shape != group["Y"].shape or not np.isfinite(prediction).all():
        raise ValueError("Incomplete/nonfinite physical-response predictions")
    reference = group["reference_index"]
    non_reference = np.arange(8) != reference
    delta = prediction - prediction[reference]
    truth = group["Y"] - group["Y"][reference]
    by_output = np.mean(np.abs(delta[non_reference] - truth[non_reference]), axis=(0, 1))
    initial_internal = group["X"][:, :, FEATURE_NAMES.index("internal_vehicle_count")]
    balance_audit = {
        "movement_action_count": int(prediction.shape[0] * prediction.shape[1]),
        "receiving_exceeds_initial_internal_plus_entry_count": int(np.count_nonzero(
            prediction[:, :, 1] > initial_internal + prediction[:, :, 0]))}
    return {"case_id": group["case_id"], "seed": group["seed"],
            "contrast_mae": float(np.mean(by_output)),
            "output_contrast_mae": dict(zip(LABEL_NAMES, map(float, by_output))),
            "raw_count_mae": float(np.mean(np.abs(prediction - group["Y"]))),
            "prediction_balance_audit": balance_audit}, prediction


def seed_errors(records, seeds):
    result = {}
    for seed in seeds:
        rows = [row for row in records if row["seed"] == seed]
        if len(rows) != 2 or len({row["case_id"] for row in rows}) != 2:
            raise ValueError("Expected exactly two distinct OOF groups per target seed")
        result[str(seed)] = {"contrast_mae": float(np.mean([row["contrast_mae"] for row in rows])),
            "output_contrast_mae": {name: float(np.mean([row["output_contrast_mae"][name] for row in rows]))
                                    for name in LABEL_NAMES},
            "raw_count_mae": float(np.mean([row["raw_count_mae"] for row in rows]))}
    return result


def paired_summary(first, second, seeds, *, draws=10000, random_seed=20261106):
    differences = np.asarray([first[str(seed)]["contrast_mae"] - second[str(seed)]["contrast_mae"] for seed in seeds])
    bootstrap = np.random.default_rng(random_seed).choice(differences, size=(draws, len(seeds))).mean(axis=1)
    return {"seed_count": len(seeds), "mean_difference": float(differences.mean()),
            "paired_bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "improved_seeds": int(np.count_nonzero(differences < 0)),
            "worse_seeds": int(np.count_nonzero(differences > 0)),
            "per_seed_difference": dict(zip(map(str, seeds), map(float, differences)))}


def run(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        spec = json.loads(args.spec.read_text())
        validate_spec(spec)
        groups = load_groups(spec, args.collection_root)
        split = folds(groups, spec["target_seeds"])
    except (ValueError, KeyError, FileNotFoundError) as error:
        result = {"protocol": PROTOCOL, "status": "HOLD_INPUT", "new_fits": 0,
                  "new_simulations": 0, "error": f"{type(error).__name__}: {error}"}
        write_json(args.output / "result.json", result)
        return result
    source = [group for group in groups if group["city"] == "cologne"]
    placebo, permutations = placebo_labels(source, spec["placebo_seed"])
    records, oof, fits = {arm: [] for arm in (*ARMS, "analytic")}, [], []
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 0, "total": 30, "unit": "fit"}), flush=True)
    for fold, (train, test, heldout) in enumerate(split):
        test_X = np.vstack([group["X"].reshape(-1, len(FEATURE_NAMES)) for group in test])
        for arm in ARMS:
            X, Y, weights, audit = training_design(train, source, arm, placebo)
            predictions = np.empty((len(test_X), 2))
            for output, name in enumerate(LABEL_NAMES):
                model = HistGradientBoostingRegressor(**MODEL_CONFIG)
                model.fit(X, Y[:, output], sample_weight=weights)
                fits.append({"fold": fold, "arm": arm, "output": name,
                             "heldout_seeds": heldout, **audit})
                write_json(args.output / "fit_progress.json", {"new_fits": len(fits), "fits": fits})
                predictions[:, output] = np.maximum(model.predict(test_X), 0.0)
                print("SCHEDULEURM_PROGRESS " + json.dumps({"current": len(fits), "total": 30,
                    "unit": "fit", "fold": fold, "arm": arm, "output": name,
                    "elapsed_seconds": time.monotonic() - started}), flush=True)
            offset = 0
            for group in test:
                size = int(np.prod(group["X"].shape[:2]))
                prediction = predictions[offset:offset + size].reshape(group["Y"].shape)
                error, prediction = response_error(group, prediction)
                records[arm].append(error)
                oof.append({"fold": fold, "arm": arm, "case_id": group["case_id"], "seed": group["seed"],
                    "reference_index": group["reference_index"], "movement_ids": group["movement_ids"],
                    "candidate_states": group["candidate_states"], "predicted_counts": prediction.tolist(),
                    "measured_counts": group["Y"].tolist()})
                offset += size
            write_json(args.output / "oof_server_only.json", {"records": oof, "fits": fits,
                        "placebo_action_permutations": permutations})
        for group in test:
            prediction = analytic_baseline(group["X"].reshape(-1, len(FEATURE_NAMES))).reshape(group["Y"].shape)
            error, _ = response_error(group, prediction)
            records["analytic"].append(error)
    seeds = spec["target_seeds"]
    summaries = {arm: seed_errors(rows, seeds) for arm, rows in records.items()}
    means = {arm: float(np.mean([row["contrast_mae"] for row in values.values()])) for arm, values in summaries.items()}
    output_means = {arm: {name: float(np.mean([row["output_contrast_mae"][name] for row in values.values()]))
                         for name in LABEL_NAMES} for arm, values in summaries.items()}
    balance_audits = {arm: {name: sum(row["prediction_balance_audit"][name] for row in rows)
                           for name in rows[0]["prediction_balance_audit"]}
                      for arm, rows in records.items()}
    comparisons = {f"source_target_minus_{arm}": paired_summary(summaries["source_target"], summaries[arm], seeds,
                   draws=spec["analysis"]["bootstrap_draws"], random_seed=spec["analysis"]["bootstrap_seed"])
                   for arm in ("target_only", "source_placebo", "analytic")}
    gate = {"all_three_paired_upper_bounds_negative": all(row["paired_bootstrap_95"][1] < 0 for row in comparisons.values()),
            "both_outputs_not_worse_than_target": all(output_means["source_target"][name] <= output_means["target_only"][name]
                                                       for name in LABEL_NAMES)}
    result = {"protocol": PROTOCOL, "status": "PASS_RESPONSE" if all(gate.values()) else "HOLD_RESPONSE",
        "new_fits": len(fits), "new_simulations": 0, "source_groups": len(source), "target_groups": 20,
        "feature_names": list(FEATURE_NAMES), "label_names": list(LABEL_NAMES),
        "mean_contrast_mae": means, "mean_output_contrast_mae": output_means,
        "prediction_balance_audit": balance_audits,
        "comparisons": comparisons, "gate": gate, "per_seed_errors": summaries, "per_case_errors": records,
        "fit_diagnostics": fits, "elapsed_sec": time.monotonic() - started,
        "network_readout_preparation_supported": all(gate.values()),
        "scope": "Development response prediction only; no selected-action, closed-loop or independent transfer confirmation.",
        "readout_boundary": "Raw counts are nonnegative but not projected to flow conservation; no 450-second outcome is used in fitting."}
    write_json(args.output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "collection-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "new_fits", "mean_contrast_mae", "gate", "error") if key in result}), flush=True)
