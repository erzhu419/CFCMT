#!/usr/bin/env python3
"""Frozen PP-star versus all-pair supervision on existing native labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158


MODEL_CONFIG = dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=160, l2_regularization=10.0,
                    random_state=20260803, loss="squared_error", early_stopping=False)
DOMAIN_GROUPS = {"jinan_3x4_real": 32, "jinan_3x4_real_2000": 24,
                 "jinan_3x4_real_2500": 24}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def group_rows(data):
    ids = np.asarray(data.metadata["action_group_ids"], dtype=str)
    return {group: np.flatnonzero(ids == group) for group in dict.fromkeys(ids.tolist())}


def feature_arrays(data, state_names, action_names):
    columns = {name: i for i, name in enumerate(data.feature_names)}
    state = data.features[:, [columns[name] for name in state_names]]
    action = data.features[:, [columns[name] for name in action_names]]
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("pair features must be finite")
    return state, action


def build_pair_design(data, state_names, action_names):
    """One common ordered-edge matrix; only S/A sample weights differ."""
    groups = group_rows(data)
    reference = np.asarray(data.metadata["is_reference"], dtype=bool)
    state, action = feature_arrays(data, state_names, action_names)
    absolute = np.asarray(data.targets["absolute_cost"], dtype=float)
    interval = np.asarray(data.targets["interval_cost"], dtype=float)
    domains = np.asarray(data.domains, dtype=str)
    domain_counts = {domain: sum(np.any(domains[rows] == domain) for rows in groups.values())
                     for domain in np.unique(domains)}
    design, targets, group_ids, pair_domains, star_weights, all_weights = [], [], [], [], [], []
    for group, rows in groups.items():
        refs = rows[reference[rows]]
        if len(rows) != 8 or len(refs) != 1 or len(np.unique(domains[rows])) != 1:
            raise ValueError("pair group requires eight actions, one PP reference and one domain")
        if not np.allclose(state[rows], state[rows[0]], rtol=0, atol=1e-10):
            raise ValueError("state parents vary across candidate actions")
        if not np.allclose(interval[rows], absolute[rows] - absolute[refs[0]], rtol=0, atol=v158.TOLERANCE):
            raise ValueError("native interval labels differ from absolute-cost minus PP")
        domain = str(domains[rows[0]])
        weight = 560.0 / (len(domain_counts) * domain_counts[domain])
        for left in rows:
            for right in rows:
                if left == right:
                    continue
                design.append(np.concatenate((state[left], action[left] - action[right])))
                targets.append(absolute[left] - absolute[right])
                group_ids.append(group)
                pair_domains.append(domain)
                star_weights.append(weight / 14 if left == refs[0] or right == refs[0] else 0.0)
                all_weights.append(weight / 56)
    output = {"X": np.asarray(design), "y": np.asarray(targets),
              "weights": {"S": np.asarray(star_weights), "A": np.asarray(all_weights)},
              "group_ids": np.asarray(group_ids), "domains": np.asarray(pair_domains)}
    if not np.isfinite(output["y"]).all():
        raise ValueError("native cost labels must be finite")
    output["diagnostics"] = {"groups": len(groups), "rows": len(targets),
        "domain_groups": {key: int(value) for key, value in domain_counts.items()},
        "arms": {arm: {"positive_rows": int(np.count_nonzero(weights)),
                         "total_weight": float(weights.sum()),
                         "per_domain_weight": {domain: float(weights[output["domains"] == domain].sum())
                                               for domain in domain_counts}}
                 for arm, weights in output["weights"].items()}}
    return output


def pair_scores(model, data, state_names, action_names):
    state, action = feature_arrays(data, state_names, action_names)
    reference = np.asarray(data.metadata["is_reference"], dtype=bool)
    forward = np.empty((data.size, len(state_names) + len(action_names)))
    for rows in group_rows(data).values():
        refs = rows[reference[rows]]
        if len(refs) != 1:
            raise ValueError("pair prediction requires one PP reference per group")
        forward[rows] = np.column_stack((state[rows], action[rows] - action[refs[0]]))
    reverse = forward.copy()
    reverse[:, len(state_names):] *= -1
    scores = 0.5 * (np.asarray(model.predict(forward)) - np.asarray(model.predict(reverse)))
    scores[reference] = 0.0
    if not np.isfinite(scores).all():
        raise ValueError("pair predictions must be finite")
    return scores


class PairModel:
    """The mean-only interface required by the unchanged V158 selector."""
    def __init__(self, model, state_names, action_names):
        self.model, self.state_names, self.action_names = model, state_names, action_names

    def predict(self, data):
        self.last_scores = pair_scores(self.model, data, self.state_names, self.action_names)
        return {"control_cost": {"mean": self.last_scores}}


def validate_records(data, records):
    groups = group_rows(data)
    if len(records) != len(groups) or {r["group_id"] for r in records} != set(groups):
        raise ValueError("OOF records require complete one-to-one original group coverage")
    for record in records:
        rows = groups[record["group_id"]]
        states = np.asarray(data.metadata["candidate_states"])[rows].tolist()
        selected = rows[states.index(record["selected_state"])]
        refs = rows[np.asarray(data.metadata["is_reference"], dtype=bool)[rows]]
        if len(refs) != 1:
            raise ValueError("OOF group requires one PP reference")
        reference = refs[0]
        if (record["reference_state"] != data.metadata["candidate_states"][reference]
                or record["seed"] != data.metadata["row_seeds"][reference]
                or record["scenario"] != data.metadata["row_scenarios"][reference]):
            raise ValueError("OOF record identity differs from native bank")
        absolute = data.targets["absolute_cost"]
        for key, value in (("selected_cost", absolute[selected]), ("phase_pressure_cost", absolute[reference]),
                           ("actual_advantage", absolute[reference] - absolute[selected])):
            if not np.isfinite(record[key]) or abs(record[key] - value) > v158.TOLERANCE:
                raise ValueError(f"OOF {key} differs from native label")
        if record["different_action"] != bool(selected != reference):
            raise ValueError("OOF override flag differs from selected state")


def validate_training_design(design):
    info = design["diagnostics"]
    if (info["groups"] != 80 or design["X"].shape != (4480, 97)
            or info["domain_groups"] != DOMAIN_GROUPS):
        raise ValueError("training design differs from frozen 80-group/97-feature fold")
    for arm, positive in (("S", 14), ("A", 56)):
        weights = design["weights"][arm]
        if not np.isclose(weights.sum(), 560, rtol=0, atol=1e-9):
            raise ValueError("training total weight differs from 560")
        for domain in DOMAIN_GROUPS:
            if not np.isclose(weights[design["domains"] == domain].sum(), 560 / 3, rtol=0, atol=1e-9):
                raise ValueError("training domain weight differs from 560/3")
        for group in dict.fromkeys(design["group_ids"].tolist()):
            mask = design["group_ids"] == group
            domain = design["domains"][mask][0]
            if (np.count_nonzero(weights[mask]) != positive
                    or not np.isclose(weights[mask].sum(), 560 / (3 * DOMAIN_GROUPS[domain]), rtol=0, atol=1e-10)):
                raise ValueError("group supervision count or weight differs from protocol")


def run(args):
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.history_root))
    import fit_tsc_predecision_history as history
    spec = json.loads(args.spec.read_text())
    try:
        if (spec["protocol"] != "tsc-matched-pair-supervision-development-v1"
                or spec["model"]["config"] != MODEL_CONFIG
                or sklearn.__version__ != "1.7.2" or spec["model"]["sklearn_version"] != "1.7.2"
                or spec["development"]["seeds"] != list(v158.DEVELOPMENT_SEEDS)
                or spec["crossfit"]["folds"] != [list(f) for f in v158.CROSSFIT_FOLDS]
                or spec["development"]["max_new_fits"] != 10
                or spec["development"]["new_simulator_trajectories"] != 0
                or spec["prediction"]["threshold"] != 0
                or spec["prediction"]["no_stay_override_of_pp_switch"] is not True
                or str(args.parent) != spec["parent_root"] or str(args.history_root) != spec["history_root"]):
            raise ValueError("run differs from frozen matched-pair protocol")
        data = history.load_mechanism_dataset(args.parent / "fit_v1/native_b100.npz")
        if (data.size != 800 or data.metadata.get("native_prefix") is not True
                or data.metadata.get("snapshot_restore_used") is not False):
            raise ValueError("expected unchanged original native B100 bank")
        histories = []
        for seed in spec["development"]["seeds"]:
            packet = json.loads((args.history_root / "history" / f"seed_{seed}" / "history.json").read_text())
            if packet["status"] != "PASS" or packet["seed"] != seed:
                raise ValueError("history collection incomplete")
            histories.extend(packet["groups"])
        data = history.join_history(data, histories)
        folds = history.seed_folds(data, spec["crossfit"]["folds"])
        old = json.loads((args.history_root / "analysis/oof_server_only.json").read_text())["records"]["history"]
        old_result = json.loads((args.history_root / "analysis/result.json").read_text())
        validate_records(data, old)
        old_costs = v158._seed_costs(old)
        if any(abs(cost - old_result["per_seed_costs"]["history"][str(seed)]) > v158.TOLERANCE
               for seed, cost in old_costs.items()):
            raise ValueError("reused history80 costs differ from parent result")
        state_names = (*v158.PAIRWISE_CAUSAL_STATE_PARENTS, *history.HISTORY_STATE_NAMES)
        action_names = (*v158.PAIRWISE_CAUSAL_ACTION_PARENTS, *v158.META_ACTION_FEATURES, *history.HISTORY_ACTION_NAMES)
        if (len(state_names), len(action_names)) != (35, 62):
            raise ValueError("history80 feature roster changed")
        prepared = []
        for training, heldout in folds:
            train = v158._arm_dataset(v158._subset_groups(data, training), "target_native")
            design = build_pair_design(train, state_names, action_names)
            validate_training_design(design)
            prepared.append((design, v158._subset_groups(data, heldout)))
    except (ValueError, KeyError, FileNotFoundError) as exc:
        result = {"status": "HOLD_INPUT", "error": f"{type(exc).__name__}: {exc}",
                  "new_fits": 0, "new_simulator_trajectories": 0}
        write_json(args.out_dir / "result.json", result)
        return result
    inputs = {"groups": 100, "candidate_rows": 800, "state_feature_names": list(state_names),
              "action_feature_names": list(action_names), "sklearn_version": sklearn.__version__,
              "folds": [design["diagnostics"] for design, _ in prepared]}
    write_json(args.out_dir / "input_validation.json", inputs)
    records, fits, predictions = {"S": [], "A": []}, [], []
    print("SCHEDULEURM_PROGRESS " + json.dumps({"current": 0, "total": 10, "unit": "fit",
                                              "elapsed_seconds": time.monotonic() - started}), flush=True)
    for index, (design, test) in enumerate(prepared):
        for arm in ("S", "A"):
            estimator = HistGradientBoostingRegressor(**MODEL_CONFIG)
            estimator.fit(design["X"], design["y"], sample_weight=design["weights"][arm])
            model = PairModel(estimator, state_names, action_names)
            selected = v158._policy_records(model, test, arm="target_native", prohibit_stay_override=True)
            validate_records(test, selected)
            records[arm].extend(selected)
            fits.append({"fold": index, "arm": arm, "heldout_seeds": spec["crossfit"]["folds"][index],
                         "weighted_training_mae": float(np.average(np.abs(design["y"] - estimator.predict(design["X"])),
                                                                    weights=design["weights"][arm])),
                         **design["diagnostics"]["arms"][arm]})
            predictions.append({"fold": index, "arm": arm,
                                "group_ids": np.asarray(test.metadata["action_group_ids"]).tolist(),
                                "candidate_states": np.asarray(test.metadata["candidate_states"]).tolist(),
                                "scores": model.last_scores.tolist()})
            write_json(args.out_dir / "oof_server_only.json", {"records": records, "fits": fits,
                                                               "predictions": predictions})
            print(json.dumps({"completed_fits": len(fits), "total_fits": 10, "fold": index, "arm": arm}), flush=True)
            print("SCHEDULEURM_PROGRESS " + json.dumps({"current": len(fits), "total": 10, "unit": "fit",
                                                      "elapsed_seconds": time.monotonic() - started}), flush=True)
    for rows in records.values():
        validate_records(data, rows)
    records["history80"] = old
    costs = {arm: v158._seed_costs(rows) for arm, rows in records.items()}
    costs["phase_pressure"] = v158._pp_seed_costs(old)
    analysis = spec["analysis"]
    comparisons = {f"A_minus_{arm}": v158._paired_summary(costs["A"], costs[arm],
        draws=analysis["bootstrap_draws"], seed=analysis["bootstrap_seed"])
        for arm in ("S", "history80", "phase_pressure")}
    behavior = {arm: history.behavior(rows) for arm, rows in records.items()}
    per_demand = {}
    for scenario in sorted(set(data.metadata["row_scenarios"])):
        per_demand[scenario] = {}
        for arm, rows in records.items():
            selected = [row for row in rows if row["scenario"] == scenario]
            values = v158._seed_costs(selected)
            per_demand[scenario][arm] = {"behavior": history.behavior(selected), "per_seed_cost": values,
                                         "mean_cost": float(np.mean(list(values.values())))}
        pp = v158._pp_seed_costs([row for row in old if row["scenario"] == scenario])
        per_demand[scenario]["phase_pressure"] = {"per_seed_cost": pp, "mean_cost": float(np.mean(list(pp.values())))}
    gate = {"all_three_paired_upper_bounds_negative": all(row["paired_bootstrap_95"][1] < 0 for row in comparisons.values()),
            "harmful_cost_not_increased_vs_S": behavior["A"]["raw_harm_sum"] <= behavior["S"]["raw_harm_sum"],
            "harmful_cost_not_increased_vs_history80": behavior["A"]["raw_harm_sum"] <= behavior["history80"]["raw_harm_sum"]}
    result = {"protocol": spec["protocol"], "status": "PASS_DEVELOPMENT" if all(gate.values()) else "HOLD_DEVELOPMENT",
              "groups": 100, "candidate_rows": 800, "new_fits": len(fits), "new_simulator_trajectories": 0,
              "mean_costs": {arm: float(np.mean(list(values.values()))) for arm, values in costs.items()},
              "per_seed_costs": costs, "comparisons": comparisons, "behavior": behavior, "per_demand": per_demand,
              "bridge_S_minus_history80": v158._paired_summary(costs["S"], costs["history80"],
                  draws=analysis["bootstrap_draws"], seed=analysis["bootstrap_seed"]),
              "gate": gate, "fit_diagnostics": fits, "input_validation": inputs,
              "elapsed_sec": time.monotonic() - started, "limits": spec["limits"],
              "independent_validation_preparation_supported": all(gate.values()), "closed_loop_authorized": False}
    write_json(args.out_dir / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "history-root", "spec", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "new_fits", "mean_costs", "gate", "error") if key in result}), flush=True)
