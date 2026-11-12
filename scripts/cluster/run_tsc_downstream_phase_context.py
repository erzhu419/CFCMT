#!/usr/bin/env python3
"""One frozen five-fold phase-context comparison; no simulator or new labels."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tsc_downstream_phase_features import FEATURE_NAMES, NetworkTopology, features_for_group
from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158
from cf_h2o.traffic_signal.action_ranker import CausalReferenceResidualRegressor, PairwisePreferenceConfig


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def group_rows(data):
    ids = np.asarray(data.metadata["action_group_ids"])
    return {g: np.flatnonzero(ids == g) for g in dict.fromkeys(ids.tolist())}


def join_features(data, packets):
    rows = group_rows(data)
    if len(packets) != len(rows) or {p["group_id"] for p in packets} != set(rows):
        raise ValueError("phase features require complete one-to-one group coverage")
    values = np.empty((data.size, len(FEATURE_NAMES)))
    for packet in packets:
        indices = rows[packet["group_id"]]
        for key, field in (("row_seeds", "seed"), ("row_scenarios", "scenario"),
                           ("row_times", "checkpoint_sec"), ("row_tls", "tls_id")):
            if not np.all(np.asarray(data.metadata[key])[indices] == packet[field]):
                raise ValueError(f"phase feature {field} differs from native bank")
        states = np.asarray(data.metadata["candidate_states"])[indices].tolist()
        if len(indices) != 8 or packet["candidate_states"] != states:
            raise ValueError("phase candidate order differs from native bank")
        array = np.asarray(packet["features"], dtype=float)
        if array.shape != (8, 6) or not np.isfinite(array).all():
            raise ValueError("phase features incomplete or nonfinite")
        values[indices] = array
    return replace(data, feature_names=(*data.feature_names, *FEATURE_NAMES),
                   features=np.column_stack((data.features, values)))


def input_support(data, folds):
    columns = [data.feature_names.index(name) for name in FEATURE_NAMES]
    by_group = {}
    for group, rows in group_rows(data).items():
        refs = rows[np.asarray(data.metadata["is_reference"], dtype=bool)[rows]]
        if len(refs) != 1:
            raise ValueError("phase group needs exactly one PP reference")
        x = data.features[rows][:, columns]
        delta = x - data.features[refs[0], columns]
        by_group[group] = {
            "scenario": data.metadata["row_scenarios"][rows[0]],
            "max_abs_delta": np.abs(delta).max(axis=0),
            "nonzero": np.any(np.abs(delta) > v158.TOLERANCE, axis=0),
        }
    def stats(items):
        return {name: {"nonzero_groups": int(sum(p["nonzero"][i] for p in items)),
                       "max_abs_action_minus_pp": float(max(p["max_abs_delta"][i] for p in items))}
                for i, name in enumerate(FEATURE_NAMES)}
    informative = {g for g, p in by_group.items() if np.any(p["nonzero"][1:])}
    training_counts = [len(training & informative) for training, _ in folds]
    return {"passed": bool(informative) and all(count > 0 for count in training_counts),
            "numeric_zero_tolerance": v158.TOLERANCE,
            "phase_informative_groups": len(informative),
            "training_fold_informative_groups": training_counts,
            "all_groups": stats(list(by_group.values())),
            "by_demand": {s: stats([p for p in by_group.values() if p["scenario"] == s])
                          for s in sorted({p["scenario"] for p in by_group.values()})}}


def validate_records(data, records):
    groups = group_rows(data)
    if len(records) != len(groups) or {r["group_id"] for r in records} != set(groups):
        raise ValueError("OOF comparison requires all original groups exactly once")
    for record in records:
        rows = groups[record["group_id"]]
        states = np.asarray(data.metadata["candidate_states"])[rows].tolist()
        selected = rows[states.index(record["selected_state"])]
        reference = rows[np.asarray(data.metadata["is_reference"], dtype=bool)[rows]][0]
        if (record["reference_state"] != data.metadata["candidate_states"][reference]
                or record["seed"] != data.metadata["row_seeds"][reference]
                or record["scenario"] != data.metadata["row_scenarios"][reference]):
            raise ValueError("OOF record identity differs from original bank")
        absolute = data.targets["absolute_cost"]
        for key, value in (("selected_cost", absolute[selected]), ("phase_pressure_cost", absolute[reference]),
                           ("actual_advantage", absolute[reference] - absolute[selected])):
            if abs(record[key] - value) > v158.TOLERANCE:
                raise ValueError(f"OOF {key} differs from original native label")
        if record["different_action"] != bool(selected != reference):
            raise ValueError("OOF override flag differs from selected state")


def fit_model(data, history):
    arm = v158._arm_dataset(data, "target_native")
    state = (*v158.PAIRWISE_CAUSAL_STATE_PARENTS, *history.HISTORY_STATE_NAMES)
    action = (*v158.PAIRWISE_CAUSAL_ACTION_PARENTS, *v158.META_ACTION_FEATURES,
              *history.HISTORY_ACTION_NAMES, *FEATURE_NAMES)
    if set((*state, *action)) - set(arm.feature_names):
        raise ValueError("model missing frozen input columns")
    model = CausalReferenceResidualRegressor(config=PairwisePreferenceConfig(**history.MODEL_CONFIG),
                                            state_feature_names=state, action_feature_names=action)
    diagnostics = model.fit(arm)
    if tuple(diagnostics["state_feature_names"]) != state or tuple(diagnostics["action_feature_names"]) != action:
        raise ValueError("model dropped frozen phase context inputs")
    return model, diagnostics


def extract_packets(args, spec):
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(args.conversion_root)
    networks, netpaths = {}, {}
    for scenario, cfg in v158._scenario_paths(args.manifest).items():
        net = (cfg.parent / ET.parse(cfg).find("./input/net-file").get("value")).resolve()
        networks[scenario], netpaths[scenario] = NetworkTopology.from_net(net), str(net)
    packets = []
    for seed in spec["development"]["seeds"]:
        parent = json.loads((args.parent / "development_v1" / f"seed_{seed}" / "result.json").read_text())
        if parent["status"] != "PASS" or parent["seed"] != seed:
            raise ValueError("native parent collection incomplete")
        traces = pickle.loads(Path(parent["full_traces_server_only"]).read_bytes())
        for group in parent["groups"]:
            snapshot = traces[group["group_id"]]["baseline_snapshot"]
            if snapshot["time_sec"] != group["checkpoint_sec"]:
                raise ValueError("executor snapshot is not the original action checkpoint")
            values, mapping = features_for_group(networks[group["scenario"]], snapshot,
                                                 group["tls_id"], group["candidate_states"])
            packets.append({**{key: group[key] for key in ("group_id", "seed", "scenario", "checkpoint_sec",
                                                          "tls_id", "candidate_states")},
                            "features": values.tolist(), "mapping": mapping})
        print(json.dumps({"extracted_seed": seed, "groups": len(packets)}), flush=True)
    return packets, netpaths


def run(args):
    started = time.monotonic()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.history_root))
    import fit_tsc_predecision_history as history
    spec = json.loads(args.spec.read_text())
    if (spec["model"]["config"] != history.MODEL_CONFIG
            or spec["development"]["seeds"] != list(v158.DEVELOPMENT_SEEDS)
            or spec["crossfit"]["folds"] != [list(f) for f in v158.CROSSFIT_FOLDS]
            or [x["name"] for x in spec["new_action_features"]] != list(FEATURE_NAMES)
            or spec["action_selection"] != {"threshold": 0, "no_stay_override_of_pp_switch": True,
                                            "reference": "phase_pressure"}):
        raise ValueError("phase context run differs from frozen protocol")
    data = history.load_mechanism_dataset(args.parent / "fit_v1/native_b100.npz")
    if data.size != 800 or data.metadata.get("native_prefix") is not True:
        raise ValueError("expected original 100-group native bank")
    histories = []
    for seed in spec["development"]["seeds"]:
        packet = json.loads((args.history_root / "history" / f"seed_{seed}" / "history.json").read_text())
        if packet["status"] != "PASS" or packet["seed"] != seed:
            raise ValueError("history recovery incomplete")
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
    try:
        packets, netpaths = extract_packets(args, spec)
        data = join_features(data, packets)
        support = input_support(data, folds)
    except (ValueError, KeyError) as exc:
        result = {"status": "HOLD_INPUT", "error": f"{type(exc).__name__}: {exc}", "new_fits": 0,
                  "new_simulator_trajectories": 0}
        write_json(args.out_dir / "result.json", result)
        return result
    write_json(args.out_dir / "features_server_only.json", packets)
    write_json(args.out_dir / "input_support.json", support)
    if not support["passed"]:
        result = {"status": "HOLD_INPUT", "input_support": support, "new_fits": 0,
                  "new_simulator_trajectories": 0, "reason": "No phase-conditioned support in bank or a training fold."}
        write_json(args.out_dir / "result.json", result)
        return result
    records, fits = [], []
    for index, (training, heldout) in enumerate(folds):
        model, diagnostics = fit_model(v158._subset_groups(data, training), history)
        test = v158._subset_groups(data, heldout)
        prediction = v158._policy_records(model, test, arm="target_native", prohibit_stay_override=True)
        validate_records(test, prediction)
        records.extend(prediction)
        fits.append({"fold": index, "heldout_seeds": spec["crossfit"]["folds"][index], "diagnostics": diagnostics})
        write_json(args.out_dir / "oof_server_only.json", {"records": records, "fits": fits})
        print(json.dumps({"completed_fits": len(fits), "total_fits": 5}), flush=True)
    validate_records(data, records)
    costs = {"phase_context": v158._seed_costs(records), "history80": old_costs,
             "phase_pressure": v158._pp_seed_costs(old)}
    analysis = spec["analysis"]
    comparisons = {f"phase_context_minus_{arm}": v158._paired_summary(
        costs["phase_context"], costs[arm], draws=analysis["bootstrap_draws"], seed=analysis["bootstrap_seed"])
        for arm in ("history80", "phase_pressure")}
    behavior = {"phase_context": history.behavior(records), "history80": history.behavior(old)}
    per_demand = {}
    for scenario in sorted(set(data.metadata["row_scenarios"])):
        per_demand[scenario] = {arm: {"behavior": history.behavior(selected),
            "per_seed_cost": v158._seed_costs(selected),
            "mean_cost": float(np.mean(list(v158._seed_costs(selected).values())))}
            for arm, rows in (("phase_context", records), ("history80", old))
            for selected in ([r for r in rows if r["scenario"] == scenario],)}
        pp = v158._pp_seed_costs([r for r in old if r["scenario"] == scenario])
        per_demand[scenario]["phase_pressure"] = {"per_seed_cost": pp, "mean_cost": float(np.mean(list(pp.values())))}
    gate = {"both_paired_upper_bounds_negative": all(r["paired_bootstrap_95"][1] < 0 for r in comparisons.values()),
            "positive_net_advantage": behavior["phase_context"]["raw_net_gain_mean"] > 0,
            "harmful_cost_not_increased": behavior["phase_context"]["raw_harm_sum"] <= behavior["history80"]["raw_harm_sum"]}
    result = {"protocol": spec["protocol"], "status": "PASS_DEVELOPMENT" if all(gate.values()) else "HOLD_DEVELOPMENT",
              "groups": 100, "candidate_rows": 800, "new_fits": len(fits), "new_simulator_trajectories": 0,
              "mean_costs": {arm: float(np.mean(list(values.values()))) for arm, values in costs.items()},
              "per_seed_costs": costs, "comparisons": comparisons, "behavior": behavior,
              "per_demand": per_demand, "input_support": support, "gate": gate,
              "state_feature_count": 35, "action_feature_count": 68, "feature_names": list(FEATURE_NAMES),
              "network_paths": netpaths, "elapsed_sec": time.monotonic() - started,
              "scope": spec["scope"], "closed_loop_authorized": False}
    write_json(args.out_dir / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "history-root", "manifest", "conversion-root", "spec", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({k: result[k] for k in ("status", "new_fits", "mean_costs", "gate", "error") if k in result}), flush=True)
