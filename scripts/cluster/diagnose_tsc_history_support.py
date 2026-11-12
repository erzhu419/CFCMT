"""Describe frozen OOF errors and training support without fitting a policy."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import numpy as np


def distances(query, reference):
    square = (query * query).sum(1)[:, None] + (reference * reference).sum(1)[None, :] - 2 * query @ reference.T
    return np.sqrt(np.maximum(square, 0) / max(query.shape[1], 1))


def support(query, train, query_seeds, train_seeds, train_groups, train_scenarios,
            query_scenarios, train_gains):
    """Train-only scaling; calibrate distances with other training seeds only."""
    scale = train.std(0)
    variable = scale > 1e-12
    mean = train.mean(0)
    x, q = (train[:, variable] - mean[variable]) / scale[variable], (query[:, variable] - mean[variable]) / scale[variable]
    if set(query_seeds) & set(train_seeds):
        raise ValueError("held-out simulator seeds leaked into support reference")
    calibration = distances(x, x)
    calibration[np.asarray(train_seeds)[:, None] == np.asarray(train_seeds)[None, :]] = np.inf
    reference_nn = calibration.min(axis=1)
    if not np.isfinite(reference_nn).all():
        raise ValueError("coverage needs other training seeds")
    d = distances(q, x)
    groups = np.asarray(train_groups)
    result = []
    for i in range(len(query)):
        # One nearest candidate per training group: seven actions are not seven states.
        neighbors = []
        for group in dict.fromkeys(groups.tolist()):
            rows = np.flatnonzero(groups == group)
            neighbors.append(int(rows[np.argmin(d[i, rows])]))
        neighbors.sort(key=lambda row: (float(d[i, row]), str(groups[row])))
        top = neighbors[:5]
        nearest = top[0]
        same = np.asarray(train_scenarios) == query_scenarios[i]
        same_reference_nn = calibration[np.ix_(same, same)].min(axis=1)
        same_top = [row for row in neighbors if same[row]][:5]
        same_distance = float(d[i, same].min())
        result.append({
            "distance": float(d[i, nearest]),
            "training_percentile": float(np.mean(reference_nn <= d[i, nearest])),
            "nearest_other_demand": bool(train_scenarios[nearest] != query_scenarios[i]),
            "same_demand_distance": same_distance,
            "same_demand_percentile": float(np.mean(same_reference_nn <= same_distance)),
            "novel_constant_columns": int(np.count_nonzero(np.abs(query[i, ~variable] - mean[~variable]) > 1e-10)),
            "neighbor_groups": [str(groups[j]) for j in top],
            "neighbor_gains_vehicle_sec": [int(train_gains[j]) for j in top] if train_gains is not None else None,
            "same_demand_neighbor_gains_vehicle_sec": [int(train_gains[j]) for j in same_top] if train_gains is not None else None,
        })
    return result


def action_outcome(costs, switch, reference, selected):
    """Costs in integer halted-vehicle-seconds; keep the original executed action."""
    costs = np.asarray(costs, dtype=np.int64)
    allowed = np.ones(len(costs), dtype=bool)
    if switch[reference] > 1e-12:
        allowed = np.asarray(switch) > 1e-12
        allowed[reference] = True
    if not allowed[selected]:
        raise ValueError("saved selected action violates no-stay execution rule")
    pp, chosen = int(costs[reference]), int(costs[selected])
    ordered = np.sort(costs[allowed])
    best = int(ordered[0])
    headroom, gain = pp - best, pp - chosen
    if gain < 0:
        category = "harmful_despite_beneficial_alternative" if headroom > 0 else "harmful_when_pp_optimal"
    elif gain == 0:
        category = "fallback_misses_benefit" if headroom > 0 else "optimal_reference"
    else:
        category = "optimal_improvement" if chosen == best else "beneficial_but_suboptimal"
    return dict(gain_vehicle_sec=gain, harm_vehicle_sec=max(-gain, 0),
                feasible_headroom_vehicle_sec=headroom, regret_vehicle_sec=chosen - best,
                missed_benefit_vehicle_sec=min(chosen, pp) - best,
                best_second_gap_vehicle_sec=int(ordered[1] - ordered[0]),
                unconstrained_headroom_vehicle_sec=pp - int(costs.min()),
                category=category)


def resolve_selected(data, record):
    rows = np.flatnonzero(np.asarray(data.metadata["action_group_ids"]) == record["group_id"])
    states = np.asarray(data.metadata["candidate_states"])[rows]
    selected = np.flatnonzero(states == record["selected_state"])
    reference = np.flatnonzero(states == record["reference_state"])
    if len(rows) != 8 or len(selected) != 1 or len(reference) != 1:
        raise ValueError("OOF group and phase cannot be joined uniquely")
    selected, reference = int(selected[0]), int(reference[0])
    for name, position in (("selected_cost", selected), ("phase_pressure_cost", reference)):
        if abs(data.targets["absolute_cost"][rows[position]] - record[name]) > 1e-12:
            raise ValueError("saved OOF cost differs from joined bank")
    if not data.metadata["is_reference"][rows[reference]]:
        raise ValueError("saved PP phase is not the bank reference")
    if any(data.metadata[key][rows[0]] != record[field]
           for key, field in (("row_seeds", "seed"), ("row_scenarios", "scenario"))):
        raise ValueError("OOF group seed or demand changed")
    return rows, reference, selected


def quantiles(values):
    values = np.asarray(values, dtype=float)
    return dict(zip(("min", "q25", "median", "q75", "max"),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1])))) if values.size else None


def summarize(rows):
    harmful = [r for r in rows if r["harm_vehicle_sec"] > 0]
    overrides = [r for r in rows if r["override"]]
    result = dict(groups=len(rows), overrides=len(overrides), harmful=len(harmful),
                  categories=dict(Counter(r["category"] for r in rows)),
                  net_gain_vehicle_sec=sum(r["gain_vehicle_sec"] for r in rows),
                  harm_vehicle_sec=sum(r["harm_vehicle_sec"] for r in rows),
                  feasible_headroom_vehicle_sec=sum(r["feasible_headroom_vehicle_sec"] for r in rows),
                  unconstrained_headroom_vehicle_sec=sum(r["unconstrained_headroom_vehicle_sec"] for r in rows),
                  missed_benefit_vehicle_sec=sum(r["missed_benefit_vehicle_sec"] for r in rows),
                  harmful_magnitude=quantiles([r["harm_vehicle_sec"] for r in harmful]),
                  best_second_gap=quantiles([r["best_second_gap_vehicle_sec"] for r in rows]),
                  harmful_within_10_vehicle_sec=sum(r["harm_vehicle_sec"] <= 10 for r in harmful))
    result["support"] = {}
    for space in ("state", "action"):
        observed = [r for r in rows if r[space] is not None]
        strata = {}
        # Descriptive train-calibrated bins only; no policy threshold is evaluated.
        for name, lo, hi in (("lower_half", -1, .5), ("middle", .5, .95), ("upper_5pct", .95, 1)):
            sample = [r for r in observed if lo < r[space]["training_percentile"] <= hi]
            strata[name] = dict(groups=len(sample), overrides=sum(r["override"] for r in sample),
                harmful=sum(r["harm_vehicle_sec"] > 0 for r in sample),
                harm_vehicle_sec=sum(r["harm_vehicle_sec"] for r in sample),
                net_gain_vehicle_sec=sum(r["gain_vehicle_sec"] for r in sample))
        result["support"][space] = dict(strata=strata,
            percentile_harmful=quantiles([r[space]["training_percentile"] for r in observed if r["harm_vehicle_sec"] > 0]),
            percentile_beneficial=quantiles([r[space]["training_percentile"] for r in observed if r["gain_vehicle_sec"] > 0]),
            nearest_other_demand=sum(r[space]["nearest_other_demand"] for r in observed),
            novel_constant_groups=sum(r[space]["novel_constant_columns"] > 0 for r in observed))
        same_tail = [r for r in observed if r[space]["same_demand_percentile"] > .95]
        result["support"][space]["same_demand"] = dict(
            upper_5pct_groups=len(same_tail),
            upper_5pct_harmful=sum(r["harm_vehicle_sec"] > 0 for r in same_tail),
            upper_5pct_harm_vehicle_sec=sum(r["harm_vehicle_sec"] for r in same_tail),
            percentile_harmful=quantiles([r[space]["same_demand_percentile"] for r in observed if r["harm_vehicle_sec"] > 0]))
        if space == "action":
            mixed = [r for r in observed if min(r[space]["neighbor_gains_vehicle_sec"]) < 0 < max(r[space]["neighbor_gains_vehicle_sec"])]
            result["support"][space].update(mixed_neighbor_sign_groups=len(mixed),
                mixed_neighbor_sign_harmful=sum(r["harm_vehicle_sec"] > 0 for r in mixed))
            same_mixed = [r for r in observed if min(r[space]["same_demand_neighbor_gains_vehicle_sec"]) < 0 < max(r[space]["same_demand_neighbor_gains_vehicle_sec"])]
            result["support"][space]["same_demand"].update(mixed_neighbor_sign_groups=len(same_mixed),
                mixed_neighbor_sign_harmful=sum(r["harm_vehicle_sec"] > 0 for r in same_mixed))
    return result


def lane_counts(conversion_root, scenarios):
    counts = {}
    for scenario in scenarios:
        cfg = conversion_root / scenario / f"{scenario}.sumocfg"
        net = cfg.parent / ET.parse(cfg).find(".//net-file").attrib["value"]
        lanes = {(c.attrib["from"], c.attrib["fromLane"]) for c in ET.parse(net).iter("connection") if c.attrib.get("tl")}
        counts[scenario] = len(lanes)
    return counts


def run(args):
    sys.path.insert(0, str(args.history_experiment))
    import fit_tsc_predecision_history as history
    v158 = history.v158
    data = history.load_mechanism_dataset(args.bank)
    previous = json.loads((args.history_experiment / "analysis/result.json").read_text())
    oof = json.loads((args.history_experiment / "analysis/oof_server_only.json").read_text())
    packets = [json.loads((args.history_experiment / "history" / f"seed_{s}" / "history.json").read_text()) for s in v158.DEVELOPMENT_SEEDS]
    data = history.join_history(data, [r for packet in packets for r in packet["groups"]])
    folds = history.seed_folds(data, v158.CROSSFIT_FOLDS)
    groups = np.asarray(data.metadata["action_group_ids"])
    seeds = np.asarray(data.metadata["row_seeds"])
    scenarios = np.asarray(data.metadata["row_scenarios"])
    references = np.asarray(data.metadata["is_reference"], dtype=bool)
    switch = data.features[:, data.feature_names.index("switch_indicator")]
    lanes = lane_counts(args.conversion_root, sorted(set(scenarios)))
    units = np.asarray([450 * lanes[s] for s in scenarios])
    raw = data.targets["absolute_cost"] * units
    if np.max(np.abs(raw - np.rint(raw))) > 1e-7:
        raise ValueError("native costs do not match the verified lane-time counting unit")
    costs = np.rint(raw).astype(np.int64)
    gains = np.zeros(data.size, dtype=np.int64)
    unique_groups = list(dict.fromkeys(groups.tolist()))
    for group in unique_groups:
        rows = np.flatnonzero(groups == group)
        gains[rows] = costs[rows[references[rows]][0]] - costs[rows]
    excluded = {"time_sin", "time_cos", "lane_count_norm", "candidate_count_norm", "graph_in_degree_norm", "graph_out_degree_norm"}
    records, features_used = {}, {}
    for arm in ("static", "history"):
        state_names = tuple(n for n in v158.PAIRWISE_CAUSAL_STATE_PARENTS if n not in excluded)
        action_names = tuple(v158.PAIRWISE_CAUSAL_ACTION_PARENTS)
        if arm == "history":
            state_names += history.HISTORY_STATE_NAMES
            action_names += history.HISTORY_ACTION_NAMES
        state = data.features[:, [data.feature_names.index(n) for n in state_names]]
        action = data.features[:, [data.feature_names.index(n) for n in action_names]].copy()
        for group in unique_groups:
            rows = np.flatnonzero(groups == group)
            action[rows] -= action[rows[references[rows]][0]].copy()
        design = np.column_stack((state, action))
        features_used[arm] = dict(state=list(state_names), action=list(action_names))
        indexed = {r["group_id"]: r for r in oof["records"][arm]}
        if set(indexed) != set(unique_groups) or len(oof["records"][arm]) != len(unique_groups):
            raise ValueError("OOF group roster is incomplete or duplicated")
        output = []
        for fold, (training, heldout) in enumerate(folds):
            train_first = np.array([np.flatnonzero(groups == g)[0] for g in unique_groups if g in training])
            queries, selected_rows, joined = [], [], []
            for group in unique_groups:
                if group not in heldout:
                    continue
                record = indexed[group]
                rows, ref, sel = resolve_selected(data, record)
                outcome = action_outcome(costs[rows], switch[rows], ref, sel)
                outcome.update(group_id=group, seed=record["seed"], scenario=record["scenario"], fold=fold,
                    override=sel != ref, predicted_gain_vehicle_sec=record["predicted_advantage"] * units[rows[0]],
                    state=None, action=None)
                queries.append(rows[0]); selected_rows.append(rows[sel]); joined.append(outcome)
            queries, selected_rows = np.asarray(queries), np.asarray(selected_rows)
            state_support = support(state[queries], state[train_first], seeds[queries], seeds[train_first], groups[train_first], scenarios[train_first], scenarios[queries], None)
            candidates = np.flatnonzero(np.isin(groups, list(training)) & ~references)
            override_indices = np.array([i for i, r in enumerate(joined) if r["override"]], dtype=int)
            selected = selected_rows[override_indices]
            action_support = support(design[selected], design[candidates], seeds[selected], seeds[candidates], groups[candidates], scenarios[candidates], scenarios[selected], gains[candidates]) if len(selected) else []
            for row, coverage in zip(joined, state_support): row["state"] = coverage
            for i, coverage in zip(override_indices, action_support): joined[i]["action"] = coverage
            output.extend(joined)
        # Reconcile errors against the already reported experiment, before diagnosis.
        if sum(r["harm_vehicle_sec"] > 0 for r in output) != previous["behavior"][arm]["harmful_overrides"]:
            raise ValueError("diagnostic changed saved harmful intervention count")
        records[arm] = output
    summaries = {arm: dict(overall=summarize(rows),
        by_demand={s: summarize([r for r in rows if r["scenario"] == s]) for s in sorted(set(scenarios))},
        by_seed={int(s): summarize([r for r in rows if r["seed"] == s]) for s in sorted(set(seeds))}) for arm, rows in records.items()}
    result = dict(protocol="tsc-history-support-diagnostic-v1", status="DIAGNOSTIC_COMPLETE", controlled_lane_counts=lanes,
        groups=100, simulator_seeds=10, refits=0, simulator_runs=0, policy_changed=False,
        all_nonreference_labels=dict(better=int(np.sum(gains[~references] > 0)), worse=int(np.sum(gains[~references] < 0)),
            equal=int(np.sum(gains[~references] == 0)), absolute_vehicle_sec=quantiles(np.abs(gains[~references]))),
        features_used=features_used, summaries=summaries,
        scope="Descriptive analysis of reused selected B100 groups. Distances diagnose feature support, not a causal data-size effect or a policy gate; one future per action.")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "records_server_only.json").write_text(json.dumps(records, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("history-experiment", "bank", "conversion-root", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "controlled_lane_counts": result["controlled_lane_counts"],
                      "overall": {a: x["overall"] for a, x in result["summaries"].items()}}, allow_nan=False))
