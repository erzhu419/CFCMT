"""Paired whole-seed history ablation on the unchanged V158 native B100 labels."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158
from cf_h2o.traffic_signal.action_ranker import CausalReferenceResidualRegressor, PairwisePreferenceConfig
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset

STATE_FIELDS = ("total_q", "total_veh", "mean_speed", "mean_occ",
                "graph_upstream_q_mean", "graph_downstream_q_mean")
ACTION_FIELDS = ("green_q", "red_q", "green_veh", "red_veh", "green_speed_mean",
                 "green_down_q", "green_down_occ", "movement_service_pressure")
HISTORY_STATE_NAMES = tuple(f"hist_{stat}_{field}" for field in STATE_FIELDS for stat in ("mean", "trend60"))
HISTORY_ACTION_NAMES = tuple(f"hist_{stat}_{field}" for field in ACTION_FIELDS for stat in ("mean", "trend60"))
HISTORY_NAMES = HISTORY_STATE_NAMES + HISTORY_ACTION_NAMES
MODEL_CONFIG = dict(learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
                    min_samples_leaf=20, l2_regularization=10.0,
                    uncertainty_quantile=0.9, random_state=20260803)


def join_history(data, records):
    """Join only complete, causally earlier histories to their original action rows."""
    groups = np.asarray(data.metadata["action_group_ids"], dtype=str)
    indexed = {row["group_id"]: row for row in records}
    if len(indexed) != len(records) or set(indexed) != set(groups):
        raise ValueError("history requires one-to-one complete action-group coverage")
    history = np.empty((data.size, len(HISTORY_NAMES)))
    for group, row in indexed.items():
        positions = np.flatnonzero(groups == group)
        if len(positions) != 8:
            raise ValueError("history requires the original eight-action groups")
        for key, field in (("row_seeds", "seed"), ("row_scenarios", "scenario"),
                           ("row_times", "checkpoint_sec"), ("row_tls", "tls_id")):
            original = np.asarray(data.metadata[key])[positions]
            if not np.all(original == row[field]):
                raise ValueError(f"history {field} does not match its original group")
        states = np.asarray(data.metadata["candidate_states"])[positions].tolist()
        if row["candidate_states"] != states:
            raise ValueError("history candidate order differs from the native bank")
        expected_times = [row["checkpoint_sec"] - offset for offset in range(60, 0, -10)]
        if row["history_times"] != expected_times:
            raise ValueError("history times must be checkpoint-60 through checkpoint-10")
        if tuple(row["feature_names"]) != HISTORY_NAMES:
            raise ValueError("history feature names are missing or reordered")
        parity = row.get("parity", {})
        if not parity or not all(value is True for value in parity.values()):
            raise ValueError("history replay failed static checkpoint parity")
        values = np.asarray(row["features"], dtype=float)
        if values.shape != (8, len(HISTORY_NAMES)) or not np.all(np.isfinite(values)):
            raise ValueError("history features are incomplete or nonfinite")
        if not np.array_equal(values[:, :12], np.repeat(values[:1, :12], 8, axis=0)):
            raise ValueError("history state features must be shared by all candidate actions")
        history[positions] = values
    return replace(data, feature_names=(*data.feature_names, *HISTORY_NAMES),
                   features=np.column_stack([data.features, history]))


def seed_folds(data, folds):
    if tuple(tuple(map(int, fold)) for fold in folds) != v158.CROSSFIT_FOLDS:
        raise ValueError("whole-seed fold roster differs from V158")
    groups = np.asarray(data.metadata["action_group_ids"], dtype=str)
    seeds = np.asarray(data.metadata["row_seeds"], dtype=int)
    mapping = {}
    for group in dict.fromkeys(groups):
        values = np.unique(seeds[groups == group])
        if len(values) != 1:
            raise ValueError("action group crosses simulator seeds")
        mapping[group] = int(values[0])
    if set(mapping.values()) != set(v158.DEVELOPMENT_SEEDS):
        raise ValueError("development seed roster differs from V158")
    result = []
    for fold in folds:
        heldout = {group for group, seed in mapping.items() if seed in fold}
        training = set(mapping) - heldout
        if len(heldout) != 20 or len(training) != 80:
            raise ValueError("whole-seed split must contain 80 training and 20 held-out groups")
        result.append((training, heldout))
    return result


def fit_model(data, history_enabled):
    arm = v158._arm_dataset(data, "target_native")
    state = (*v158.PAIRWISE_CAUSAL_STATE_PARENTS, *HISTORY_STATE_NAMES)
    action = (*v158.PAIRWISE_CAUSAL_ACTION_PARENTS, *v158.META_ACTION_FEATURES, *HISTORY_ACTION_NAMES)
    missing = set((*state, *action)) - set(arm.feature_names)
    if missing:
        raise ValueError(f"model missing declared features: {sorted(missing)}")
    if not history_enabled:
        arm.features[:, [arm.feature_names.index(name) for name in HISTORY_NAMES]] = 0.0
    model = CausalReferenceResidualRegressor(config=PairwisePreferenceConfig(**MODEL_CONFIG),
                                            state_feature_names=state, action_feature_names=action)
    diagnostics = model.fit(arm)
    if tuple(diagnostics["state_feature_names"]) != state or tuple(diagnostics["action_feature_names"]) != action:
        raise ValueError("fitted model did not retain the declared history inputs")
    return model, diagnostics


def policy_records(model, data, history_enabled):
    if not history_enabled:
        data = v158._copy_dataset(data)
        data.features[:, [data.feature_names.index(name) for name in HISTORY_NAMES]] = 0.0
    return v158._policy_records(model, data, arm="target_native", prohibit_stay_override=True)


def behavior(records):
    gains = np.asarray([row["actual_advantage"] for row in records], dtype=float)
    overrides = sum(row["different_action"] for row in records)
    harmful = int(np.count_nonzero(gains < -v158.TOLERANCE))
    return dict(groups=len(records), overrides=int(overrides), harmful_overrides=harmful,
                harmful_fraction=harmful / overrides if overrides else None,
                raw_benefit_sum=float(np.maximum(gains, 0).sum()),
                raw_harm_sum=float(np.maximum(-gains, 0).sum()),
                raw_net_gain_mean=float(gains.mean()))


def run(args):
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    protocol = json.loads(args.protocol.read_text())
    if protocol["model"]["config"] != MODEL_CONFIG or tuple(protocol["development"]["seeds"]) != v158.DEVELOPMENT_SEEDS:
        raise ValueError("history ablation changed the fixed model or seed budget")
    parent = json.loads(args.parent_fit_result.read_text())
    if parent["protocol"] != v158.FIT_RESULT_PROTOCOL:
        raise ValueError("expected the original V158 fit result")
    data = load_mechanism_dataset(args.bank)
    if (data.size != 800 or data.metadata.get("native_prefix") is not True
            or data.metadata.get("snapshot_restore_used") is not False):
        raise ValueError("expected the unchanged native B100 bank")
    histories = []
    for seed in v158.DEVELOPMENT_SEEDS:
        packet = json.loads((args.history_root / f"seed_{seed}" / "history.json").read_text())
        if packet["status"] != "PASS" or packet["seed"] != seed:
            raise ValueError(f"seed {seed}: incomplete history collection")
        histories.extend(packet["groups"])
    data = join_history(data, histories)
    folds = seed_folds(data, protocol["crossfit"]["folds"])
    records, fit_diagnostics = {"static": [], "history": []}, []
    for index, (training, heldout) in enumerate(folds):
        train, test = v158._subset_groups(data, training), v158._subset_groups(data, heldout)
        diagnostics = {}
        for arm in records:
            model, diagnostics[arm] = fit_model(train, arm == "history")
            records[arm].extend(policy_records(model, test, arm == "history"))
        fit_diagnostics.append(dict(fold=index, heldout_seeds=protocol["crossfit"]["folds"][index], models=diagnostics))
    costs = {arm: v158._seed_costs(rows) for arm, rows in records.items()}
    costs["phase_pressure"] = v158._pp_seed_costs(records["static"])
    expected = {int(seed): cost for seed, cost in parent["oof"]["seed_costs"]["target_native"].items()}
    if set(expected) != set(costs["static"]):
        raise ValueError("parent static seed roster changed")
    reproduction = max(abs(costs["static"][seed] - expected[seed]) for seed in expected)
    if reproduction > v158.TOLERANCE:
        raise ValueError(f"static arm failed V158 reproduction: {reproduction}")
    analysis = protocol["analysis"]
    comparisons = {f"history_minus_{arm}": v158._paired_summary(
        costs["history"], costs[arm], draws=analysis["bootstrap_draws"], seed=analysis["bootstrap_seed"])
        for arm in ("static", "phase_pressure")}
    behaviors = {arm: behavior(rows) for arm, rows in records.items()}
    per_demand = {}
    for scenario in sorted(set(data.metadata["row_scenarios"])):
        selected = {arm: [row for row in rows if row["scenario"] == scenario] for arm, rows in records.items()}
        per_demand[scenario] = {arm: dict(behavior(rows), per_seed_cost=v158._seed_costs(rows),
            per_seed_behavior={seed: behavior([row for row in rows if row["seed"] == seed])
                               for seed in v158.DEVELOPMENT_SEEDS}) for arm, rows in selected.items()}
        per_demand[scenario]["phase_pressure_seed_cost"] = v158._pp_seed_costs(selected["static"])
    checks = dict(both_paired_upper_bounds_below_zero=all(row["paired_bootstrap_95"][1] < 0 for row in comparisons.values()),
                  positive_net_gain=behaviors["history"]["raw_net_gain_mean"] > 0,
                  harmful_cost_not_increased=behaviors["history"]["raw_harm_sum"] <= behaviors["static"]["raw_harm_sum"])
    passed = all(checks.values())
    result = dict(protocol=protocol["protocol"], status="PASS_DEVELOPMENT" if passed else "HOLD_DEVELOPMENT",
                  scope="reused V158 development bank; whole-seed OOF, not independent confirmation",
                  groups=100, rows=800, simulator_seeds=10, folds=5,
                  estimand="equal seeds; within seed 10 groups weighted 4/3/3 by demand",
                  mean_costs={arm: float(np.mean(list(values.values()))) for arm, values in costs.items()},
                  per_seed_costs=costs, comparisons=comparisons, behavior=behaviors, per_demand=per_demand,
                  static_parent_max_absolute_difference=reproduction,
                  state_feature_count=35, action_feature_count=62, history_feature_names=list(HISTORY_NAMES),
                  gate=checks, independent_validation_preparation_supported=passed,
                  closed_loop_authorized=False)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    for name, value in (("result.json", result), ("oof_server_only.json", dict(records=records, fits=fit_diagnostics))):
        (args.out_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "history-root", "parent-fit-result", "protocol", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    result = run(parser.parse_args())
    print(json.dumps({key: result[key] for key in ("status", "mean_costs", "comparisons", "gate")}, allow_nan=False))


if __name__ == "__main__":
    main()
