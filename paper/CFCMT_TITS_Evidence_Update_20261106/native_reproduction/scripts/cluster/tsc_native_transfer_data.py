"""Adapt existing Cologne/Jinan native labels to a shared physical-cost task."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

from cf_h2o.traffic_signal.action_contrast import rule_reference_indices
from cf_h2o.traffic_signal.action_ranker import (
    PAIRWISE_CAUSAL_ACTION_PARENTS, PAIRWISE_CAUSAL_STATE_PARENTS,
)
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset


HORIZON = 450


def common_feature_names(source, target):
    """Declared causal physical parents, intersected by names rather than outcomes."""
    common = set(source.feature_names) & set(target.feature_names)
    state = tuple(name for name in PAIRWISE_CAUSAL_STATE_PARENTS if name in common)
    action = tuple(name for name in PAIRWISE_CAUSAL_ACTION_PARENTS if name in common)
    if state != PAIRWISE_CAUSAL_STATE_PARENTS or action != PAIRWISE_CAUSAL_ACTION_PARENTS:
        raise ValueError("native transfer requires the audited 23 state and 40 action parents")
    return state, action


def canonicalize(data, city, lane_count, state_names, action_names):
    """Keep absolute per-lane costs for evaluation; learn PP-relative vehicle-seconds."""
    groups = np.asarray(data.metadata["action_group_ids"], dtype=str)
    phases = np.asarray(data.metadata["candidate_states"], dtype=str)
    ordered = list(dict.fromkeys(groups.tolist()))
    references = rule_reference_indices(data, policy="phase_pressure")
    refmask = np.zeros(data.size, dtype=bool)
    refrows = np.empty(data.size, dtype=int)
    if city == "cologne":
        absolute = np.asarray(data.targets["interval_cost"], dtype=float)
        if not np.array_equal(absolute, data.targets["prefix_mean_cost_450s"]):
            raise ValueError("Cologne interval_cost must be the absolute native 450-second mean")
        seeds = [int(re.search(r":seed(\d+):", group).group(1)) for group in groups]
        scenarios = [str(data.metadata["scenario"])] * data.size
        expected_scale = 8
    elif city == "jinan":
        absolute = np.asarray(data.targets["absolute_cost"], dtype=float)
        seeds = list(data.metadata["row_seeds"])
        scenarios = list(data.metadata["row_scenarios"])
        expected_scale = 144
    else:
        raise ValueError("this fixed adaptation only covers Cologne and Jinan")
    if lane_count != expected_scale or not np.isfinite(absolute).all() or np.any(absolute < 0):
        raise ValueError("native cost unit or absolute labels differ from the audited contract")
    names = (*state_names, *action_names)
    x = data.features[:, [data.feature_names.index(name) for name in names]].copy()
    if not np.isfinite(x).all():
        raise ValueError("physical input features must be finite")
    for group in ordered:
        rows = np.flatnonzero(groups == group)
        ref = references[group]
        if len(rows) != 8 or len(set(phases[rows])) != 8 or len(set(np.asarray(seeds)[rows])) != 1:
            raise ValueError("native action groups require eight distinct actions and one seed")
        if len(set(np.asarray(scenarios)[rows])) != 1:
            raise ValueError("native action group crosses scenarios")
        if not np.allclose(x[rows, :len(state_names)], x[rows[0], :len(state_names)], rtol=0, atol=1e-10):
            raise ValueError("declared pre-action state varies across candidate actions")
        refmask[ref], refrows[rows] = True, ref
    residual = absolute - absolute[refrows]
    if city == "jinan":
        if not np.array_equal(refmask, np.asarray(data.metadata["is_reference"], dtype=bool)):
            raise ValueError("stored Jinan PP rows differ from stable first-maximum phase pressure")
        if not np.allclose(residual, data.targets["interval_cost"], rtol=0, atol=1e-12):
            raise ValueError("Jinan interval_cost is not its native candidate-minus-PP residual")
    meta = {key: list(data.metadata[key]) for key in
            ("action_group_ids", "candidate_states", "row_tls", "row_times")}
    meta.update(is_reference=refmask.tolist(), reference_rows=refrows.tolist(),
                row_seeds=seeds, row_scenarios=scenarios,
                native_prefix=True, snapshot_restore_used=False,
                controlled_lane_count=lane_count, horizon_sec=HORIZON,
                interval_cost_units="halted_vehicle_seconds_relative_to_PP",
                absolute_cost_units="mean_halted_vehicles_per_controlled_lane",
                source_city=city)
    canonical = replace(data, feature_names=names, features=x,
                        priors={"interval_cost": np.zeros(data.size)},
                        targets={"interval_cost": residual * lane_count * HORIZON,
                                 "absolute_cost": absolute.copy()}, metadata=meta)
    ranges = [float(np.ptp(residual[groups == g])) for g in ordered]
    audit = {"groups": len(ordered), "rows": data.size,
             "controlled_lane_count": lane_count, "horizon_sec": HORIZON,
             "physical_cost_factor": lane_count * HORIZON,
             "reference_rule": "phase_pressure; first maximum green_q in original candidate order",
             "reference_groups": int(refmask.sum()),
             "group_range_per_lane_min_median_max": [float(np.min(ranges)), float(np.median(ranges)), float(np.max(ranges))],
             "maximum_abs_physical_delta": float(np.abs(canonical.targets["interval_cost"]).max()),
             "seed_group_counts": {str(seed): int(sum(np.asarray(seeds)[np.flatnonzero(groups == g)[0]] == seed for g in ordered))
                                   for seed in sorted(set(seeds))}}
    return canonical, audit


def controlled_lane_count(sumocfg):
    cfg = Path(sumocfg)
    net_name = ET.parse(cfg).find("./input/net-file").get("value")
    net = (cfg.parent / net_name).resolve()
    connections = ET.parse(net).getroot().findall("connection")
    lanes = {(c.get("from"), c.get("fromLane")) for c in connections if c.get("tl")}
    signals = {c.get("tl") for c in connections if c.get("tl")}
    return len(lanes), len(signals), str(net)


def load_native_inputs(source_inputs, snapshot_root):
    """Load the fixed native P100+K95 and V158 banks, without model loading."""
    spec = source_inputs
    if str(snapshot_root) != spec["snapshot_root"]:
        raise ValueError("native input runtime snapshot differs from the frozen source inputs")
    # Parent stages this unchanged existing helper beside this adapter.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cologne_native_bank import assemble_expanded_native_bank
    source = assemble_expanded_native_bank(
        load_mechanism_dataset(Path(spec["source_base_bank"])),
        [load_mechanism_dataset(Path(part["bank"])) for part in spec["source_shards"]],
        spec["original_group_ids"] + spec["additional_group_ids"])
    target = load_mechanism_dataset(Path(spec["target_bank"]))
    contract = source.metadata["native_prefix_protocol"]
    expected = {"controlled_lane_count": 8, "first_interval_sec": 10, "horizon_sec": 450,
                "cost_samples": 450, "prefix_policy": "phase_pressure",
                "continuation_policy": "phase_pressure", "snapshot_restore_used": False,
                "cost_aggregation": "mean_native_halted_vehicles_per_controlled_lane"}
    if any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("Cologne native label execution contract changed")
    if (target.size != 800 or target.metadata.get("native_prefix") is not True
            or target.metadata.get("snapshot_restore_used") is not False):
        raise ValueError("Jinan requires the unchanged original native B100")
    topology = {"cologne": controlled_lane_count(spec["source_sumocfg"])}
    if topology["cologne"][:2] != (8, 1):
        raise ValueError("Cologne network is no longer the audited one-signal/eight-lane task")
    for scenario, path in spec["target_sumocfgs"].items():
        topology[scenario] = controlled_lane_count(path)
        if topology[scenario][:2] != (144, 12):
            raise ValueError("Jinan controlled network is no longer 12 signals/144 lanes")
    if set(target.domains) != set(spec["target_sumocfgs"]):
        raise ValueError("Jinan scenario roster differs from its three native demands")
    states, actions = common_feature_names(source, target)
    source, source_audit = canonicalize(source, "cologne", 8, states, actions)
    target, target_audit = canonicalize(target, "jinan", 144, states, actions)
    if source_audit["groups"] != 195 or target_audit["groups"] != 100:
        raise ValueError("direct transfer requires all 195 source and 100 target groups")
    audit = {"status": "PASS_INPUT", "source": source_audit, "target": target_audit,
             "state_feature_names": list(states), "action_feature_names": list(actions),
             "state_feature_count": len(states), "action_feature_count": len(actions),
             "topology": {k: {"controlled_lanes": v[0], "signals": v[1], "net": v[2]} for k, v in topology.items()},
             "source_native_contract": contract,
             "legacy_prediction_columns_used": False, "new_fits": 0, "new_simulations": 0,
             "limits": "Source is one junction; matching units does not equate its propagation effects with Jinan's network. Source and target retain their original selected development-state distributions."}
    return source, target, audit
