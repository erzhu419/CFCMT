#!/usr/bin/env python3
"""Recover causal history by passive PP replay; reuse native labels only on parity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158
from cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches import (
    compare_physical, physical_snapshot, phase_pressure_decision,
)
from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset


STATE_FIELDS = (
    "total_q", "total_veh", "mean_speed", "mean_occ",
    "graph_upstream_q_mean", "graph_downstream_q_mean",
)
ACTION_FIELDS = (
    "green_q", "red_q", "green_veh", "red_veh", "green_speed_mean",
    "green_down_q", "green_down_occ", "movement_service_pressure",
)
FIELDS = (*STATE_FIELDS, *ACTION_FIELDS)
FEATURE_NAMES = tuple(
    f"hist_{summary}_{field}" for field in FIELDS for summary in ("mean", "trend60")
)
OFFSETS = np.arange(-60, 0, 10)


def history_summary(samples, times, checkpoint):
    """Input is six times x current candidate roster x fourteen physical fields."""
    values = np.asarray(samples, dtype=float)
    if not np.array_equal(np.asarray(times), int(checkpoint) + OFFSETS):
        raise ValueError("history must use exactly the six preceding observations")
    if values.shape != (6, 8, len(FIELDS)) or not np.isfinite(values).all():
        raise ValueError("history observations are incomplete or nonfinite")
    centered = OFFSETS - OFFSETS.mean()
    trend = np.einsum("t,tcf->cf", centered, values) / np.sum(centered ** 2) * 60.0
    return np.stack((values.mean(axis=0), trend), axis=-1).reshape(8, -1)


class HistoryProbe:
    selection_layer = "passive_predecision_history_recovery"

    def __init__(self, api, groups, bank, traces):
        self.api, self.groups, self.bank, self.traces = api, groups, bank, traces
        self.by_point = {(g["checkpoint_sec"], g["tls_id"]): g for g in groups}
        self.buffer = {}
        self.recovered = []
        self.indices = [v3.FEATURE_NAMES_V3.index(name) for name in FIELDS]

    def prepare_interval(self, **kwargs):
        states, self.executors = kwargs["states"], kwargs["executors"]
        self.now = int(v158._strict_time(states))
        # Only observed numeric copies are retained; no mutable executor is stored.
        needed = {
            g["tls_id"] for g in self.groups
            if self.now in (int(g["checkpoint_sec"]) + OFFSETS)
        }
        values = {}
        for tls in needed:
            state = states[tls]
            values[tls] = {
                candidate.state: v3.candidate_features_v3(
                    state, candidate, self.executors[tls], control_interval_sec=10
                )[self.indices].copy()
                for candidate in state.info.candidates
            }
        self.buffer[self.now] = values
        self.buffer = {t: value for t, value in self.buffer.items() if t >= self.now - 60}

    def select(self, dataset, *, reference_index):
        tls = str(dataset.metadata["row_tls"][0])
        group = self.by_point.get((self.now, tls))
        if group is not None:
            group_id = group["group_id"]
            candidate_states = list(dataset.metadata["candidate_states"])
            rows = np.flatnonzero(np.asarray(self.bank.metadata["action_group_ids"]) == group_id)
            bank_indices = [self.bank.feature_names.index(n) for n in dataset.feature_names]
            expected = self.bank.features[rows][:, bank_indices]
            physical = compare_physical(
                self.traces[group_id]["baseline_snapshot"],
                physical_snapshot(self.api, self.executors), tolerance=0.0,
            )
            parity = {
                "candidate_states": candidate_states == group["candidate_states"],
                "reference_index": int(reference_index) == group["reference_index"],
                "features": np.array_equal(dataset.features, expected),
                "context": np.array_equal(dataset.context, self.bank.context[rows]),
                "physical": physical["passed"],
            }
            if not all(parity.values()):
                raise ValueError(json.dumps({"group": group_id, "parity": parity, "physical": physical}))
            times = (self.now + OFFSETS).tolist()
            samples = [
                [self.buffer[t][tls][candidate] for candidate in candidate_states]
                for t in times
            ]
            self.recovered.append({
                "group_id": group_id, "seed": group["seed"], "scenario": group["scenario"],
                "checkpoint_sec": self.now, "tls_id": tls,
                "candidate_states": candidate_states, "history_times": times,
                "feature_names": list(FEATURE_NAMES),
                "features": history_summary(samples, times, self.now).tolist(),
                "parity": parity,
            })
        return phase_pressure_decision(
            dataset, reference_index=reference_index, arm="phase_pressure",
            rejection="passive_history_recovery",
        )


def collect_seed(*, seed, parent_root, bank, manifest, output, api):
    started = time.monotonic()
    source = parent_root / "development_v1" / f"seed_{seed}" / "result.json"
    old = json.loads(source.read_text())
    if old["status"] != "PASS" or old["seed"] != seed:
        raise ValueError("parent collection is incomplete")
    traces = pickle.loads(Path(old["full_traces_server_only"]).read_bytes())
    paths = v158._scenario_paths(manifest)
    recovered, receipts = [], []
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=False)
    for scenario in old["scenarios"]:
        groups = [g for g in old["groups"] if g["scenario"] == scenario]
        probe = HistoryProbe(api, groups, bank, traces)
        duration = max(g["checkpoint_sec"] for g in groups) + 10
        metrics = v158._evaluate_originator(
            api=api, v3=v3, sumocfg=paths[scenario], scenario=scenario, seed=seed,
            duration_sec=duration, originator=probe, observer=None,
            tripinfo_path=seed_dir / f"{scenario}_tripinfo.xml",
        )
        if (not metrics["ok"] or metrics.get("accepted_intervention_trace", [])
                or len(probe.recovered) != len(groups)):
            raise ValueError("passive replay failed or missed a checkpoint")
        recovered.extend(probe.recovered)
        receipts.append({"scenario": scenario, "duration_sec": duration,
                         "groups": len(probe.recovered), "learned_interventions": 0})
        print(json.dumps({"seed": seed, **receipts[-1]}), flush=True)
    result = {"status": "PASS", "seed": seed, "groups": recovered,
              "trajectories": receipts, "elapsed_sec": time.monotonic() - started,
              "label_branches_run": 0, "parent_seed_result": str(source)}
    (seed_dir / "history.json").write_text(json.dumps(result, allow_nan=False) + "\n")
    return {"seed": seed, "groups": len(recovered), "elapsed_sec": result["elapsed_sec"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if not set(args.seeds).issubset(protocol["development"]["seeds"]):
        raise ValueError("seed outside declared development bank")
    if protocol["history"]["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("history design differs from protocol")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(args.conversion_root)
    api = load_libsumo()
    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("requires original SUMO 1.22.0 runtime")
    bank = load_mechanism_dataset(args.parent_root / "fit_v1/native_b100.npz")
    for seed in args.seeds:
        receipt = collect_seed(seed=seed, parent_root=args.parent_root, bank=bank,
                               manifest=args.manifest, output=args.output, api=api)
        print(json.dumps({"completed": receipt}), flush=True)


if __name__ == "__main__":
    main()
