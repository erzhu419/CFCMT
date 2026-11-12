#!/usr/bin/env python3
"""Replay the nine frozen native action cases; keep second-level traces on server."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np

from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3
from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158
from cf_h2o.eval.traffic_signal_v157c_jinan_native_action_branches import (
    _new_window_record, _record_step_diagnostics, _sample, _window_summary,
    compare_physical, physical_snapshot,
)
from cf_h2o.sumo_runtime import load_libsumo, libsumo_version
from cf_h2o.traffic_signal.dataset_cache import load_mechanism_dataset


WINDOWS = ((1, 10), (11, 60), (61, 120), (121, 450))


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")


def reconstruct_group(case, old, bank):
    group = dict(next(g for g in old["groups"] if g["group_id"] == case["group_id"]))
    rows = np.flatnonzero(np.asarray(bank.metadata["action_group_ids"]) == case["group_id"])
    states = [bank.metadata["candidate_states"][i] for i in rows]
    if states != group["candidate_states"]:
        raise ValueError("native bank candidate order changed")
    reference, selected = group["reference_index"], states.index(case["selected_state"])
    if states[reference] != case["reference_state"] or selected == reference:
        raise ValueError("frozen reference/selected action changed")
    for index, name in ((reference, "reference_cost_450"), (selected, "selected_cost_450")):
        if abs(group["native_costs"][index] - case[name]) > 1e-12:
            raise ValueError("case endpoint differs from native parent")
    # The originator receives action contrasts, including reference_* / delta_*.
    # Only the nine legacy model prediction columns were added by the native bank.
    names = tuple(name for name in bank.feature_names if name not in v158.RAW_PREDICTION_FEATURES)
    columns = [bank.feature_names.index(name) for name in names]
    group["dataset"] = SimpleNamespace(
        feature_names=names, context_names=bank.context_names,
        features=bank.features[rows][:, columns], context=bank.context[rows],
    )
    return group, selected


def phase_snapshot(api, executor, tls):
    value = executor.snapshot()
    value.pop("audit")
    value["sumo_state"] = str(api.trafficlight.getRedYellowGreenState(tls))
    return value


class TimelineOriginator(v158.FixedActionBranchOriginator):
    def prepare_interval(self, **kwargs):
        super().prepare_interval(**kwargs)
        self.info = kwargs["states"][self.focal_tls].info


class TimelineObserver:
    def __init__(self, api, originator):
        self.api, self.originator = api, originator
        self.record, self.start, self.rows = _new_window_record(), {}, []
        self.applied = None

    def __call__(self, *, stage, time_sec, executors):
        now, checkpoint = int(round(time_sec)), self.originator.checkpoint_sec
        api, tls = self.api, self.originator.focal_tls
        if stage == "before_simulation_step" and checkpoint <= now < checkpoint + 450:
            # The request has already been applied, but this second is not simulated yet.
            self.applied = (now, phase_snapshot(api, executors[tls], tls))
        if stage != "after_executor_advance":
            return
        if now == checkpoint:
            self.start = physical_snapshot(api, executors)
        if not checkpoint < now <= checkpoint + 450:
            return
        if self.applied is None or self.applied[0] != now - 1:
            raise ValueError("missing applied phase for the simulated second")
        sample = _sample(api, self.originator.controlled_lanes, float(now))
        self.record["samples"].append(sample)
        _record_step_diagnostics(api, self.record)
        info = self.originator.info
        outgoing = sorted({str(link[1]) for links in info.controlled_links for link in links if link})
        self.rows.append({
            "offset_sec": now - checkpoint, **sample,
            "applied": self.applied[1],
            "end": phase_snapshot(api, executors[tls], tls),
            "incoming_q": {lane: int(api.lane.getLastStepHaltingNumber(lane))
                           for lane in info.incoming_lanes},
            "outgoing_q": {lane: int(api.lane.getLastStepHaltingNumber(lane)) for lane in outgoing},
            # SUMO 1.22 libsumo returns occupied length / lane length, not percent.
            "outgoing_occupancy_fraction": {lane: float(api.lane.getLastStepOccupancy(lane))
                                            for lane in outgoing},
        })


def phase_runs(rows, states):
    runs = []
    for row in rows:
        phase = row["applied"]
        key = (phase["mode"], phase["sumo_state"], phase["target_green_state"])
        if runs and runs[-1]["key"] == key:
            runs[-1]["end_sec"] = row["offset_sec"]
        else:
            runs.append({"key": key, "start_sec": row["offset_sec"],
                         "end_sec": row["offset_sec"], "mode": key[0],
                         "green_index": states.index(key[1]) if key[1] in states else None,
                         "target_index": states.index(key[2])})
    return [{k: v for k, v in row.items() if k != "key"} for row in runs]


def summarize_pair(case, branches, lane_count, states):
    pp, selected = branches["pp"]["rows"], branches["selected"]["rows"]
    if any([r["offset_sec"] for r in rows] != list(range(1, 451)) for rows in (pp, selected)):
        raise ValueError("paired timeline must contain exactly seconds 1 through 450")
    result = {**case, "controlled_lane_count": lane_count, "windows": [],
              "phases": {arm: phase_runs(branches[arm]["rows"], states) for arm in branches}}
    cumulative = 0
    for lo, hi in WINDOWS:
        metrics = {}
        for arm, rows in (("pp", pp), ("selected", selected)):
            window = rows[lo - 1:hi]
            metrics[arm] = {
                "network_halted_vehicle_sec": round(sum(r["cost"] for r in window) * lane_count),
                "focal_halted_vehicle_sec": sum(sum(r["incoming_q"].values()) for r in window),
                "outgoing_halted_vehicle_sec": sum(sum(r["outgoing_q"].values()) for r in window),
                "outgoing_occupancy_mean_fraction": float(np.mean([
                    np.mean(list(r["outgoing_occupancy_fraction"].values())) for r in window])),
                "applied_mode_seconds": dict(Counter(r["applied"]["mode"] for r in window)),
            }
        delta = metrics["selected"]["network_halted_vehicle_sec"] - metrics["pp"]["network_halted_vehicle_sec"]
        local = metrics["selected"]["focal_halted_vehicle_sec"] - metrics["pp"]["focal_halted_vehicle_sec"]
        cumulative += delta
        result["windows"].append({"start_sec": lo, "end_sec": hi, "arms": metrics,
                                  "delta_vehicle_sec": delta, "cumulative_delta_vehicle_sec": cumulative,
                                  "focal_delta_vehicle_sec": local, "other_tls_delta_vehicle_sec": delta - local})
    result["delta_vehicle_sec"] = cumulative
    result["sign_convention"] = "selected minus PP; positive means harm"
    result["candidate_states"] = states
    return result


def collect_case(index, case, *, parent, bank, paths, output, api):
    started = time.monotonic()
    case_dir = output / f"case_{index:02d}"
    case_dir.mkdir(parents=True, exist_ok=False)
    old = json.loads((parent / "development_v1" / f"seed_{case['seed']}" / "result.json").read_text())
    if old["status"] != "PASS":
        raise ValueError("native parent incomplete")
    original = pickle.loads(Path(old["full_traces_server_only"]).read_bytes())[case["group_id"]]
    group, selected = reconstruct_group(case, old, bank)
    branches = {}
    for arm, candidate, expected in (("pp", group["reference_index"], case["reference_cost_450"]),
                                     ("selected", selected, case["selected_cost_450"])):
        originator = TimelineOriginator(group, candidate)
        observer = TimelineObserver(api, originator)
        runtime = v158._evaluate_originator(
            api=api, v3=v3, sumocfg=paths[case["scenario"]], scenario=case["scenario"],
            seed=case["seed"], duration_sec=case["checkpoint_sec"] + 450,
            originator=originator, observer=observer, tripinfo_path=case_dir / f"{arm}_tripinfo.xml",
        )
        if runtime.get("ok") is not True:
            failure = {"status": "REPLAY_ERROR", "runtime": runtime,
                       "dataset_match": originator.dataset_match,
                       "sample_count": len(observer.rows),
                       "checkpoint_reached": bool(observer.start)}
            write_json(case_dir / f"{arm}_receipt.json", failure)
            print(json.dumps(failure), flush=True)
            raise ValueError(f"case {index} {arm}: {runtime.get('error')}")
        summary = _window_summary(observer.record, observer.start)
        physical = compare_physical(original["baseline_snapshot"], observer.start, tolerance=0.0)
        trace = runtime.get("accepted_intervention_trace", [])
        checks = {
            "evaluator_complete": runtime.get("ok") is True,
            "physical_start_exact": physical["passed"],
            "candidate_dataset_exact": bool(originator.dataset_match and all(originator.dataset_match.values())),
            "decision_recorded": originator.decision is not None,
            "window_complete": summary["sample_count"] == 450,
            "population_accounting": summary["active_population_accounting_passed"],
            "no_teleport": runtime.get("starting_teleports") == runtime.get("ending_teleports") == 0,
            "original_endpoint_reproduced": abs(summary["mean_halted_per_controlled_lane"] - expected) < 1e-12,
            "intervention_count_and_point": trace == [] if arm == "pp" else (
                len(trace) == 1 and trace[0]["time_sec"] == case["checkpoint_sec"]
                and trace[0]["tls_id"] == case["tls_id"] and trace[0]["execution_effective"] is True),
            "phase_writer_parity": all(r["applied"]["current_state"] == r["applied"]["sumo_state"]
                                       and r["end"]["current_state"] == r["end"]["sumo_state"] for r in observer.rows),
        }
        if arm == "pp":
            checks["baseline_cost_path_exact"] = all(
                a["time_sec"] == b["time_sec"] and a["cost"] == b["cost"]
                for a, b in zip(observer.rows, original["baseline_samples"])) and len(original["baseline_samples"]) == 450
        receipt = {"checks": checks, "summary": summary, "physical": physical,
                   "decision": originator.decision, "intervention_trace": trace,
                   "controlled_lane_count": len(originator.controlled_lanes)}
        write_json(case_dir / f"{arm}_receipt.json", receipt)
        write_json(case_dir / f"{arm}_timeline_server_only.json", observer.rows)
        print(json.dumps({"case": index, "arm": arm, "checks": checks}), flush=True)
        if not all(checks.values()):
            raise ValueError(f"case {index} {arm}: replay validity failed; no mechanism interpretation")
        branches[arm] = {"rows": observer.rows, "receipt": receipt}
    result = summarize_pair(case, branches, len(originator.controlled_lanes), group["candidate_states"])
    result.update(status="PASS", case_index=index, elapsed_sec=time.monotonic() - started,
                  receipts={arm: b["receipt"] for arm, b in branches.items()})
    write_json(case_dir / "result.json", result)
    print(json.dumps({"completed_case": index, "delta_vehicle_sec": result["delta_vehicle_sec"],
                      "elapsed_sec": result["elapsed_sec"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "manifest", "conversion-root", "spec", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--case-indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if tuple(map(tuple, spec["fixed_reporting_windows_sec"])) != WINDOWS:
        raise ValueError("reporting windows differ from frozen spec")
    if len(set(args.case_indices)) != len(args.case_indices) or not set(args.case_indices) <= set(range(9)):
        raise ValueError("case indices must be distinct members of the fixed nine cases")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(args.conversion_root)
    api = load_libsumo()
    if "1.22.0" not in str(libsumo_version()):
        raise ValueError("requires original SUMO 1.22.0 runtime")
    bank = load_mechanism_dataset(args.parent / "fit_v1/native_b100.npz")
    paths = v158._scenario_paths(args.manifest)
    for index in args.case_indices:
        collect_case(index, spec["cases"][index], parent=args.parent, bank=bank,
                     paths=paths, output=args.output, api=api)


if __name__ == "__main__":
    main()
