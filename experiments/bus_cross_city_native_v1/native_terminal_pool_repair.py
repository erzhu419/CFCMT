"""Copy the existing target pool and correct its eight full-horizon terminal tails."""

import json
import shutil

import h5py
import numpy as np

from native_runtime import HERE, bootstrap


def main():
    folder = HERE / "native_terminal_repair"
    admission = json.loads((folder / "case00_replay" / "comparison.json").read_text())
    assets = json.loads((HERE / "native_terminal_asset_audit.json").read_text())
    if (not admission["complete"] or admission["case"] != 0
            or admission["new_horizon_seconds"] != 24000 or not assets["complete"]):
        raise RuntimeError("Case00 terminal replay has not passed")
    original = HERE / "target_pool.h5"
    destination = folder / "target_pool.h5"
    if destination.exists():
        raise RuntimeError("Corrected pool already exists; keep its original provenance")
    api = bootstrap()
    changed = {"rewards", "wait_passenger_seconds", "delta_t_seconds",
               "raw_next_observations", "next_observations"}
    rows = []
    with h5py.File(original, "r") as source:
        config = api.contract.TransitMDPConfigV7(**json.loads(source.attrs["config"]))
        station_max = json.loads(source.attrs["station_max_by_line"])
        terminals = np.flatnonzero(source["terminals"][:])
        if len(terminals) != 8:
            raise RuntimeError("The original target pool does not contain eight terminal rows")
        for i in terminals:
            case = int(source["collection_case"][i])
            day = json.loads((HERE / "runs" / f"case{case:02d}" / "summary.json").read_text())
            evidence = next(d for d in assets["cases"] if d["case"] == case)
            counts = day["passengers"]["state_counts_at_horizon"]
            end, horizon = day["horizon_seconds"], day["episode_seconds"]
            if (counts["in_vehicle"] != 0 or counts["other_active"] != 0
                    or evidence["person_xml_count"] != day["passengers"]["departed_passengers"]
                    or evidence["asset_person_count"] != evidence["person_xml_count"]
                    or evidence["max_person_depart_seconds"] >= end
                    or assets["common_assets"]["last_bus_depart_seconds"] >= end
                    or assets["common_assets"]["person_plan_shapes"] != [["walk", "ride"]]):
                raise RuntimeError(f"case{case:02d} does not have a static empty-service waiting tail")
            extension = horizon - end
            added_wait = counts["waiting"] * extension
            snapshot = api.contract.passenger_time_snapshot_v5(dict(time_seconds=horizon,
                waiting_passengers=counts["waiting"], in_vehicle_passengers=0,
                accrued_wait_seconds=day["passengers"]["accrued_wait_seconds_total"] + added_wait,
                accrued_in_vehicle_seconds=day["passengers"]["accrued_in_vehicle_seconds_total"]))
            raw = api.contract.terminal_raw_state_v7(source["raw_observations"][i], snapshot)
            values = dict(rewards=source["rewards"][i] - 2 * added_wait,
                wait_passenger_seconds=source["wait_passenger_seconds"][i] + added_wait,
                delta_t_seconds=source["delta_t_seconds"][i] + extension,
                raw_next_observations=raw,
                next_observations=api.contract.raw_state_to_canonical_v7(raw, station_max, config))
            if case == 0:
                with h5py.File(folder / "case00_replay" / "transitions.h5", "r") as replay:
                    for key in ("rewards", "wait_passenger_seconds", "delta_t_seconds"):
                        if values[key] != replay[key][-1]:
                            raise RuntimeError("Analytic repair differs from measured case00: " + key)
                    if not np.array_equal(raw, replay["next_observations"][-1]):
                        raise RuntimeError("Analytic terminal state differs from measured case00")
            rows.append(dict(case=case, pool_row=int(i), split_id=int(source["split_id"][i]),
                source_row_index=int(source["source_row_index"][i]),
                old_horizon_seconds=end, new_horizon_seconds=horizon, waiting=counts["waiting"],
                added_wait_passenger_seconds=added_wait, reward_before=float(source["rewards"][i]),
                reward_after=float(values["rewards"]), values=values))
    shutil.copyfile(original, destination)
    with h5py.File(destination, "r+") as repaired:
        for row in rows:
            for key, value in row.pop("values").items():
                repaired[key][row["pool_row"]] = value
        report = dict(complete=True, original_pool=str(original), corrected_pool=str(destination),
            total_unique_rows=len(repaired["rewards"]), corrected_terminal_rows=rows,
            method="Measured original-action case00 replay; analytic static waiting extension for all eight terminal tails",
            replay_simulations=1, new_training_updates=0, original_pool_preserved=True,
            unchanged_fields="All observations/actions/roles/indices and all nonterminal next states/rewards/durations",
            asset_audit=assets)
        summary = json.loads(repaired.attrs["pool_summary"])
        summary.update(terminal_horizon_corrected=True, terminal_correction=report["method"],
            original_pool=str(original), training_launched=False)
        for day in summary["cases"]:
            row = next(r for r in rows if r["case"] == day["case"])
            day.update(original_horizon_seconds=row["old_horizon_seconds"], actual_horizon_seconds=24000)
        repaired.attrs["pool_summary"] = json.dumps(summary)
        repaired.attrs["terminal_horizon_correction"] = json.dumps(report)
    with h5py.File(original, "r") as source, h5py.File(destination, "r") as repaired:
        differences = {}
        for key in source:
            a, b = source[key][:], repaired[key][:]
            delta = a != b
            mask = delta if delta.ndim == 1 else np.any(delta, axis=tuple(range(1, delta.ndim)))
            different_rows = np.flatnonzero(mask)
            differences[key] = different_rows.tolist()
            if len(different_rows) and (key not in changed or not np.array_equal(different_rows, terminals)):
                raise RuntimeError("Unexpected corrected pool difference: " + key)
        if not np.array_equal(repaired["discounts"][:], 1-repaired["terminals"][:]):
            raise RuntimeError("Corrected discount/terminal masks differ")
        if not np.array_equal(repaired["rewards"][:],
                -(2*repaired["wait_passenger_seconds"][:] + repaired["in_vehicle_passenger_seconds"][:])):
            raise RuntimeError("Corrected rewards do not equal passenger ledger increments")
        report["field_changed_rows"] = differences
    (folder / "pool_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    (folder / "pool_repair.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(complete=True,corrected_terminal_rows=len(rows),total_unique_rows=report["total_unique_rows"])),flush=True)


if __name__ == "__main__":
    main()
