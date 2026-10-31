"""Assemble the fixed 16384-row target union from eight complete native days."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from native_runtime import HERE, bootstrap

ROWS_PER_DAY = 2048
ROLE_TO_SPLIT_ID = {"fit": 0, "stop": 1, "calibrate": 2}


def selected_indices(total_rows, count=ROWS_PER_DAY):
    """Keep chronological, equally spaced original transitions, including tail."""
    if total_rows < count:
        return None
    return np.linspace(0, total_rows - 1, count, dtype=np.int64)


def read_metadata(data):
    return {name: json.loads(value) for name, value in data.attrs.items()}


def assemble_pool(protocol, input_root, output_root, canonicalize, schema_version):
    """Select by row positions only; the original reward/next-state pair stays intact."""
    input_root, output_root = Path(input_root), Path(output_root)
    cases = protocol["collection_cases"]
    planned = protocol["planned_training"]
    if any(case["target_rows"] != ROWS_PER_DAY for case in cases):
        raise ValueError("each frozen day must contribute exactly 2048 rows")
    if len({(case["demand_seed"], case["sumo_seed"]) for case in cases}) != len(cases):
        raise ValueError("collection cases reuse the same demand/SUMO day")
    allocations = {role: sum(case["split"] == role for case in cases) * ROWS_PER_DAY
        for role in ROLE_TO_SPLIT_ID}
    if allocations != planned["target_splits"] or sum(allocations.values()) != planned["target_unique_transition_budget"]:
        raise ValueError("the case allocation differs from the fixed target budget")
    inventory = []
    for case in cases:
        path = input_root / f"case{case['case']:02d}" / "transitions.h5"
        day_summary = json.loads(path.with_name("summary.json").read_text())
        if not day_summary["complete"] or day_summary["diagnostic"]:
            raise ValueError(f"case{case['case']:02d} is not a complete formal day")
        if day_summary["horizon_seconds"] != case["episode_seconds"]:
            raise ValueError(f"case{case['case']:02d} actual collection horizon "
                f"{day_summary['horizon_seconds']} differs from {case['episode_seconds']}")
        with h5py.File(path, "r") as data:
            meta = read_metadata(data)
            for key in ("case", "split", "method", "episode_seconds", "demand_seed", "sumo_seed", "action_seed"):
                if meta[key] != case[key]:
                    raise ValueError(f"case{case['case']:02d} metadata differs at {key}")
            count = len(data["observations"])
            if int(data["terminals"][-1]) != 1 or np.sum(data["terminals"][:]) != 1:
                raise ValueError("a native day must retain its single final terminal tail")
            inventory.append(dict(case=case["case"], split=case["split"], source_rows=count,
                path=str(path), metadata=meta))
    short_days = [{key: row[key] for key in ("case", "split", "source_rows")}
        for row in inventory if row["source_rows"] < ROWS_PER_DAY]
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "pool_summary.json"
    if short_days:
        summary = dict(complete=False, decision="INSUFFICIENT", insufficient_cases=short_days,
            rows_per_case=ROWS_PER_DAY, target_total_unique_budget=planned["target_unique_transition_budget"],
            actual_cases=len(cases), training_launched=False)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    first = inventory[0]["metadata"]
    chunks = []
    day_reports = []
    for case, row in zip(cases, inventory):
        meta = row["metadata"]
        for key in ("station_max_by_line", "line_ids", "config", "sumo_version"):
            if meta[key] != first[key]:
                raise ValueError(f"case{case['case']:02d} uses different {key}")
        indices = selected_indices(row["source_rows"])
        with h5py.File(row["path"], "r") as data:
            selected = {name: dataset[indices] for name, dataset in data.items()}
        selected["raw_observations"] = selected.pop("observations")
        selected["raw_next_observations"] = selected.pop("next_observations")
        selected["observations"] = canonicalize(selected["raw_observations"], meta).astype(np.float32)
        selected["next_observations"] = canonicalize(selected["raw_next_observations"], meta).astype(np.float32)
        if (selected["observations"].shape != (ROWS_PER_DAY, 229)
                or selected["raw_observations"].shape != (ROWS_PER_DAY, 245)):
            raise ValueError("the target pool has incorrect canonical/raw state dimensions")
        global_id = case["trajectory_id"]
        for name, value in (("split_id", ROLE_TO_SPLIT_ID[case["split"]]),
                ("policy_id", 0), ("policy_seed", case["action_seed"]),
                ("sumo_seed", case["sumo_seed"]), ("demand_realization_seed", case["demand_seed"]),
                ("episode_id", global_id), ("trajectory_id", global_id),
                ("global_trajectory_id", global_id), ("collection_case", case["case"])):
            selected[name] = np.full(ROWS_PER_DAY, value, dtype=np.int64)
        selected["target_role"] = np.full(ROWS_PER_DAY, case["split"], dtype="S9")
        selected["source_row_index"] = indices
        eligible = (selected["delta_t_seconds"] > 0) & (selected["terminals"] == 0)
        selected["world_model_fit_eligible"] = eligible & (case["split"] == "fit")
        chunks.append(selected)
        day_reports.append(dict(case=case["case"], split=case["split"], global_trajectory_id=global_id,
            source_rows=row["source_rows"], selected_rows=ROWS_PER_DAY,
            physical_dt_positive_rows=int(np.sum(selected["delta_t_seconds"] > 0)),
            simultaneous_dt_zero_rows=int(np.sum(selected["delta_t_seconds"] == 0)),
            eligible_positive_dt_nonterminal_rows=int(np.sum(eligible)),
            final_terminal_source_row=int(indices[-1]), asset_manifest=meta["asset_manifest"],
            scheduled_service_date=meta["scheduled_service_date"], demand_seed=case["demand_seed"],
            sumo_seed=case["sumo_seed"], action_seed=case["action_seed"]))
    arrays = {name: np.concatenate([chunk[name] for chunk in chunks]) for name in chunks[0]}
    split_reports = {}
    for role, split_id in ROLE_TO_SPLIT_ID.items():
        mask = arrays["split_id"] == split_id
        split_reports[role] = dict(rows=int(np.sum(mask)),
            trajectory_ids=np.unique(arrays["trajectory_id"][mask]).tolist(),
            physical_dt_positive_rows=int(np.sum(mask & (arrays["delta_t_seconds"] > 0))),
            simultaneous_dt_zero_rows=int(np.sum(mask & (arrays["delta_t_seconds"] == 0))),
            eligible_positive_dt_nonterminal_rows=int(np.sum(mask & (arrays["delta_t_seconds"] > 0)
                & (arrays["terminals"] == 0))))
    eligible_fit = split_reports["fit"]["eligible_positive_dt_nonterminal_rows"]
    summary = dict(complete=True, decision="READY", total_unique_rows=len(arrays["rewards"]),
        target_total_unique_budget=planned["target_unique_transition_budget"],
        rows_per_case=ROWS_PER_DAY, cases=day_reports, splits=split_reports,
        world_model_fit_row_cap=min(planned["world_model_fit_rows_within_fit"], eligible_fit),
        world_model_fit_rows_used=0, training_launched=False, actor_split_id=0,
        split_id_to_target_role={str(value): key for key, value in ROLE_TO_SPLIT_ID.items()},
        final_test_included=False, normalization_data_role="fit", dataset_file="target_pool.h5",
        selection="Equally spaced chronological row indices with the final terminal row; each row retains its original adjacent next state and reward",
        reward_accounting="Selected rewards describe selected original intervals; their sum is not a full-day passenger cost")
    with h5py.File(output_root / "target_pool.h5", "w") as data:
        for name, array in arrays.items():
            if array.dtype.kind == "O":
                data.create_dataset(name, data=array, dtype=h5py.string_dtype("utf-8"))
            else:
                data.create_dataset(name, data=array, compression="gzip")
        data.attrs["schema_version"] = schema_version
        for name, value in dict(config=first["config"], station_max_by_line=first["station_max_by_line"],
                policy_names={"mixture": 0}, source_case_metadata=[row["metadata"] for row in inventory],
                target_role_to_split_id=ROLE_TO_SPLIT_ID, final_test_included=False,
                target_total_unique_budget=planned["target_unique_transition_budget"],
                normalization_data_role="fit", pool_summary=summary).items():
            data.attrs[name] = json.dumps(value)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    protocol = json.loads((HERE / "protocol.json").read_text())
    api = bootstrap()
    def canonicalize(raw, metadata):
        return api.contract.raw_state_to_canonical_v7(raw, metadata["station_max_by_line"],
            api.contract.TransitMDPConfigV7(**metadata["config"]))
    summary = assemble_pool(protocol, args.input_root, args.output,
        canonicalize, api.contract.SCHEMA_VERSION)
    print(json.dumps({key: summary[key] for key in ("complete", "decision")}), flush=True)


if __name__ == "__main__":
    main()
