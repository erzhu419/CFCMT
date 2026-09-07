"""Run the v37 leave-one-group-out pairwise deployment screen."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_cross_fitted_pairwise import target_group_records
from cf_h2o.eval.traffic_signal_cross_fitted_pairwise_screen import (
    _fit_explicit_target_groups,
)
from cf_h2o.eval.traffic_signal_leave_one_group_pairwise import (
    LOGO_GROUP_COUNT,
    LOGO_TRAIN_GROUP_COUNT,
    select_leave_one_group_pairwise,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_gate import (
    TARGET_GROUP_BUDGET,
)
from cf_h2o.eval.traffic_signal_pairwise_deployment_screen import (
    DEPLOYMENT_FAMILIES,
    _group_regret_rows,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _group_subset_v3,
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    evaluate_target_action_groups,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import (
    load_traffic_signal_manifest,
)
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid


_LOGO_PROCESS_STATE: dict[str, Any] | None = None


def _logo_worker(group_id: str) -> dict[str, Any]:
    if _LOGO_PROCESS_STATE is None:
        raise RuntimeError("LOGO worker state was not initialized")
    started = time.monotonic()
    state = _LOGO_PROCESS_STATE
    selected_group_ids = state["selected_group_ids"]
    training_ids = tuple(
        value for value in selected_group_ids if value != str(group_id)
    )
    if len(training_ids) != LOGO_TRAIN_GROUP_COUNT:
        raise ValueError("LOGO training group count changed")
    models = _fit_explicit_target_groups(
        bank=state["bank"],
        target=state["target"],
        target_group_ids=training_ids,
        manifest=state["manifest"],
        source_rule_costs=state["source_rule_costs"],
        source_rule_specs=state["source_rule_specs"],
    )
    validation = _group_subset_v3(
        state["bank"][state["target"]],
        selected_groups={str(group_id)},
        metadata_updates={
            "target_data_role": "leave_one_group_out_validation",
            "logo_heldout_group_id": str(group_id),
        },
    )
    rows = _group_regret_rows(validation, models=models)
    if len(rows) != 1 or str(rows[0]["group_id"]) != str(group_id):
        raise ValueError("LOGO worker returned the wrong target group")
    row = rows[0]
    record = state["record_by_id"][str(group_id)]
    row.update(
        {
            "simulator_seed": int(record["simulator_seed"]),
            "tls_id": str(record["tls_id"]),
            "training_group_count": len(training_ids),
        }
    )
    return {
        "heldout_group_id": str(group_id),
        "training_group_ids": list(training_ids),
        "logo_row": row,
        "model_diagnostics": models.diagnostics,
        "elapsed_sec": float(time.monotonic() - started),
        "worker_pid": os.getpid(),
    }


def run_target_logo_screen(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest_path: Path,
    target: str,
    cache_workers: int,
    logo_workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    manifest = load_traffic_signal_manifest(manifest_path)
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=(2027, 3037, 4047),
        collection_shards=16,
        workers=cache_workers,
    )
    if target not in bank:
        raise ValueError(f"unknown target: {target}")
    baseline = json.loads(baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    split = _target_adaptation_split_v3(
        bank[target],
        group_budget=TARGET_GROUP_BUDGET,
        selection_seed=20260803,
        calibration_seeds=(4047,),
        calibration_fraction=0.4,
    )
    selected_group_ids = tuple(
        sorted(
            str(value)
            for value in split["selected"].metadata[
                "target_adaptation_selected_group_ids"
            ]
        )
    )
    if len(selected_group_ids) != LOGO_GROUP_COUNT:
        raise ValueError("LOGO screen requires exactly 60 selected groups")
    records = target_group_records(
        bank[target],
        selected_group_ids=selected_group_ids,
    )
    record_by_id = {str(row["group_id"]): row for row in records}
    actual_logo_workers = min(max(int(logo_workers), 1), LOGO_GROUP_COUNT)
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("LOGO process parallelism requires Linux fork support")
    global _LOGO_PROCESS_STATE
    _LOGO_PROCESS_STATE = {
        "bank": bank,
        "target": target,
        "selected_group_ids": selected_group_ids,
        "record_by_id": record_by_id,
        "manifest": manifest,
        "source_rule_costs": source_rule_costs,
        "source_rule_specs": source_rule_specs,
    }
    try:
        with ProcessPoolExecutor(
            max_workers=actual_logo_workers,
            mp_context=mp.get_context("fork"),
        ) as pool:
            futures = {
                group_id: pool.submit(_logo_worker, group_id)
                for group_id in selected_group_ids
            }
            logo_fits = []
            for index, group_id in enumerate(selected_group_ids, start=1):
                row = futures[group_id].result()
                logo_fits.append(row)
                print(
                    "CFCMT_LOGO_PROGRESS "
                    f"target={target} completed={index}/{LOGO_GROUP_COUNT} "
                    f"group={group_id} elapsed={row['elapsed_sec']:.1f}s",
                    flush=True,
                )
    finally:
        _LOGO_PROCESS_STATE = None
    logo_rows = [row["logo_row"] for row in logo_fits]
    decision = select_leave_one_group_pairwise(logo_rows)

    final_fit_started = time.monotonic()
    final_models = _fit_explicit_target_groups(
        bank=bank,
        target=target,
        target_group_ids=selected_group_ids,
        manifest=manifest,
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
    )
    evaluation = evaluate_target_action_groups(
        bank[target],
        models=final_models,
        excluded_group_ids=selected_group_ids,
    )
    selected_family = str(decision["selected_family"])
    return {
        "protocol": "tsc-v37r33-logo-pairwise-deployment-screen-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "manifest": manifest.to_dict(),
        "manifest_path": str(manifest_path.resolve()),
        "cache_audit": cache_audit,
        "baseline_result": str(baseline_result.resolve()),
        "target": target,
        "city_group": manifest.city_groups[target],
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": list(selected_group_ids),
        "model_families": list(DEPLOYMENT_FAMILIES),
        "parallelism": {
            "logo_workers_requested": int(logo_workers),
            "logo_workers_actual": actual_logo_workers,
            "executor": "forked_process_pool",
        },
        "logo_fits": logo_fits,
        "logo_rows": logo_rows,
        "deployment_decision": decision,
        "final_fit_group_count": len(selected_group_ids),
        "final_model_diagnostics": final_models.diagnostics,
        "final_fit_elapsed_sec": float(time.monotonic() - final_fit_started),
        "offline_evaluation": evaluation,
        "selected_family": selected_family,
        "selected_evaluation": evaluation["families"][selected_family],
        "elapsed_sec": float(time.monotonic() - started),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--cache-workers", type=int, default=16)
    parser.add_argument("--logo-workers", type=int, default=12)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_target_logo_screen(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest_path=args.manifest,
        target=args.target,
        cache_workers=args.cache_workers,
        logo_workers=args.logo_workers,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "target": payload["target"],
                "selected_family": payload["selected_family"],
                "logo_gate_passed": payload["deployment_decision"]["gate"][
                    "passed"
                ],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
