"""Run one v78 shard of leakage-isolated multihorizon nested selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    parse_action_group_seed,
)
from cf_h2o.eval.traffic_signal_multihorizon_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multihorizon_latent_screen import (
    AUTHORIZATION_DECISION as SCREEN_AUTHORIZATION_DECISION,
    BASE_LATENT_MODEL_FAMILY,
    RESULT_PROTOCOL as SCREEN_RESULT_PROTOCOL,
    STATE_MODEL_FAMILY,
    _atomic_oof_npz,
    _expanded_specs,
    _file_sha256,
    _fit_fold,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    CONTRAST_FEATURES_V3,
    rollout_prefix_target_name,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (
    _row_subset,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic import (
    select_ranker_gate,
)
from cf_h2o.traffic_signal.action_contrast import (
    action_group_ids,
    build_action_contrast_dataset,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import LATENT_PROTOCOL
from cf_h2o.traffic_signal.state_conditioned_spatiotemporal_latent import (
    STATE_CONDITIONED_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_latent_screen import (
    PROTOCOL as SCREEN_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_nested_selection import (
    PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_multihorizon_waiting_cache import (
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_state_conditioned_latent_diagnostic import (
    MANIFEST_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    PARTITION_PROTOCOL,
)


OUTER_RESULT_PROTOCOL = "tsc-v83r79-multihorizon-nested-outer-result-v1"
NODE_RESULT_PROTOCOL = "tsc-v83r79-multihorizon-nested-node-result-v1"
_PROCESS_CONTEXT: dict[str, Any] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def exclude_outer_seed(
    contrast: MechanismDataset,
    *,
    outer_seed: int,
) -> MechanismDataset:
    groups = action_group_ids(contrast).astype(str)
    group_seeds = np.asarray(
        [parse_action_group_seed(group) for group in groups], dtype=int
    )
    reduced = _row_subset(contrast, group_seeds != int(outer_seed))
    reduced_groups = action_group_ids(reduced).astype(str)
    reduced_seeds = {
        parse_action_group_seed(group) for group in np.unique(reduced_groups)
    }
    if int(outer_seed) in reduced_seeds or not reduced_seeds:
        raise RuntimeError("outer seed exclusion failed")
    return reduced


def fit_nested_inner_fold(
    *,
    outer_seed: int,
    inner_seed: int,
    outer_excluded_contrast: MechanismDataset,
    expanded_specs: Sequence[Mapping[str, Any]],
    scenario: str,
    expected_action_count: int,
) -> dict[str, Any]:
    if int(outer_seed) == int(inner_seed):
        raise ValueError("outer and inner seeds must differ")
    groups = action_group_ids(outer_excluded_contrast).astype(str)
    observed = {
        parse_action_group_seed(group) for group in np.unique(groups)
    }
    if int(outer_seed) in observed or int(inner_seed) not in observed:
        raise RuntimeError("nested inner fold received a leaked seed partition")
    fold = _fit_fold(
        heldout_seed=int(inner_seed),
        contrast=outer_excluded_contrast,
        expanded_specs=expanded_specs,
        scenario=scenario,
        expected_action_count=int(expected_action_count),
    )
    return {
        **fold,
        "outer_seed": int(outer_seed),
        "inner_seed": int(inner_seed),
        "outer_seed_absent_from_inner_dataset": True,
        "training_seed_count": len(observed) - 1,
    }


def _process_inner(inner_seed: int) -> dict[str, Any]:
    if _PROCESS_CONTEXT is None:
        raise RuntimeError("nested process context is not initialized")
    return fit_nested_inner_fold(
        outer_seed=int(_PROCESS_CONTEXT["outer_seed"]),
        inner_seed=int(inner_seed),
        outer_excluded_contrast=_PROCESS_CONTEXT["contrast"],
        expanded_specs=_PROCESS_CONTEXT["expanded_specs"],
        scenario=str(_PROCESS_CONTEXT["scenario"]),
        expected_action_count=int(_PROCESS_CONTEXT["expected_action_count"]),
    )


def _fit_contract_exact(
    folds: Sequence[Mapping[str, Any]],
    *,
    expected_targets: set[str],
) -> bool:
    valid = True
    for fold in folds:
        for result in fold["model_results"]:
            diagnostics = result["fit_diagnostics"]
            family = str(result["model_family"])
            target_name = str(diagnostics.get("target_name", ""))
            if family == BASE_LATENT_MODEL_FAMILY:
                family_valid = diagnostics.get("protocol") == LATENT_PROTOCOL
            elif family == STATE_MODEL_FAMILY:
                family_valid = bool(
                    diagnostics.get("protocol") == STATE_CONDITIONED_PROTOCOL
                    and diagnostics.get("calibration_mode")
                    == "leave_self_out_within_cell"
                )
            else:
                family_valid = False
            valid = valid and family_valid and target_name in expected_targets
    return bool(valid)


def _fold_summaries(folds: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: value for key, value in fold.items() if key != "model_results"
        }
        | {
            "model_results": [
                {
                    "model_key": result["model_key"],
                    "model_family": result["model_family"],
                    "fit_diagnostics": result["fit_diagnostics"],
                    "validation_record_count": len(result["records"]),
                }
                for result in fold["model_results"]
            ]
        }
        for fold in folds
    ]


def _completed_outer(final_dir: Path, *, outer_seed: int) -> dict[str, Any] | None:
    result_path = final_dir / "selection_v1.json"
    oof_path = final_dir / "inner_oof_v1.npz"
    if not final_dir.exists():
        return None
    if not result_path.is_file() or not oof_path.is_file():
        raise RuntimeError(f"incomplete nested outer artifact: {final_dir}")
    result = _read_json(result_path)
    if (
        result.get("protocol") != OUTER_RESULT_PROTOCOL
        or int(result.get("outer_seed", -1)) != int(outer_seed)
        or result.get("integrity_gate", {}).get("passed") is not True
        or result.get("inner_oof_artifact", {}).get("sha256")
        != _file_sha256(oof_path)
    ):
        raise ValueError(f"completed nested outer artifact changed: {final_dir}")
    return result


def run_outer_selection(
    *,
    outer_seed: int,
    all_seeds: Sequence[int],
    contrast: MechanismDataset,
    expanded_specs: Sequence[Mapping[str, Any]],
    scenario: str,
    expected_action_count: int,
    selection: Mapping[str, Any],
    fold_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    final_dir = Path(output_root) / f"outer_seed_{int(outer_seed)}"
    completed = _completed_outer(final_dir, outer_seed=int(outer_seed))
    if completed is not None:
        return completed
    outer_excluded = exclude_outer_seed(contrast, outer_seed=int(outer_seed))
    inner_seeds = tuple(
        int(seed) for seed in all_seeds if int(seed) != int(outer_seed)
    )
    workers = min(max(int(fold_workers), 1), len(inner_seeds))
    global _PROCESS_CONTEXT
    _PROCESS_CONTEXT = {
        "outer_seed": int(outer_seed),
        "contrast": outer_excluded,
        "expanded_specs": tuple(dict(spec) for spec in expanded_specs),
        "scenario": str(scenario),
        "expected_action_count": int(expected_action_count),
    }
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as pool:
            folds = list(pool.map(_process_inner, inner_seeds))
    finally:
        _PROCESS_CONTEXT = None
    oof_records = [
        row
        for fold in folds
        for model_result in fold["model_results"]
        for row in model_result["records"]
    ]
    groups = action_group_ids(outer_excluded).astype(str)
    expected_groups = len(np.unique(groups))
    expected_records = expected_groups * len(expanded_specs)
    unique_keys = {
        (str(row["model_key"]), str(row["group_id"])) for row in oof_records
    }
    expected_targets = {
        str(spec["target_name"]) for spec in expanded_specs
    }
    integrity = {
        "outer_seed_absent_from_every_inner_dataset": all(
            bool(fold["outer_seed_absent_from_inner_dataset"]) for fold in folds
        ),
        "inner_seed_set_exact": {
            int(fold["inner_seed"]) for fold in folds
        }
        == set(inner_seeds),
        "inner_fold_count_exact": len(folds) == len(inner_seeds),
        "training_seed_count_exact": {
            int(fold["training_seed_count"]) for fold in folds
        }
        == {len(all_seeds) - 2},
        "inner_training_validation_disjoint": all(
            bool(fold["training_validation_disjoint"])
            and bool(fold["heldout_absent_from_training"])
            for fold in folds
        ),
        "validation_groups_exact": sum(
            int(fold["validation_group_count"]) for fold in folds
        )
        == expected_groups,
        "oof_record_count_exact": len(oof_records) == expected_records,
        "oof_group_model_keys_unique": len(unique_keys) == expected_records,
        "expected_actions_per_group": {
            int(row["action_count"]) for row in oof_records
        }
        == {int(expected_action_count)},
        "target_specific_fit_contract_exact": _fit_contract_exact(
            folds, expected_targets=expected_targets
        ),
    }
    integrity["passed"] = all(integrity.values())
    if not integrity["passed"]:
        raise RuntimeError(f"nested outer integrity failed for seed {outer_seed}")
    gate_selection = select_ranker_gate(
        oof_records,
        model_specs=expanded_specs,
        seeds=inner_seeds,
        selection=selection,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".outer_seed_{int(outer_seed)}.", dir=output_root
        )
    )
    try:
        temporary_oof = temporary_dir / "inner_oof_v1.npz"
        _atomic_oof_npz(temporary_oof, oof_records)
        final_oof = final_dir / temporary_oof.name
        result = {
            "protocol": OUTER_RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "outer_seed": int(outer_seed),
            "inner_seeds": list(inner_seeds),
            "training_seed_count_per_inner_fit": len(all_seeds) - 2,
            "scenario": str(scenario),
            "expanded_candidate_count": len(expanded_specs),
            "expected_inner_group_count": expected_groups,
            "inner_fold_workers": workers,
            "inner_oof_artifact": {
                "path": str(final_oof),
                "sha256": _file_sha256(temporary_oof),
                "record_count": len(oof_records),
                "format": "compressed_columnar_npz_no_pickle",
            },
            "folds": _fold_summaries(folds),
            "gate_selection": gate_selection,
            "integrity_gate": integrity,
            "decision": gate_selection["decision"],
        }
        atomic_write_json(temporary_dir / "selection_v1.json", result)
        if final_dir.exists():
            raise FileExistsError(f"nested outer result appeared: {final_dir}")
        temporary_dir.rename(final_dir)
        return result
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def run_node_shard(
    *,
    nested_protocol_path: Path,
    screen_protocol_path: Path,
    screen_result_path: Path,
    screen_oof_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    cache_root: Path,
    partition_path: Path,
    manifest_path: Path,
    conversion_root: Path,
    node: str,
    outer_seeds: Sequence[int],
    cache_workers: int,
    fold_workers: int,
    output_root: Path,
) -> dict[str, Any]:
    nested_protocol = _read_json(nested_protocol_path)
    screen_protocol = _read_json(screen_protocol_path)
    screen_result = _read_json(screen_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    partition = _read_json(partition_path)
    manifest_document = _read_json(manifest_path)
    frozen = dict(nested_protocol.get("frozen_inputs", {}))
    nested = dict(nested_protocol.get("nested_selection", {}))
    assigned = tuple(
        int(value) for value in nested.get("node_assignments", {}).get(node, ())
    )
    evidence_ok = bool(
        nested_protocol.get("protocol") == PROTOCOL
        and screen_protocol.get("protocol") == SCREEN_PROTOCOL
        and screen_result.get("protocol") == SCREEN_RESULT_PROTOCOL
        and screen_result.get("status") == "PASS"
        and screen_result.get("decision") == SCREEN_AUTHORIZATION_DECISION
        and screen_result.get("integrity_gate", {}).get("passed") is True
        and cache_protocol.get("protocol") == CACHE_PROTOCOL
        and cache_audit.get("protocol") == CACHE_AUDIT_PROTOCOL
        and cache_audit.get("status") == "PASS"
        and partition.get("protocol") == PARTITION_PROTOCOL
        and manifest_document.get("protocol") == MANIFEST_PROTOCOL
        and _sha256(screen_protocol_path) == frozen.get("screen_protocol_sha256")
        and _sha256(screen_result_path) == frozen.get("screen_result_sha256")
        and _file_sha256(screen_oof_path) == frozen.get("screen_oof_sha256")
        and _sha256(cache_protocol_path) == frozen.get("cache_protocol_sha256")
        and _sha256(cache_audit_path) == frozen.get("cache_audit_sha256")
        and _sha256(partition_path) == frozen.get("partition_sha256")
        and _sha256(manifest_path) == frozen.get("manifest_sha256")
        and tuple(int(value) for value in outer_seeds) == assigned
    )
    if not evidence_ok:
        raise ValueError("nested node evidence changed")
    expected_root = Path(nested_protocol["output_roots"]["remote_shard_root"]) / node
    if Path(output_root) != expected_root:
        raise ValueError("nested node output root changed")
    all_seeds = tuple(int(value) for value in nested["outer_seeds"])
    sealed_roots = (
        Path(partition["confirmatory"]["root"]),
        Path(partition["prospective"]["root"]),
    )
    if (
        set(all_seeds) != {
            int(value) for value in partition["redevelopment"]["seeds"]
        }
        or any(path.exists() for path in sealed_roots)
    ):
        raise ValueError("nested node partition changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    scenario = str(nested["scenarios"][0])
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("nested node scenario is not unique")
    manifest = replace(full_manifest, scenarios=selected_specs)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=all_seeds,
        collection_shards=int(nested["collection_shards"]),
        workers=max(int(cache_workers), 1),
    )
    dataset = _merge_city_datasets(
        bank, (scenario,), city=str(nested["city"])
    )
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy=str(nested["reference_policy"]),
        contrast_features=CONTRAST_FEATURES_V3,
    )
    groups = action_group_ids(contrast).astype(str)
    observed_seeds = {
        parse_action_group_seed(group) for group in np.unique(groups)
    }
    expanded_specs = _expanded_specs(nested)
    expected_targets = {
        rollout_prefix_target_name(int(value))
        for value in nested["rollout_prefix_horizons_sec"]
    }
    if (
        observed_seeds != set(all_seeds)
        or len(expanded_specs) != int(nested["expanded_candidate_count"])
        or not expected_targets.issubset(contrast.targets)
    ):
        raise ValueError("nested node candidate or data contract changed")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    outer_results = []
    for index, outer_seed in enumerate(assigned, start=1):
        result = run_outer_selection(
            outer_seed=int(outer_seed),
            all_seeds=all_seeds,
            contrast=contrast,
            expanded_specs=expanded_specs,
            scenario=scenario,
            expected_action_count=int(nested["expected_action_count_per_group"]),
            selection=nested["selection"],
            fold_workers=int(fold_workers),
            output_root=output_root,
        )
        result_path = output_root / f"outer_seed_{outer_seed}" / "selection_v1.json"
        outer_results.append(
            {
                "outer_seed": int(outer_seed),
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
                "inner_oof_sha256": result["inner_oof_artifact"]["sha256"],
                "selected_candidate": (
                    result["gate_selection"]["selected"].get("key")
                    if result["gate_selection"]["selected"]
                    else None
                ),
            }
        )
        print(
            json.dumps(
                {
                    "node": node,
                    "outer_complete": f"{index}/{len(assigned)}",
                    "outer_seed": int(outer_seed),
                    "selected": outer_results[-1]["selected_candidate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return {
        "protocol": NODE_RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS",
        "node": str(node),
        "nested_protocol_sha256": _sha256(nested_protocol_path),
        "screen_result_sha256": _sha256(screen_result_path),
        "screen_oof_sha256": _file_sha256(screen_oof_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "assigned_outer_seeds": list(assigned),
        "outer_result_count": len(outer_results),
        "outer_results": outer_results,
        "bank_audit": bank_audit,
        "integrity_gate": {
            "frozen_evidence_chain": evidence_ok,
            "assigned_outer_seed_set_exact": tuple(int(v) for v in outer_seeds)
            == assigned,
            "all_outer_results_complete": len(outer_results) == len(assigned),
            "sealed_roots_absent": not any(path.exists() for path in sealed_roots),
            "confirmatory_or_prospective_data_used": False,
            "passed": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-protocol", type=Path, required=True)
    parser.add_argument("--screen-protocol", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-oof", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--outer-seeds", nargs="+", type=int, required=True)
    parser.add_argument("--cache-workers", type=int, default=32)
    parser.add_argument("--fold-workers", type=int, default=21)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite nested node result: {args.out}")
    result = run_node_shard(
        nested_protocol_path=args.nested_protocol,
        screen_protocol_path=args.screen_protocol,
        screen_result_path=args.screen_result,
        screen_oof_path=args.screen_oof,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        cache_root=args.cache_root,
        partition_path=args.partition,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        node=str(args.node),
        outer_seeds=args.outer_seeds,
        cache_workers=args.cache_workers,
        fold_workers=args.fold_workers,
        output_root=args.output_root,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "node": result["node"],
                "outer_results": result["outer_result_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
