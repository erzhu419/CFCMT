"""Aggregate and independently audit the v78 nested selection shards."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_multihorizon_latent_screen import (
    AUTHORIZATION_DECISION as SCREEN_AUTHORIZATION_DECISION,
    RESULT_PROTOCOL as SCREEN_RESULT_PROTOCOL,
    _expanded_specs,
    _file_sha256,
)
from cf_h2o.eval.traffic_signal_multihorizon_nested_selection import (
    NODE_RESULT_PROTOCOL,
    OUTER_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic import (
    _bootstrap_seed_means,
    _seed_means,
    summarize_gate_candidate,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from scripts.cluster.freeze_tsc_external_v9_multihorizon_nested_selection import (
    NODES,
    PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-multihorizon-nested-selection-audit-v1"
AUTHORIZATION_DECISION = "authorize_multihorizon_latent_closed_loop_development"
REJECTION_DECISION = "retain_phase_pressure_and_reject_nested_multihorizon_latent"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _required_oof_columns() -> tuple[str, ...]:
    return (
        "model_key",
        "model_family",
        "simulator_seed",
        "group_id",
        "selected_differs",
        "predicted_delta",
        "selected_uncertainty",
        "selected_context_trust",
        "actual_group_normalized_delta",
        "heldout_seed",
    )


def reconstruct_gate_selection_from_npz(
    path: Path,
    *,
    model_specs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        missing = set(_required_oof_columns()).difference(archive.files)
        if missing:
            raise ValueError(f"nested OOF sidecar is missing columns: {sorted(missing)}")
        columns = {name: archive[name] for name in _required_oof_columns()}
    grid = []
    for spec in model_specs:
        model_key = str(spec["key"])
        rows = np.flatnonzero(columns["model_key"] == model_key)
        if rows.size == 0:
            raise ValueError(f"nested OOF sidecar is missing model {model_key!r}")
        records = [
            {
                "model_key": model_key,
                "model_family": str(columns["model_family"][row]),
                "simulator_seed": int(columns["simulator_seed"][row]),
                "group_id": str(columns["group_id"][row]),
                "selected_differs": bool(columns["selected_differs"][row]),
                "predicted_delta": float(columns["predicted_delta"][row]),
                "selected_uncertainty": float(
                    columns["selected_uncertainty"][row]
                ),
                "selected_context_trust": float(
                    columns["selected_context_trust"][row]
                ),
                "actual_group_normalized_delta": float(
                    columns["actual_group_normalized_delta"][row]
                ),
            }
            for row in rows
        ]
        for candidate in selection["gate_candidates"]:
            grid.append(
                summarize_gate_candidate(
                    records,
                    seeds=seeds,
                    risk_multiplier=float(candidate["risk_multiplier"]),
                    minimum_context_trust=float(
                        candidate["minimum_context_trust"]
                    ),
                    selection_rule=selection["selection_rule"],
                    bootstrap=selection["bootstrap"],
                )
            )
    feasible = [row for row in grid if bool(row["feasible"])]
    selected = min(
        feasible,
        key=lambda row: (
            float(row["robust_score"]),
            float(row["mean_delta"]),
            float(row["harmful_group_fraction"]),
            -int(row["retained_group_count"]),
            -float(row["risk_multiplier"]),
            -float(row["minimum_context_trust"]),
            str(row["key"]),
        ),
        default=None,
    )
    return {
        "decision": (
            "freeze_selected_direct_ranker"
            if selected is not None
            else "fallback_to_phase_pressure"
        ),
        "selected": selected,
        "feasible_candidate_count": len(feasible),
        "grid": grid,
    }


def _screen_columns(path: Path) -> dict[str, np.ndarray]:
    required = _required_oof_columns()
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"v77 OOF sidecar is missing columns: {sorted(missing)}")
        return {name: archive[name] for name in required}


def apply_outer_selection(
    *,
    outer_seed: int,
    selected: Mapping[str, Any] | None,
    screen_columns: Mapping[str, np.ndarray],
    first_model_key: str,
) -> list[dict[str, Any]]:
    model_key = str(selected["model_key"]) if selected else str(first_model_key)
    mask = np.asarray(
        (screen_columns["simulator_seed"] == int(outer_seed))
        & (screen_columns["model_key"] == model_key),
        dtype=bool,
    )
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError(
            f"v77 OOF sidecar has no outer records for seed {outer_seed}, {model_key}"
        )
    if {
        int(value) for value in screen_columns["heldout_seed"][rows]
    } != {int(outer_seed)}:
        raise ValueError("v77 outer prediction was not trained seed-out")
    risk = float(selected["risk_multiplier"]) if selected else 0.0
    trust_threshold = (
        float(selected["minimum_context_trust"]) if selected else 0.0
    )
    result = []
    for row in rows:
        accepted = bool(
            selected is not None
            and bool(screen_columns["selected_differs"][row])
            and float(screen_columns["selected_context_trust"][row])
            >= trust_threshold
            and float(screen_columns["predicted_delta"][row])
            + risk * float(screen_columns["selected_uncertainty"][row])
            < 0.0
        )
        actual = float(screen_columns["actual_group_normalized_delta"][row])
        result.append(
            {
                "outer_seed": int(outer_seed),
                "simulator_seed": int(outer_seed),
                "group_id": str(screen_columns["group_id"][row]),
                "selected_candidate_key": (
                    str(selected["key"]) if selected else "phase_pressure_fallback"
                ),
                "model_key": str(selected["model_key"]) if selected else "",
                "risk_multiplier": risk,
                "minimum_context_trust": trust_threshold,
                "ranker_accepted": accepted,
                "actual_group_normalized_delta": actual,
                "deployed_delta": actual if accepted else 0.0,
            }
        )
    return result


def summarize_nested_deployment(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    selection_rule: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
) -> dict[str, Any]:
    seed_means = _seed_means(records, seeds=seeds, delta_key="deployed_delta")
    values = np.asarray(list(seed_means.values()), dtype=float)
    accepted = [row for row in records if bool(row["ranker_accepted"])]
    retained_seeds = {int(row["simulator_seed"]) for row in accepted}
    harmful_fraction = (
        float(
            np.mean(
                [
                    float(row["actual_group_normalized_delta"]) > 0.0
                    for row in accepted
                ]
            )
        )
        if accepted
        else 1.0
    )
    interval = _bootstrap_seed_means(
        seed_means,
        replicates=int(bootstrap["replicates"]),
        bootstrap_seed=int(bootstrap["seed"]),
    )
    mean_delta = float(np.mean(values))
    seed_std = float(np.std(values))
    worst_seed = float(np.max(values))
    retained_fraction = len(retained_seeds) / max(len(seeds), 1)
    gates = {
        "minimum_retained_groups": len(accepted)
        >= int(selection_rule["minimum_retained_groups_per_city"]),
        "minimum_retained_seed_fraction": retained_fraction
        >= float(selection_rule["minimum_retained_seed_fraction"]),
        "mean_delta": mean_delta
        <= float(selection_rule["maximum_equal_seed_scenario_mean_delta"]),
        "bootstrap_upper": float(interval["ci95"][1])
        <= float(selection_rule["maximum_bootstrap_95pct_upper_mean_delta"]),
        "worst_seed": worst_seed
        <= float(selection_rule["maximum_worst_seed_mean_delta"]),
        "harmful_fraction": harmful_fraction
        <= float(selection_rule["maximum_harmful_group_fraction"]),
    }
    gates["passed"] = all(gates.values())
    return {
        "decision_count": len(records),
        "retained_group_count": len(accepted),
        "retained_seed_count": len(retained_seeds),
        "retained_seed_fraction": retained_fraction,
        "harmful_group_fraction": harmful_fraction,
        "mean_delta": mean_delta,
        "seed_delta_std": seed_std,
        "worst_seed_delta": worst_seed,
        "robust_score": mean_delta + 0.5 * seed_std + max(worst_seed, 0.0),
        "bootstrap": interval,
        "seed_mean_delta": {
            str(seed): float(seed_means[seed]) for seed in sorted(seed_means)
        },
        "gates": gates,
    }


def _atomic_deployment_npz(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "outer_seed": np.asarray(
            [int(row["outer_seed"]) for row in records], dtype=np.int64
        ),
        "group_id": np.asarray([str(row["group_id"]) for row in records]),
        "selected_candidate_key": np.asarray(
            [str(row["selected_candidate_key"]) for row in records]
        ),
        "model_key": np.asarray([str(row["model_key"]) for row in records]),
        "risk_multiplier": np.asarray(
            [float(row["risk_multiplier"]) for row in records], dtype=float
        ),
        "minimum_context_trust": np.asarray(
            [float(row["minimum_context_trust"]) for row in records], dtype=float
        ),
        "ranker_accepted": np.asarray(
            [bool(row["ranker_accepted"]) for row in records], dtype=bool
        ),
        "actual_group_normalized_delta": np.asarray(
            [float(row["actual_group_normalized_delta"]) for row in records],
            dtype=float,
        ),
        "deployed_delta": np.asarray(
            [float(row["deployed_delta"]) for row in records], dtype=float
        ),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
        raise


def run_audit(
    *,
    nested_protocol_path: Path,
    screen_result_path: Path,
    screen_oof_path: Path,
    local_shard_root: Path,
    deployment_oof_out: Path,
) -> dict[str, Any]:
    protocol = _read_json(nested_protocol_path)
    screen_result = _read_json(screen_result_path)
    frozen = dict(protocol.get("frozen_inputs", {}))
    nested = dict(protocol.get("nested_selection", {}))
    evidence_ok = bool(
        protocol.get("protocol") == PROTOCOL
        and screen_result.get("protocol") == SCREEN_RESULT_PROTOCOL
        and screen_result.get("status") == "PASS"
        and screen_result.get("decision") == SCREEN_AUTHORIZATION_DECISION
        and screen_result.get("integrity_gate", {}).get("passed") is True
        and _sha256(screen_result_path) == frozen.get("screen_result_sha256")
        and _file_sha256(screen_oof_path) == frozen.get("screen_oof_sha256")
        and Path(local_shard_root).resolve()
        == Path(protocol["output_roots"]["local_shard_root"]).resolve()
    )
    if not evidence_ok:
        raise ValueError("nested audit parent evidence changed")
    seeds = tuple(int(value) for value in nested["outer_seeds"])
    expanded_specs = _expanded_specs(nested)
    node_summaries = []
    recorded_outer: dict[int, tuple[dict[str, Any], Path]] = {}
    node_integrity = True
    for node in NODES:
        summary_path = Path(local_shard_root) / node / "summary_v1.json"
        summary = _read_json(summary_path)
        assigned = tuple(
            int(value) for value in nested["node_assignments"][node]
        )
        valid = bool(
            summary.get("protocol") == NODE_RESULT_PROTOCOL
            and summary.get("status") == "PASS"
            and summary.get("node") == node
            and summary.get("nested_protocol_sha256")
            == _sha256(nested_protocol_path)
            and tuple(summary.get("assigned_outer_seeds", ())) == assigned
            and summary.get("integrity_gate", {}).get("passed") is True
            and int(summary.get("outer_result_count", -1)) == len(assigned)
        )
        node_integrity = node_integrity and valid
        by_seed = {
            int(item["outer_seed"]): dict(item)
            for item in summary.get("outer_results", ())
        }
        if set(by_seed) != set(assigned):
            raise ValueError(f"nested node {node} outer summaries changed")
        for outer_seed in assigned:
            result_path = (
                Path(local_shard_root)
                / node
                / f"outer_seed_{outer_seed}"
                / "selection_v1.json"
            )
            result = _read_json(result_path)
            if (
                result.get("protocol") != OUTER_RESULT_PROTOCOL
                or int(result.get("outer_seed", -1)) != outer_seed
                or result.get("integrity_gate", {}).get("passed") is not True
                or _sha256(result_path) != by_seed[outer_seed]["result_sha256"]
            ):
                raise ValueError(f"nested outer result changed: {result_path}")
            if outer_seed in recorded_outer:
                raise ValueError(f"duplicate nested outer seed: {outer_seed}")
            recorded_outer[outer_seed] = (result, result_path)
        node_summaries.append(
            {
                "node": node,
                "path": str(summary_path.resolve()),
                "sha256": _sha256(summary_path),
                "outer_seeds": list(assigned),
                "valid": valid,
            }
        )
    if set(recorded_outer) != set(seeds):
        raise ValueError("nested outer seed coverage changed")
    screen_columns = _screen_columns(screen_oof_path)
    first_model_key = str(expanded_specs[0]["key"])
    deployment_records = []
    outer_audits = []
    selection_counts: Counter[str] = Counter()
    all_reconstructed = True
    all_inner_artifacts = True
    for outer_seed in seeds:
        result, result_path = recorded_outer[outer_seed]
        node = next(
            node
            for node in NODES
            if outer_seed in nested["node_assignments"][node]
        )
        inner_oof = result_path.parent / "inner_oof_v1.npz"
        artifact = dict(result["inner_oof_artifact"])
        artifact_valid = bool(
            inner_oof.is_file()
            and _file_sha256(inner_oof) == artifact.get("sha256")
            and artifact.get("sha256")
            == next(
                item["inner_oof_sha256"]
                for item in _read_json(
                    Path(local_shard_root) / node / "summary_v1.json"
                )["outer_results"]
                if int(item["outer_seed"]) == outer_seed
            )
        )
        all_inner_artifacts = all_inner_artifacts and artifact_valid
        if not artifact_valid:
            raise ValueError(f"nested inner OOF artifact changed: {inner_oof}")
        reconstructed = reconstruct_gate_selection_from_npz(
            inner_oof,
            model_specs=expanded_specs,
            seeds=result["inner_seeds"],
            selection=nested["selection"],
        )
        exact = reconstructed == result["gate_selection"]
        all_reconstructed = all_reconstructed and exact
        if not exact:
            raise ValueError(
                f"nested inner selection cannot be reconstructed for {outer_seed}"
            )
        selected = reconstructed["selected"]
        selected_key = (
            str(selected["key"]) if selected else "phase_pressure_fallback"
        )
        selection_counts[selected_key] += 1
        outer_records = apply_outer_selection(
            outer_seed=outer_seed,
            selected=selected,
            screen_columns=screen_columns,
            first_model_key=first_model_key,
        )
        deployment_records.extend(outer_records)
        outer_audits.append(
            {
                "outer_seed": outer_seed,
                "inner_seed_count": len(result["inner_seeds"]),
                "inner_oof_record_count": int(artifact["record_count"]),
                "inner_selection_reconstructed_exact": exact,
                "selected_candidate": selected_key,
                "outer_group_count": len(outer_records),
                "outer_retained_group_count": sum(
                    bool(row["ranker_accepted"]) for row in outer_records
                ),
                "outer_mean_delta": float(
                    np.mean([float(row["deployed_delta"]) for row in outer_records])
                ),
            }
        )
    group_keys = {
        (int(row["outer_seed"]), str(row["group_id"]))
        for row in deployment_records
    }
    expected_groups = int(screen_result["action_group_count"])
    deployment_exact = bool(
        len(deployment_records) == expected_groups
        and len(group_keys) == expected_groups
        and {int(row["outer_seed"]) for row in deployment_records} == set(seeds)
    )
    summary = summarize_nested_deployment(
        deployment_records,
        seeds=seeds,
        selection_rule=nested["selection"]["selection_rule"],
        bootstrap=nested["selection"]["bootstrap"],
    )
    _atomic_deployment_npz(deployment_oof_out, deployment_records)
    final_candidate = dict(nested["final_full_development_candidate"])
    final_matches_screen = bool(
        final_candidate.get("key")
        == screen_result.get("advance_gate", {})
        .get("selected_candidate", {})
        .get("key")
    )
    integrity = {
        "frozen_evidence_chain": evidence_ok,
        "six_node_summaries_exact": node_integrity
        and len(node_summaries) == len(NODES),
        "outer_seed_set_exact": set(recorded_outer) == set(seeds),
        "inner_selection_reconstructed_exact": all_reconstructed,
        "inner_oof_artifacts_exact": all_inner_artifacts,
        "outer_deployment_groups_exact": deployment_exact,
        "final_candidate_matches_v77_full_development_selection": final_matches_screen,
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value
        for key, value in integrity.items()
        if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    advance = bool(integrity["passed"] and summary["gates"]["passed"])
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": AUTHORIZATION_DECISION if advance else REJECTION_DECISION,
        "nested_protocol_sha256": _sha256(nested_protocol_path),
        "screen_result_sha256": _sha256(screen_result_path),
        "screen_oof_sha256": _file_sha256(screen_oof_path),
        "node_summaries": node_summaries,
        "outer_audits": outer_audits,
        "selection_stability": {
            "candidate_counts": dict(sorted(selection_counts.items())),
            "distinct_selected_candidate_count": len(selection_counts),
            "fallback_outer_fold_count": selection_counts.get(
                "phase_pressure_fallback", 0
            ),
        },
        "nested_deployment": summary,
        "deployment_oof_artifact": {
            "path": str(Path(deployment_oof_out).resolve()),
            "sha256": _file_sha256(deployment_oof_out),
            "record_count": len(deployment_records),
            "format": "compressed_columnar_npz_no_pickle",
        },
        "final_full_development_candidate": final_candidate,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": advance,
            "next_stage": (
                "multihorizon_latent_closed_loop_redevelopment"
                if advance
                else "none"
            ),
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-protocol", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-oof", type=Path, required=True)
    parser.add_argument("--local-shard-root", type=Path, required=True)
    parser.add_argument("--deployment-oof-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists() or args.deployment_oof_out.exists():
        raise FileExistsError("refusing to overwrite nested audit outputs")
    result = run_audit(
        nested_protocol_path=args.nested_protocol,
        screen_result_path=args.screen_result,
        screen_oof_path=args.screen_oof,
        local_shard_root=args.local_shard_root,
        deployment_oof_out=args.deployment_oof_out,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "mean_delta": result["nested_deployment"]["mean_delta"],
                "selected_candidates": result["selection_stability"][
                    "candidate_counts"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
