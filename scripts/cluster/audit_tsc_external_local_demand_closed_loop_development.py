#!/usr/bin/env python3
"""Audit the frozen v75 paired development matrix and its advance gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
import json
from pathlib import Path
import socket
import sys
from typing import Any, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (  # noqa: E402
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_local_demand_closed_loop_development import (  # noqa: E402
    CANDIDATE_KEY,
    PHASE_POLICY,
    PROTOCOL,
    RESULT_PROTOCOL,
    all_rollout_identities,
)
from scripts.cluster.run_tsc_external_local_demand_closed_loop_development_shard import (  # noqa: E402
    SHARD_PROTOCOL,
)


AUDIT_PROTOCOL = "tsc-v75r71-local-demand-closed-loop-development-audit-v1"
PASS_DECISION = "authorize_local_demand_veto_untouched_validation_freeze_only"
FAIL_DECISION = "reject_local_demand_development_and_preserve_sealed_seeds"
EXPECTED_NODES = tuple(f"node{index:03d}" for index in range(1, 7))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safety(metrics: Mapping[str, Any]) -> dict[str, int]:
    audit = dict(metrics.get("safety_audit", {}))
    return {
        "collision_incidents": int(audit.get("unique_collision_incidents", 0)),
        "starting_teleports": int(metrics.get("starting_teleports", -1)),
        "ending_teleports": int(metrics.get("ending_teleports", -1)),
    }


def _bootstrap(values: Mapping[int, float], spec: Mapping[str, Any]) -> dict[str, Any]:
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=float)
    rng = np.random.default_rng(int(spec["seed"]))
    indices = rng.integers(
        0,
        ordered.size,
        size=(int(spec["replicates"]), ordered.size),
    )
    draws = ordered[indices].mean(axis=1)
    return {
        "unit": str(spec["unit"]),
        "replicates": int(spec["replicates"]),
        "seed": int(spec["seed"]),
        "observed": float(np.mean(ordered)),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "probability_nonnegative": float(np.mean(draws >= 0.0)),
    }


def _active_city_summary(
    *,
    city: str,
    pairs: Sequence[Mapping[str, Any]],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    rule: Mapping[str, Any],
    bootstrap_spec: Mapping[str, Any],
) -> dict[str, Any]:
    by_key = {(int(row["seed"]), str(row["scenario"])): row for row in pairs}
    expected = {(int(seed), str(scenario)) for seed in seeds for scenario in scenarios}
    if set(by_key) != expected:
        raise ValueError(f"v75 active city pair matrix changed for {city}")
    seed_rows = {}
    for seed in seeds:
        rows = [by_key[(int(seed), str(scenario))] for scenario in scenarios]
        seed_rows[str(seed)] = {
            "city_mean_relative_delta": float(
                np.mean([float(row["relative_delta"]) for row in rows])
            ),
            "executed_overrides": int(sum(int(row["executed_overrides"]) for row in rows)),
            "scenario_relative_deltas": {
                str(row["scenario"]): float(row["relative_delta"]) for row in rows
            },
        }
    seed_delta = {
        int(seed): float(seed_rows[str(seed)]["city_mean_relative_delta"])
        for seed in seeds
    }
    mean_delta = float(np.mean(list(seed_delta.values())))
    worst_seed = float(max(seed_delta.values()))
    improved_fraction = float(
        np.mean(np.asarray(list(seed_delta.values()), dtype=float) < 0.0)
    )
    bootstrap = _bootstrap(seed_delta, bootstrap_spec)
    method_collisions = sum(
        int(row["method_safety"]["collision_incidents"]) for row in pairs
    )
    phase_collisions = sum(
        int(row["phase_safety"]["collision_incidents"]) for row in pairs
    )
    method_teleports = sum(
        int(row["method_safety"]["starting_teleports"])
        + int(row["method_safety"]["ending_teleports"])
        for row in pairs
    )
    phase_teleports = sum(
        int(row["phase_safety"]["starting_teleports"])
        + int(row["phase_safety"]["ending_teleports"])
        for row in pairs
    )
    gates = {
        "target_veto_active_in_every_row": all(
            bool(row["target_veto_active"]) for row in pairs
        ),
        "every_scenario_seed_intervened": all(
            int(row["executed_overrides"])
            >= int(rule["minimum_executed_interventions_per_scenario_seed"])
            for row in pairs
        ),
        "minimum_city_mean_improvement": mean_delta
        <= -float(rule["minimum_city_mean_relative_improvement"]),
        "bootstrap_upper_within_limit": float(bootstrap["ci95"][1])
        <= float(rule["maximum_bootstrap_95pct_upper_mean_relative_delta"]),
        "worst_seed_regression_within_limit": worst_seed
        <= float(rule["maximum_seed_level_city_mean_relative_regression"]),
        "minimum_improved_seed_fraction": improved_fraction
        >= float(rule["minimum_improved_seed_fraction"]),
        "collisions_noninferior_to_phase": method_collisions <= phase_collisions,
        "teleports_noninferior_to_phase": method_teleports <= phase_teleports,
    }
    gates["passed"] = all(gates.values())
    return {
        "city": city,
        "role": "oof_active_improvement",
        "paired_scenario_seed_units": len(pairs),
        "city_mean_relative_delta": mean_delta,
        "seed_delta_std": float(np.std(list(seed_delta.values()))),
        "worst_seed_relative_delta": worst_seed,
        "improved_seed_fraction": improved_fraction,
        "total_executed_overrides": int(
            sum(int(row["executed_overrides"]) for row in pairs)
        ),
        "bootstrap_mean_relative_delta": bootstrap,
        "seed_rows": seed_rows,
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def _fallback_city_summary(
    *, city: str, pairs: Sequence[Mapping[str, Any]], atol: float
) -> dict[str, Any]:
    gates = {
        "target_veto_disabled_in_every_row": all(
            not bool(row["target_veto_active"]) for row in pairs
        ),
        "zero_executed_interventions": all(
            int(row["executed_overrides"]) == 0 for row in pairs
        ),
        "waiting_time_exact_safe_abstention": all(
            abs(float(row["method_mean_waiting_time"]) - float(row["phase_mean_waiting_time"]))
            <= atol
            for row in pairs
        ),
        "travel_time_exact_safe_abstention": all(
            abs(float(row["method_mean_travel_time"]) - float(row["phase_mean_travel_time"]))
            <= atol
            for row in pairs
        ),
        "safety_exact_safe_abstention": all(
            row["method_safety"] == row["phase_safety"] for row in pairs
        ),
    }
    gates["passed"] = all(gates.values())
    return {
        "city": city,
        "role": "oof_fallback_exact_safe_abstention",
        "paired_scenario_seed_units": len(pairs),
        "city_mean_relative_delta": float(
            np.mean([float(row["relative_delta"]) for row in pairs])
        ),
        "total_executed_overrides": int(
            sum(int(row["executed_overrides"]) for row in pairs)
        ),
        "gates": gates,
        "feasible": bool(gates["passed"]),
    }


def audit(
    *, protocol_path: Path, launch_path: Path, results_root: Path
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    launch = _read_json(launch_path)
    identities = all_rollout_identities(protocol)
    identity_order = {identity: index for index, identity in enumerate(identities)}
    rows = []
    errors = []
    for identity in identities:
        city, scenario, seed, policy = identity
        root = Path(results_root) / city / scenario / f"seed_{seed}" / policy
        result_path = root / "result.json"
        tripinfo_path = root / "tripinfo.xml"
        try:
            result = _read_json(result_path)
            ET.parse(tripinfo_path)
            metrics = dict(result["metrics"])
            expected_node = EXPECTED_NODES[identity_order[identity] % len(EXPECTED_NODES)]
            gate = {
                "protocol": result.get("protocol") == RESULT_PROTOCOL,
                "identity": (
                    result.get("city"),
                    result.get("scenario"),
                    int(result.get("seed", -1)),
                    result.get("policy"),
                )
                == identity,
                "development_only": result.get("selection_eligible_evidence") is True
                and result.get("untouched_validation_evidence") is False,
                "protocol_hash": result.get("protocol_sha256")
                == _sha256(protocol_path),
                "runtime_sources": bool(result.get("runtime_executable_source_gate"))
                and all(result["runtime_executable_source_gate"].values()),
                "tripinfo_hash": result.get("tripinfo_evidence", {}).get("sha256")
                == _sha256(tripinfo_path),
                "metrics_ok": metrics.get("ok") is True,
                "no_teleports": int(metrics.get("starting_teleports", -1)) == 0
                and int(metrics.get("ending_teleports", -1)) == 0,
                "physical_shard": str(result.get("hostname")) == expected_node,
                "no_refit": result.get("controller_refit") is False,
            }
            gate["passed"] = all(gate.values())
            guard = dict(metrics.get("guard_audit", {}))
            rows.append(
                {
                    "city": city,
                    "scenario": scenario,
                    "seed": int(seed),
                    "policy": policy,
                    "hostname": result.get("hostname"),
                    "mean_waiting_time": float(
                        metrics["mean_tripinfo_waiting_time"]
                    ),
                    "mean_travel_time": float(metrics["mean_tripinfo_duration"]),
                    "executed_overrides": int(guard.get("executed_overrides", 0)),
                    "rejected_long_horizon_veto": int(
                        guard.get("rejected_long_horizon_veto", 0)
                    ),
                    "target_veto_active": bool(result["target_veto_active"]),
                    "safety": _safety(metrics),
                    "result_path": str(result_path.resolve()),
                    "result_sha256": _sha256(result_path),
                    "tripinfo_path": str(tripinfo_path.resolve()),
                    "tripinfo_sha256": _sha256(tripinfo_path),
                    "gate": gate,
                }
            )
        except Exception as exc:
            errors.append({"identity": list(identity), "error": repr(exc)})
    index = {
        (row["city"], row["scenario"], int(row["seed"]), row["policy"]): row
        for row in rows
    }
    paired = []
    for city, scenarios in sorted(protocol["development"]["city_scenarios"].items()):
        for scenario in scenarios:
            for seed in protocol["development"]["seeds"]:
                phase = index.get((city, scenario, int(seed), PHASE_POLICY))
                method = index.get((city, scenario, int(seed), CANDIDATE_KEY))
                if phase is None or method is None:
                    continue
                denominator = max(abs(float(phase["mean_waiting_time"])), 1e-12)
                paired.append(
                    {
                        "city": city,
                        "scenario": scenario,
                        "seed": int(seed),
                        "relative_delta": (
                            float(method["mean_waiting_time"])
                            - float(phase["mean_waiting_time"])
                        )
                        / denominator,
                        "method_mean_waiting_time": method["mean_waiting_time"],
                        "phase_mean_waiting_time": phase["mean_waiting_time"],
                        "method_mean_travel_time": method["mean_travel_time"],
                        "phase_mean_travel_time": phase["mean_travel_time"],
                        "executed_overrides": method["executed_overrides"],
                        "target_veto_active": method["target_veto_active"],
                        "method_safety": method["safety"],
                        "phase_safety": phase["safety"],
                    }
                )
    development = protocol["development"]
    city_summaries = {}
    for city, scenarios in sorted(development["city_scenarios"].items()):
        city_pairs = [row for row in paired if row["city"] == city]
        if city in set(development["active_cities"]):
            city_summaries[city] = _active_city_summary(
                city=city,
                pairs=city_pairs,
                scenarios=scenarios,
                seeds=development["seeds"],
                rule=development["selection_rule"]["active_city"],
                bootstrap_spec=development["bootstrap"],
            )
        else:
            city_summaries[city] = _fallback_city_summary(
                city=city,
                pairs=city_pairs,
                atol=float(
                    development["selection_rule"]["fallback_city"][
                        "mean_waiting_time_must_equal_phase_atol"
                    ]
                ),
            )
    shard_paths = sorted((Path(results_root) / "_shards").glob("shard_*.json"))
    shards = [_read_json(path) for path in shard_paths]
    observed_files = sorted(path for path in Path(results_root).rglob("*") if path.is_file())
    expected_file_count = 2 * len(identities) + len(EXPECTED_NODES)
    sealed_paths = [Path(value) for value in protocol["future_outputs"].values()]
    integrity = {
        "no_errors": not errors,
        "all_rows_present": len(rows) == len(identities) == len(index),
        "all_row_gates": bool(rows) and all(row["gate"]["passed"] for row in rows),
        "paired_matrix_complete": len(paired) == len(identities) // 2,
        "all_six_shards": len(shards) == len(EXPECTED_NODES)
        and {row.get("hostname") for row in shards} == set(EXPECTED_NODES)
        and all(row.get("protocol") == SHARD_PROTOCOL and row.get("passed") for row in shards),
        "result_file_count": len(observed_files) == expected_file_count,
        "launch_submitted": launch.get("submitted") is True,
        "launch_protocol_hash": launch.get("frozen_protocol_sha256")
        == _sha256(protocol_path),
        "six_physical_nodes": set(launch.get("physical_nodes", ()))
        == set(EXPECTED_NODES),
        "sealed_outputs_absent": all(not path.exists() for path in sealed_paths),
    }
    integrity["passed"] = all(integrity.values())
    advance = {
        "integrity_passed": bool(integrity["passed"]),
        "all_city_contracts": bool(city_summaries)
        and all(row["feasible"] for row in city_summaries.values()),
    }
    advance["passed"] = all(advance.values())
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": PASS_DECISION if advance["passed"] else FAIL_DECISION,
        "claim_boundary": (
            "Six-seed adaptation/development evidence only. Validation and prospective "
            "seeds remain sealed regardless of this result."
        ),
        "input_hashes": {
            "protocol": _sha256(protocol_path),
            "launch": _sha256(launch_path),
        },
        "rows": rows,
        "paired_rows": paired,
        "city_summaries": city_summaries,
        "errors": errors,
        "integrity_gate": integrity,
        "advance_gate": advance,
        "result_file_count": len(observed_files),
        "result_tree_sha256": _sha256_tree(Path(results_root)),
    }


def _sha256_tree(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(item for item in Path(root).rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v75 audit: {args.out}")
    payload = audit(
        protocol_path=args.protocol,
        launch_path=args.launch,
        results_root=args.results_root,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "decision": payload["decision"],
                "advance": payload["advance_gate"]["passed"],
                "cities": {
                    city: {
                        "feasible": row["feasible"],
                        "mean_delta": row["city_mean_relative_delta"],
                    }
                    for city, row in payload["city_summaries"].items()
                },
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
