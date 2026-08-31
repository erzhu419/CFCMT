#!/usr/bin/env python3
"""Run one process-isolated shard of source-city component candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_causal_source_component_rollout import (
    RESULT_PROTOCOL,
    run_source_city_component_rollout,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import ContrastGuardConfig


SHARD_PROTOCOL = "tsc-v93-source-city-component-process-shard-v1"
CANDIDATE_PROTOCOL = "tsc-v93-source-city-candidate-grid-v1"
SCENARIOS = (
    "la_1x4",
    "jinan_3x4_real",
    "jinan_3x4_real_2000",
    "jinan_3x4_real_2500",
)
CITIES = ("los_angeles", "jinan")
BASELINE_CANDIDATE_POLICIES = ("phase_pressure",)
BASELINE_RESULT_PROTOCOL = (
    "tsc-v111-target-offline-source-guard-baseline-rollout-v1"
)


def load_candidate_specs(path: Path) -> tuple[dict[str, Any], ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("protocol") != CANDIDATE_PROTOCOL:
        raise ValueError("source-city candidate protocol changed")
    candidates = tuple(dict(value) for value in payload.get("candidates", ()))
    keys = tuple(str(value.get("key", "")) for value in candidates)
    if (
        not candidates
        or len(keys) != len(set(keys))
        or any(not re.fullmatch(r"[a-z0-9_]+", key) for key in keys)
    ):
        raise ValueError("candidate keys are empty, duplicated, or unsafe")
    for candidate in candidates:
        by_city = candidate.get("weights_by_city")
        if not isinstance(by_city, Mapping) or set(by_city) != set(CITIES):
            raise ValueError("each candidate must define both target cities")
        for city in CITIES:
            weights = {
                str(name): float(value)
                for name, value in dict(by_city[city]).items()
            }
            if any(value < 0.0 for value in weights.values()) or sum(
                weights.values()
            ) > 1.0 + 1e-12:
                raise ValueError("candidate source weights are not convex")
            candidate["weights_by_city"][city] = weights
        baseline_policy = candidate.get("baseline_policy")
        if baseline_policy is not None:
            if (
                str(baseline_policy) not in BASELINE_CANDIDATE_POLICIES
                or any(candidate["weights_by_city"][city] for city in CITIES)
                or "guard_by_city" in candidate
            ):
                raise ValueError("baseline candidate contract changed")
            candidate["baseline_policy"] = str(baseline_policy)
        if "guard_by_city" in candidate:
            raw_guards = candidate["guard_by_city"]
            if not isinstance(raw_guards, Mapping) or set(raw_guards) != set(CITIES):
                raise ValueError("candidate guard must define both target cities")
            normalized_guards = {}
            for city in CITIES:
                raw_guard = raw_guards[city]
                if raw_guard is None:
                    normalized_guards[city] = None
                    continue
                if not isinstance(raw_guard, Mapping):
                    raise ValueError("candidate city guard must be an object or null")
                values = dict(raw_guard)
                values.pop("enabled", None)
                expected = {
                    "risk_multiplier",
                    "min_context_trust",
                    "margin",
                    "max_relative_rule_gap",
                }
                if set(values) != expected:
                    raise ValueError("candidate city guard fields changed")
                guard = ContrastGuardConfig(enabled=True, **values)
                normalized_guards[city] = {
                    "enabled": True,
                    "risk_multiplier": float(guard.risk_multiplier),
                    "min_context_trust": float(guard.min_context_trust),
                    "margin": float(guard.margin),
                    "max_relative_rule_gap": float(guard.max_relative_rule_gap),
                }
            candidate["guard_by_city"] = normalized_guards
    model_candidates = tuple(
        candidate for candidate in candidates if "baseline_policy" not in candidate
    )
    guarded = tuple(
        candidate for candidate in model_candidates if "guard_by_city" in candidate
    )
    if guarded:
        if len(guarded) != len(model_candidates):
            raise ValueError("all confirmation candidates must define city guards")
        reference = guarded[0]["guard_by_city"]
        if any(candidate["guard_by_city"] != reference for candidate in guarded[1:]):
            raise ValueError("confirmation candidates must share city guards")
    return candidates


def all_identities(
    seeds: Sequence[int],
    candidate_keys: Sequence[str],
    *,
    scenarios: Sequence[str] = SCENARIOS,
) -> tuple[tuple[str, int, str], ...]:
    normalized_seeds = tuple(int(value) for value in seeds)
    normalized_keys = tuple(str(value) for value in candidate_keys)
    normalized_scenarios = tuple(str(value) for value in scenarios)
    if (
        not normalized_seeds
        or len(set(normalized_seeds)) != len(normalized_seeds)
        or not normalized_keys
        or len(set(normalized_keys)) != len(normalized_keys)
        or not normalized_scenarios
        or len(set(normalized_scenarios)) != len(normalized_scenarios)
        or set(normalized_scenarios) - set(SCENARIOS)
    ):
        raise ValueError("source-component seed/candidate grid is invalid")
    return tuple(
        (scenario, seed, key)
        for scenario in normalized_scenarios
        for seed in normalized_seeds
        for key in normalized_keys
    )


def shard_identities(
    seeds: Sequence[int],
    candidate_keys: Sequence[str],
    *,
    shard_index: int,
    shard_count: int,
    scenarios: Sequence[str] = SCENARIOS,
) -> tuple[tuple[str, int, str], ...]:
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid source-component shard index/count")
    return tuple(
        identity
        for index, identity in enumerate(
            all_identities(seeds, candidate_keys, scenarios=scenarios)
        )
        if index % shard_count == shard_index
    )


def _run_one(payload: Mapping[str, Any]) -> dict[str, Any]:
    scenario, seed, candidate_key = payload["identity"]
    result_path = Path(str(payload["result_path"]))
    tripinfo_path = Path(str(payload["tripinfo_path"]))
    if result_path.exists() or tripinfo_path.exists():
        raise FileExistsError(f"refusing to overwrite rollout: {result_path.parent}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    city = "los_angeles" if scenario == "la_1x4" else "jinan"
    candidate = payload["candidate_by_key"][candidate_key]
    weights = dict(candidate["weights_by_city"][city])
    guard_values = (
        candidate["guard_by_city"][city]
        if "guard_by_city" in candidate
        else payload.get("guard_config")
    )
    guard = ContrastGuardConfig(**guard_values) if guard_values is not None else None
    rollout_kwargs = {
        "protocol_spec_path": Path(str(payload["protocol"])),
        "offline_authorization_path": Path(str(payload["offline_authorization"])),
        "expected_offline_authorization_sha256": str(
            payload["offline_authorization_sha256"]
        ),
        "joint_freeze_audit_path": Path(str(payload["joint_audit"])),
        "expected_joint_freeze_sha256": str(payload["joint_audit_sha256"]),
        "freeze_root": Path(str(payload["freeze_root"])),
        "external_manifest_path": Path(str(payload["external_manifest"])),
        "conversion_root": Path(str(payload["conversion_root"])),
        "expected_external_manifest_sha256": str(
            payload["external_manifest_sha256"]
        ),
        "conversion_manifest_path": Path(str(payload["conversion_manifest"])),
        "expected_conversion_manifest_sha256": str(
            payload["conversion_manifest_sha256"]
        ),
        "expected_conversion_tree_sha256": str(payload["conversion_tree_sha256"]),
        "expected_sumo_version": "1.22.0",
        "scenario": str(scenario),
        "seed": int(seed),
        "tripinfo_out": tripinfo_path,
    }
    started = time.monotonic()
    baseline_policy = candidate.get("baseline_policy")
    if baseline_policy is not None:
        diagnostic = ClosedLoopDiagnosticSpec(
            name=f"cfcmt_source_component_{candidate_key}_baseline",
            source_policy=str(baseline_policy),
            coordination_mode="sparse",
            result_protocol=BASELINE_RESULT_PROTOCOL,
            allowed_seeds=tuple(int(value) for value in payload["allowed_seeds"]),
            analysis_status="prospective-v111-target-offline-guard-confirmation",
        )
        result = run_external_closed_loop_rollout(
            **rollout_kwargs,
            policy=diagnostic.name,
            diagnostic_spec=diagnostic,
        )
        result["analysis_protocol"] = BASELINE_RESULT_PROTOCOL
        result["candidate_key"] = str(candidate_key)
        result["source_city_weights"] = {}
        result["source_mass"] = 0.0
        result["source_guard_config"] = None
        result["source_only_guard"] = None
        result["target_only_anchor"] = None
    else:
        result = run_source_city_component_rollout(
            main_model_path=Path(str(payload[f"{city}_main_model"])),
            expected_main_model_sha256=str(payload[f"{city}_main_model_sha256"]),
            component_bundle_path=Path(str(payload[f"{city}_component_bundle"])),
            expected_component_bundle_sha256=str(
                payload[f"{city}_component_bundle_sha256"]
            ),
            source_city_weights=weights,
            candidate_key=str(candidate_key),
            allowed_seeds=tuple(int(value) for value in payload["allowed_seeds"]),
            guard_config=guard,
            source_only_guard_path=(
                Path(str(payload["source_only_guard"]))
                if payload.get("source_only_guard") is not None
                else None
            ),
            expected_source_only_guard_sha256=payload.get(
                "source_only_guard_sha256"
            ),
            strict_target_only_model_path=(
                Path(str(payload[f"{city}_strict_target_only_model"]))
                if payload.get(f"{city}_strict_target_only_model") is not None
                else None
            ),
            expected_strict_target_only_model_sha256=payload.get(
                f"{city}_strict_target_only_model_sha256"
            ),
            rollout_kwargs=rollout_kwargs,
        )
    _atomic_json(result_path, result)
    metrics = result["metrics"]
    return {
        "city": city,
        "scenario": str(scenario),
        "seed": int(seed),
        "candidate_key": str(candidate_key),
        "source_city_weights": weights,
        "guard_config": guard_values,
        "baseline_policy": baseline_policy,
        "target_only_anchor": result.get("target_only_anchor"),
        "hostname": socket.gethostname(),
        "elapsed_sec": float(time.monotonic() - started),
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "tripinfo_path": str(tripinfo_path),
        "tripinfo_sha256": _sha256(tripinfo_path),
        "metrics": {
            key: metrics[key]
            for key in (
                "mean_tripinfo_waiting_time",
                "mean_tripinfo_duration",
                "mean_tripinfo_time_loss",
                "system_vehicle_hours",
                "completion_ratio",
                "collision_incidents",
                "starting_teleports",
                "ending_teleports",
            )
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--candidate-spec-sha256", required=True)
    parser.add_argument("--guard-risk-multiplier", type=float)
    parser.add_argument("--guard-min-context-trust", type=float, default=0.0)
    parser.add_argument("--guard-margin", type=float, default=0.0)
    parser.add_argument("--guard-max-relative-rule-gap", type=float, default=1.0)
    parser.add_argument("--source-only-guard", type=Path)
    parser.add_argument("--source-only-guard-sha256")
    for city in CITIES:
        parser.add_argument(f"--{city.replace('_', '-')}-main-model", type=Path, required=True)
        parser.add_argument(f"--{city.replace('_', '-')}-main-model-sha256", required=True)
        parser.add_argument(f"--{city.replace('_', '-')}-component-bundle", type=Path, required=True)
        parser.add_argument(f"--{city.replace('_', '-')}-component-bundle-sha256", required=True)
        parser.add_argument(
            f"--{city.replace('_', '-')}-strict-target-only-model", type=Path
        )
        parser.add_argument(
            f"--{city.replace('_', '-')}-strict-target-only-model-sha256"
        )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--offline-authorization", type=Path, required=True)
    parser.add_argument("--offline-authorization-sha256", required=True)
    parser.add_argument("--joint-audit", type=Path, required=True)
    parser.add_argument("--joint-audit-sha256", required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest-sha256", required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--conversion-manifest", type=Path, required=True)
    parser.add_argument("--conversion-manifest-sha256", required=True)
    parser.add_argument("--conversion-tree-sha256", required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args(argv)
    for city in CITIES:
        strict_values = (
            getattr(args, f"{city}_strict_target_only_model"),
            getattr(args, f"{city}_strict_target_only_model_sha256"),
        )
        if any(value is not None for value in strict_values) and not all(
            value is not None for value in strict_values
        ):
            raise ValueError(
                f"{city} strict target-only model arguments are incomplete"
            )
    if _sha256(args.candidate_spec) != str(args.candidate_spec_sha256):
        raise ValueError("source-city candidate specification identity changed")
    source_guard_values = (
        args.source_only_guard,
        args.source_only_guard_sha256,
    )
    if any(value is not None for value in source_guard_values) and not all(
        value is not None for value in source_guard_values
    ):
        raise ValueError("source-only guard arguments are incomplete")
    if args.source_only_guard is not None and _sha256(
        args.source_only_guard
    ) != str(args.source_only_guard_sha256):
        raise ValueError("source-only guard identity changed")
    candidates = load_candidate_specs(args.candidate_spec)
    candidate_by_key = {str(row["key"]): row for row in candidates}
    identities = shard_identities(
        args.seeds,
        tuple(candidate_by_key),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        scenarios=args.scenarios,
    )
    guard_config = (
        {
            "enabled": True,
            "risk_multiplier": float(args.guard_risk_multiplier),
            "min_context_trust": float(args.guard_min_context_trust),
            "margin": float(args.guard_margin),
            "max_relative_rule_gap": float(args.guard_max_relative_rule_gap),
        }
        if args.guard_risk_multiplier is not None
        else None
    )
    guarded_candidates = tuple(
        candidate for candidate in candidates if "guard_by_city" in candidate
    )
    if guarded_candidates and guard_config is not None:
        raise ValueError("do not mix candidate city guards with a global guard")
    if args.source_only_guard is not None and (
        guarded_candidates or guard_config is not None
    ):
        raise ValueError("do not mix source-only and legacy guard inputs")
    candidate_guard_by_city = (
        dict(guarded_candidates[0]["guard_by_city"])
        if guarded_candidates
        else None
    )
    common = {
        "allowed_seeds": tuple(args.seeds),
        "candidate_by_key": candidate_by_key,
        "guard_config": guard_config,
        "candidate_guard_by_city": candidate_guard_by_city,
        "source_only_guard": (
            str(args.source_only_guard)
            if args.source_only_guard is not None
            else None
        ),
        "source_only_guard_sha256": args.source_only_guard_sha256,
        "scenarios": list(args.scenarios),
        "protocol": str(args.protocol),
        "offline_authorization": str(args.offline_authorization),
        "offline_authorization_sha256": args.offline_authorization_sha256,
        "joint_audit": str(args.joint_audit),
        "joint_audit_sha256": args.joint_audit_sha256,
        "freeze_root": str(args.freeze_root),
        "external_manifest": str(args.external_manifest),
        "external_manifest_sha256": args.external_manifest_sha256,
        "conversion_root": str(args.conversion_root),
        "conversion_manifest": str(args.conversion_manifest),
        "conversion_manifest_sha256": args.conversion_manifest_sha256,
        "conversion_tree_sha256": args.conversion_tree_sha256,
    }
    for city in CITIES:
        prefix = city
        common[f"{prefix}_main_model"] = str(getattr(args, f"{prefix}_main_model"))
        common[f"{prefix}_main_model_sha256"] = getattr(
            args, f"{prefix}_main_model_sha256"
        )
        common[f"{prefix}_component_bundle"] = str(
            getattr(args, f"{prefix}_component_bundle")
        )
        common[f"{prefix}_component_bundle_sha256"] = getattr(
            args, f"{prefix}_component_bundle_sha256"
        )
        strict_model = getattr(args, f"{prefix}_strict_target_only_model")
        common[f"{prefix}_strict_target_only_model"] = (
            str(strict_model) if strict_model is not None else None
        )
        common[f"{prefix}_strict_target_only_model_sha256"] = getattr(
            args, f"{prefix}_strict_target_only_model_sha256"
        )
    payloads = []
    for scenario, seed, candidate_key in identities:
        root = args.results_root / scenario / f"seed_{seed}" / candidate_key
        payloads.append(
            {
                **common,
                "identity": (scenario, seed, candidate_key),
                "result_path": str(root / "result.json"),
                "tripinfo_path": str(root / "tripinfo.xml"),
            }
        )
    started = time.monotonic()
    os.chdir(Path(__file__).resolve().parents[2])
    workers = min(max(int(args.workers), 1), len(payloads))
    with mp.get_context("spawn").Pool(
        processes=workers, maxtasksperchild=1
    ) as pool:
        rows = list(pool.imap_unordered(_run_one, payloads, chunksize=1))
    rows.sort(key=lambda row: (row["scenario"], row["seed"], row["candidate_key"]))
    observed = {
        (row["scenario"], int(row["seed"]), str(row["candidate_key"]))
        for row in rows
    }
    if observed != set(identities):
        raise RuntimeError("source-component shard coverage failed")
    summary = {
        "protocol": SHARD_PROTOCOL,
        "result_protocol": RESULT_PROTOCOL,
        "candidate_protocol": CANDIDATE_PROTOCOL,
        "candidate_spec": str(args.candidate_spec.resolve()),
        "candidate_spec_sha256": str(args.candidate_spec_sha256),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "workers": workers,
        "fresh_process_per_rollout": True,
        "guard_config": guard_config,
        "candidate_guard_by_city": candidate_guard_by_city,
        "source_only_guard": (
            str(args.source_only_guard.resolve())
            if args.source_only_guard is not None
            else None
        ),
        "source_only_guard_sha256": args.source_only_guard_sha256,
        "scenarios": list(args.scenarios),
        "task_count": len(rows),
        "elapsed_sec": float(time.monotonic() - started),
        "passed": True,
        "rows": rows,
    }
    summary_path = args.results_root / "_shards" / f"shard_{args.shard_index}.json"
    if summary_path.exists():
        raise FileExistsError(f"refusing to overwrite shard summary: {summary_path}")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(summary_path, summary)
    print(
        json.dumps(
            {
                "status": "PASS",
                "shard_index": args.shard_index,
                "task_count": len(rows),
                "elapsed_sec": summary["elapsed_sec"],
            },
            sort_keys=True,
        )
    )
    print(f"Eval complete: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
