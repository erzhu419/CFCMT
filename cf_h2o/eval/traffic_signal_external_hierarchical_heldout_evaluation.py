"""Confirm the frozen hierarchical controller on six fresh offline seeds.

The controller, numeric gates, seed list, and analysis thresholds are frozen
before this module is run.  The result can authorize only the predeclared
closed-loop development grid; it is not a closed-loop efficacy claim.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import re
import socket
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_estimand_aligned_confirmation import (
    _read_json,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_guard_diagnostic import (
    HIERARCHICAL,
    hierarchical_variant_rows,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_guard_freeze import (
    MODEL_PROTOCOL,
    RESULT_PROTOCOL as FREEZE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    oof_action_records,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import (
    ContrastGuardConfig,
    PriorRegularizationConfig,
)
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.target_action_support import (
    HierarchicalTargetActionSupport,
    TargetActionSupport,
)


FROZEN_PROTOCOL = "tsc-v61r57-external-v9-hierarchical-guard-confirmation-v1"
FREEZE_AUDIT_PROTOCOL = (
    "tsc-v60r56-external-v9-hierarchical-guard-joint-audit-v1"
)
FREEZE_AUDIT_DECISION = "authorize_hierarchical_fresh_offline_seed_collection"
CACHE_AUDIT_PROTOCOL = (
    "tsc-v61r57-external-v9-hierarchical-offline-cache-audit-v1"
)
CACHE_AUDIT_DECISION = "authorize_hierarchical_action_level_confirmation"
RESULT_PROTOCOL = (
    "tsc-v62r58-external-v9-hierarchical-fresh-offline-confirmation-v1"
)
AUTHORIZATION_DECISION = "authorize_closed_loop_development_grid_replay_only"
FAILURE_DECISION = (
    "retain_hierarchical_fresh_offline_failure_and_prohibit_closed_loop_replay"
)
BOOTSTRAP_SEED = 20260815
BOOTSTRAP_CHUNK_SIZE = 256
_GROUP_SEED_RE = re.compile(r"(?:^|:)seed(?P<seed>[0-9]+)(?=:)")


def parse_action_group_seed(group_id: str) -> int:
    """Extract the exact simulator seed encoded in a cache action-group ID."""

    match = _GROUP_SEED_RE.search(str(group_id))
    if match is None:
        raise ValueError(f"action group lacks an encoded seed: {group_id!r}")
    return int(match.group("seed"))


def _equal_seed_scenario_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenarios: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap action groups while giving every seed-scenario equal weight."""

    expected = {
        (int(simulator_seed), str(scenario))
        for simulator_seed in seeds
        for scenario in scenarios
    }
    by_stratum: dict[tuple[int, str], np.ndarray] = {}
    for key in sorted(expected):
        values = np.asarray(
            [
                float(row["actual_delta"])
                for row in rows
                if int(row["simulator_seed"]) == key[0]
                and str(row["scenario"]) == key[1]
            ],
            dtype=float,
        )
        if values.size == 0:
            raise ValueError(f"fresh offline bootstrap lacks stratum {key}")
        if not np.isfinite(values).all():
            raise ValueError(f"fresh offline bootstrap has non-finite values in {key}")
        by_stratum[key] = values
    observed_keys = {
        (int(row["simulator_seed"]), str(row["scenario"])) for row in rows
    }
    if observed_keys != expected:
        raise ValueError("fresh offline rows contain an undeclared seed or scenario")

    stratum_means = {key: float(values.mean()) for key, values in by_stratum.items()}
    seed_means = {
        int(simulator_seed): float(
            np.mean(
                [
                    stratum_means[(int(simulator_seed), str(scenario))]
                    for scenario in scenarios
                ]
            )
        )
        for simulator_seed in seeds
    }
    observed = float(np.mean(list(stratum_means.values())))
    rng = np.random.default_rng(int(seed))
    draws = np.zeros(int(replicates), dtype=float)
    for values in by_stratum.values():
        for start in range(0, int(replicates), BOOTSTRAP_CHUNK_SIZE):
            stop = min(start + BOOTSTRAP_CHUNK_SIZE, int(replicates))
            indices = rng.integers(
                0, values.size, size=(stop - start, values.size)
            )
            draws[start:stop] += values[indices].mean(axis=1)
    draws /= float(len(by_stratum))
    return {
        "protocol": "paired-action-group-equal-seed-scenario-bootstrap-v1",
        "unit": "counterfactual action group within simulator seed and scenario",
        "seed_weighting": "equal",
        "scenario_weighting_within_city": "equal",
        "replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "seed_count": len(tuple(seeds)),
        "scenario_count": len(tuple(scenarios)),
        "stratum_count": len(by_stratum),
        "group_counts": {
            f"seed{key[0]}::{key[1]}": int(values.size)
            for key, values in by_stratum.items()
        },
        "stratum_mean_delta": {
            f"seed{key[0]}::{key[1]}": value
            for key, value in stratum_means.items()
        },
        "seed_mean_delta": {str(key): value for key, value in seed_means.items()},
        "observed_mean_delta_vs_phase_pressure": observed,
        "worst_seed_scenario_mean_delta": float(max(stratum_means.values())),
        "nonpositive_seed_fraction": float(
            np.mean(np.asarray(list(seed_means.values())) <= 0.0)
        ),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "one_sided_probability_delta_nonnegative": float(np.mean(draws >= 0.0)),
    }


def _city_gate(
    *,
    rows: Sequence[Mapping[str, Any]],
    bootstrap: Mapping[str, Any],
    gate_spec: Mapping[str, Any],
) -> dict[str, Any]:
    decisions = len(rows)
    overrides = sum(bool(row["selected"]) for row in rows)
    override_fraction = float(overrides / max(decisions, 1))
    observed = float(bootstrap["observed_mean_delta_vs_phase_pressure"])
    upper = float(bootstrap["ci95"][1])
    worst = float(bootstrap["worst_seed_scenario_mean_delta"])
    nonpositive = float(bootstrap["nonpositive_seed_fraction"])
    result = {
        "decisions": decisions,
        "accepted_overrides": overrides,
        "override_fraction": override_fraction,
        "observed_mean_delta_vs_phase_pressure": observed,
        "bootstrap_95pct_upper_delta": upper,
        "worst_seed_scenario_mean_delta": worst,
        "nonpositive_seed_fraction": nonpositive,
        "mean_improvement_passed": observed
        <= -float(gate_spec["minimum_equal_seed_scenario_mean_improvement"]),
        "bootstrap_upper_passed": upper
        <= float(gate_spec["maximum_bootstrap_95pct_upper_delta"]),
        "worst_seed_scenario_passed": worst
        <= float(gate_spec["maximum_seed_scenario_mean_delta"]),
        "nonpositive_seed_fraction_passed": nonpositive
        >= float(gate_spec["minimum_nonpositive_seed_fraction"]),
        "minimum_override_fraction_passed": override_fraction
        >= float(gate_spec["minimum_override_fraction"]),
        "maximum_override_fraction_passed": override_fraction
        <= float(gate_spec["maximum_override_fraction"]),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "mean_improvement_passed",
            "bootstrap_upper_passed",
            "worst_seed_scenario_passed",
            "nonpositive_seed_fraction_passed",
            "minimum_override_fraction_passed",
            "maximum_override_fraction_passed",
        )
    )
    return result


def _load_frozen_city_model(
    *,
    city: str,
    freeze_root: Path,
    protocol: Mapping[str, Any],
    freeze_audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = dict(protocol["city_artifacts"][city])
    audited = dict(freeze_audit["cities"][city])
    certificate_path = Path(freeze_root) / city / "freeze.json"
    model_path = Path(freeze_root) / city / "model.pkl"
    if (
        _sha256(certificate_path) != str(artifact["certificate"]["sha256"])
        or _sha256(certificate_path) != str(audited["certificate"]["sha256"])
        or _sha256(model_path) != str(artifact["model"]["sha256"])
        or _sha256(model_path) != str(audited["model"]["sha256"])
    ):
        raise ValueError(f"hierarchical artifact identity changed for {city}")
    certificate = _read_json(certificate_path)
    model = pickle.loads(model_path.read_bytes())
    if (
        certificate.get("protocol") != FREEZE_PROTOCOL
        or certificate.get("city") != city
        or not bool(certificate.get("freeze_gate", {}).get("passed", False))
        or model.get("protocol") != MODEL_PROTOCOL
        or model.get("city") != city
        or not isinstance(model.get("models"), OfflineScreeningModels)
        or not isinstance(model.get("regularizer"), PriorRegularizationConfig)
        or not isinstance(model.get("guard"), ContrastGuardConfig)
        or not isinstance(model.get("pressure_target_support"), TargetActionSupport)
        or not isinstance(model.get("conformal_target_support"), TargetActionSupport)
        or not isinstance(model.get("target_support"), HierarchicalTargetActionSupport)
    ):
        raise ValueError(f"hierarchical model contract changed for {city}")
    return certificate, model


def run_hierarchical_fresh_offline_confirmation(
    *,
    evaluation_cache_root: Path,
    cache_audit_path: Path,
    expected_cache_audit_sha256: str,
    external_manifest_path: Path,
    conversion_root: Path,
    protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    freeze_audit = _read_json(freeze_audit_path)
    cache_audit = _read_json(cache_audit_path)
    protocol_sha = _sha256(protocol_path)
    freeze_audit_sha = _sha256(freeze_audit_path)
    cache_audit_sha = _sha256(cache_audit_path)
    if (
        protocol.get("protocol") != FROZEN_PROTOCOL
        or freeze_audit.get("protocol") != FREEZE_AUDIT_PROTOCOL
        or freeze_audit.get("status") != "PASS"
        or freeze_audit.get("decision") != FREEZE_AUDIT_DECISION
        or not bool(freeze_audit.get("integrity_gate", {}).get("passed", False))
        or freeze_audit_sha
        != str(protocol["hierarchical_joint_audit"]["sha256"])
        or cache_audit_sha != str(expected_cache_audit_sha256)
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("decision") != CACHE_AUDIT_DECISION
        or not bool(cache_audit.get("integrity_gate", {}).get("passed", False))
        or cache_audit.get("frozen_protocol_sha256") != protocol_sha
    ):
        raise ValueError("hierarchical fresh-offline evidence chain changed")

    offline = dict(protocol["fresh_offline_confirmation"])
    gate_spec = dict(offline["action_level_gate"])
    seeds = tuple(int(value) for value in offline["seeds"])
    city_scenarios = {
        str(city): tuple(str(value) for value in scenarios)
        for city, scenarios in protocol["city_scenarios"].items()
    }
    expected_cache_files = len(seeds) * sum(
        len(scenarios) for scenarios in city_scenarios.values()
    ) * int(offline["collection_shards_per_scenario_seed"])
    cache_sha, cache_files = aggregate_cache_sha256(evaluation_cache_root)
    if (
        cache_sha != str(cache_audit["cache_sha256"])
        or cache_files != int(cache_audit["cache_file_count"])
        or cache_files != expected_cache_files
    ):
        raise ValueError("hierarchical fresh-offline cache identity changed")

    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(
        Path(conversion_root).resolve()
    )
    manifest = load_traffic_signal_manifest(external_manifest_path)
    bank, bank_audit = load_frozen_counterfactual_bank(
        evaluation_cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(offline["collection_shards_per_scenario_seed"]),
        workers=max(int(workers), 1),
    )
    city_results: dict[str, Any] = {}
    for city_index, (city, scenarios) in enumerate(sorted(city_scenarios.items())):
        certificate, model = _load_frozen_city_model(
            city=city,
            freeze_root=freeze_root,
            protocol=protocol,
            freeze_audit=freeze_audit,
        )
        dataset = _merge_city_datasets(bank, scenarios, city=city)
        common_kwargs = {
            "models": model["models"],
            "selected_candidate": str(model["selected_candidate"]),
            "fold_index": 0,
            "validation_seed": 0,
            "scenarios": scenarios,
        }
        pressure_records = oof_action_records(
            dataset,
            target_support=model["pressure_target_support"],
            **common_kwargs,
        )
        conformal_records = oof_action_records(
            dataset,
            target_support=model["conformal_target_support"],
            **common_kwargs,
        )
        variant_rows = hierarchical_variant_rows(
            pressure_records=pressure_records,
            conformal_records=conformal_records,
            regularizer=model["regularizer"],
            guard=model["guard"],
        )
        rows = variant_rows[HIERARCHICAL]
        for row in rows:
            row["simulator_seed"] = parse_action_group_seed(str(row["group_id"]))
            if not str(row["group_id"]).startswith(f"{row['scenario']}:"):
                raise ValueError("action group scenario identity changed")
        bootstrap = _equal_seed_scenario_bootstrap(
            rows,
            seeds=seeds,
            scenarios=scenarios,
            replicates=int(gate_spec["bootstrap_replicates"]),
            seed=BOOTSTRAP_SEED + city_index,
        )
        city_gate = _city_gate(rows=rows, bootstrap=bootstrap, gate_spec=gate_spec)
        accepted = [row for row in rows if bool(row["selected"])]
        city_results[city] = {
            "scenarios": list(scenarios),
            "seeds": list(seeds),
            "selected_candidate": str(model["selected_candidate"]),
            "model_sha256": str(protocol["city_artifacts"][city]["model"]["sha256"]),
            "certificate_sha256": str(
                protocol["city_artifacts"][city]["certificate"]["sha256"]
            ),
            "deployment": certificate["deployment"],
            "decision_count": len(rows),
            "accepted_harmful_actions": int(
                sum(float(row["raw_actual_delta"]) > 0.0 for row in accepted)
            ),
            "accepted_beneficial_actions": int(
                sum(float(row["raw_actual_delta"]) < 0.0 for row in accepted)
            ),
            "bootstrap": bootstrap,
            "gate": city_gate,
            "action_rows": rows,
        }
    overall_gate = {
        "all_cities_present": set(city_results) == set(city_scenarios),
        "all_city_gates_passed": bool(city_results)
        and all(row["gate"]["passed"] for row in city_results.values()),
        "fresh_seed_matrix_exact": tuple(cache_audit.get("seeds", ())) == seeds,
        "no_target_refit": all(
            bool(row["deployment"].get("policy") == "cfcmt_hierarchical_guard_mpc")
            for row in city_results.values()
        ),
    }
    overall_gate["passed"] = all(overall_gate.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "runtime": _runtime_metadata(),
        "claim_boundary": (
            "Fresh-seed counterfactual action-regret confirmation of the immutable "
            "hierarchical controller. A pass authorizes only the frozen closed-loop "
            "development grid and does not establish closed-loop efficacy."
        ),
        "frozen_protocol_sha256": protocol_sha,
        "freeze_audit_sha256": freeze_audit_sha,
        "cache_audit_sha256": cache_audit_sha,
        "evaluation_cache_sha256": cache_sha,
        "evaluation_cache_file_count": cache_files,
        "evaluation_cache_audit": bank_audit,
        "analysis_contract": {
            "controller_refit": False,
            "variant": HIERARCHICAL,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "gate": gate_spec,
            "reporting_unit": "city",
            "seed_weighting": "equal",
            "scenario_weighting_within_city": "equal",
        },
        "city_results": city_results,
        "offline_confirmation_gate": overall_gate,
        "decision": (
            AUTHORIZATION_DECISION if overall_gate["passed"] else FAILURE_DECISION
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-cache-root", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--expected-cache-audit-sha256", required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze-audit", type=Path, required=True)
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(
            f"refusing to overwrite hierarchical offline result: {args.out}"
        )
    payload = run_hierarchical_fresh_offline_confirmation(
        evaluation_cache_root=args.evaluation_cache_root,
        cache_audit_path=args.cache_audit,
        expected_cache_audit_sha256=args.expected_cache_audit_sha256,
        external_manifest_path=args.external_manifest,
        conversion_root=args.conversion_root,
        protocol_path=args.protocol,
        freeze_audit_path=args.freeze_audit,
        freeze_root=args.freeze_root,
        workers=args.workers,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "gate_passed": payload["offline_confirmation_gate"]["passed"],
                "decision": payload["decision"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["offline_confirmation_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
