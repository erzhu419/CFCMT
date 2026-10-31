"""Independent Salt Lake confirmation for the globally frozen pairwise family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_group_normalized_selection import (
    GROUP_NORMALIZED_RIGID_FAMILY,
)
from cf_h2o.eval.traffic_signal_pairwise_preference_selection import (
    PAIRWISE_PREFERENCE_FAMILY,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _target_adaptation_split_v3,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    evaluate_target_action_groups,
    fit_target_screening_models,
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.generalized_pressure import generalized_pressure_grid
from cf_h2o.traffic_signal.mechanism_world_model import MechanismFitConfig


TARGETS = (
    "saltlake_400s_200w_q1_weekday_peak",
    "saltlake_state_university_q1_weekday_peak",
)
TARGET_CITY_GROUP = "salt_lake_city"
TARGET_GROUP_BUDGET = 60
FALLBACK_FAMILY = GROUP_NORMALIZED_RIGID_FAMILY
CANDIDATE_FAMILY = PAIRWISE_PREFERENCE_FAMILY
CONFIRMATION_FAMILIES = (FALLBACK_FAMILY, CANDIDATE_FAMILY)
SOURCE_SEEDS = (2027, 3037, 4047)
COLLECTION_SHARDS = 16
EXPECTED_COMBINED_FILE_COUNT = 18 * len(SOURCE_SEEDS) * COLLECTION_SHARDS
MIN_MACRO_RELATIVE_IMPROVEMENT = 0.10
MIN_IMPROVED_NETWORK_COUNT = 2
MAX_ABSOLUTE_NETWORK_REGRESSION = 0.05
NUMERICAL_TOLERANCE = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_cache_sha256(cache_root: Path) -> tuple[str, int]:
    files = sorted(Path(cache_root).glob("*.npz"), key=lambda path: path.name)
    names = [path.name for path in files]
    if not files or len(names) != len(set(names)):
        raise ValueError("combined cache must contain unique NPZ basenames")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest(), len(files)


def summarize_saltlake_confirmation(
    regrets: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    if set(regrets) != set(TARGETS):
        raise ValueError("Salt Lake confirmation target set changed")
    normalized: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        row = regrets[target]
        if set(row) != set(CONFIRMATION_FAMILIES):
            raise ValueError(f"confirmation family set changed for {target}")
        normalized[target] = {}
        for family in CONFIRMATION_FAMILIES:
            value = float(row[family])
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid regret for {target}/{family}: {value}")
            normalized[target][family] = value

    fallback_macro = float(
        np.mean([normalized[target][FALLBACK_FAMILY] for target in TARGETS])
    )
    candidate_macro = float(
        np.mean([normalized[target][CANDIDATE_FAMILY] for target in TARGETS])
    )
    if fallback_macro <= NUMERICAL_TOLERANCE:
        raise ValueError("same-information fallback macro regret is zero")
    improvements = {
        target: normalized[target][FALLBACK_FAMILY]
        - normalized[target][CANDIDATE_FAMILY]
        for target in TARGETS
    }
    relative_improvement = (fallback_macro - candidate_macro) / fallback_macro
    improved_count = sum(
        value > NUMERICAL_TOLERANCE for value in improvements.values()
    )
    max_regression = max(max(-value, 0.0) for value in improvements.values())
    gate = {
        "macro_relative_improvement_at_least_0_10": (
            relative_improvement + NUMERICAL_TOLERANCE
            >= MIN_MACRO_RELATIVE_IMPROVEMENT
        ),
        "both_networks_strictly_improve": (
            improved_count >= MIN_IMPROVED_NETWORK_COUNT
        ),
        "maximum_absolute_network_regression_at_most_0_05": (
            max_regression
            <= MAX_ABSOLUTE_NETWORK_REGRESSION + NUMERICAL_TOLERANCE
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "protocol": "globally-frozen-pairwise-salt-lake-confirmation-gate-v1",
        "passed": gate["passed"],
        "decision": (
            "confirm_global_pairwise_target_simulator_adaptation"
            if gate["passed"]
            else "reject_global_pairwise_salt_lake_confirmation"
        ),
        "fallback_macro_regret": fallback_macro,
        "candidate_macro_regret": candidate_macro,
        "macro_relative_improvement": relative_improvement,
        "macro_relative_improvement_pct": 100.0 * relative_improvement,
        "network_improvements": improvements,
        "improved_network_count": improved_count,
        "maximum_absolute_network_regression": max_regression,
        "gate": gate,
    }


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    confirmation = payload.get("confirmation", {})
    if (
        confirmation.get("candidate_family") != CANDIDATE_FAMILY
        or confirmation.get("same_information_fallback") != FALLBACK_FAMILY
        or int(confirmation.get("target_group_budget", -1))
        != TARGET_GROUP_BUDGET
        or tuple(confirmation.get("target_scenarios", ())) != TARGETS
        or confirmation.get("target_specific_family_selection") is not False
    ):
        raise ValueError("v38 protocol specification differs from executable contract")
    expected_gate = {
        "minimum_macro_relative_improvement": MIN_MACRO_RELATIVE_IMPROVEMENT,
        "minimum_improved_network_count": MIN_IMPROVED_NETWORK_COUNT,
        "required_network_count": len(TARGETS),
        "maximum_absolute_network_regression": MAX_ABSOLUTE_NETWORK_REGRESSION,
    }
    if confirmation.get("primary_gate") != expected_gate:
        raise ValueError("v38 primary confirmation gate changed")
    return payload


def run_target_confirmation(
    *,
    cache_root: Path,
    baseline_result: Path,
    manifest_path: Path,
    protocol_spec_path: Path,
    target: str,
    expected_cache_sha256: str,
    expected_baseline_sha256: str,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    if target not in TARGETS:
        raise ValueError(f"target is outside the frozen Salt Lake set: {target}")
    protocol_spec = _load_protocol(protocol_spec_path)
    baseline_sha256 = _sha256(baseline_result)
    if baseline_sha256 != str(expected_baseline_sha256):
        raise ValueError("source-rule baseline hash mismatch")
    cache_sha256, cache_file_count = aggregate_cache_sha256(cache_root)
    if cache_sha256 != str(expected_cache_sha256):
        raise ValueError("combined counterfactual cache hash mismatch")
    if cache_file_count != EXPECTED_COMBINED_FILE_COUNT:
        raise ValueError(
            f"combined cache file count changed: {cache_file_count} "
            f"!= {EXPECTED_COMBINED_FILE_COUNT}"
        )

    manifest = load_traffic_signal_manifest(manifest_path)
    if tuple(name for name in manifest.sumocfgs if name in TARGETS) != TARGETS:
        raise ValueError("Salt Lake targets are absent or reordered in the manifest")
    if {manifest.city_groups[name] for name in TARGETS} != {TARGET_CITY_GROUP}:
        raise ValueError("Salt Lake city-group identity changed")
    bank, cache_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=SOURCE_SEEDS,
        collection_shards=COLLECTION_SHARDS,
        workers=workers,
    )
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
    if len(selected_group_ids) != TARGET_GROUP_BUDGET or len(
        set(selected_group_ids)
    ) != TARGET_GROUP_BUDGET:
        raise ValueError("target adaptation did not select exactly 60 unique groups")

    baseline = json.loads(baseline_result.read_text(encoding="utf-8"))
    source_rule_costs = baseline["source_rule_policy_costs"]
    source_rule_specs = {spec.key: spec for spec in generalized_pressure_grid()}
    models = fit_target_screening_models(
        bank,
        target,
        fit_config=MechanismFitConfig(),
        source_rule_costs=source_rule_costs,
        source_rule_specs=source_rule_specs,
        model_families=CONFIRMATION_FAMILIES,
        target_group_budget=TARGET_GROUP_BUDGET,
        target_adaptation_selection_seed=20260803,
        target_calibration_seeds=(4047,),
        target_calibration_fraction=0.4,
        source_selector_min_context_domains=5,
        scenario_city_groups=manifest.city_groups,
        target_adaptation_group_ids=selected_group_ids,
    )
    diagnostics = models.diagnostics
    expected_source_scenarios = sorted(set(manifest.sumocfgs) - set(TARGETS))
    if diagnostics.get("heldout_city_scenarios") != sorted(TARGETS):
        raise ValueError("same-city Salt Lake holdout contract failed")
    if diagnostics.get("source_scenarios") != expected_source_scenarios:
        raise ValueError("development source scenario set changed")
    if len(expected_source_scenarios) != 16:
        raise ValueError("confirmation requires exactly 16 development sources")
    source_city_groups = {
        manifest.city_groups[name] for name in expected_source_scenarios
    }
    if len(source_city_groups) != 6 or TARGET_CITY_GROUP in source_city_groups:
        raise ValueError("Salt Lake leaked into the six source city groups")
    if diagnostics.get("target_adaptation_protocol") != (
        "explicit-complete-action-group-list-v1"
    ):
        raise ValueError("final model did not use explicit complete target groups")
    if int(diagnostics.get("target_adaptation_groups", -1)) != TARGET_GROUP_BUDGET:
        raise ValueError("final model did not fit all 60 target groups")
    if diagnostics.get("target_adaptation_group_ids") != list(selected_group_ids):
        raise ValueError("fitted target groups differ from the frozen selection")

    evaluation = evaluate_target_action_groups(
        bank[target],
        models=models,
        excluded_group_ids=selected_group_ids,
    )
    if int(evaluation.get("excluded_group_count", -1)) != TARGET_GROUP_BUDGET:
        raise ValueError("evaluation did not exclude all 60 adaptation groups")
    evaluation_group_count = int(evaluation.get("evaluation_group_count", -1))
    if evaluation_group_count <= 0:
        raise ValueError("no disjoint Salt Lake evaluation groups remain")
    if set(evaluation.get("families", {})) != set(CONFIRMATION_FAMILIES):
        raise ValueError("confirmation evaluation family set changed")
    for family in CONFIRMATION_FAMILIES:
        if int(evaluation["families"][family].get("group_count", -1)) != (
            evaluation_group_count
        ):
            raise ValueError(f"evaluation group count changed for {family}")

    return {
        "protocol": "tsc-v38r34-global-pairwise-salt-lake-target-confirmation-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "protocol_spec_path": str(protocol_spec_path.resolve()),
        "protocol_spec_sha256": _sha256(protocol_spec_path),
        "protocol_spec": protocol_spec,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest.to_dict(),
        "cache_root": str(Path(cache_root).resolve()),
        "cache_aggregate_sha256": cache_sha256,
        "cache_file_count": cache_file_count,
        "cache_audit": cache_audit,
        "baseline_result": str(baseline_result.resolve()),
        "baseline_sha256": baseline_sha256,
        "target": target,
        "target_city_group": TARGET_CITY_GROUP,
        "heldout_city_scenarios": sorted(TARGETS),
        "source_scenarios": expected_source_scenarios,
        "source_city_groups": sorted(source_city_groups),
        "target_group_budget": TARGET_GROUP_BUDGET,
        "selected_group_ids": list(selected_group_ids),
        "model_families": list(CONFIRMATION_FAMILIES),
        "model_diagnostics": diagnostics,
        "offline_evaluation": evaluation,
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
    parser.add_argument("--protocol-spec", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-cache-sha256", required=True)
    parser.add_argument("--expected-baseline-sha256", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = run_target_confirmation(
        cache_root=args.cache_root,
        baseline_result=args.baseline_result,
        manifest_path=args.manifest,
        protocol_spec_path=args.protocol_spec,
        target=args.target,
        expected_cache_sha256=args.expected_cache_sha256,
        expected_baseline_sha256=args.expected_baseline_sha256,
        workers=args.workers,
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
                "evaluation_group_count": payload["offline_evaluation"][
                    "evaluation_group_count"
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
