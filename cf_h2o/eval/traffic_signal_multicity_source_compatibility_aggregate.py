"""Aggregate the complete V144 ordered source-to-target inventory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_multicity_source_compatibility_inventory import (
    RESULT_PROTOCOL as TARGET_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    TARGET_BUDGET,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


RESULT_PROTOCOL = "tsc-v144-multicity-source-compatibility-aggregate-v1"
MINIMUM_USEFUL_SOURCE_IMPROVEMENT = 0.0005
MINIMUM_TARGETS_WITH_HEADROOM = 5


def _read_result(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V144 result is not a JSON object: {path}")
    return value


def aggregate_source_compatibility_inventory(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    by_target = {str(row.get("target_city")): row for row in rows}
    if len(rows) != len(EXPECTED_CITY_GROUPS) or set(by_target) != set(
        EXPECTED_CITY_GROUPS
    ):
        raise ValueError("V144 requires exactly one result for every target city")
    shared_inputs = None
    shared_signatures = None
    target_rows = {}
    targets_with_headroom = 0
    for target in EXPECTED_CITY_GROUPS:
        row = by_target[target]
        expected_sources = set(EXPECTED_CITY_GROUPS) - {target}
        if (
            row.get("protocol") != TARGET_RESULT_PROTOCOL
            or int(row.get("target_budget", -1)) != TARGET_BUDGET
            or set(row.get("source_city_groups", ())) != expected_sources
            or set(row.get("source_arm_summaries", {})) != expected_sources
            or set(row.get("source_placebo_arm_summaries", {}))
            != expected_sources
            or row.get("information_budget", {}).get(
                "evaluation_groups_used_for_fit_or_selection"
            )
            is not False
        ):
            raise ValueError(f"{target}: V144 information boundary changed")
        inputs = dict(row.get("inputs", {}))
        signatures = dict(row.get("city_covariate_signatures", {}))
        if set(signatures) != set(EXPECTED_CITY_GROUPS) or any(
            signature.get("information_boundary")
            != "features_and_static_context_only_no_priors_or_targets"
            for signature in signatures.values()
        ):
            raise ValueError(f"{target}: V144 covariate signature boundary changed")
        if shared_inputs is None:
            shared_inputs = inputs
            shared_signatures = signatures
        elif inputs != shared_inputs or signatures != shared_signatures:
            raise ValueError("V144 target rows do not share inputs and signatures")
        target_only = float(row["arm_summaries"]["target_only_causal"]["mean"])
        source_rows = {}
        for source in sorted(expected_sources):
            source_mean = float(row["source_arm_summaries"][source]["mean"])
            placebo_mean = float(
                row["source_placebo_arm_summaries"][source]["mean"]
            )
            source_rows[source] = {
                "source_mean": source_mean,
                "target_only_mean": target_only,
                "source_placebo_mean": placebo_mean,
                "source_minus_target_only": source_mean - target_only,
                "source_minus_matched_placebo": source_mean - placebo_mean,
            }
        best_source = min(
            source_rows,
            key=lambda source: (
                source_rows[source]["source_minus_target_only"], source
            ),
        )
        best_effect = float(
            source_rows[best_source]["source_minus_target_only"]
        )
        if best_effect <= -MINIMUM_USEFUL_SOURCE_IMPROVEMENT:
            targets_with_headroom += 1
        target_rows[target] = {
            "sources": source_rows,
            "descriptive_oracle_best_source": best_source,
            "descriptive_oracle_best_effect": best_effect,
        }
    all_effects = np.asarray(
        [
            values["source_minus_target_only"]
            for target in target_rows.values()
            for values in target["sources"].values()
        ],
        dtype=float,
    )
    admitted = targets_with_headroom >= MINIMUM_TARGETS_WITH_HEADROOM
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "ordered-pair-source-compatibility-development-aggregate",
        "target_budget": TARGET_BUDGET,
        "target_city_groups": list(EXPECTED_CITY_GROUPS),
        "ordered_pair_count": int(all_effects.size),
        "ordered_pair_effects": target_rows,
        "city_covariate_signatures": shared_signatures,
        "headroom_admission": {
            "passed": admitted,
            "decision": (
                "authorize_source_only_leave_one_city_gate_development"
                if admitted
                else "close_source_compatibility_gate_development"
            ),
            "targets_with_useful_source_headroom": targets_with_headroom,
            "minimum_targets_with_headroom": MINIMUM_TARGETS_WITH_HEADROOM,
            "minimum_useful_source_improvement": MINIMUM_USEFUL_SOURCE_IMPROVEMENT,
            "ordered_pair_mean_effect": float(np.mean(all_effects)),
            "ordered_pair_improving_fraction": float(np.mean(all_effects < 0.0)),
        },
        "inputs": shared_inputs,
        "claim_boundary": (
            "V144 oracle rows only establish selection headroom. They are not a "
            "deployable policy and cannot be reported as gate performance."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite V144 aggregate: {args.out}")
    result = aggregate_source_compatibility_inventory(
        [_read_result(path) for path in args.results]
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": "PASS" if result["headroom_admission"]["passed"] else "REJECT",
                "protocol": result["protocol"],
                "headroom_admission": result["headroom_admission"],
                "result": str(args.out),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
