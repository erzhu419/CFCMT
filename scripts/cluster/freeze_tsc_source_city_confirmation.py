#!/usr/bin/env python3
"""Freeze the V93 source selector before disjoint-seed confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from scripts.cluster.audit_tsc_source_city_component_screen import AUDIT_PROTOCOL
from scripts.cluster.audit_tsc_source_city_guard_screen import (
    AUDIT_PROTOCOL as GUARD_AUDIT_PROTOCOL,
)
from scripts.cluster.run_tsc_source_city_component_shard import (
    CANDIDATE_PROTOCOL,
    CITIES,
    load_candidate_specs,
)


PROTOCOL = "tsc-v93-causal-source-selector-disjoint-confirmation-freeze-v1"
CONFIRMATION_SEED_COUNT = 56


def build_confirmation_freeze(
    *,
    identity_audit: dict[str, Any],
    weight_audit: dict[str, Any],
    guard_audit: dict[str, Any],
    identity_candidates: Sequence[dict[str, Any]],
    confirmation_seeds: Sequence[int],
) -> dict[str, Any]:
    for label, audit in (
        ("identity", identity_audit),
        ("weight", weight_audit),
    ):
        if (
            audit.get("protocol") != AUDIT_PROTOCOL
            or audit.get("scientific_status") != "development-only-not-confirmation"
        ):
            raise ValueError(f"{label} development audit contract changed")
    if (
        guard_audit.get("protocol") != GUARD_AUDIT_PROTOCOL
        or guard_audit.get("scientific_status")
        != "development-only-not-confirmation"
    ):
        raise ValueError("guard development audit contract changed")

    selected_weights = {
        str(city): {
            str(group): float(value) for group, value in dict(weights).items()
        }
        for city, weights in weight_audit["selected_weights_by_city"].items()
    }
    if set(selected_weights) != set(CITIES):
        raise ValueError("confirmation target cities changed")
    selected_guards = {
        str(city): row.get("guard_config")
        for city, row in guard_audit["selected_guard_by_city"].items()
    }
    if set(selected_guards) != set(CITIES):
        raise ValueError("confirmation city guard set changed")

    source_groups = sorted(
        {
            str(group)
            for candidate in identity_candidates
            for weights in candidate["weights_by_city"].values()
            for group in weights
        }
    )
    if len(source_groups) < 2:
        raise ValueError("confirmation equal-source baseline is not identifiable")
    selected_group_set = {
        group for weights in selected_weights.values() for group in weights
    }
    if not selected_group_set.issubset(set(source_groups)):
        raise ValueError("selected source is absent from the identity screen")

    development_seeds = tuple(
        int(value)
        for value in guard_audit["guard_selection_by_city"]["jinan"]["seeds"]
    )
    for city in CITIES:
        if tuple(
            int(value)
            for value in guard_audit["guard_selection_by_city"][city]["seeds"]
        ) != development_seeds:
            raise ValueError("guard development seed sets differ by city")
    confirmation = tuple(int(value) for value in confirmation_seeds)
    if (
        len(confirmation) != CONFIRMATION_SEED_COUNT
        or len(set(confirmation)) != len(confirmation)
        or set(confirmation) & set(development_seeds)
    ):
        raise ValueError("confirmation seeds are not a disjoint 56-seed set")

    empty = {city: {} for city in CITIES}
    equal_weights = {
        city: {group: 1.0 / len(source_groups) for group in source_groups}
        for city in CITIES
    }
    candidates = [
        {
            "key": "target_only",
            "weights_by_city": empty,
            "guard_by_city": selected_guards,
        },
        {
            "key": "equal_all_sources",
            "weights_by_city": equal_weights,
            "guard_by_city": selected_guards,
        },
        {
            "key": "frozen_selector",
            "weights_by_city": selected_weights,
            "guard_by_city": selected_guards,
        },
    ]
    return {
        "protocol": CANDIDATE_PROTOCOL,
        "freeze_protocol": PROTOCOL,
        "scientific_status": "frozen-before-disjoint-seed-confirmation",
        "parent_identity_audit_sha256": identity_audit.get("_sha256"),
        "parent_weight_audit_sha256": weight_audit.get("_sha256"),
        "parent_guard_audit_sha256": guard_audit.get("_sha256"),
        "development_seeds": list(development_seeds),
        "confirmation_seeds": list(confirmation),
        "source_groups": source_groups,
        "selected_weights_by_city": selected_weights,
        "selected_guard_by_city": selected_guards,
        "candidate_semantics": {
            "target_only": "fixed target anchor with the same city guard",
            "equal_all_sources": "untuned equal mixture over all screened source cities",
            "frozen_selector": "development-frozen source identity, mass, and city guard",
        },
        "analysis_contract": {
            "primary_metric": "mean_tripinfo_waiting_time",
            "pairing_unit": "scenario_seed",
            "city_aggregation": "equal_scenario_within_city_then_equal_city_macro",
            "uncertainty": "paired_seed_bootstrap_and_wilcoxon_holm",
            "safety_metrics": [
                "collision_incidents",
                "starting_teleports",
                "ending_teleports",
            ],
            "no_post_confirmation_retuning": True,
        },
        "candidates": candidates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-audit", type=Path, required=True)
    parser.add_argument("--weight-audit", type=Path, required=True)
    parser.add_argument("--guard-audit", type=Path, required=True)
    parser.add_argument("--identity-candidate-spec", type=Path, required=True)
    parser.add_argument("--confirmation-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite confirmation freeze: {args.out}")
    identity = json.loads(args.identity_audit.read_text(encoding="utf-8"))
    weight = json.loads(args.weight_audit.read_text(encoding="utf-8"))
    guard = json.loads(args.guard_audit.read_text(encoding="utf-8"))
    identity["_sha256"] = _sha256(args.identity_audit)
    weight["_sha256"] = _sha256(args.weight_audit)
    guard["_sha256"] = _sha256(args.guard_audit)
    payload = build_confirmation_freeze(
        identity_audit=identity,
        weight_audit=weight,
        guard_audit=guard,
        identity_candidates=load_candidate_specs(args.identity_candidate_spec),
        confirmation_seeds=args.confirmation_seeds,
    )
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "confirmation_seed_count": len(payload["confirmation_seeds"]),
                "selected_weights_by_city": payload["selected_weights_by_city"],
                "selected_guard_by_city": payload["selected_guard_by_city"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
