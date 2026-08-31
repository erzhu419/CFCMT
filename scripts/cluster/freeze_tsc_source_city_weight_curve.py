#!/usr/bin/env python3
"""Freeze a continuous source-mass curve from the V93 identity screen."""

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
from scripts.cluster.run_tsc_source_city_component_shard import CANDIDATE_PROTOCOL


PROTOCOL = "tsc-v93-source-city-selected-weight-curve-freeze-v1"
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def build_weight_curve(screen_audit: dict[str, Any]) -> dict[str, Any]:
    if (
        screen_audit.get("protocol") != AUDIT_PROTOCOL
        or screen_audit.get("scientific_status") != "development-only-not-confirmation"
    ):
        raise ValueError("source-city screen audit contract changed")
    selected = {
        str(city): {
            str(group): float(weight) for group, weight in dict(weights).items()
        }
        for city, weights in screen_audit["selected_weights_by_city"].items()
    }
    if set(selected) != {"los_angeles", "jinan"}:
        raise ValueError("source-city screen target cities changed")
    source_by_city = {}
    for city, weights in selected.items():
        if len(weights) > 1:
            raise ValueError("identity screen must select at most one source per city")
        source_by_city[city] = next(iter(weights), None)
    candidates = []
    for weight in WEIGHTS:
        key = "target_only" if weight == 0.0 else f"selected_w{str(weight).replace('.', 'p')}"
        candidates.append(
            {
                "key": key,
                "weights_by_city": {
                    city: (
                        {source: float(weight)}
                        if source is not None and weight > 0.0
                        else {}
                    )
                    for city, source in source_by_city.items()
                },
            }
        )
    return {
        "protocol": CANDIDATE_PROTOCOL,
        "freeze_protocol": PROTOCOL,
        "scientific_status": "development-only-selected-source-weight-curve",
        "parent_screen_audit_sha256": screen_audit.get("_sha256"),
        "selected_source_by_city": source_by_city,
        "weights": list(WEIGHTS),
        "selection_rule": {
            "baseline": "target_only",
            "minimum_mean_improvement": 0.005,
            "maximum_seed_regression": 0.01,
            "added_collision_or_teleport_incidents": 0,
        },
        "candidates": candidates,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite weight curve: {args.out}")
    audit = json.loads(args.screen_audit.read_text(encoding="utf-8"))
    audit["_sha256"] = _sha256(args.screen_audit)
    payload = build_weight_curve(audit)
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_source_by_city": payload["selected_source_by_city"],
                "candidate_count": len(payload["candidates"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
