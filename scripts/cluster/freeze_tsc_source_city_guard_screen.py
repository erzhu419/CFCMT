#!/usr/bin/env python3
"""Freeze selected source weights before the V93 guard screen."""

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


PROTOCOL = "tsc-v93-source-city-selected-weight-guard-screen-freeze-v1"


def build_guard_screen(weight_audit: dict[str, Any]) -> dict[str, Any]:
    if (
        weight_audit.get("protocol") != AUDIT_PROTOCOL
        or weight_audit.get("scientific_status")
        != "development-only-not-confirmation"
    ):
        raise ValueError("source-city weight audit contract changed")
    selected = {
        str(city): {
            str(group): float(weight) for group, weight in dict(weights).items()
        }
        for city, weights in weight_audit["selected_weights_by_city"].items()
    }
    if set(selected) != {"los_angeles", "jinan"}:
        raise ValueError("source-city guard target cities changed")
    for weights in selected.values():
        if (
            len(weights) > 1
            or any(value < 0.0 for value in weights.values())
            or sum(weights.values()) > 1.0 + 1e-12
        ):
            raise ValueError("selected source-city weights are not convex")
    empty = {city: {} for city in selected}
    return {
        "protocol": CANDIDATE_PROTOCOL,
        "freeze_protocol": PROTOCOL,
        "scientific_status": "development-only-selected-weight-guard-screen",
        "parent_weight_audit_sha256": weight_audit.get("_sha256"),
        "selected_weights_by_city": selected,
        "candidates": [
            {"key": "target_only", "weights_by_city": empty},
            {"key": "selected_source", "weights_by_city": selected},
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite guard screen: {args.out}")
    audit = json.loads(args.weight_audit.read_text(encoding="utf-8"))
    audit["_sha256"] = _sha256(args.weight_audit)
    payload = build_guard_screen(audit)
    _atomic_json(args.out, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_weights_by_city": payload["selected_weights_by_city"],
                "candidate_count": len(payload["candidates"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
