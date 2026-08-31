#!/usr/bin/env python3
"""Re-audit frozen v98 offline source selection with family-wise control."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from cf_h2o.eval.traffic_signal_causal_source_offline_selection import PROTOCOL
from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    select_source_city_weight_with_familywise_control,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


AUDIT_PROTOCOL = "tsc-v108-v98-familywise-source-selection-robustness-audit-v1"


def audit_frozen_selections(inputs: Sequence[Path]) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    input_paths: dict[str, str] = {}
    for path in inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise ValueError(f"unexpected offline selector protocol: {path}")
        city = str(payload["city"])
        if city in selections:
            raise ValueError(f"duplicate frozen city: {city}")
        selections[city] = select_source_city_weight_with_familywise_control(
            payload["selection_matrix"],
            minimum_mean_improvement=0.005,
            maximum_fold_regression=0.01,
            familywise_alpha=0.05,
        )
        input_paths[city] = str(path.resolve())
    if not selections:
        raise ValueError("no frozen offline selections were supplied")
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "post-hoc robustness audit of frozen target-offline folds; "
            "no closed-loop outcomes read"
        ),
        "input_paths": input_paths,
        "selection_by_city": selections,
        "closed_loop_outcomes_consumed": 0,
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    payload = audit_frozen_selections(args.inputs)
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                city: {
                    "source": selection["selected_source_city_group"],
                    "weight": selection["selected_source_weight"],
                    "critical": selection["simultaneous_confidence_multiplier"],
                }
                for city, selection in payload["selection_by_city"].items()
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
