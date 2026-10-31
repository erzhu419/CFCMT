#!/usr/bin/env python3
"""Audit and select the V93 fixed-mass source-city component screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


from cf_h2o.eval.traffic_signal_causal_source_weighting import (
    select_source_city_candidate_from_closed_loop,
)
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from scripts.cluster.run_tsc_source_city_component_shard import (
    CANDIDATE_PROTOCOL,
    SCENARIOS,
    SHARD_PROTOCOL,
    all_identities,
    load_candidate_specs,
)


AUDIT_PROTOCOL = "tsc-v93-source-city-component-screen-audit-v1"
SCENARIOS_BY_CITY = {
    "los_angeles": ("la_1x4",),
    "jinan": (
        "jinan_3x4_real",
        "jinan_3x4_real_2000",
        "jinan_3x4_real_2500",
    ),
}


def _load_rows(
    *,
    shards_root: Path,
    expected_shard_count: int,
    candidate_spec_sha256: str,
) -> list[dict[str, Any]]:
    paths = sorted(Path(shards_root).glob("shard_*.json"))
    if len(paths) != int(expected_shard_count):
        raise ValueError("source-city screen shard count changed")
    rows = []
    indices = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        indices.add(int(payload.get("shard_index", -1)))
        if not (
            payload.get("protocol") == SHARD_PROTOCOL
            and payload.get("candidate_protocol") == CANDIDATE_PROTOCOL
            and payload.get("candidate_spec_sha256")
            == str(candidate_spec_sha256)
            and payload.get("passed") is True
            and int(payload.get("shard_count", -1)) == int(expected_shard_count)
            and int(payload.get("task_count", -1)) == len(payload.get("rows", ()))
        ):
            raise ValueError(f"invalid source-city screen shard: {path}")
        rows.extend(dict(row) for row in payload["rows"])
    if indices != set(range(int(expected_shard_count))):
        raise ValueError("source-city screen shard indices are incomplete")
    return rows


def build_audit(
    *,
    rows: Sequence[dict[str, Any]],
    seeds: Sequence[int],
    candidates: Sequence[dict[str, Any]],
    candidate_spec_sha256: str,
) -> dict[str, Any]:
    candidate_keys = tuple(str(row["key"]) for row in candidates)
    expected = set(all_identities(seeds, candidate_keys))
    observed = {
        (str(row["scenario"]), int(row["seed"]), str(row["candidate_key"]))
        for row in rows
    }
    if len(rows) != len(expected) or observed != expected:
        raise ValueError("source-city screen rollout identity coverage failed")
    if {str(row["scenario"]) for row in rows} != set(SCENARIOS):
        raise ValueError("source-city screen scenario coverage changed")
    selections = {
        city: select_source_city_candidate_from_closed_loop(
            rows,
            city=city,
            scenarios=scenarios,
            candidate_keys=candidate_keys,
        )
        for city, scenarios in SCENARIOS_BY_CITY.items()
    }
    candidate_by_key = {str(row["key"]): row for row in candidates}
    selected_weights_by_city = {
        city: dict(
            candidate_by_key[selection["selected_candidate_key"]][
                "weights_by_city"
            ][city]
        )
        for city, selection in selections.items()
    }
    return {
        "protocol": AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "development-only-not-confirmation",
        "candidate_spec_sha256": str(candidate_spec_sha256),
        "seed_count": len(tuple(seeds)),
        "scenario_count": len(SCENARIOS),
        "candidate_count": len(candidate_keys),
        "matrix_size": len(expected),
        "selections": selections,
        "selected_weights_by_city": selected_weights_by_city,
        "claim_boundary": (
            "The selected source identities and weights are development choices. "
            "They require a disjoint-seed or new-city confirmation before inferential use."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--expected-shard-count", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    candidate_sha = _sha256(args.candidate_spec)
    candidates = load_candidate_specs(args.candidate_spec)
    rows = _load_rows(
        shards_root=args.shards_root,
        expected_shard_count=args.expected_shard_count,
        candidate_spec_sha256=candidate_sha,
    )
    audit = build_audit(
        rows=rows,
        seeds=tuple(args.seeds),
        candidates=candidates,
        candidate_spec_sha256=candidate_sha,
    )
    _atomic_json(args.out, audit)
    print(
        json.dumps(
            {
                "status": "PASS",
                "matrix_size": audit["matrix_size"],
                "selected_weights_by_city": audit["selected_weights_by_city"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
