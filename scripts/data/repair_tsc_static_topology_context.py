#!/usr/bin/env python3
"""Repair static topology context in an existing counterfactual cache."""

from __future__ import annotations

import argparse
from pathlib import Path

from cf_h2o.eval.traffic_signal_topology_context_repair import (
    corrected_scenario_contexts,
    repair_cache_tree,
)
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--control-interval-sec", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    if args.audit_out.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.audit_out}")
    manifest = load_traffic_signal_manifest(args.manifest)
    contexts = corrected_scenario_contexts(
        manifest,
        control_interval_sec=args.control_interval_sec,
    )
    audit = repair_cache_tree(
        source_root=args.source_root,
        destination_root=args.destination_root,
        contexts=contexts,
        workers=args.workers,
    )
    audit["manifest"] = str(args.manifest.resolve())
    audit["control_interval_sec"] = int(args.control_interval_sec)
    atomic_write_json(args.audit_out, audit)
    print(args.audit_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
