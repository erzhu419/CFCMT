#!/usr/bin/env python3
"""Freeze the post-v72 local-demand and robust-target diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_local_demand_model_diagnostic import (  # noqa: E402
    DIAGNOSTIC_PROTOCOL,
    VARIANT_SPECS,
    variant_feature_names,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


V72_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v55_external_v9_"
    "spatiotemporal_veto_diagnostic_v2.json"
)


def _file_count(path: Path) -> int:
    return (
        sum(1 for item in Path(path).rglob("*") if item.is_file())
        if Path(path).exists()
        else 0
    )


def build_protocol(
    *,
    v72_result_path: Path,
    diagnostic_output: Path,
    successor_output_root: Path,
    future_closed_loop_root: Path,
    future_validation_root: Path,
) -> dict[str, Any]:
    v72_protocol = _read_json(V72_PROTOCOL_PATH)
    v72_result = _read_json(v72_result_path)
    cache_path = Path(v72_protocol["cache_protocol"]["path"])
    cache_audit_path = Path(v72_protocol["cache_audit"]["path"])
    proposal_path = Path(v72_protocol["proposal_parent_protocol"]["path"])
    freeze_audit_path = Path(v72_protocol["hierarchical_freeze_audit"]["path"])
    if (
        _sha256(v72_result_path)
        != "7903481e31e20f6779475a2345a2a3243ed05e99485f67808eba4302f44fb8c0"
        or v72_result.get("status") != "PASS"
        or v72_result.get("decision")
        != "diagnostic_only_no_closed_loop_or_validation_authorization"
        or v72_result.get("integrity_gate", {}).get("passed") is not True
        or v72_result.get("integrity_gate", {}).get(
            "static_global_reproduces_v70"
        )
        is not True
        or any(
            bool(variant["passed_original_v70_rule"])
            for city in v72_result["cities"].values()
            for name, variant in city["variants"].items()
            if name != "counterfactual_label_oracle"
        )
        or _sha256(cache_path) != v72_protocol["cache_protocol"]["sha256"]
        or _sha256(cache_audit_path) != v72_protocol["cache_audit"]["sha256"]
        or _sha256(proposal_path)
        != v72_protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != v72_protocol["hierarchical_freeze_audit"]["sha256"]
    ):
        raise ValueError("v73 local-demand diagnostic parent evidence changed")
    future_counts = {
        "diagnostic_output": int(Path(diagnostic_output).is_file()),
        "successor_output_root": _file_count(successor_output_root),
        "future_closed_loop_root": _file_count(future_closed_loop_root),
        "future_validation_root": _file_count(future_validation_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"v73 future evidence already exists: {future_counts}")
    sources = (
        "cf_h2o/eval/traffic_signal_external_local_demand_model_diagnostic.py",
        "cf_h2o/traffic_signal/local_demand_timeline_context.py",
        "scripts/cluster/run_tsc_external_local_demand_model_diagnostic.py",
    )
    diagnostic = v72_protocol["diagnostic"]
    return {
        "protocol": DIAGNOSTIC_PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v72_negative_pre_local_demand_model_diagnostic_outcome_freeze",
        "v72_diagnostic_protocol": {
            "path": str(V72_PROTOCOL_PATH),
            "sha256": _sha256(V72_PROTOCOL_PATH),
        },
        "v72_diagnostic_result": {
            "path": str(v72_result_path),
            "sha256": _sha256(v72_result_path),
            "decision": v72_result["decision"],
        },
        "cache_protocol": {
            "path": str(cache_path),
            "sha256": _sha256(cache_path),
        },
        "cache_audit": {
            "path": str(cache_audit_path),
            "sha256": _sha256(cache_audit_path),
            "cache_sha256": v72_protocol["cache_audit"]["cache_sha256"],
        },
        "proposal_parent_protocol": {
            "path": str(proposal_path),
            "sha256": _sha256(proposal_path),
        },
        "hierarchical_freeze_audit": {
            "path": str(freeze_audit_path),
            "sha256": _sha256(freeze_audit_path),
        },
        "runtime_executable_sources": {
            path: _sha256(Path(path)) for path in sources
        },
        "diagnostic": {
            "classification": (
                "post_v72_disclosed_adaptation_seed_local_demand_model_diagnostic"
            ),
            "seeds": diagnostic["seeds"],
            "city_scenarios": diagnostic["city_scenarios"],
            "cross_fitting": "leave_one_simulator_seed_out",
            "target": diagnostic["target"],
            "variants": {
                name: {
                    **spec,
                    "feature_names": list(variant_feature_names(name)),
                    "feature_count": len(variant_feature_names(name)),
                }
                for name, spec in VARIANT_SPECS.items()
            },
            "acceptance_quantiles": diagnostic["acceptance_quantiles"],
            "model": diagnostic["model"],
            "bootstrap": diagnostic["bootstrap"],
            "selection_rule": diagnostic["selection_rule"],
            "topology_baseline_must_exactly_reproduce_v72_grid": True,
            "module_advancement_rule": (
                "a_successor_can_be_frozen_only_if_at_least_one_nonbaseline_"
                "variant_passes_the_original_v70_selection_rule_and_the_"
                "topology_baseline_exactly_reproduces_v72"
            ),
        },
        "future_file_counts_at_freeze": future_counts,
        "validation": v72_protocol["validation"],
        "prospective_confirmation": v72_protocol["prospective_confirmation"],
        "claim_boundary": (
            "Adaptation-seed diagnostic after disclosed v70 and v72 negative "
            "results. No model is frozen for closed-loop use, and sealed seeds "
            "remain untouched."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v72-result", type=Path, required=True)
    parser.add_argument("--diagnostic-output", type=Path, required=True)
    parser.add_argument("--successor-output-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--future-validation-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v73 protocol: {args.out}")
    payload = build_protocol(
        v72_result_path=args.v72_result,
        diagnostic_output=args.diagnostic_output,
        successor_output_root=args.successor_output_root,
        future_closed_loop_root=args.future_closed_loop_root,
        future_validation_root=args.future_validation_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "variants": list(payload["diagnostic"]["variants"]),
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
