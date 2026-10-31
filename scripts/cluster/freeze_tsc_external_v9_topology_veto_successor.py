#!/usr/bin/env python3
"""Freeze the post-v76 topology-only successor before fitting artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_historical_demand_model_diagnostic import (  # noqa: E402
    BASE_VARIANT,
    HISTORICAL_VARIANT,
    RESULT_PROTOCOL as V76_RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_external_topology_veto_freeze import (  # noqa: E402
    FREEZE_PROTOCOL,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


V59_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v59_external_v9_"
    "historical_prefix_model_diagnostic.json"
)
ABANDONED_V75_ROOT = Path(
    "cf_h2o/results/cluster/tsc_v75r71_external_v9_local_demand_"
    "closed_loop_development_20260810"
)


def _file_count(path: Path) -> int:
    return (
        sum(1 for item in Path(path).rglob("*") if item.is_file())
        if Path(path).exists()
        else 0
    )


def build_protocol(
    *,
    diagnostic_result_path: Path,
    diagnostic_audit_path: Path,
    freeze_result_path: Path,
    artifact_root: Path,
    future_closed_loop_root: Path,
    future_validation_root: Path,
    future_prospective_root: Path,
) -> dict[str, Any]:
    v59_protocol = _read_json(V59_PROTOCOL_PATH)
    diagnostic_result = _read_json(diagnostic_result_path)
    diagnostic_audit = _read_json(diagnostic_audit_path)
    cache_path = Path(v59_protocol["cache_protocol"]["path"])
    cache_audit_path = Path(v59_protocol["cache_audit"]["path"])
    proposal_path = Path(v59_protocol["proposal_parent_protocol"]["path"])
    freeze_audit_path = Path(v59_protocol["hierarchical_freeze_audit"]["path"])
    city_results = diagnostic_result.get("cities", {})
    jinan = city_results.get("jinan", {}).get("variants", {})
    los_angeles = city_results.get("los_angeles", {}).get("variants", {})
    jinan_base = jinan.get(BASE_VARIANT, {})
    jinan_history = jinan.get(HISTORICAL_VARIANT, {})
    la_base = los_angeles.get(BASE_VARIANT, {})
    audit_gate = diagnostic_audit.get("integrity_gate", {})
    if (
        diagnostic_result.get("protocol") != V76_RESULT_PROTOCOL
        or diagnostic_result.get("status") != "PASS"
        or diagnostic_audit.get("status") != "FAIL"
        or diagnostic_audit.get("decision") != "reject"
        or diagnostic_audit.get("diagnostic_result", {}).get("sha256")
        != _sha256(diagnostic_result_path)
        or audit_gate.get("integrity_passed") is not True
        or audit_gate.get("runtime_features_exclude_future_terms") is not True
        or audit_gate.get("sealed_outputs_absent_at_freeze") is not True
        or audit_gate.get("jinan_historical_module_advances") is not False
        or jinan_base.get("passed_selection_rule") is not True
        or float(
            jinan_base.get("selected_candidate", {}).get(
                "acceptance_quantile", -1.0
            )
        )
        != 0.075
        or jinan_history.get("passed_selection_rule") is not False
        or la_base.get("passed_selection_rule") is not False
        or int(v59_protocol["abandoned_v75"]["result_file_count_at_correction_freeze"])
        != 0
    ):
        raise ValueError("v77 successor parent evidence changed")

    future_counts = {
        "abandoned_v75_root": _file_count(ABANDONED_V75_ROOT),
        "freeze_result": int(Path(freeze_result_path).is_file()),
        "artifact_root": _file_count(artifact_root),
        "future_closed_loop_root": _file_count(future_closed_loop_root),
        "future_validation_root": _file_count(future_validation_root),
        "future_prospective_root": _file_count(future_prospective_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"v77 future evidence already exists: {future_counts}")
    sources = (
        "cf_h2o/eval/traffic_signal_external_topology_veto_freeze.py",
        "cf_h2o/eval/traffic_signal_external_historical_demand_model_diagnostic.py",
        "cf_h2o/traffic_signal/topology_target_veto.py",
        "cf_h2o/traffic_signal/demand_timeline_context.py",
        "cf_h2o/traffic_signal/proposal_conditional_target_veto.py",
        "cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py",
        "scripts/cluster/run_tsc_external_topology_veto_freeze.py",
    )
    model_spec = v59_protocol["diagnostic"]
    city_decisions = {
        "jinan": {
            "decision": "freeze_active_veto",
            "variant": BASE_VARIANT,
            "proposal_count": int(city_results["jinan"]["proposal_count"]),
            "selected_candidate": jinan_base["selected_candidate"],
            "selection_source": "v76_prespecified_no_local_baseline",
        },
        "los_angeles": {
            "decision": "force_phase_pressure_fallback",
            "variant": BASE_VARIANT,
            "proposal_count": int(city_results["los_angeles"]["proposal_count"]),
            "selected_candidate": None,
            "selection_source": "v76_prespecified_no_local_baseline_failed",
        },
    }
    return {
        "protocol": FREEZE_PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v76_failed_extension_audit_pre_topology_full_fit",
        "development_disposition": (
            "The historical-prefix extension is rejected. The simpler prespecified "
            "topology-only baseline is retained as a newly disclosed development "
            "successor because it independently passed the original OOF rule."
        ),
        "v59_diagnostic_protocol": {
            "path": str(V59_PROTOCOL_PATH),
            "sha256": _sha256(V59_PROTOCOL_PATH),
        },
        "v76_diagnostic_result": {
            "path": str(diagnostic_result_path),
            "sha256": _sha256(diagnostic_result_path),
        },
        "v76_diagnostic_audit": {
            "path": str(diagnostic_audit_path),
            "sha256": _sha256(diagnostic_audit_path),
            "status": diagnostic_audit["status"],
            "decision": diagnostic_audit["decision"],
        },
        "cache_protocol": {"path": str(cache_path), "sha256": _sha256(cache_path)},
        "cache_audit": {
            "path": str(cache_audit_path),
            "sha256": _sha256(cache_audit_path),
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
        "model": {
            "variant": BASE_VARIANT,
            "variant_spec": model_spec["variants"][BASE_VARIANT],
            "config": model_spec["model"],
            "city_decisions": city_decisions,
            "fit": "all_six_adaptation_seeds_after_seed_loo_selection",
            "target_transform": model_spec["target_transform"],
            "scenario_id_is_not_a_feature": True,
            "runtime_information_budget": (
                "current_candidate_state_and_static_tls_topology"
            ),
        },
        "future_file_counts_at_freeze": future_counts,
        "validation": v59_protocol["validation"],
        "prospective_confirmation": v59_protocol["prospective_confirmation"],
        "claim_boundary": (
            "This is a disclosed post-diagnostic development choice based only on "
            "six adaptation seeds. Jinan is active and Los Angeles is disabled. "
            "Closed-loop and sealed-seed outcomes remain untouched."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--diagnostic-audit", type=Path, required=True)
    parser.add_argument("--freeze-result", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--future-closed-loop-root", type=Path, required=True)
    parser.add_argument("--future-validation-root", type=Path, required=True)
    parser.add_argument("--future-prospective-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v77 protocol: {args.out}")
    payload = build_protocol(
        diagnostic_result_path=args.diagnostic_result,
        diagnostic_audit_path=args.diagnostic_audit,
        freeze_result_path=args.freeze_result,
        artifact_root=args.artifact_root,
        future_closed_loop_root=args.future_closed_loop_root,
        future_validation_root=args.future_validation_root,
        future_prospective_root=args.future_prospective_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "variant": payload["model"]["variant"],
                "active_cities": ["jinan"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
