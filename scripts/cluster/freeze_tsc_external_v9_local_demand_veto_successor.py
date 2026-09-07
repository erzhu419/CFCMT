#!/usr/bin/env python3
"""Freeze the v74 local-demand successor before fitting artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_external_local_demand_veto_freeze import (  # noqa: E402
    FREEZE_PROTOCOL,
    WINNER_VARIANT,
)
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json  # noqa: E402
from scripts.cluster.launch_tsc_external_network_admission_v6 import (  # noqa: E402
    _read_json,
    _sha256,
)


V73_PROTOCOL_PATH = Path(
    "cf_h2o/config/traffic_signal_tsc_v56_external_v9_"
    "local_demand_model_diagnostic.json"
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
) -> dict[str, Any]:
    v73_protocol = _read_json(V73_PROTOCOL_PATH)
    diagnostic_result = _read_json(diagnostic_result_path)
    diagnostic_audit = _read_json(diagnostic_audit_path)
    cache_path = Path(v73_protocol["cache_protocol"]["path"])
    cache_audit_path = Path(v73_protocol["cache_audit"]["path"])
    proposal_path = Path(v73_protocol["proposal_parent_protocol"]["path"])
    freeze_audit_path = Path(v73_protocol["hierarchical_freeze_audit"]["path"])
    if (
        _sha256(diagnostic_result_path)
        != "5dcf83e9da51b835be974a4cdb654a44fcd5f7d0791033ce6c8ffaab323bc82f"
        or _sha256(diagnostic_audit_path)
        != "616884af2d7940013f55c5f9f335b019c5fe2281fee46fced1710a2e67662345"
        or diagnostic_result.get("status") != "PASS"
        or diagnostic_audit.get("status") != "PASS"
        or diagnostic_audit.get("decision")
        != "authorize_oof_qualified_city_successor_freeze_only"
        or diagnostic_audit.get("diagnostic_result", {}).get("sha256")
        != _sha256(diagnostic_result_path)
        or diagnostic_audit.get("city_decisions", {}).get("jinan", {}).get(
            "decision"
        )
        != "freeze_active_veto"
        or diagnostic_audit.get("city_decisions", {}).get(
            "los_angeles", {}
        ).get("decision")
        != "force_phase_pressure_fallback"
    ):
        raise ValueError("v74 successor parent evidence changed")
    future_counts = {
        "freeze_result": int(Path(freeze_result_path).is_file()),
        "artifact_root": _file_count(artifact_root),
        "future_closed_loop_root": _file_count(future_closed_loop_root),
        "future_validation_root": _file_count(future_validation_root),
    }
    if any(future_counts.values()):
        raise FileExistsError(f"v74 future evidence already exists: {future_counts}")
    sources = (
        "cf_h2o/eval/traffic_signal_external_local_demand_veto_freeze.py",
        "cf_h2o/eval/traffic_signal_external_local_demand_model_diagnostic.py",
        "cf_h2o/traffic_signal/local_demand_target_veto.py",
        "cf_h2o/traffic_signal/local_demand_timeline_context.py",
        "cf_h2o/traffic_signal/demand_timeline_context.py",
        "cf_h2o/eval/traffic_signal_resco_cfcmt_v3.py",
        "scripts/cluster/run_tsc_external_local_demand_veto_freeze.py",
    )
    return {
        "protocol": FREEZE_PROTOCOL,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "post_v73_oof_audit_pre_full_adaptation_model_fit",
        "v73_diagnostic_protocol": {
            "path": str(V73_PROTOCOL_PATH),
            "sha256": _sha256(V73_PROTOCOL_PATH),
        },
        "v73_diagnostic_result": {
            "path": str(diagnostic_result_path),
            "sha256": _sha256(diagnostic_result_path),
        },
        "v73_diagnostic_audit": {
            "path": str(diagnostic_audit_path),
            "sha256": _sha256(diagnostic_audit_path),
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
            "variant": WINNER_VARIANT,
            "variant_spec": v73_protocol["diagnostic"]["variants"][WINNER_VARIANT],
            "config": v73_protocol["diagnostic"]["model"],
            "city_decisions": diagnostic_audit["city_decisions"],
            "fit": "all_six_adaptation_seeds_after_seed_loo_selection",
            "scenario_id_is_not_a_feature": True,
            "local_demand_context_is_label_free": True,
        },
        "future_file_counts_at_freeze": future_counts,
        "validation": v73_protocol["validation"],
        "prospective_confirmation": v73_protocol["prospective_confirmation"],
        "claim_boundary": (
            "The selected Jinan veto is fit on all disclosed adaptation seeds. "
            "Los Angeles is frozen disabled. Closed-loop and sealed seeds remain untouched."
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
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite v74 protocol: {args.out}")
    payload = build_protocol(
        diagnostic_result_path=args.diagnostic_result,
        diagnostic_audit_path=args.diagnostic_audit,
        freeze_result_path=args.freeze_result,
        artifact_root=args.artifact_root,
        future_closed_loop_root=args.future_closed_loop_root,
        future_validation_root=args.future_validation_root,
    )
    atomic_write_json(args.out, payload)
    print(
        json.dumps(
            {
                "protocol": payload["protocol"],
                "variant": payload["model"]["variant"],
                "sha256": _sha256(args.out),
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
