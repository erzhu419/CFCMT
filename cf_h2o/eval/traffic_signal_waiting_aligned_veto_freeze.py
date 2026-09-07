"""Freeze the full-fit waiting-aligned topology veto after OOF selection."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
from typing import Any, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_external_historical_demand_model_diagnostic import (
    BASE_VARIANT,
)
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    frozen_proposal_records,
)
from cf_h2o.eval.traffic_signal_external_topology_veto_freeze import (
    fit_city_topology_veto,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_veto_diagnostic import (
    RESULT_PROTOCOL as DIAGNOSTIC_RESULT_PROTOCOL,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.dataset_cache import atomic_write_json
from cf_h2o.traffic_signal.demand_timeline_context import (
    build_tls_local_topology_context,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_redevelopment_cache import (
    PROTOCOL as CACHE_PROTOCOL,
)
from scripts.cluster.freeze_tsc_external_v9_waiting_aligned_veto_diagnostic import (
    PROTOCOL as DIAGNOSTIC_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-waiting-aligned-veto-freeze-result-v1"
ARTIFACT_ROLE = "waiting_aligned_static_topology_veto_only"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def run_freeze(
    *,
    diagnostic_protocol_path: Path,
    diagnostic_result_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    cache_root: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    manifest_path: Path,
    conversion_root: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(diagnostic_protocol_path)
    diagnostic = _read_json(diagnostic_result_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    parent = _read_json(proposal_parent_protocol_path)
    parent_audit = _read_json(hierarchical_freeze_audit_path)
    if (
        protocol.get("protocol") != DIAGNOSTIC_PROTOCOL
        or diagnostic.get("protocol") != DIAGNOSTIC_RESULT_PROTOCOL
        or diagnostic.get("status") != "PASS"
        or diagnostic.get("decision")
        != "authorize_waiting_aligned_veto_artifact_freeze"
        or diagnostic.get("integrity_gate", {}).get("passed") is not True
        or diagnostic.get("advance_gate", {}).get("passed") is not True
        or diagnostic.get("diagnostic_protocol_sha256")
        != _sha256(diagnostic_protocol_path)
        or cache_protocol.get("protocol") != CACHE_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or diagnostic.get("cache_audit_sha256") != _sha256(cache_audit_path)
        or parent_audit.get("status") != "PASS"
    ):
        raise ValueError("waiting-aligned veto freeze evidence changed")
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite waiting veto artifact: {output_root}")
    training = dict(protocol["training"])
    city = str(training["city"])
    scenario = str(training["scenarios"][0])
    seeds = tuple(int(value) for value in training["seeds"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("waiting-aligned freeze scenario changed")
    manifest = replace(full_manifest, scenarios=selected_specs)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(training["collection_shards"]),
        workers=max(1, int(workers)),
    )
    _, frozen_model = _load_frozen_city_model(
        city=city,
        freeze_root=hierarchical_freeze_root,
        protocol=parent,
        freeze_audit=parent_audit,
    )
    dataset = _merge_city_datasets(bank, (scenario,), city=city)
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    records = frozen_proposal_records(
        contrast, frozen_model=frozen_model, scenarios=(scenario,)
    )
    topology = {
        scenario: build_tls_local_topology_context(
            Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        )
    }
    selected = dict(diagnostic["advance_gate"]["selected_candidate"])
    city_decision = {
        "variant": BASE_VARIANT,
        "proposal_count": int(diagnostic["proposal_count"]),
        "decision": "freeze_active_veto",
        "selected_candidate": selected,
    }
    summary, artifact = fit_city_topology_veto(
        city=city,
        scenarios=(scenario,),
        records=records,
        topology=topology,
        model_spec=protocol["diagnostic"]["model"],
        city_decision=city_decision,
    )
    artifact.update(
        {
            "artifact_role": ARTIFACT_ROLE,
            "counterfactual_cost_mode": "halted_queue",
            "counterfactual_estimand": training["target"],
            "rollout_value_horizon_sec": 450,
            "training_seed_count": len(seeds),
            "training_seeds": seeds,
            "training_scenarios": (scenario,),
            "proposal_origin_changed": False,
        }
    )
    summary.update(
        {
            "artifact_role": ARTIFACT_ROLE,
            "counterfactual_cost_mode": "halted_queue",
            "counterfactual_estimand": training["target"],
            "rollout_value_horizon_sec": 450,
            "training_seed_count": len(seeds),
        }
    )
    city_root = output_root / city
    city_root.mkdir(parents=True, exist_ok=False)
    model_path = city_root / "model.pkl"
    model_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
    certificate = {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    certificate_path = city_root / "freeze.json"
    atomic_write_json(certificate_path, certificate)
    integrity = {
        "diagnostic_gate_passed": True,
        "scenario_exact": summary["scenarios"] == [scenario],
        "proposal_count_exact": int(summary["proposal_training_row_count"])
        == int(diagnostic["proposal_count"]),
        "selected_quantile_exact": float(
            summary["final_config"]["acceptance_quantile"]
        )
        == float(selected["acceptance_quantile"]),
        "veto_active": bool(summary["final_config"]["enabled"]),
        "cost_mode_explicit": artifact["counterfactual_cost_mode"] == "halted_queue",
        "proposal_origin_unchanged": artifact["proposal_origin_changed"] is False,
    }
    integrity["passed"] = all(integrity.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": (
            "authorize_waiting_aligned_veto_closed_loop_redevelopment"
            if integrity["passed"]
            else "reject_waiting_aligned_veto_artifact"
        ),
        "diagnostic_result_sha256": _sha256(diagnostic_result_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "bank_audit": bank_audit,
        "city": city,
        "scenarios": [scenario],
        "artifact_role": ARTIFACT_ROLE,
        "artifact": {
            "root": str(output_root.resolve()),
            "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "certificate": {
                "path": str(certificate_path.resolve()),
                "sha256": _sha256(certificate_path),
            },
        },
        "summary": summary,
        "integrity_gate": integrity,
        "claim_boundary": (
            "Full-fit artifact on disclosed redevelopment seeds only. Closed-loop "
            "efficacy and untouched-seed generalization remain untested."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--diagnostic-result", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite waiting veto freeze result: {args.out}")
    result = run_freeze(
        diagnostic_protocol_path=args.diagnostic_protocol,
        diagnostic_result_path=args.diagnostic_result,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        cache_root=args.cache_root,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        output_root=args.output_root,
        workers=args.workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "city": result["city"],
                "quantile": result["summary"]["final_config"][
                    "acceptance_quantile"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
