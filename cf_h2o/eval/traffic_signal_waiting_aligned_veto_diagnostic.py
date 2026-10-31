"""Seed-blocked OOF diagnostic for the waiting-aligned proposal veto."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
    _fit_variant,
)
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    frozen_proposal_records,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_counterfactual_cache_audit import (
    RESULT_PROTOCOL as CACHE_AUDIT_PROTOCOL,
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
    PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v83r79-waiting-aligned-veto-diagnostic-result-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def run_diagnostic(
    *,
    diagnostic_protocol_path: Path,
    cache_protocol_path: Path,
    cache_audit_path: Path,
    cache_root: Path,
    proposal_parent_protocol_path: Path,
    hierarchical_freeze_audit_path: Path,
    hierarchical_freeze_root: Path,
    manifest_path: Path,
    conversion_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = _read_json(diagnostic_protocol_path)
    cache_protocol = _read_json(cache_protocol_path)
    cache_audit = _read_json(cache_audit_path)
    parent = _read_json(proposal_parent_protocol_path)
    parent_audit = _read_json(hierarchical_freeze_audit_path)
    frozen = dict(protocol.get("frozen_inputs", {}))
    if (
        protocol.get("protocol") != PROTOCOL
        or cache_protocol.get("protocol") != CACHE_PROTOCOL
        or cache_audit.get("protocol") != CACHE_AUDIT_PROTOCOL
        or cache_audit.get("status") != "PASS"
        or cache_audit.get("gate", {}).get("passed") is not True
        or _sha256(cache_protocol_path) != frozen.get("cache_protocol_sha256")
        or _sha256(cache_audit_path) != frozen.get("cache_audit_sha256")
        or _sha256(proposal_parent_protocol_path)
        != frozen.get("proposal_parent_protocol_sha256")
        or _sha256(hierarchical_freeze_audit_path)
        != frozen.get("hierarchical_freeze_audit_sha256")
        or parent_audit.get("status") != "PASS"
    ):
        raise ValueError("waiting-aligned veto diagnostic evidence changed")
    training = dict(protocol["training"])
    scenario = str(training["scenarios"][0])
    seeds = tuple(int(value) for value in training["seeds"])
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root))
    full_manifest = load_traffic_signal_manifest(manifest_path)
    selected_specs = tuple(
        item for item in full_manifest.scenarios if item.scenario == scenario
    )
    if len(selected_specs) != 1:
        raise ValueError("waiting-aligned scenario is not unique in the manifest")
    manifest = replace(full_manifest, scenarios=selected_specs)
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=seeds,
        collection_shards=int(training["collection_shards"]),
        workers=max(int(workers), 1),
    )
    _, frozen_model = _load_frozen_city_model(
        city=str(training["city"]),
        freeze_root=hierarchical_freeze_root,
        protocol=parent,
        freeze_audit=parent_audit,
    )
    dataset = _merge_city_datasets(bank, (scenario,), city=str(training["city"]))
    contrast = build_action_contrast_dataset(
        dataset,
        reference_policy="phase_pressure",
        contrast_features=CONTRAST_FEATURES_V3,
    )
    all_records = frozen_proposal_records(
        contrast, frozen_model=frozen_model, scenarios=(scenario,)
    )
    proposals = [row for row in all_records if bool(row["short_proposed"])]
    topology = {
        scenario: build_tls_local_topology_context(
            Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        )
    }
    variant = _fit_variant(
        variant=BASE_VARIANT,
        proposals=proposals,
        all_records=all_records,
        scenarios=(scenario,),
        seeds=seeds,
        profiles={},
        temporal={},
        topology=topology,
        historical={},
        diagnostic=protocol["diagnostic"],
    )
    selected = dict(variant["selected_candidate"])
    advance = bool(variant["passed_selection_rule"] and selected["feasible"])
    integrity = {
        "cache_audit_passed": True,
        "seed_set_exact": {int(row["simulator_seed"]) for row in all_records}
        == set(seeds),
        "scenario_set_exact": {str(row["scenario"]) for row in all_records}
        == {scenario},
        "proposal_rows_nonempty": bool(proposals),
        "oof_rows_exact": int(variant["oof_row_count"]) == len(proposals),
        "only_static_topology_variant": variant["variant"] == BASE_VARIANT,
        "confirmatory_or_prospective_data_used": False,
    }
    integrity["passed"] = all(
        value for key, value in integrity.items() if key != "confirmatory_or_prospective_data_used"
    ) and not integrity["confirmatory_or_prospective_data_used"]
    if not integrity["passed"]:
        advance = False
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": (
            "authorize_waiting_aligned_veto_artifact_freeze"
            if advance
            else "retain_phase_pressure_and_reject_waiting_aligned_veto"
        ),
        "diagnostic_protocol_sha256": _sha256(diagnostic_protocol_path),
        "cache_audit_sha256": _sha256(cache_audit_path),
        "bank_audit": bank_audit,
        "all_action_group_count": len(all_records),
        "proposal_count": len(proposals),
        "proposal_fraction": len(proposals) / max(len(all_records), 1),
        "variant": variant,
        "integrity_gate": integrity,
        "advance_gate": {
            "passed": advance,
            "selected_candidate": selected,
        },
        "claim_boundary": protocol["claim_boundary"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-protocol", type=Path, required=True)
    parser.add_argument("--cache-protocol", type=Path, required=True)
    parser.add_argument("--cache-audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--proposal-parent-protocol", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-audit", type=Path, required=True)
    parser.add_argument("--hierarchical-freeze-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--conversion-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite veto diagnostic: {args.out}")
    result = run_diagnostic(
        diagnostic_protocol_path=args.diagnostic_protocol,
        cache_protocol_path=args.cache_protocol,
        cache_audit_path=args.cache_audit,
        cache_root=args.cache_root,
        proposal_parent_protocol_path=args.proposal_parent_protocol,
        hierarchical_freeze_audit_path=args.hierarchical_freeze_audit,
        hierarchical_freeze_root=args.hierarchical_freeze_root,
        manifest_path=args.manifest,
        conversion_root=args.conversion_root,
        workers=args.workers,
    )
    atomic_write_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "proposals": result["proposal_count"],
                "selected": result["advance_gate"]["selected_candidate"].get(
                    "key"
                ),
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
