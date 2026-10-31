"""Fit and freeze the v73-qualified local-demand execution veto."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import socket
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _merge_city_datasets,
    _sha256,
)
from cf_h2o.eval.traffic_signal_external_demand_conditioned_veto_freeze import (
    _model_config,
    _profile,
)
from cf_h2o.eval.traffic_signal_external_hierarchical_heldout_evaluation import (
    _load_frozen_city_model,
)
from cf_h2o.eval.traffic_signal_external_local_demand_model_diagnostic import (
    VARIANT_SPECS,
    _transformed_target,
    variant_feature_matrix,
    variant_feature_names,
)
from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    frozen_proposal_records,
)
from cf_h2o.eval.traffic_signal_external_proposal_conditional_veto_freeze import (
    _balanced_weights,
)
from cf_h2o.eval.traffic_signal_external_trust_region_freeze import (
    TARGET_SUPPORT_FEATURES,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import CONTRAST_FEATURES_V3
from cf_h2o.eval.traffic_signal_saltlake_global_pairwise_confirmation import (
    aggregate_cache_sha256,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    load_frozen_counterfactual_bank,
)
from cf_h2o.traffic_signal.action_contrast import build_action_contrast_dataset
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest
from cf_h2o.traffic_signal.demand_schedule_context import DemandScheduleProfile
from cf_h2o.traffic_signal.demand_timeline_context import (
    TLSLocalTopologyContext,
    build_tls_local_topology_context,
)
from cf_h2o.traffic_signal.local_demand_target_veto import (
    ARTIFACT_PROTOCOL,
    LocalDemandTargetVeto,
    LocalDemandVetoConfig,
)
from cf_h2o.traffic_signal.local_demand_timeline_context import (
    LocalDemandTimelineContext,
    build_local_demand_timeline_context,
)
from cf_h2o.traffic_signal.proposal_conditional_target_veto import (
    MODEL_PROTOCOL,
    ProposalConditionalRegressor,
)


FREEZE_PROTOCOL = "tsc-v74r70-external-v9-local-demand-veto-freeze-v1"
RESULT_PROTOCOL = "tsc-v74r70-external-v9-local-demand-veto-result-v1"
AUTHORIZATION_DECISION = "authorize_local_demand_veto_closed_loop_development_only"
FAILURE_DECISION = "retain_phase_pressure_and_prohibit_v74_closed_loop_development"
WINNER_VARIANT = "topology_local_squared_raw_scaled"


def fit_city_local_demand_veto(
    *,
    city: str,
    scenarios: Sequence[str],
    records: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, DemandScheduleProfile],
    topology: Mapping[str, TLSLocalTopologyContext],
    local_demand: Mapping[str, LocalDemandTimelineContext],
    model_spec: Mapping[str, Any],
    city_decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    proposals = tuple(row for row in records if bool(row["short_proposed"]))
    if (
        not proposals
        or str(city_decision["variant"]) != WINNER_VARIANT
        or int(city_decision["proposal_count"]) != len(proposals)
        or set(profiles) != set(scenarios)
        or set(topology) != set(scenarios)
        or set(local_demand) != set(scenarios)
    ):
        raise ValueError(f"v74 local-demand inputs changed for {city}")
    features = variant_feature_matrix(
        proposals,
        variant=WINNER_VARIANT,
        profiles=profiles,
        temporal={},
        topology=topology,
        local_demand=local_demand,
    )
    target, target_scales = _transformed_target(
        proposals, target=str(VARIANT_SPECS[WINNER_VARIANT]["target"])
    )
    model = ProposalConditionalRegressor(
        feature_names=variant_feature_names(WINNER_VARIANT),
        config=_model_config(model_spec),
    )
    final_fit = model.fit(
        features, target, sample_weight=_balanced_weights(proposals)
    )
    active = str(city_decision["decision"]) == "freeze_active_veto"
    selected = city_decision.get("selected_candidate")
    if active != (selected is not None):
        raise ValueError(f"v74 active-city decision changed for {city}")
    acceptance_quantile = (
        float(selected["acceptance_quantile"]) if active else 0.025
    )
    fitted_scores = model.predict_matrix(features)
    score_threshold = float(np.quantile(fitted_scores, acceptance_quantile))
    config = LocalDemandVetoConfig(
        enabled=active,
        acceptance_quantile=acceptance_quantile,
        score_threshold=score_threshold,
    )
    scenario_vetoes = {
        scenario: LocalDemandTargetVeto(
            model=model,
            config=config,
            candidate_feature_names=TARGET_SUPPORT_FEATURES,
            demand_profile=profiles[scenario],
            topology_context=topology[scenario],
            local_demand_context=local_demand[scenario],
            target_transform=str(VARIANT_SPECS[WINNER_VARIANT]["target"]),
            target_scale=float(target_scales[scenario]),
        )
        for scenario in scenarios
    }
    summary = {
        "city": str(city),
        "scenarios": list(scenarios),
        "variant": WINNER_VARIANT,
        "variant_spec": dict(VARIANT_SPECS[WINNER_VARIANT]),
        "all_action_group_count": len(records),
        "proposal_training_row_count": len(proposals),
        "candidate_feature_names": list(TARGET_SUPPORT_FEATURES),
        "combined_feature_names": list(variant_feature_names(WINNER_VARIANT)),
        "model_config": asdict(_model_config(model_spec)),
        "target_scales": target_scales,
        "oof_gate_selection": dict(city_decision),
        "final_fit": final_fit,
        "final_config": asdict(config),
        "scenario_veto_diagnostics": {
            scenario: veto.diagnostics() for scenario, veto in scenario_vetoes.items()
        },
    }
    artifact = {
        "protocol": ARTIFACT_PROTOCOL,
        "model_protocol": MODEL_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": str(city),
        "classification": "target_simulator_labeled_few_shot_adaptation",
        "zero_shot": False,
        "scenario_id_is_model_feature": False,
        "proposal_role": "immutable_v60_hierarchical_cfcmt",
        "veto_role": "local_demand_veto_only_never_action_originator",
        "variant": WINNER_VARIANT,
        "scenario_vetoes": scenario_vetoes,
        "model": model,
        "config": config,
        "oof_gate_selection": dict(city_decision),
        "final_fit": final_fit,
        "target_scales": target_scales,
    }
    return summary, artifact


def run_local_demand_veto_freeze(
    *,
    freeze_protocol_path: Path,
    cache_root: Path,
    cache_audit_path: Path,
    proposal_parent_protocol_path: Path,
    freeze_audit_path: Path,
    freeze_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    diagnostic_result_path: Path,
    diagnostic_audit_path: Path,
    output_root: Path,
    workers: int,
) -> dict[str, Any]:
    protocol = json.loads(Path(freeze_protocol_path).read_text(encoding="utf-8"))
    cache_protocol_path = Path(protocol["cache_protocol"]["path"])
    cache_protocol = json.loads(cache_protocol_path.read_text(encoding="utf-8"))
    cache_audit = json.loads(Path(cache_audit_path).read_text(encoding="utf-8"))
    proposal_parent = json.loads(
        Path(proposal_parent_protocol_path).read_text(encoding="utf-8")
    )
    parent_freeze_audit = json.loads(
        Path(freeze_audit_path).read_text(encoding="utf-8")
    )
    diagnostic_result = json.loads(
        Path(diagnostic_result_path).read_text(encoding="utf-8")
    )
    diagnostic_audit = json.loads(
        Path(diagnostic_audit_path).read_text(encoding="utf-8")
    )
    if (
        protocol.get("protocol") != FREEZE_PROTOCOL
        or _sha256(cache_protocol_path) != protocol["cache_protocol"]["sha256"]
        or _sha256(cache_audit_path) != protocol["cache_audit"]["sha256"]
        or cache_audit.get("status") != "PASS"
        or _sha256(proposal_parent_protocol_path)
        != protocol["proposal_parent_protocol"]["sha256"]
        or _sha256(freeze_audit_path)
        != protocol["hierarchical_freeze_audit"]["sha256"]
        or parent_freeze_audit.get("status") != "PASS"
        or _sha256(diagnostic_result_path)
        != protocol["v73_diagnostic_result"]["sha256"]
        or _sha256(diagnostic_audit_path)
        != protocol["v73_diagnostic_audit"]["sha256"]
        or diagnostic_audit.get("status") != "PASS"
        or diagnostic_audit.get("decision")
        != "authorize_oof_qualified_city_successor_freeze_only"
    ):
        raise ValueError("v74 local-demand freeze evidence changed")
    source_gate = {
        path: Path(path).is_file() and _sha256(Path(path)) == expected
        for path, expected in protocol["runtime_executable_sources"].items()
    }
    if not source_gate or not all(source_gate.values()):
        raise ValueError(f"v74 local-demand sources changed: {source_gate}")
    cache_sha, cache_count = aggregate_cache_sha256(cache_root)
    if (
        cache_sha != cache_audit["cache_sha256"]
        or cache_count != cache_audit["cache_file_count"]
    ):
        raise ValueError("v74 local-demand cache changed")
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite v74 freeze: {output_root}")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(Path(conversion_root).resolve())
    manifest = load_traffic_signal_manifest(external_manifest_path)
    collection = cache_protocol["long_horizon_collection"]
    bank, bank_audit = load_frozen_counterfactual_bank(
        cache_root,
        manifest,
        seeds=tuple(int(value) for value in collection["seeds"]),
        collection_shards=int(collection["collection_shards_per_scenario_seed"]),
        workers=max(int(workers), 1),
    )
    profiles = {
        scenario: _profile(payload)
        for scenario, payload in cache_protocol["demand_schedule_context"][
            "profiles"
        ].items()
    }
    topology = {}
    local_demand = {}
    for scenario, profile in profiles.items():
        sumocfg = Path(conversion_root) / scenario / f"{scenario}.sumocfg"
        topology[scenario] = build_tls_local_topology_context(sumocfg)
        local_demand[scenario] = build_local_demand_timeline_context(sumocfg)
        if (
            profile.input_sha256 != topology[scenario].input_sha256
            or profile.input_sha256 != local_demand[scenario].input_sha256
        ):
            raise ValueError(f"v74 context identity changed for {scenario}")
    output_root.mkdir(parents=True, exist_ok=True)
    city_results = {}
    city_artifacts = {}
    for city, scenarios_raw in sorted(cache_protocol["city_scenarios"].items()):
        scenarios = tuple(str(value) for value in scenarios_raw)
        _, frozen_model = _load_frozen_city_model(
            city=str(city),
            freeze_root=freeze_root,
            protocol=proposal_parent,
            freeze_audit=parent_freeze_audit,
        )
        dataset = _merge_city_datasets(bank, scenarios, city=str(city))
        contrast = build_action_contrast_dataset(
            dataset,
            reference_policy="phase_pressure",
            contrast_features=CONTRAST_FEATURES_V3,
        )
        records = frozen_proposal_records(
            contrast, frozen_model=frozen_model, scenarios=scenarios
        )
        summary, artifact = fit_city_local_demand_veto(
            city=str(city),
            scenarios=scenarios,
            records=records,
            profiles={scenario: profiles[scenario] for scenario in scenarios},
            topology={scenario: topology[scenario] for scenario in scenarios},
            local_demand={
                scenario: local_demand[scenario] for scenario in scenarios
            },
            model_spec=protocol["model"]["config"],
            city_decision=diagnostic_audit["city_decisions"][str(city)],
        )
        city_root = output_root / str(city)
        city_root.mkdir(parents=True, exist_ok=False)
        model_path = city_root / "model.pkl"
        model_path.write_bytes(pickle.dumps(artifact, protocol=pickle.HIGHEST_PROTOCOL))
        freeze_payload = {
            "protocol": RESULT_PROTOCOL,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            **summary,
        }
        freeze_path = city_root / "freeze.json"
        _atomic_json(freeze_path, freeze_payload)
        city_results[str(city)] = freeze_payload
        city_artifacts[str(city)] = {
            "model": {"path": str(model_path.resolve()), "sha256": _sha256(model_path)},
            "certificate": {
                "path": str(freeze_path.resolve()),
                "sha256": _sha256(freeze_path),
            },
            "active": bool(summary["final_config"]["enabled"]),
        }
    active_cities = sorted(
        city for city, artifact in city_artifacts.items() if artifact["active"]
    )
    expected_active = sorted(
        city
        for city, row in diagnostic_audit["city_decisions"].items()
        if row["decision"] == "freeze_active_veto"
    )
    integrity = {
        "source_gate_passed": all(source_gate.values()),
        "cache_identity_matches": True,
        "all_cities_present": set(city_results) == set(cache_protocol["city_scenarios"]),
        "active_cities_exactly_match_oof_audit": active_cities == expected_active,
        "active_city_nonempty": bool(active_cities),
        "inactive_cities_force_disabled_veto": all(
            bool(row["final_config"]["enabled"])
            == (city in set(expected_active))
            for city, row in city_results.items()
        ),
        "sealed_results_absent": all(
            int(value) == 0
            for value in protocol["future_file_counts_at_freeze"].values()
        ),
    }
    integrity["passed"] = all(integrity.values())
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "status": "PASS" if integrity["passed"] else "FAIL",
        "decision": (
            AUTHORIZATION_DECISION if integrity["passed"] else FAILURE_DECISION
        ),
        "claim_boundary": (
            "Full-adaptation model freeze only. Closed-loop efficacy and sealed-seed "
            "generalization remain untested."
        ),
        "freeze_protocol_sha256": _sha256(freeze_protocol_path),
        "diagnostic_result_sha256": _sha256(diagnostic_result_path),
        "diagnostic_audit_sha256": _sha256(diagnostic_audit_path),
        "cache_sha256": cache_sha,
        "cache_file_count": cache_count,
        "bank_audit": bank_audit,
        "source_gate": source_gate,
        "active_cities": active_cities,
        "city_artifacts": city_artifacts,
        "city_results": city_results,
        "integrity_gate": integrity,
    }
