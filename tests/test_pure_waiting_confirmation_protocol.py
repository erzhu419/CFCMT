from __future__ import annotations

import json
from pathlib import Path
import pickle

import pytest

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze import (
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
    freeze_confirmation,
)
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_audit import (
    audit_confirmation,
)
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_rollout import (
    RESULT_PROTOCOL as ROLLOUT_PROTOCOL,
    STATIC_INPUT_AUDIT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    H2OPLUS_MODEL_PROTOCOL,
    MODEL_BUNDLE_PROTOCOL,
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
    STRICT_TARGET_MODEL_PROTOCOL,
    TARGET_ONLY_FAMILY,
    _contract_sha256,
)
from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    PURE_WAITING_RESULT_PROTOCOL,
)
from scripts.cluster.run_tsc_pure_waiting_confirmation_shard import (
    SHARD_PROTOCOL,
    shard_identities,
)
from scripts.cluster.run_tsc_pure_waiting_confirmation_freeze import (
    PROTOCOL_RELATIVE as REMOTE_FREEZE_PROTOCOL_RELATIVE,
    build_remote_command as build_remote_freeze_command,
)
from scripts.cluster.launch_tsc_pure_waiting_confirmation import build_specs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = PROJECT_ROOT / (
    "cf_h2o/config/traffic_signal_tsc_v117_"
    "pure_waiting_heldout_city_confirmation.json"
)


def test_v117_remote_freeze_reads_server_side_model_artifacts() -> None:
    snapshot = Path("/remote/snapshot")
    command = build_remote_freeze_command(
        snapshot=snapshot,
        selector_result=Path("/remote/results/selector.json"),
        selector_sha256="a" * 64,
        deployment_fit_result=Path("/remote/results/los_angeles/result.json"),
        deployment_fit_sha256="b" * 64,
        remote_out=Path("/remote/results/freeze.json"),
    )
    assert str(snapshot / REMOTE_FREEZE_PROTOCOL_RELATIVE) in command
    assert "/remote/results/selector.json" in command
    assert "/remote/results/los_angeles/result.json" in command
    assert "a" * 64 in command
    assert "b" * 64 in command
    assert "traffic_signal_pure_waiting_confirmation_freeze" in command


def test_v117_scheduler_specs_emit_completion_marker() -> None:
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        freeze_path=Path("/remote/freeze.json"),
        freeze_sha256="a" * 64,
        conversion_root=Path("/remote/conversion"),
        conversion_manifest=Path("/remote/conversion/manifest.json"),
        remote_results_root=Path("/remote/results"),
        workers_per_node=20,
    )
    assert len(specs) == 6
    assert all(spec["cmd"].endswith("&& printf 'TASK_DONE\\n'") for spec in specs)


def _write_pickle(path: Path, payload: dict) -> dict:
    path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _parents(tmp_path: Path) -> tuple[Path, Path]:
    groups = tuple(f"g{index}" for index in range(100))
    adaptation = "shared-adaptation"
    blend_selection = {
        "protocol": "pure-waiting-target-label-free-source-loco-blend-selection-v1",
        "target_transition_labels_consumed": 0,
        "source_city_groups": list(DEVELOPMENT_CITY_GROUPS),
        "candidate_count": len(CANDIDATE_KEYS),
        "gate_passed": False,
        "selected_candidate": ANCHOR_KEY,
        "fallback_candidate": ANCHOR_KEY,
        "decision": "suppress_residual_and_freeze_rigid_anchor",
        "evaluation_audit": {},
        "selection": {},
    }
    blend_sha256 = _contract_sha256(blend_selection)
    blend_contract = {
        name: blend_selection[name]
        for name in (
            "protocol",
            "target_transition_labels_consumed",
            "source_city_groups",
            "candidate_count",
            "gate_passed",
            "selected_candidate",
            "fallback_candidate",
            "decision",
        )
    }
    blend_contract["selection_sha256"] = blend_sha256
    specs = {
        "strict_target_b100": (STRICT_TARGET_MODEL_PROTOCOL, groups),
        "source_components_b0": (MODEL_BUNDLE_PROTOCOL, ()),
        "source_components_b100": (MODEL_BUNDLE_PROTOCOL, groups),
        "h2oplus_dense_b0": (H2OPLUS_MODEL_PROTOCOL, ()),
        "h2oplus_dense_b100": (H2OPLUS_MODEL_PROTOCOL, groups),
    }
    artifacts = {}
    for key, (protocol, selected) in specs.items():
        payload = {
            "protocol": protocol,
            "city": "los_angeles",
            "target_group_budget": len(selected),
            "selected_group_ids": selected,
            "target_name": "prefix_mean_cost_450s",
            "prior_policy": "phase_pressure",
            "adaptation_contract_sha256": adaptation,
        }
        if key.startswith("source_components_"):
            payload.update(
                {
                    "selected_candidate": ANCHOR_KEY,
                    "candidate_selection_sha256": blend_sha256,
                    "blend_candidate_selection": blend_contract,
                }
            )
        if key == "strict_target_b100":
            payload.update(
                {
                    "family": TARGET_ONLY_FAMILY,
                    "selected_candidate": ANCHOR_KEY,
                    "blend_candidate_selection_sha256": blend_sha256,
                    "architecture_match": {
                        "protocol": (
                            "pure-waiting-architecture-matched-target-only-v1"
                        ),
                        "base_families": [
                            "causal_group_normalized_rigid_advantage",
                            "causal_antisymmetric_pairwise_advantage",
                        ],
                        "selected_candidate": ANCHOR_KEY,
                        "blend_candidate_selection_sha256": blend_sha256,
                        "source_transition_rows_consumed": 0,
                    },
                    "fit_diagnostics": {
                        "source_row_count_consumed": 0,
                        "target_group_count": 100,
                    },
                }
            )
        artifacts[key] = _write_pickle(
            tmp_path / f"{key}.pkl",
            payload,
        )
    fit_path = tmp_path / "fit.json"
    fit_path.write_text(
        json.dumps(
            {
                "protocol": FIT_RESULT_PROTOCOL,
                "city": "los_angeles",
                "target_name": "prefix_mean_cost_450s",
                "adaptation_contract_sha256": adaptation,
                "blend_candidate_selection": blend_selection,
                "blend_candidate_selection_sha256": blend_sha256,
                "model_artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    target_key = "target"
    source_key = "source"
    b0_key = "b0"
    selector_path = tmp_path / "selector.json"
    selector_path.write_text(
        json.dumps(
            {
                "protocol": PURE_WAITING_RESULT_PROTOCOL,
                "city": "jinan",
                "seeds": [11, 12],
                "selector_cache_audit": {
                    "decision": "authorize_v116_pure_waiting_nested_source_selection"
                },
                "source_transfer_gate_passed": True,
                "full_development_selection": {
                    "selected_profile_key": source_key,
                    "selected_source_city_group": "all_sources_mean",
                },
                "profile_definitions": {
                    target_key: {
                        "source_city_group": None,
                        "target_profile_key": None,
                    },
                    source_key: {
                        "source_city_group": "all_sources_mean",
                        "target_profile_key": target_key,
                    },
                },
                "zero_shot_source_admission": {
                    "source_transfer_gate_passed": True,
                    "full_development_selection": {
                        "selected_profile_key": b0_key,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return selector_path, fit_path


def test_v117_freeze_activates_only_predeclared_selector_outputs(tmp_path) -> None:
    selector_path, fit_path = _parents(tmp_path)
    freeze = freeze_confirmation(
        protocol_path=PROTOCOL_PATH,
        selector_result_path=selector_path,
        expected_selector_result_sha256=_sha256(selector_path),
        deployment_fit_result_path=fit_path,
        expected_deployment_fit_result_sha256=_sha256(fit_path),
    )
    assert freeze["decision"] == AUTHORIZATION_DECISION
    assert freeze["matrix_size"] == 64 * 6
    assert freeze["inactive_arm_fallbacks"] == {}
    shards = [
        set(shard_identities(freeze, shard_index=index, shard_count=6))
        for index in range(6)
    ]
    assert sum(len(rows) for rows in shards) == freeze["matrix_size"]
    assert len(set().union(*shards)) == freeze["matrix_size"]
    assert all(not (left & right) for i, left in enumerate(shards) for right in shards[i + 1 :])


def test_v117_freeze_rejects_unequal_b100_information(tmp_path) -> None:
    selector_path, fit_path = _parents(tmp_path)
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    h2o_path = Path(fit["model_artifacts"]["h2oplus_dense_b100"]["path"])
    payload = pickle.loads(h2o_path.read_bytes())
    payload["selected_group_ids"] = tuple(f"x{index}" for index in range(100))
    fit["model_artifacts"]["h2oplus_dense_b100"] = _write_pickle(
        h2o_path, payload
    )
    fit_path.write_text(json.dumps(fit), encoding="utf-8")
    with pytest.raises(ValueError, match="adaptation contract"):
        freeze_confirmation(
            protocol_path=PROTOCOL_PATH,
            selector_result_path=selector_path,
            expected_selector_result_sha256=_sha256(selector_path),
            deployment_fit_result_path=fit_path,
            expected_deployment_fit_result_sha256=_sha256(fit_path),
        )


def test_v117_audit_uses_complete_paired_seed_matrix(tmp_path) -> None:
    arms = (
        "phase_pressure",
        "target_only_b100",
        "source_selected_b100",
        "h2oplus_dense_b100",
    )
    seeds = (101, 102)
    freeze = {
        "protocol": FREEZE_PROTOCOL,
        "decision": AUTHORIZATION_DECISION,
        "deployment": {"city": "los_angeles", "scenario": "la_1x4"},
        "fresh_seeds": seeds,
        "active_arms": arms,
        "inactive_arm_fallbacks": {},
        "matrix_size": len(seeds) * len(arms),
        "confirmation_gate": {
            "bootstrap_replicates": 100,
            "bootstrap_seed": 7,
            "minimum_active_source_seed_fraction": 0.5,
            "maximum_95pct_upper_relative_delta_source_b100_vs_h2oplus_b100": 0.0,
            "maximum_95pct_upper_relative_delta_source_b100_vs_target_only_b100": 0.0,
            "maximum_95pct_upper_relative_delta_source_b100_vs_phase_pressure": 0.01,
        },
        "claim_scope": "test",
    }
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    freeze_sha = _sha256(freeze_path)
    results_root = tmp_path / "results"
    waiting = {
        "phase_pressure": 10.0,
        "target_only_b100": 9.0,
        "source_selected_b100": 8.0,
        "h2oplus_dense_b100": 9.0,
    }
    for shard_index in range(6):
        audit_path = results_root / "_input_audits" / f"shard_{shard_index}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(
                {
                    "protocol": STATIC_INPUT_AUDIT_PROTOCOL,
                    "status": "PASS",
                    "freeze_sha256": freeze_sha,
                }
            ),
            encoding="utf-8",
        )
        identities = shard_identities(
            freeze, shard_index=shard_index, shard_count=6
        )
        summary_rows = []
        for seed, arm in identities:
            root = results_root / f"seed_{seed}" / arm
            root.mkdir(parents=True, exist_ok=True)
            tripinfo = root / "tripinfo.xml"
            tripinfo.write_text("<tripinfos/>", encoding="utf-8")
            metrics = {
                "ok": True,
                "mean_tripinfo_waiting_time": waiting[arm],
                "collision_incidents": 0,
                "starting_teleports": 0,
                "ending_teleports": 0,
                "guard_audit": {
                    "effective_overrides": (
                        1 if arm == "source_selected_b100" else 0
                    )
                },
            }
            (root / "result.json").write_text(
                json.dumps(
                    {
                        "protocol": ROLLOUT_PROTOCOL,
                        "freeze_sha256": freeze_sha,
                        "seed": seed,
                        "arm": arm,
                        "city": "los_angeles",
                        "scenario": "la_1x4",
                        "tripinfo_evidence": {"sha256": _sha256(tripinfo)},
                        "metrics": metrics,
                    }
                ),
                encoding="utf-8",
            )
            summary_rows.append({"seed": seed, "arm": arm})
        summary_path = results_root / "_shards" / f"shard_{shard_index}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "protocol": SHARD_PROTOCOL,
                    "freeze_sha256": freeze_sha,
                    "shard_index": shard_index,
                    "shard_count": 6,
                    "hostname": f"node{shard_index}",
                    "static_input_audit_path": str(audit_path),
                    "static_input_audit_sha256": _sha256(audit_path),
                    "rows": summary_rows,
                }
            ),
            encoding="utf-8",
        )
    audit = audit_confirmation(
        freeze_path=freeze_path,
        expected_freeze_sha256=freeze_sha,
        results_root=results_root,
    )
    assert audit["status"] == "PASS"
    assert audit["confirmation_gate_passed"] is True
    assert audit["decision"] == "authorize_v117_cross_city_source_benefit_claim"
