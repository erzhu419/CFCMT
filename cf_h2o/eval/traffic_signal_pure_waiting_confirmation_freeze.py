"""Freeze the V117 selector-heldout Los Angeles confirmation matrix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    _atomic_json,
    _sha256,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_component_fit import (
    H2OPLUS_MODEL_PROTOCOL,
    MODEL_BUNDLE_PROTOCOL,
    RESULT_PROTOCOL as FIT_RESULT_PROTOCOL,
    STRICT_TARGET_MODEL_PROTOCOL,
    _contract_sha256,
    validate_architecture_matched_target_contract,
    validate_target_label_free_blend_contract,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_source_selector import (
    PURE_WAITING_RESULT_PROTOCOL,
)


PROTOCOL = "tsc-v117-pure-waiting-selector-heldout-city-confirmation-v2"
FREEZE_PROTOCOL = "tsc-v117-pure-waiting-confirmation-freeze-v2"
AUTHORIZATION_DECISION = "authorize_v117_selector_heldout_city_rollouts"
ARM_ORDER = (
    "phase_pressure",
    "target_only_b100",
    "source_selected_b100",
    "source_selected_b0",
    "h2oplus_dense_b100",
    "h2oplus_dense_b0",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _artifact(
    record: Mapping[str, Any],
    *,
    expected_protocol: str,
    city: str,
    budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(str(record["path"]))
    expected_sha = str(record["sha256"])
    if not path.is_file() or _sha256(path) != expected_sha:
        raise ValueError(f"V117 model artifact identity changed: {path}")
    payload = pickle.loads(path.read_bytes())
    if (
        payload.get("protocol") != expected_protocol
        or payload.get("city") != city
        or int(payload.get("target_group_budget", -1)) != int(budget)
        or len(tuple(payload.get("selected_group_ids", ()))) != int(budget)
        or payload.get("target_name") != "prefix_mean_cost_450s"
        or payload.get("prior_policy") != "phase_pressure"
    ):
        raise ValueError(f"V117 model artifact contract changed: {path}")
    frozen = {
        "path": str(path.resolve()),
        "sha256": expected_sha,
        "size_bytes": int(path.stat().st_size),
        "protocol": expected_protocol,
        "target_group_budget": int(budget),
        "adaptation_contract_sha256": str(
            payload["adaptation_contract_sha256"]
        ),
    }
    return payload, frozen


def _selected_target_profile(selector: Mapping[str, Any]) -> str | None:
    selection = dict(selector["full_development_selection"])
    selected_key = selection.get("selected_profile_key")
    if selected_key is None:
        return None
    profile = dict(selector["profile_definitions"][str(selected_key)])
    if profile.get("source_city_group") is None:
        return str(selected_key)
    matched = profile.get("target_profile_key")
    if matched is None or str(matched) not in selector["profile_definitions"]:
        raise ValueError("V116 selected source lacks its matched target profile")
    return str(matched)


def freeze_confirmation(
    *,
    protocol_path: Path,
    selector_result_path: Path,
    expected_selector_result_sha256: str,
    deployment_fit_result_path: Path,
    expected_deployment_fit_result_sha256: str,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    selector = _read_json(selector_result_path)
    fit = _read_json(deployment_fit_result_path)
    if (
        protocol.get("protocol") != PROTOCOL
        or _sha256(selector_result_path) != str(expected_selector_result_sha256)
        or selector.get("protocol") != PURE_WAITING_RESULT_PROTOCOL
        or selector.get("city") != protocol["selector"]["development_city"]
        or selector.get("selector_cache_audit", {}).get("decision")
        != "authorize_v116_pure_waiting_nested_source_selection"
        or _sha256(deployment_fit_result_path)
        != str(expected_deployment_fit_result_sha256)
        or fit.get("protocol") != FIT_RESULT_PROTOCOL
        or fit.get("city") != protocol["deployment"]["city"]
        or fit.get("target_name") != "prefix_mean_cost_450s"
    ):
        raise ValueError("V117 parent evidence changed")
    seeds = tuple(int(value) for value in protocol["fresh_seeds"])
    selector_seeds = set(int(value) for value in selector["seeds"])
    if (
        len(seeds) != int(protocol["fresh_seed_generation"]["seed_count"])
        or len(seeds) != len(set(seeds))
        or set(seeds) & selector_seeds
        or set(seeds) & {5057, 6067}
    ):
        raise ValueError("V117 fresh-seed isolation changed")

    records = dict(fit["model_artifacts"])
    payloads: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, dict[str, Any]] = {}
    specs = {
        "strict_target_b100": (STRICT_TARGET_MODEL_PROTOCOL, 100),
        "source_components_b0": (MODEL_BUNDLE_PROTOCOL, 0),
        "source_components_b100": (MODEL_BUNDLE_PROTOCOL, 100),
        "h2oplus_dense_b0": (H2OPLUS_MODEL_PROTOCOL, 0),
        "h2oplus_dense_b100": (H2OPLUS_MODEL_PROTOCOL, 100),
    }
    for key, (artifact_protocol, budget) in specs.items():
        payloads[key], artifacts[key] = _artifact(
            records[key],
            expected_protocol=artifact_protocol,
            city=str(protocol["deployment"]["city"]),
            budget=budget,
        )
    validate_architecture_matched_target_contract(
        payloads["strict_target_b100"]
    )
    fit_blend_selection = dict(fit.get("blend_candidate_selection", {}))
    fit_blend_sha256 = str(fit.get("blend_candidate_selection_sha256", ""))
    if (
        _contract_sha256(fit_blend_selection) != fit_blend_sha256
        or any(
            validate_target_label_free_blend_contract(payload).get(
                "selection_sha256"
            )
            != fit_blend_sha256
            for payload in (
                payloads["source_components_b0"],
                payloads["source_components_b100"],
            )
        )
    ):
        raise ValueError("V117 target-label-free blend evidence changed")
    adaptation_hashes = {
        str(payload["adaptation_contract_sha256"])
        for payload in payloads.values()
    }
    if (
        len(adaptation_hashes) != 1
        or adaptation_hashes != {str(fit["adaptation_contract_sha256"])}
        or tuple(payloads["strict_target_b100"]["selected_group_ids"])
        != tuple(payloads["source_components_b100"]["selected_group_ids"])
        or tuple(payloads["strict_target_b100"]["selected_group_ids"])
        != tuple(payloads["h2oplus_dense_b100"]["selected_group_ids"])
        or payloads["strict_target_b100"].get("selected_candidate")
        != payloads["source_components_b100"].get("selected_candidate")
        or payloads["strict_target_b100"].get(
            "blend_candidate_selection_sha256"
        )
        != payloads["source_components_b100"].get(
            "candidate_selection_sha256"
        )
    ):
        raise ValueError("V117 B100 arms do not share one adaptation contract")

    full_selection = dict(selector["full_development_selection"])
    selected_source = full_selection.get("selected_source_city_group")
    b100_source_active = bool(
        selector.get("source_transfer_gate_passed", False)
        and selected_source is not None
    )
    target_profile_key = _selected_target_profile(selector)
    zero = dict(selector.get("zero_shot_source_admission") or {})
    b0_source_active = bool(
        zero.get("source_transfer_gate_passed", False)
        and zero.get("full_development_selection", {}).get(
            "selected_profile_key"
        )
        is not None
    )
    activation = {
        "phase_pressure": True,
        "target_only_b100": target_profile_key is not None,
        "source_selected_b100": b100_source_active,
        "source_selected_b0": b0_source_active,
        "h2oplus_dense_b100": True,
        "h2oplus_dense_b0": True,
    }
    active_arms = tuple(arm for arm in ARM_ORDER if activation[arm])
    inactive_fallbacks = {
        arm: "phase_pressure" for arm in ARM_ORDER if not activation[arm]
    }
    if not {"phase_pressure", "h2oplus_dense_b0", "h2oplus_dense_b100"} <= set(
        active_arms
    ):
        raise AssertionError("V117 unconditional arms disappeared")
    return {
        "protocol": FREEZE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": AUTHORIZATION_DECISION,
        "scientific_status": "frozen_before_any_v117_rollout",
        "protocol_spec": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": _sha256(protocol_path),
        },
        "selector_result": {
            "path": str(Path(selector_result_path).resolve()),
            "sha256": str(expected_selector_result_sha256),
            "development_city": selector["city"],
            "source_transfer_gate_passed": b100_source_active,
            "zero_shot_source_gate_passed": b0_source_active,
            "selected_b100_profile_key": full_selection.get(
                "selected_profile_key"
            ),
            "selected_b100_source_city_group": selected_source,
            "matched_target_profile_key": target_profile_key,
            "selected_b0_profile_key": zero.get(
                "full_development_selection", {}
            ).get("selected_profile_key"),
        },
        "deployment_fit_result": {
            "path": str(Path(deployment_fit_result_path).resolve()),
            "sha256": str(expected_deployment_fit_result_sha256),
            "adaptation_contract_sha256": str(fit["adaptation_contract_sha256"]),
        },
        "blend_candidate_selection": {
            "sha256": fit_blend_sha256,
            "selected_candidate": fit_blend_selection["selected_candidate"],
            "gate_passed": bool(fit_blend_selection["gate_passed"]),
            "target_transition_labels_consumed": int(
                fit_blend_selection["target_transition_labels_consumed"]
            ),
            "decision": fit_blend_selection["decision"],
        },
        "model_artifacts": artifacts,
        "deployment": dict(protocol["deployment"]),
        "fresh_seeds": list(seeds),
        "active_arms": list(active_arms),
        "inactive_arm_fallbacks": inactive_fallbacks,
        "matrix_size": len(seeds) * len(active_arms),
        "information_arms": dict(protocol["information_arms"]),
        "primary_comparisons": list(protocol["primary_comparisons"]),
        "secondary_comparisons": list(protocol["secondary_comparisons"]),
        "confirmation_gate": dict(protocol["confirmation_gate"]),
        "claim_scope": str(protocol["claim_scope"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--selector-result", type=Path, required=True)
    parser.add_argument("--selector-result-sha256", required=True)
    parser.add_argument("--deployment-fit-result", type=Path, required=True)
    parser.add_argument("--deployment-fit-result-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("refusing to overwrite V117 confirmation freeze")
    result = freeze_confirmation(
        protocol_path=args.protocol,
        selector_result_path=args.selector_result,
        expected_selector_result_sha256=args.selector_result_sha256,
        deployment_fit_result_path=args.deployment_fit_result,
        expected_deployment_fit_result_sha256=(
            args.deployment_fit_result_sha256
        ),
    )
    _atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "active_arms": result["active_arms"],
                "matrix_size": result["matrix_size"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
