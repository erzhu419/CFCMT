"""Run one V91 fresh-seed CFCMT versus target-only confirmation rollout."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_external_closed_loop_confirmation import (
    ClosedLoopDiagnosticSpec,
    METHOD_POLICY,
    run_external_closed_loop_rollout,
)
from cf_h2o.eval.traffic_signal_external_source_contribution_ablation import (
    POLICY as TARGET_ONLY_POLICY,
)


RESULT_PROTOCOL = "tsc-v91r87-source-contribution-fresh-rollout-v1"
ANALYSIS_PROTOCOL = "tsc-v91r87-source-contribution-fresh-confirmation-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def all_rollout_identities(
    protocol: dict[str, Any],
) -> tuple[tuple[str, str, int, str], ...]:
    confirmation = protocol["confirmation"]
    identities = tuple(
        (str(city), str(scenario), int(seed), str(policy))
        for city, scenarios in confirmation["city_scenarios"].items()
        for scenario in scenarios
        for seed in confirmation["seeds"]
        for policy in confirmation["policies"]
    )
    if (
        len(identities) != int(confirmation["matrix_size"])
        or len(identities) != len(set(identities))
    ):
        raise ValueError("V91 rollout identity matrix changed")
    return identities


def _spec(
    *, policy: str, source_policy: str, seeds: tuple[int, ...]
) -> ClosedLoopDiagnosticSpec:
    return ClosedLoopDiagnosticSpec(
        name=policy,
        source_policy=source_policy,
        coordination_mode="sparse",
        result_protocol=RESULT_PROTOCOL,
        allowed_seeds=seeds,
        analysis_status="fresh_seed_source_contribution_confirmation",
    )


def run_confirmation_rollout(
    *,
    confirmation_protocol_path: Path,
    v43_protocol_path: Path,
    offline_authorization_path: Path,
    expected_offline_authorization_sha256: str,
    joint_freeze_audit_path: Path,
    expected_joint_freeze_sha256: str,
    freeze_root: Path,
    external_manifest_path: Path,
    conversion_root: Path,
    expected_external_manifest_sha256: str,
    conversion_manifest_path: Path,
    expected_conversion_manifest_sha256: str,
    expected_conversion_tree_sha256: str,
    scenario: str,
    seed: int,
    policy: str,
    tripinfo_out: Path,
) -> dict[str, Any]:
    from scripts.cluster.freeze_tsc_external_source_contribution_confirmation import (
        PROTOCOL,
    )

    protocol = _read_json(confirmation_protocol_path)
    if protocol.get("protocol") != PROTOCOL:
        raise ValueError("V91 confirmation protocol changed")
    confirmation = protocol["confirmation"]
    seeds = tuple(int(value) for value in confirmation["seeds"])
    matching = [
        city
        for city, scenarios in confirmation["city_scenarios"].items()
        if scenario in scenarios
    ]
    if (
        len(matching) != 1
        or int(seed) not in seeds
        or policy not in tuple(confirmation["policies"])
    ):
        raise ValueError("V91 rollout is outside the frozen matrix")
    source_policy = (
        METHOD_POLICY if policy == METHOD_POLICY else TARGET_ONLY_POLICY
    )
    result = run_external_closed_loop_rollout(
        protocol_spec_path=v43_protocol_path,
        offline_authorization_path=offline_authorization_path,
        expected_offline_authorization_sha256=expected_offline_authorization_sha256,
        joint_freeze_audit_path=joint_freeze_audit_path,
        expected_joint_freeze_sha256=expected_joint_freeze_sha256,
        freeze_root=freeze_root,
        external_manifest_path=external_manifest_path,
        conversion_root=conversion_root,
        expected_external_manifest_sha256=expected_external_manifest_sha256,
        conversion_manifest_path=conversion_manifest_path,
        expected_conversion_manifest_sha256=expected_conversion_manifest_sha256,
        expected_conversion_tree_sha256=expected_conversion_tree_sha256,
        expected_sumo_version="1.22.0",
        scenario=scenario,
        seed=int(seed),
        policy=policy,
        tripinfo_out=tripinfo_out,
        diagnostic_spec=_spec(
            policy=policy, source_policy=source_policy, seeds=seeds
        ),
    )
    if result["city"] != matching[0]:
        raise RuntimeError("V91 city identity changed during rollout")
    result.update(
        {
            "analysis_protocol": ANALYSIS_PROTOCOL,
            "confirmation_protocol_sha256": _sha256(confirmation_protocol_path),
            "fresh_confirmation_evidence": True,
            "original_v43_confirmation_unchanged": True,
        }
    )
    return result
