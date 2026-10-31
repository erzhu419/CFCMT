"""Run one frozen V117 pure-waiting Los Angeles confirmation rollout."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
from typing import Any, Mapping

from cf_h2o.eval.traffic_signal_causal_source_component_rollout import (
    load_pure_waiting_h2oplus_runtime_override,
    load_pure_waiting_selector_runtime_override,
)
from cf_h2o.eval.traffic_signal_counterfactual_cache import _tree_sha256
from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _sha256
from cf_h2o.eval.traffic_signal_pure_waiting_confirmation_freeze import (
    AUTHORIZATION_DECISION,
    FREEZE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_resco_cfcmt_v2 import _runtime_metadata
from cf_h2o.eval.traffic_signal_resco_cfcmt_v3 import evaluate_policy_v3
from cf_h2o.sumo_runtime import libsumo_version, load_libsumo
from cf_h2o.traffic_signal.benchmark_manifest import load_traffic_signal_manifest


RESULT_PROTOCOL = "tsc-v117-pure-waiting-heldout-city-rollout-v2"
STATIC_INPUT_AUDIT_PROTOCOL = "tsc-v117-static-input-shard-audit-v2"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def audit_static_inputs(
    *,
    freeze_path: Path,
    expected_freeze_sha256: str,
    manifest_path: Path,
    conversion_root: Path,
    conversion_manifest_path: Path,
) -> dict[str, Any]:
    """Validate immutable simulator inputs once before a process shard starts."""

    freeze = _read_json(freeze_path)
    deployment = dict(freeze.get("deployment", {}))
    conversion_root = Path(conversion_root).resolve()
    if (
        _sha256(freeze_path) != str(expected_freeze_sha256)
        or freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("decision") != AUTHORIZATION_DECISION
        or _sha256(manifest_path) != deployment.get("manifest_sha256")
        or str(conversion_root) != str(deployment.get("conversion_root"))
        or str(Path(conversion_manifest_path).resolve())
        != str(Path(deployment.get("conversion_manifest", "")).resolve())
        or _sha256(conversion_manifest_path)
        != deployment.get("conversion_manifest_sha256")
        or _tree_sha256(conversion_root)
        != deployment.get("conversion_tree_sha256")
        or libsumo_version() != deployment.get("sumo_version")
    ):
        raise ValueError("V117 static simulator input identity changed")
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(manifest_path)
    scenario = str(deployment["scenario"])
    if (
        scenario not in manifest.sumocfgs
        or manifest.city_groups[scenario] != deployment.get("city")
        or conversion_root not in Path(manifest.sumocfgs[scenario]).resolve().parents
    ):
        raise ValueError("V117 deployment scenario mapping changed")
    return {
        "protocol": STATIC_INPUT_AUDIT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "hostname": socket.gethostname(),
        "freeze_sha256": str(expected_freeze_sha256),
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "conversion_root": str(conversion_root),
        "conversion_manifest_path": str(Path(conversion_manifest_path).resolve()),
        "conversion_manifest_sha256": _sha256(conversion_manifest_path),
        "conversion_tree_sha256": deployment["conversion_tree_sha256"],
        "sumo_version": libsumo_version(),
        "scenario": scenario,
        "city": deployment["city"],
    }


def _runtime_for_arm(
    freeze: Mapping[str, Any], *, arm: str
) -> tuple[str, Any | None, Mapping[str, Any] | None]:
    deployment = dict(freeze["deployment"])
    city = str(deployment["city"])
    artifacts = dict(freeze["model_artifacts"])
    adaptation_sha = str(
        freeze["deployment_fit_result"]["adaptation_contract_sha256"]
    )
    if arm == "phase_pressure":
        return "phase_pressure", None, None
    if arm in {"target_only_b100", "source_selected_b100", "source_selected_b0"}:
        budget = 0 if arm.endswith("_b0") else 100
        bundle = artifacts[f"source_components_b{budget}"]
        strict = artifacts["strict_target_b100"] if budget else None
        selector = dict(freeze["selector_result"])
        runtime = load_pure_waiting_selector_runtime_override(
            component_bundle_path=Path(bundle["path"]),
            expected_component_bundle_sha256=str(bundle["sha256"]),
            selector_result_path=Path(selector["path"]),
            expected_selector_result_sha256=str(selector["sha256"]),
            city=city,
            arm=arm,
            prediction_horizon_sec=int(deployment["prediction_horizon_sec"]),
            strict_target_only_model_path=(
                Path(strict["path"]) if strict is not None else None
            ),
            expected_strict_target_only_model_sha256=(
                str(strict["sha256"]) if strict is not None else None
            ),
        )
        return runtime.runtime_policy, runtime.models, runtime.provenance
    if arm in {"h2oplus_dense_b0", "h2oplus_dense_b100"}:
        model = artifacts[arm]
        runtime = load_pure_waiting_h2oplus_runtime_override(
            model_path=Path(model["path"]),
            expected_model_sha256=str(model["sha256"]),
            expected_adaptation_contract_sha256=adaptation_sha,
            city=city,
            arm=arm,
            prediction_horizon_sec=int(deployment["prediction_horizon_sec"]),
        )
        return runtime.runtime_policy, runtime.models, runtime.provenance
    raise ValueError(f"unknown V117 arm: {arm}")


def run_confirmation_rollout(
    *,
    freeze_path: Path,
    expected_freeze_sha256: str,
    static_input_audit_path: Path,
    expected_static_input_audit_sha256: str,
    manifest_path: Path,
    arm: str,
    seed: int,
    tripinfo_out: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    freeze = _read_json(freeze_path)
    audit = _read_json(static_input_audit_path)
    deployment = dict(freeze.get("deployment", {}))
    if (
        _sha256(freeze_path) != str(expected_freeze_sha256)
        or freeze.get("protocol") != FREEZE_PROTOCOL
        or freeze.get("decision") != AUTHORIZATION_DECISION
        or arm not in freeze.get("active_arms", ())
        or int(seed) not in {int(value) for value in freeze.get("fresh_seeds", ())}
        or _sha256(static_input_audit_path)
        != str(expected_static_input_audit_sha256)
        or audit.get("protocol") != STATIC_INPUT_AUDIT_PROTOCOL
        or audit.get("status") != "PASS"
        or audit.get("freeze_sha256") != str(expected_freeze_sha256)
        or audit.get("hostname") != socket.gethostname()
        or _sha256(manifest_path) != deployment.get("manifest_sha256")
    ):
        raise ValueError("V117 rollout identity or shard input audit changed")
    conversion_root = Path(str(audit["conversion_root"]))
    os.environ["CFCMT_EXTERNAL_CONVERSION_ROOT"] = str(conversion_root)
    manifest = load_traffic_signal_manifest(manifest_path)
    scenario = str(deployment["scenario"])
    runtime_policy, models, provenance = _runtime_for_arm(freeze, arm=arm)
    tripinfo_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_tripinfo = tripinfo_out.with_suffix(
        tripinfo_out.suffix + f".tmp-{os.getpid()}"
    )
    try:
        metrics = evaluate_policy_v3(
            sumo_api=load_libsumo(),
            sumocfg=manifest.sumocfgs[scenario],
            scenario=scenario,
            policy=runtime_policy,
            models=models,
            duration_sec=float(deployment["duration_sec"]),
            control_interval_sec=int(deployment["control_interval_sec"]),
            warmup_sec=float(deployment["warmup_sec"]),
            seed=int(seed),
            tripinfo_output=temporary_tripinfo,
            residual_coordination_mode=str(
                deployment["residual_coordination_mode"]
            ),
        )
        if not bool(metrics.get("ok", False)):
            raise RuntimeError(f"V117 rollout failed: {metrics.get('error')}")
        if bool(deployment["require_zero_starting_and_ending_teleports"]) and (
            int(metrics.get("starting_teleports", -1)) != 0
            or int(metrics.get("ending_teleports", -1)) != 0
        ):
            raise RuntimeError("V117 rollout contains teleports")
        if not temporary_tripinfo.is_file() or temporary_tripinfo.stat().st_size <= 0:
            raise RuntimeError("V117 rollout lacks tripinfo evidence")
        temporary_tripinfo.replace(tripinfo_out)
    except Exception:
        temporary_tripinfo.unlink(missing_ok=True)
        raise
    return {
        "protocol": RESULT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "runtime": _runtime_metadata(),
        "freeze_sha256": str(expected_freeze_sha256),
        "static_input_audit": {
            "path": str(Path(static_input_audit_path).resolve()),
            "sha256": str(expected_static_input_audit_sha256),
        },
        "city": deployment["city"],
        "scenario": scenario,
        "seed": int(seed),
        "arm": arm,
        "runtime_policy": runtime_policy,
        "target_transition_label_budget": int(
            freeze["information_arms"][arm]["target_transition_label_budget"]
        ),
        "prediction_horizon_sec": (
            int(models.prediction_horizon_sec) if models is not None else None
        ),
        "runtime_provenance": dict(provenance) if provenance is not None else None,
        "tripinfo_evidence": {
            "path": str(Path(tripinfo_out).resolve()),
            "sha256": _sha256(tripinfo_out),
            "size_bytes": int(tripinfo_out.stat().st_size),
        },
        "metrics": metrics,
        "elapsed_sec": float(time.monotonic() - started),
    }
