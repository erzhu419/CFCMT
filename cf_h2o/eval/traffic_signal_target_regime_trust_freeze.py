"""Full-fit freeze for the target-regime trust layer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import _atomic_json, _sha256
from cf_h2o.traffic_signal.target_regime_trust import (
    TargetRegimeTrustModel,
    fit_target_regime_trust,
)


ARTIFACT_PROTOCOL = "cfcmt-target-regime-trust-artifact-v1"
FREEZE_PROTOCOL = "tsc-v80r76-target-regime-trust-full-fit-freeze-v1"


def build_full_fit_artifact(
    diagnostic_result: Mapping[str, Any],
    *,
    city: str,
    scenarios: Sequence[str],
    seeds: Sequence[int],
    l2: float,
) -> dict[str, Any]:
    if (
        diagnostic_result.get("decision")
        != "authorize_target_regime_trust_full_fit_freeze_only"
        or diagnostic_result.get("advance_gate", {}).get("passed") is not True
    ):
        raise ValueError("target regime diagnostic did not authorize a full fit")
    rows = [dict(row) for row in diagnostic_result["source_rows"]]
    expected = {
        (int(seed), str(scenario)) for seed in seeds for scenario in scenarios
    }
    indexed = {
        (int(row["seed"]), str(row["scenario"])): row for row in rows
    }
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("target regime full-fit source matrix changed")
    model = fit_target_regime_trust(
        [float(row["phase_mean_queue_per_lane"]) for row in rows],
        [float(row["relative_delta"]) for row in rows],
        l2=float(l2),
    )
    scenario_rows = {}
    for scenario in scenarios:
        feature = float(
            np.mean(
                [
                    float(indexed[(int(seed), str(scenario))]["phase_mean_queue_per_lane"])
                    for seed in seeds
                ]
            )
        )
        prediction = model.predict_relative_delta(feature)
        scenario_rows[str(scenario)] = {
            "historical_phase_mean_queue_per_lane": feature,
            "history_source_seeds": [int(seed) for seed in seeds],
            "predicted_relative_delta": prediction,
            "trust_local_controller": bool(prediction < 0.0),
        }
    decisions = [
        bool(row["trust_local_controller"]) for row in scenario_rows.values()
    ]
    if not any(decisions) or all(decisions):
        raise ValueError("target regime full fit did not produce a selective gate")
    return {
        "protocol": ARTIFACT_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "city": str(city),
        "classification": (
            "target_simulator_labeled_full_horizon_few_shot_regime_adaptation"
        ),
        "model": model.to_dict(),
        "scenario_history": scenario_rows,
        "scenario_id_is_model_feature": False,
        "scenario_id_role": "lookup_key_for_frozen_target_history_summary_only",
        "heldout_or_validation_metrics_used": False,
        "runtime_rule": (
            "use local CFCMT only when the frozen target-history regime model "
            "predicts a negative waiting-time relative delta; otherwise execute "
            "PhasePressure exactly"
        ),
    }


def write_full_fit_freeze(
    *,
    artifact: Mapping[str, Any],
    artifact_path: Path,
    freeze_path: Path,
    protocol_path: Path,
    diagnostic_path: Path,
) -> dict[str, Any]:
    if artifact_path.exists() or freeze_path.exists():
        raise FileExistsError("refusing to overwrite target regime trust freeze")
    _atomic_json(artifact_path, dict(artifact))
    loaded = json.loads(artifact_path.read_text(encoding="utf-8"))
    TargetRegimeTrustModel.from_dict(loaded["model"])
    if loaded.get("protocol") != ARTIFACT_PROTOCOL:
        raise ValueError("target regime artifact round trip failed")
    payload = {
        "protocol": FREEZE_PROTOCOL,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "decision": "authorize_untouched_validation_protocol_freeze_only",
        "protocol_spec": {
            "path": str(Path(protocol_path).resolve()),
            "sha256": _sha256(protocol_path),
        },
        "diagnostic": {
            "path": str(Path(diagnostic_path).resolve()),
            "sha256": _sha256(diagnostic_path),
        },
        "artifact": {
            "path": str(Path(artifact_path).resolve()),
            "sha256": _sha256(artifact_path),
            "size_bytes": int(artifact_path.stat().st_size),
            "round_trip_passed": True,
        },
        "active_scenarios": sorted(
            scenario
            for scenario, row in loaded["scenario_history"].items()
            if row["trust_local_controller"]
        ),
        "fallback_scenarios": sorted(
            scenario
            for scenario, row in loaded["scenario_history"].items()
            if not row["trust_local_controller"]
        ),
    }
    _atomic_json(freeze_path, payload)
    return payload


def load_frozen_regime_trust(
    *, artifact_path: Path, expected_sha256: str
) -> tuple[dict[str, Any], TargetRegimeTrustModel]:
    if _sha256(artifact_path) != str(expected_sha256):
        raise ValueError("target regime trust artifact hash changed")
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    if (
        payload.get("protocol") != ARTIFACT_PROTOCOL
        or payload.get("scenario_id_is_model_feature") is not False
        or payload.get("heldout_or_validation_metrics_used") is not False
    ):
        raise ValueError("target regime trust artifact contract changed")
    return payload, TargetRegimeTrustModel.from_dict(payload["model"])
