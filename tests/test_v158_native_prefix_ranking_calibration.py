from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cf_h2o.eval import traffic_signal_v158_native_prefix_ranking_calibration as v158
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "cf_h2o/config/traffic_signal_tsc_v158_native_prefix_ranking_calibration.json"
)


def _dataset(group_count: int = 4) -> MechanismDataset:
    rows = group_count * 8
    feature_names = (
        "total_q",
        "switch_indicator",
        *v158.RAW_PREDICTION_FEATURES,
    )
    features = np.zeros((rows, len(feature_names)), dtype=float)
    groups: list[str] = []
    references: list[int] = []
    is_reference: list[bool] = []
    seeds: list[int] = []
    scenarios: list[str] = []
    states: list[str] = []
    target = np.zeros(rows, dtype=float)
    for group_index in range(group_count):
        offset = group_index * 8
        reference = offset + 2
        features[offset : offset + 8, 0] = float(group_index + 1)
        features[offset : offset + 8, 1] = np.arange(8) != 0
        features[offset : offset + 8, 2:] = (
            np.arange(8)[:, None] * np.arange(1, 10)[None, :] / 10.0
        )
        target[offset : offset + 8] = np.linspace(0.4, -0.3, 8)
        target[reference] = 0.0
        groups.extend([f"group-{group_index}"] * 8)
        references.extend([reference] * 8)
        is_reference.extend(index == 2 for index in range(8))
        seeds.extend([100 + group_index] * 8)
        scenarios.extend(["jinan_3x4_real"] * 8)
        states.extend(f"state-{index}" for index in range(8))
    return MechanismDataset(
        feature_names=feature_names,
        features=features,
        context_names=("context",),
        context=np.arange(rows, dtype=float).reshape(-1, 1),
        priors={"interval_cost": np.zeros(rows)},
        targets={
            "interval_cost": target,
            "absolute_cost": target + 2.0,
        },
        domains=np.asarray(scenarios, dtype=str),
        metadata={
            "action_group_ids": groups,
            "is_reference": is_reference,
            "reference_rows": references,
            "row_tls": ["tls"] * rows,
            "row_times": [300.0] * rows,
            "candidate_states": states,
            "row_seeds": seeds,
            "row_scenarios": scenarios,
            "action_group_count": group_count,
        },
    )


def test_frozen_protocol_validates_exact_b100_counts():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    v158.validate_protocol(protocol)
    changed = json.loads(json.dumps(protocol))
    changed["development"]["action_groups"] = 99
    with pytest.raises(ValueError, match="frozen protocol changed"):
        v158.validate_protocol(changed)


def test_probe_binds_reference_index_to_selected_tls(monkeypatch):
    probe = v158.DevelopmentProbeOriginator(
        api=object(),
        models={arm: object() for arm in v158.BASE_ARMS},
        scenario="jinan_3x4_real",
        seed=180314,
        bins=((300, 600),),
    )
    probe._slot = 0
    probe._time = 300
    probe._expected = ("tls_a", "tls_b")
    probe._executors = {}
    monkeypatch.setattr(v158, "physical_snapshot", lambda *_args: {"snapshot": True})
    monkeypatch.setattr(
        v158,
        "_prediction_vectors",
        lambda _model, data: {
            field: np.zeros(data.size, dtype=float) for field in v158.PREDICTION_FIELDS
        },
    )

    def fake_decision(_model, data, *, reference_index, arm):
        tls = data.metadata["row_tls"][0]
        proposed = 1 if tls == "tls_a" and arm == "uniform_source" else reference_index
        return v158.V157CActionDecision(
            reference_index=reference_index,
            proposed_index=proposed,
            selected_index=proposed,
            learned_differs=proposed != reference_index,
            eligible=proposed != reference_index,
            priority=2.0 if tls == "tls_a" else 1.0,
            rejection=None,
            arm=arm,
            tls_id=tls,
            time_sec=300.0,
            proposed_state=data.metadata["candidate_states"][proposed],
            selected_state=data.metadata["candidate_states"][proposed],
            reference_state=data.metadata["candidate_states"][reference_index],
            predicted_score=-1.0,
            reference_score=0.0,
        )

    monkeypatch.setattr(v158, "score_model_decision", fake_decision)
    base = _dataset(1)
    datasets = []
    for tls in ("tls_a", "tls_b"):
        datasets.append(
            MechanismDataset(
                feature_names=base.feature_names,
                features=base.features,
                context_names=base.context_names,
                context=base.context,
                priors=base.priors,
                targets=base.targets,
                domains=base.domains,
                metadata={
                    **base.metadata,
                    "row_tls": [tls] * 8,
                    "row_times": [300.0] * 8,
                    "candidate_states": [f"{tls}-{index}" for index in range(8)],
                },
            )
        )
    probe.select(datasets[0], reference_index=2)
    probe.select(datasets[1], reference_index=0)
    assert probe.selected[0]["tls_id"] == "tls_a"
    assert probe.selected[0]["reference_index"] == 2


def test_group_subset_reindexes_reference_rows():
    subset = v158._subset_groups(_dataset(2), {"group-1"})
    assert subset.size == 8
    assert subset.metadata["reference_rows"] == [2] * 8
    assert subset.metadata["is_reference"] == [False, False, True, False, False, False, False, False]


def test_capacity_matched_arms_fit_with_the_same_feature_count():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    data = _dataset(8)
    diagnostics = {}
    for arm in v158.CALIBRATED_ARMS:
        model, diagnostics[arm], arm_data = v158._fit_calibrator(protocol, data, arm)
        prediction = model.predict(arm_data)["control_cost"]["mean"]
        assert prediction.shape == (data.size,)
        assert np.all(np.isfinite(prediction))
        assert arm_data.feature_names[-6:] == v158.META_ACTION_FEATURES
    assert {row["action_feature_count"] for row in diagnostics.values()} == {7}


def test_zero_threshold_and_no_stay_override_preserve_pp_switch():
    data = _dataset(1)
    arm_data = v158._arm_dataset(data, "source_native")
    scores = np.asarray([-1.0, 0.2, 0.0, 0.3, 0.4, 0.5, 0.6, 0.7])
    selected = v158._select_row(
        arm_data,
        scores,
        np.arange(8),
        prohibit_stay_override=True,
    )
    assert selected == 2
    scores[1] = -2.0
    selected = v158._select_row(
        arm_data,
        scores,
        np.arange(8),
        prohibit_stay_override=True,
    )
    assert selected == 1


def test_gate_requires_source_to_beat_all_three_controls():
    protocol = json.loads(CONFIG.read_text(encoding="utf-8"))
    passing = {
        name: {
            "mean_difference": -0.001,
            "paired_bootstrap_95": [-0.002, -0.0001],
            "improved_seeds": 8,
        }
        for name in (
            "source_minus_target",
            "source_minus_placebo",
            "source_minus_phase_pressure",
        )
    }
    assert v158._gate(protocol, passing)["passed"] is True
    passing["source_minus_placebo"]["paired_bootstrap_95"][1] = 0.0001
    assert v158._gate(protocol, passing)["passed"] is False
