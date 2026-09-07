from __future__ import annotations

from pathlib import Path

import numpy as np

from cf_h2o.eval.traffic_signal_multihorizon_latent_screen import (
    _atomic_oof_npz,
)
from cf_h2o.eval.traffic_signal_multihorizon_nested_selection import (
    exclude_outer_seed,
    fit_nested_inner_fold,
)
from cf_h2o.eval.traffic_signal_multihorizon_nested_selection_audit import (
    reconstruct_gate_selection_from_npz,
    summarize_nested_deployment,
)
from cf_h2o.eval.traffic_signal_state_conditioned_latent_diagnostic import (
    BASE_LATENT_MODEL_FAMILY,
)
from cf_h2o.eval.traffic_signal_waiting_aligned_action_ranker_diagnostic import (
    select_ranker_gate,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.spatiotemporal_action_latent import (
    TLS_TIME_PHASE_PAIR_SCOPE,
)


def _contrast() -> MechanismDataset:
    features = []
    groups = []
    references = []
    states = []
    tls = []
    times = []
    target_10 = []
    target_30 = []
    for seed in (11, 22, 33, 44):
        for time_sec in (100.0, 400.0):
            group = f"test:seed{seed}:{int(time_sec)}:tls0"
            for action in range(3):
                features.append([float(action)])
                groups.append(group)
                references.append(action == 0)
                states.append(f"phase{action}")
                tls.append("tls0")
                times.append(time_sec)
                target_10.append((0.0, -1.0, 0.5)[action])
                target_30.append((0.0, 0.5, -1.0)[action])
    targets = {
        "prefix_mean_cost_10s": np.asarray(target_10, dtype=float),
        "prefix_mean_cost_30s": np.asarray(target_30, dtype=float),
    }
    return MechanismDataset(
        feature_names=("action",),
        features=np.asarray(features, dtype=float),
        context_names=("demand",),
        context=np.zeros((len(features), 1), dtype=float),
        priors={name: np.zeros(len(features), dtype=float) for name in targets},
        targets=targets,
        domains=np.asarray(["test"] * len(features)),
        metadata={
            "action_group_ids": groups,
            "is_reference": references,
            "candidate_states": states,
            "row_tls": tls,
            "row_times": times,
        },
    )


def _spec(key: str, target_name: str) -> dict[str, object]:
    return {
        "key": key,
        "family": BASE_LATENT_MODEL_FAMILY,
        "target_name": target_name,
        "config": {
            "scope": TLS_TIME_PHASE_PAIR_SCOPE,
            "shrinkage": 0.0,
            "time_bin_sec": 300,
            "uncertainty_quantile": 0.9,
        },
    }


def _selection() -> dict[str, object]:
    return {
        "gate_candidates": [
            {"risk_multiplier": 0.0, "minimum_context_trust": 0.0}
        ],
        "selection_rule": {
            "minimum_retained_groups_per_city": 1,
            "minimum_retained_seed_fraction": 1.0,
            "maximum_equal_seed_scenario_mean_delta": -0.01,
            "maximum_bootstrap_95pct_upper_mean_delta": 0.0,
            "maximum_worst_seed_mean_delta": 0.0,
            "maximum_harmful_group_fraction": 0.5,
        },
        "bootstrap": {"replicates": 100, "seed": 123},
    }


def test_nested_inner_fold_excludes_outer_and_inner_training_seed() -> None:
    contrast = _contrast()
    reduced = exclude_outer_seed(contrast, outer_seed=11)
    fold = fit_nested_inner_fold(
        outer_seed=11,
        inner_seed=22,
        outer_excluded_contrast=reduced,
        expanded_specs=(
            _spec("base::h10s", "prefix_mean_cost_10s"),
            _spec("base::h30s", "prefix_mean_cost_30s"),
        ),
        scenario="test",
        expected_action_count=3,
    )
    records = [
        row for result in fold["model_results"] for row in result["records"]
    ]

    assert fold["outer_seed_absent_from_inner_dataset"] is True
    assert fold["heldout_absent_from_training"] is True
    assert fold["training_validation_disjoint"] is True
    assert fold["training_seed_count"] == 2
    assert fold["training_group_count"] == 4
    assert fold["validation_group_count"] == 2
    assert {row["simulator_seed"] for row in records} == {22}


def _oof_record(model: str, seed: int, delta: float) -> dict[str, object]:
    return {
        "model_key": model,
        "model_family": "test",
        "target_name": "prefix_mean_cost_10s",
        "scenario": "test",
        "group_id": f"test:seed{seed}:100:tls0",
        "selected_candidate_state": "phase1",
        "reference_candidate_state": "phase0",
        "oracle_candidate_state": "phase1",
        "heldout_seed": seed,
        "simulator_seed": seed,
        "action_count": 3,
        "selected_latent_level": 1,
        "selected_support_count": 3,
        "selected_differs": True,
        "selected_is_oracle": delta < 0.0,
        "predicted_delta": -1.0,
        "selected_uncertainty": 0.1,
        "selected_context_trust": 1.0,
        "selected_context_distance": 0.0,
        "actual_group_normalized_delta": delta,
        "actual_raw_delta": delta,
        "oracle_group_normalized_delta": min(delta, 0.0),
        "normalized_regret_to_oracle": max(delta, 0.0),
    }


def test_nested_sidecar_reconstructs_gate_selection_exactly(tmp_path: Path) -> None:
    seeds = (11, 22, 33)
    specs = (
        {"key": "good", "family": "test"},
        {"key": "bad", "family": "test"},
    )
    records = [
        *[_oof_record("good", seed, -0.2) for seed in seeds],
        *[_oof_record("bad", seed, 0.2) for seed in seeds],
    ]
    path = tmp_path / "inner_oof.npz"
    _atomic_oof_npz(path, records)
    expected = select_ranker_gate(
        records,
        model_specs=specs,
        seeds=seeds,
        selection=_selection(),
    )
    reconstructed = reconstruct_gate_selection_from_npz(
        path,
        model_specs=specs,
        seeds=seeds,
        selection=_selection(),
    )

    assert reconstructed == expected
    assert reconstructed["selected"]["model_key"] == "good"


def test_nested_deployment_summary_applies_original_gate_thresholds() -> None:
    records = [
        {
            "simulator_seed": seed,
            "ranker_accepted": True,
            "actual_group_normalized_delta": -0.2,
            "deployed_delta": -0.2,
        }
        for seed in (11, 22, 33)
    ]
    summary = summarize_nested_deployment(
        records,
        seeds=(11, 22, 33),
        selection_rule=_selection()["selection_rule"],
        bootstrap=_selection()["bootstrap"],
    )

    assert summary["gates"]["passed"] is True
    assert summary["retained_seed_fraction"] == 1.0
    assert np.isclose(summary["mean_delta"], -0.2)
