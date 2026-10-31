from copy import deepcopy

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_multicity_oof_compatibility_gate import (
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_oof_compatibility_gate_aggregate import (
    aggregate_oof_compatibility_gate,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    _label_free_oof_compatibility_diagnostic,
)


def _summary(mean: float) -> dict:
    return {"mean": mean}


def _diagnostic(trust: float) -> dict:
    chance = 0.5
    agreement = chance + trust * (1.0 - chance)
    return {
        "protocol": "five-fold-oof-policy-compatibility-v1",
        "information_boundary": (
            "b20_model_predictions_on_disjoint_b5_features_and_phase_pressure_"
            "reference_only"
        ),
        "fold_count": 5,
        "group_count": 25,
        "labels_from_scored_groups_used": False,
        "action_agreement_fraction": agreement,
        "chance_action_agreement": chance,
        "chance_corrected_trust_mass": trust,
        "pairwise_rank_agreement_fraction": agreement,
        "mean_normalized_target_regret": 1.0 - trust,
        "q90_normalized_target_regret": 1.0 - trust,
        "mean_normalized_target_margin": 0.5,
        "target_intervention_fraction": 0.5,
        "source_intervention_fraction": 0.5,
    }


def _rows() -> list[dict]:
    rows = []
    for target in EXPECTED_CITY_GROUPS:
        sources = [city for city in EXPECTED_CITY_GROUPS if city != target]
        preferred = "cologne" if target != "cologne" else "atlanta"
        source_means = {}
        source_placebos = {}
        shrunk_means = {}
        shrunk_placebos = {}
        diagnostics = {}
        for source in sources:
            historically_positive = source in {"atlanta", "cologne"}
            full_mean = 0.09 if historically_positive else 0.12
            source_means[source] = _summary(full_mean)
            source_placebos[source] = _summary(full_mean + 0.01)
            shrunk_means[source] = _summary(0.09 if source == preferred else 0.1)
            shrunk_placebos[source] = _summary(0.11)
            diagnostics[source] = _diagnostic(0.8 if source == preferred else 0.2)
        rows.append(
            {
                "protocol": RESULT_PROTOCOL,
                "target_city": target,
                "target_budget": 25,
                "source_city_groups": sources,
                "information_budget": {
                    "evaluation_groups_used_for_fit_or_selection": False,
                    "adaptation_groups_used_for_selector_diagnostics": True,
                    "adaptation_compatibility_scored_group_labels_used": False,
                    "adaptation_compatibility_fold_count": 5,
                },
                "arm_summaries": {
                    "phase_pressure": _summary(0.0),
                    "target_only_causal": _summary(0.1),
                    "pooled_h2oplus": _summary(0.11),
                    "uniform_cfcmt": _summary(0.105),
                    "matched_source_placebo": _summary(0.11),
                },
                "source_arm_summaries": source_means,
                "source_placebo_arm_summaries": source_placebos,
                "compatibility_shrunk_source_arm_summaries": shrunk_means,
                "compatibility_shrunk_placebo_arm_summaries": shrunk_placebos,
                "adaptation_compatibility_diagnostics": {
                    "protocol": "b25-five-fold-oof-policy-compatibility-v1",
                    "evaluation_features_or_labels_used": False,
                    "scored_adaptation_labels_used": False,
                    "adaptation_group_count": 25,
                    "fold_count": 5,
                    "source_arms": diagnostics,
                },
                "inputs": {"source": "frozen"},
            }
        )
    return rows


def test_oof_compatibility_is_chance_corrected_and_regret_aware() -> None:
    folds = (
        {
            "fold_index": 0,
            "policy_rows": (np.array([0, 1]), np.array([2, 3])),
            "references": np.array([True, False, True, False]),
        },
    )
    target = {0: np.array([0.0, 1.0, 1.0, 0.0])}
    source = {0: np.array([0.0, 1.0, 0.0, 1.0])}
    result = _label_free_oof_compatibility_diagnostic(
        target_scores=target,
        source_scores=source,
        folds=folds,
    )
    assert result["action_agreement_fraction"] == 0.5
    assert result["chance_action_agreement"] == 0.5
    assert result["chance_corrected_trust_mass"] == 0.0
    assert result["mean_normalized_target_regret"] == 0.5
    assert result["labels_from_scored_groups_used"] is False


def test_v146_selects_positive_history_source_by_oof_compatibility() -> None:
    result = aggregate_oof_compatibility_gate(_rows())
    assert result["development_gate"]["passed"] is True
    for target, decision in result["city_decisions"].items():
        expected = "cologne" if target != "cologne" else "atlanta"
        assert decision["selected_source"] == expected
        assert decision["selected_trust_mass"] == 0.8
        assert target not in decision["meta_label_cities"]
        for source, history in decision["source_histories"].items():
            assert target not in history["history_targets"]
            assert source not in history["history_targets"]


def test_v146_rejects_scored_fold_or_evaluation_information() -> None:
    rows = _rows()
    changed = deepcopy(rows)
    changed[0]["adaptation_compatibility_diagnostics"][
        "scored_adaptation_labels_used"
    ] = True
    with pytest.raises(ValueError, match="information boundary changed"):
        aggregate_oof_compatibility_gate(changed)
    changed = deepcopy(rows)
    changed[0]["information_budget"][
        "evaluation_groups_used_for_fit_or_selection"
    ] = True
    with pytest.raises(ValueError, match="information boundary changed"):
        aggregate_oof_compatibility_gate(changed)
