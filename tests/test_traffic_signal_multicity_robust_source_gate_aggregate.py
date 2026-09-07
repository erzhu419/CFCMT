from copy import deepcopy

import pytest

from cf_h2o.eval.traffic_signal_multicity_robust_source_gate import RESULT_PROTOCOL
from cf_h2o.eval.traffic_signal_multicity_robust_source_gate_aggregate import (
    aggregate_robust_source_gate,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
)


def _summary(mean: float) -> dict:
    return {"mean": mean}


def _diagnostic(fraction: float) -> dict:
    return {
        "protocol": "adaptation-action-reference-intervention-v1",
        "information_boundary": (
            "fitted_b25_model_predictions_and_phase_pressure_reference_only"
        ),
        "group_count": 25,
        "intervention_fraction": fraction,
    }


def _rows() -> list[dict]:
    rows = []
    stable_source = "cologne"
    fallback_source = "atlanta"
    high_intervention_target = "atlanta"
    for target in EXPECTED_CITY_GROUPS:
        sources = [city for city in EXPECTED_CITY_GROUPS if city != target]
        source_means = {}
        source_placebos = {}
        diagnostics = {
            "target_only": _diagnostic(0.4),
            "h2oplus": _diagnostic(0.4),
        }
        for source in sources:
            if source == stable_source or (
                target == stable_source and source == fallback_source
            ):
                mean = 0.09
            else:
                mean = 0.12
            source_means[source] = _summary(mean)
            source_placebos[source] = _summary(mean + 0.01)
            if target == high_intervention_target:
                fraction = 0.95
            elif target == stable_source and source == fallback_source:
                fraction = 0.1
            else:
                fraction = 0.2 if source == stable_source else 0.5
            diagnostics[source] = _diagnostic(fraction)
        rows.append(
            {
                "protocol": RESULT_PROTOCOL,
                "target_city": target,
                "target_budget": 25,
                "source_city_groups": sources,
                "information_budget": {
                    "evaluation_groups_used_for_fit_or_selection": False,
                    "adaptation_groups_used_for_selector_diagnostics": True,
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
                "adaptation_selector_diagnostics": {
                    "protocol": "b25-adaptation-only-label-free-policy-diagnostics-v1",
                    "evaluation_features_or_labels_used": False,
                    "adaptation_group_count": 25,
                    "arms": diagnostics,
                },
                "inputs": {"source": "frozen"},
            }
        )
    return rows


def test_v145_gate_uses_null_robust_history_and_fallback_stages() -> None:
    result = aggregate_robust_source_gate(_rows())
    assert result["development_gate"]["passed"] is True
    assert (
        result["city_decisions"]["atlanta"]["selection_stage"]
        == "source_null_high_intervention_veto"
    )
    assert (
        result["city_decisions"]["cologne"]["selection_stage"]
        == "minimum_intervention_fallback"
    )
    assert result["city_decisions"]["cologne"]["selected_source"] == "atlanta"
    assert (
        result["city_decisions"]["hangzhou"]["selection_stage"]
        == "robust_history"
    )
    for target, decision in result["city_decisions"].items():
        assert target not in decision["meta_label_cities"]
        for source, history in decision["source_histories"].items():
            assert target not in history["history_targets"]
            assert source not in history["history_targets"]


def test_v145_gate_rejects_evaluation_selector_information() -> None:
    rows = _rows()
    changed = deepcopy(rows)
    changed[0]["adaptation_selector_diagnostics"][
        "evaluation_features_or_labels_used"
    ] = True
    with pytest.raises(ValueError, match="information boundary changed"):
        aggregate_robust_source_gate(changed)
