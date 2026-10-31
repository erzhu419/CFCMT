from copy import deepcopy

import pytest

from cf_h2o.eval.traffic_signal_multicity_mechanism_compatibility_gate import (
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_mechanism_compatibility_gate_aggregate import (
    aggregate_mechanism_compatibility_gate,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
    MECHANISM_COMPATIBILITY_NAMES,
    _mechanism_oof_compatibility_diagnostic,
)


def _summary(mean: float) -> dict:
    return {"mean": mean}


def _diagnostic(mass: float) -> dict:
    trusts = {name: mass for name in MECHANISM_COMPATIBILITY_NAMES}
    return {
        "protocol": "five-fold-oof-labeled-mechanism-compatibility-v1",
        "information_boundary": (
            "b20_fixed_parent_mechanism_fits_scored_on_disjoint_b5_"
            "next_state_labels_no_evaluation_groups"
        ),
        "fold_count": 5,
        "group_count": 25,
        "labels_from_scored_groups_used": True,
        "estimator": "fixed-parent-rank0-domain-balanced-residual-v1",
        "mechanism_names": list(MECHANISM_COMPATIBILITY_NAMES),
        "target_only_losses": {
            name: 1.0 for name in MECHANISM_COMPATIBILITY_NAMES
        },
        "source_plus_target_losses": {
            name: 1.0 - mass for name in MECHANISM_COMPATIBILITY_NAMES
        },
        "mechanism_trust": trusts,
        "source_mass": mass,
        "fold_losses": [],
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
            robust = source in {"atlanta", "cologne"}
            source_means[source] = _summary(0.09 if robust else 0.12)
            source_placebos[source] = _summary(0.11)
            shrunk_means[source] = _summary(0.09 if robust else 0.12)
            shrunk_placebos[source] = _summary(0.11)
            diagnostics[source] = _diagnostic(
                0.8 if source == preferred else 0.2
            )
        rows.append(
            {
                "protocol": RESULT_PROTOCOL,
                "target_city": target,
                "target_budget": 25,
                "source_city_groups": sources,
                "information_budget": {
                    "evaluation_groups_used_for_fit_or_selection": False,
                    "adaptation_groups_used_for_selector_diagnostics": True,
                    "adaptation_mechanism_scored_group_labels_used": True,
                    "adaptation_mechanism_fold_count": 5,
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
                "mechanism_shrunk_source_arm_summaries": shrunk_means,
                "mechanism_shrunk_placebo_arm_summaries": shrunk_placebos,
                "adaptation_mechanism_diagnostics": {
                    "protocol": (
                        "b25-five-fold-oof-labeled-mechanism-compatibility-v1"
                    ),
                    "evaluation_features_or_labels_used": False,
                    "scored_adaptation_labels_used": True,
                    "adaptation_group_count": 25,
                    "fold_count": 5,
                    "mechanism_names": list(MECHANISM_COMPATIBILITY_NAMES),
                    "source_arms": diagnostics,
                },
                "inputs": {"source": "frozen"},
            }
        )
    return rows


def test_mechanism_oof_trust_uses_labeled_loss_improvement() -> None:
    folds = tuple(
        {"heldout_groups": tuple(f"g{index}-{row}" for row in range(5))}
        for index in range(5)
    )
    target = {
        index: {name: 4.0 for name in MECHANISM_COMPATIBILITY_NAMES}
        for index in range(5)
    }
    source = {
        index: {
            name: 2.0 if mechanism_index < 3 else 6.0
            for mechanism_index, name in enumerate(MECHANISM_COMPATIBILITY_NAMES)
        }
        for index in range(5)
    }
    result = _mechanism_oof_compatibility_diagnostic(
        target_losses=target,
        source_losses=source,
        folds=folds,
    )
    assert result["labels_from_scored_groups_used"] is True
    assert list(result["mechanism_trust"].values()) == [0.5, 0.5, 0.5, 0.0, 0.0]
    assert result["source_mass"] == pytest.approx(0.3)


def test_v148_selects_robust_history_source_by_mechanism_mass() -> None:
    result = aggregate_mechanism_compatibility_gate(_rows())
    assert result["development_gate"]["passed"] is True
    for target, decision in result["city_decisions"].items():
        expected = "cologne" if target != "cologne" else "atlanta"
        assert decision["selected_source"] == expected
        assert decision["selected_mechanism_mass"] == 0.8
        assert target not in decision["meta_label_cities"]
        for source, history in decision["source_histories"].items():
            assert target not in history["history_targets"]
            assert source not in history["history_targets"]


def test_v148_uses_exact_source_null_without_robust_history() -> None:
    rows = _rows()
    for row in rows:
        for source in row["mechanism_shrunk_source_arm_summaries"]:
            row["mechanism_shrunk_source_arm_summaries"][source]["mean"] = 0.12
            row["mechanism_shrunk_placebo_arm_summaries"][source]["mean"] = 0.11
    result = aggregate_mechanism_compatibility_gate(rows)
    assert result["development_gate"]["passed"] is False
    assert all(
        decision["selected_source"] is None
        for decision in result["city_decisions"].values()
    )


def test_v148_rejects_mislabeled_information_boundary() -> None:
    rows = _rows()
    changed = deepcopy(rows)
    changed[0]["adaptation_mechanism_diagnostics"][
        "scored_adaptation_labels_used"
    ] = False
    with pytest.raises(ValueError, match="information boundary changed"):
        aggregate_mechanism_compatibility_gate(changed)
    changed = deepcopy(rows)
    changed[0]["information_budget"][
        "evaluation_groups_used_for_fit_or_selection"
    ] = True
    with pytest.raises(ValueError, match="information boundary changed"):
        aggregate_mechanism_compatibility_gate(changed)
