from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from cf_h2o.eval.traffic_signal_cross_city_meta_source_utility import (
    METHOD_PROTOCOL,
    RESULT_PROTOCOL,
    UTILITY_GATE_PROTOCOL,
    _MetaDomain,
    _meta_source_audit,
    _records,
    aggregate_results,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    SOURCE_PRIOR_STRENGTH,
)
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    STATE_UTILITY_FEATURES,
    source_blind_candidate_scores,
)


def _result(city: str, *, admitted: bool) -> dict:
    rigid = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    source = {
        str(seed): value - (0.02 if admitted else 0.0)
        for seed, value in rigid.items()
    }
    placebo = {
        str(seed): value + (0.01 if admitted else 0.0)
        for seed, value in source.items()
    }
    blind = {
        str(seed): value + (0.015 if admitted else 0.0)
        for seed, value in source.items()
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": 100,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
            "selector": {"admitted": admitted},
        },
        "information_budget": {"target_utility_label_count": 0},
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_meta_source_pairwise": {"seed_values": source},
            "selected_meta_matched_placebo": {"seed_values": placebo},
            "selected_meta_source_blind": {"seed_values": blind},
        },
    }


def test_v151a_aggregate_passes_two_source_specific_target_cities() -> None:
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert result["source_admission_count"] == 2
    assert result["development_gate"]["decision"] == (
        "authorize_v151a_source_meta_utility_closed_loop_development"
    )


def test_v151a_aggregate_rejects_source_blind_tie() -> None:
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]
    for index in range(2):
        rows[index]["arm_summaries"]["selected_meta_source_blind"] = deepcopy(
            rows[index]["arm_summaries"]["selected_meta_source_pairwise"]
        )

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["checks"][
        "mean_selected_effect_beats_source_blind"
    ] is False


def test_v151a_aggregate_rejects_target_utility_labels() -> None:
    rows = [_result(city, admitted=False) for city in EXPECTED_CITY_GROUPS]
    rows[0]["information_budget"]["target_utility_label_count"] = 1

    try:
        aggregate_results(rows)
    except ValueError as exc:
        assert "seven valid B100 results" in str(exc)
    else:
        raise AssertionError("V151A accepted target utility labels")


def _meta_prediction(city: str) -> SimpleNamespace:
    group_count = 12
    features = np.zeros((2 * group_count, len(STATE_UTILITY_FEATURES)))
    groups = np.repeat([f"{city}:g{index}" for index in range(group_count)], 2)
    contrast = MechanismDataset(
        feature_names=STATE_UTILITY_FEATURES,
        features=features,
        context_names=("context",),
        context=np.zeros((2 * group_count, 1)),
        priors={},
        targets={},
        domains=np.full(2 * group_count, city, dtype=object),
        metadata={
            "action_group_ids": groups,
            "is_reference": np.tile([False, True], group_count),
        },
    )
    rigid = np.tile([0.0, 1.0], group_count)
    candidate = np.tile([1.0, 0.0], group_count)
    return SimpleNamespace(
        contrast=contrast,
        rigid_score=rigid,
        candidate_scores={"source|queue_service": candidate},
        placebo_scores={"source|queue_service": candidate.copy()},
        actual=np.tile([0.0, -1.0], group_count),
        groups=groups,
        references=np.tile([False, True], group_count),
    )


def test_v151a_meta_audit_runs_six_leave_city_out_folds() -> None:
    domains = []
    for fold_id, city in enumerate(EXPECTED_CITY_GROUPS[:6]):
        prediction = _meta_prediction(city)
        blind = source_blind_candidate_scores(
            prediction.rigid_score,
            prediction.candidate_scores,
        )
        domains.append(
            _MetaDomain(
                city=city,
                prediction=prediction,
                source_records=_records(
                    prediction,
                    prediction.candidate_scores,
                    city=city,
                    fold_id=fold_id,
                ),
                placebo_records=_records(
                    prediction,
                    prediction.placebo_scores,
                    city=city,
                    fold_id=fold_id,
                ),
                blind_records=_records(
                    prediction,
                    blind,
                    city=city,
                    fold_id=fold_id,
                ),
                source_order=("source",),
            )
        )

    audit = _meta_source_audit(domains)

    assert audit["fold_count"] == 6
    assert [row["heldout_utility_city"] for row in audit["folds"]] == list(
        EXPECTED_CITY_GROUPS[:6]
    )
    assert audit["mean_gain_vs_rigid"] > 0.0
    assert audit["mean_gain_vs_source_blind"] == 0.0
    assert audit["passed"] is False

    parallel = _meta_source_audit(domains, workers=6)
    assert parallel == audit
