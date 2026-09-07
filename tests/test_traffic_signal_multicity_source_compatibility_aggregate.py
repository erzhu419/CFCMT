from copy import deepcopy

from cf_h2o.eval.traffic_signal_multicity_source_compatibility_aggregate import (
    aggregate_source_compatibility_inventory,
)
from cf_h2o.eval.traffic_signal_multicity_source_compatibility_inventory import (
    RESULT_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_multicity_uniform_source_ensemble import (
    EXPECTED_CITY_GROUPS,
)


def _signature(city: str) -> dict:
    return {
        "protocol": "scenario-equal-label-free-city-covariate-signature-v1",
        "information_boundary": "features_and_static_context_only_no_priors_or_targets",
        "scenario_count": 1,
        "scenarios": [city],
        "context_names": ["context"],
        "context_mean": [1.0],
        "context_std": [0.0],
        "feature_names": ["state"],
        "feature_mean": [1.0],
        "feature_std": [1.0],
        "feature_q25": [0.0],
        "feature_q75": [2.0],
    }


def _rows() -> list[dict]:
    signatures = {city: _signature(city) for city in EXPECTED_CITY_GROUPS}
    rows = []
    for target in EXPECTED_CITY_GROUPS:
        sources = [city for city in EXPECTED_CITY_GROUPS if city != target]
        rows.append(
            {
                "protocol": RESULT_PROTOCOL,
                "target_city": target,
                "target_budget": 25,
                "source_city_groups": sources,
                "information_budget": {
                    "evaluation_groups_used_for_fit_or_selection": False
                },
                "arm_summaries": {"target_only_causal": {"mean": 0.1}},
                "source_arm_summaries": {
                    source: {"mean": 0.09 + 0.001 * index}
                    for index, source in enumerate(sources)
                },
                "source_placebo_arm_summaries": {
                    source: {"mean": 0.11} for source in sources
                },
                "city_covariate_signatures": signatures,
                "inputs": {"source": "frozen"},
            }
        )
    return rows


def test_v144_aggregate_admits_complete_ordered_pair_headroom() -> None:
    result = aggregate_source_compatibility_inventory(_rows())
    assert result["ordered_pair_count"] == 42
    assert result["headroom_admission"]["passed"] is True
    assert result["headroom_admission"]["targets_with_useful_source_headroom"] == 7


def test_v144_aggregate_rejects_target_label_signature() -> None:
    rows = _rows()
    changed = deepcopy(rows[0]["city_covariate_signatures"])
    changed[EXPECTED_CITY_GROUPS[0]]["information_boundary"] = "includes_targets"
    rows[0]["city_covariate_signatures"] = changed
    try:
        aggregate_source_compatibility_inventory(rows)
    except ValueError as exc:
        assert "signature boundary changed" in str(exc)
    else:
        raise AssertionError("V144 accepted a target-label city signature")
