from cf_h2o.eval.traffic_signal_anchored_pairwise_selection import (
    ANCHOR_KEY,
    CANDIDATE_KEYS,
    DEVELOPMENT_CITY_GROUPS,
    constant_candidate_key,
    summarize_anchored_development,
)


def _rows(candidate_regrets: dict[str, float]):
    rows = {}
    selected = constant_candidate_key(0.5)
    for city in DEVELOPMENT_CITY_GROUPS:
        rows[city] = {
            candidate: {
                "regret": (
                    0.2
                    if candidate == ANCHOR_KEY
                    else candidate_regrets.get(city, 0.25)
                    if candidate == selected
                    else 0.3
                ),
                "mean_alpha": 0.0 if candidate == ANCHOR_KEY else 0.5,
            }
            for candidate in CANDIDATE_KEYS
        }
    return rows


def test_anchored_development_passes_stable_global_candidate() -> None:
    result = summarize_anchored_development(
        _rows({city: 0.15 for city in DEVELOPMENT_CITY_GROUPS})
    )

    assert result["passed"] is True
    assert result["global_selection"]["selected_candidate"] == (
        constant_candidate_key(0.5)
    )
    assert result["loco"]["improved_city_count"] == 7


def test_anchored_development_rejects_brittle_candidate() -> None:
    regressions = {
        city: (0.15 if index < 4 else 0.23)
        for index, city in enumerate(DEVELOPMENT_CITY_GROUPS)
    }
    result = summarize_anchored_development(_rows(regressions))

    assert result["passed"] is False
    assert result["global_selection"]["selected_candidate"] == ANCHOR_KEY
