"""Check the registered aggregation and reject a missing MP rollout."""

import numpy as np
import pytest

from experiments.tsc_v93_max_pressure_20261030.report import (
    CITY_SCENARIOS, METRICS, index_rows, relative_deltas, value,
)


def make_rows(seeds, methods):
    rows = []
    for method in methods:
        for city, scenarios in CITY_SCENARIOS.items():
            for scenario in scenarios:
                for seed in seeds:
                    waiting = 100.0 if city == "los_angeles" else (30.0 if method == "frozen_selector" else 60.0)
                    row = dict(method=method, city=city, scenario=scenario, seed=seed)
                    row.update({key: waiting for key in METRICS})
                    rows.append(row)
    return rows


def test_city_relative_effect_is_not_ratio_of_macro_means():
    seeds = [11, 22]
    methods = ("frozen_selector", "max_pressure")
    indexed = index_rows(make_rows(seeds, methods), seeds, methods)
    deltas = relative_deltas(indexed, seeds)
    assert np.allclose(deltas["macro"], [-0.25, -0.25])
    pooled_ratio = value(indexed, "frozen_selector", "macro", 11, METRICS[0]) / value(
        indexed, "max_pressure", "macro", 11, METRICS[0]) - 1
    assert pooled_ratio == -0.1875


def test_partial_mp_cohort_cannot_be_reported():
    seeds = list(range(56))
    rows = make_rows(seeds, ("max_pressure",))
    assert len(rows) == 224
    with pytest.raises(ValueError, match="complete method/scenario/seed matrix"):
        index_rows(rows[:-1], seeds, ("max_pressure",))
