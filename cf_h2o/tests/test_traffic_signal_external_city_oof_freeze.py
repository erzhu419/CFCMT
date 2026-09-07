from __future__ import annotations

from collections import Counter

import pytest

from cf_h2o.eval.traffic_signal_external_city_oof_freeze import (
    ADAPTATION_SEEDS,
    _balanced_quotas,
    assign_scenario_seed_stratified_folds,
)


def _records():
    scenarios = ("jinan_real", "jinan_2000", "jinan_2500")
    quotas = (34, 33, 33)
    records = []
    for scenario_index, (scenario, quota) in enumerate(zip(scenarios, quotas, strict=True)):
        first_seed_count = quota // 2 + int(quota % 2 and scenario_index % 2 == 0)
        counts = (first_seed_count, quota - first_seed_count)
        for seed, count in zip(ADAPTATION_SEEDS, counts, strict=True):
            for index in range(count):
                records.append(
                    {
                        "group_id": f"{scenario}:seed{seed}:{index}:tls_{index % 12}",
                        "simulator_seed": seed,
                        "snapshot_time_sec": float(index * 10),
                        "tls_id": f"tls_{index % 12}",
                    }
                )
    return scenarios, records


def test_balanced_quotas_preserve_order_and_total():
    assert _balanced_quotas(100, ("a", "b", "c")) == {
        "a": 34,
        "b": 33,
        "c": 33,
    }


def test_external_fold_assignment_balances_scenario_and_seed_strata():
    scenarios, records = _records()
    result = assign_scenario_seed_stratified_folds(records, scenarios=scenarios)

    assert result["fold_group_counts"] == [20] * 5
    assert len(result["assignments"]) == 100
    for counts in result["scenario_fold_group_counts"].values():
        assert max(counts) - min(counts) <= 1
    for rows in result["scenario_seed_fold_group_counts"].values():
        for counts in rows.values():
            assert max(counts) - min(counts) <= 1

    assigned = Counter(result["assignments"].values())
    assert assigned == Counter({fold: 20 for fold in range(5)})


def test_external_fold_assignment_rejects_unidentifiable_scenario():
    scenarios, records = _records()
    records[0] = {**records[0], "group_id": "unknown:seed5057:0:tls_0"}
    with pytest.raises(ValueError, match="identify one external scenario"):
        assign_scenario_seed_stratified_folds(records, scenarios=scenarios)
