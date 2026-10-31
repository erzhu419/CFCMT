from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cf_h2o.eval.traffic_signal_external_full_budget_freeze import (
    ADAPTATION_SEEDS,
    BASE_FAMILIES,
    BENCHMARK_FAMILIES,
    EVALUATION_SEED,
    V9_EVALUATION_SEED,
    V9_METHOD_MODEL_PROTOCOL,
    V9_PROTOCOL,
    _artifact_protocols,
    _project_models,
    assign_leave_one_seed_out_folds,
    select_all_city_adaptation_groups,
)
from cf_h2o.eval.traffic_signal_tsc_mechanism_offline_ablation import (
    OfflineScreeningModels,
)


def _groups(scenario: str, counts: tuple[int, int]) -> list[str]:
    return [
        f"{scenario}:seed{seed}:{index}:tls_{index % 4}"
        for seed, count in zip(ADAPTATION_SEEDS, counts, strict=True)
        for index in range(count)
    ]


def test_full_budget_selection_uses_every_valid_group_without_subsampling():
    scenarios = ("jinan_real", "jinan_2000", "jinan_2500")
    bank = {
        scenarios[0]: SimpleNamespace(
            metadata={"action_group_ids": _groups(scenarios[0], (7, 8))}
        ),
        scenarios[1]: SimpleNamespace(
            metadata={"action_group_ids": _groups(scenarios[1], (5, 6))}
        ),
        scenarios[2]: SimpleNamespace(
            metadata={"action_group_ids": _groups(scenarios[2], (9, 9))}
        ),
    }

    selected, audit = select_all_city_adaptation_groups(
        bank, city="jinan", scenarios=scenarios
    )

    expected = {
        group
        for scenario in scenarios
        for group in bank[scenario].metadata["action_group_ids"]
    }
    assert set(selected) == expected
    assert len(selected) == 44
    assert audit["selected_group_count"] == 44
    assert audit["protocol"] == "all-valid-city-adaptation-groups-no-subsampling-v1"


def test_leave_one_seed_out_fold_never_splits_a_simulator_seed():
    scenarios = ("la",)
    records = [
        {
            "group_id": group,
            "simulator_seed": seed,
            "snapshot_time_sec": float(index * 10),
            "tls_id": f"tls_{index % 4}",
        }
        for seed, count in zip(ADAPTATION_SEEDS, (13, 17), strict=True)
        for index, group in enumerate(_groups("la", (count, 0)) if seed == ADAPTATION_SEEDS[0] else _groups("la", (0, count)))
    ]

    result = assign_leave_one_seed_out_folds(records, scenarios=scenarios)

    assert result["fold_group_counts"] == [13, 17]
    for group, fold in result["assignments"].items():
        expected_fold = ADAPTATION_SEEDS.index(
            next(seed for seed in ADAPTATION_SEEDS if f":seed{seed}:" in group)
        )
        assert fold == expected_fold


def test_model_projection_keeps_only_requested_family_contract():
    all_families = tuple(dict.fromkeys((*BASE_FAMILIES, *BENCHMARK_FAMILIES)))
    models = OfflineScreeningModels(
        family_models={name: object() for name in all_families},
        prior_spec=object(),
        objective_modes={name: "advantage" for name in all_families},
        diagnostics={"marker": "shared-full-refit"},
    )

    method = _project_models(models, BASE_FAMILIES)
    baseline = _project_models(models, BENCHMARK_FAMILIES)

    assert tuple(method.family_models) == BASE_FAMILIES
    assert tuple(baseline.family_models) == BENCHMARK_FAMILIES
    assert method.diagnostics == baseline.diagnostics == {
        "marker": "shared-full-refit"
    }


def test_machine_protocol_reserves_a_fresh_confirmation_seed():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "cf_h2o/config/traffic_signal_tsc_v29_external_full_budget_confirmation.json").read_text(
            encoding="utf-8"
        )
    )
    target = payload["target_protocol"]
    assert target["evaluation_seeds"] == [EVALUATION_SEED]
    assert set(target["adaptation_seeds"]).isdisjoint(
        target["diagnostic_only_seeds"] + target["evaluation_seeds"]
    )
    assert target["target_group_budget_per_city"] == "all_available"
    assert target["deployment_model_ensemble_size"] == 1


def test_v9_machine_protocol_uses_fresh_seed_and_distinct_artifact_identity():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (
            root
            / "cf_h2o/config/traffic_signal_tsc_v41_external_v9_full_budget_refit.json"
        ).read_text(encoding="utf-8")
    )
    target = payload["target_protocol"]

    assert target["evaluation_seeds"] == [V9_EVALUATION_SEED]
    assert V9_EVALUATION_SEED != EVALUATION_SEED
    assert set(target["adaptation_seeds"]).isdisjoint(
        target["diagnostic_only_seeds"] + target["evaluation_seeds"]
    )
    assert _artifact_protocols(V9_PROTOCOL)["method_model"] == (
        V9_METHOD_MODEL_PROTOCOL
    )
