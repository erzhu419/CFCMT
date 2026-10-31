from __future__ import annotations

import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_external_policy_horizon_expanded_development import (
    active_candidate_keys,
    all_rollout_identities,
)
from scripts.cluster.run_tsc_external_policy_horizon_expanded_development_shard import (
    shard_identities,
)
from scripts.cluster.audit_tsc_external_policy_horizon_expanded_development import (
    select_city,
)


EXPANDED = Path(
    "cf_h2o/config/traffic_signal_tsc_v36_external_policy_horizon_expanded_development.json"
)
ADAPTATION = Path(
    "cf_h2o/config/traffic_signal_tsc_v34_external_policy_horizon_adaptation.json"
)


def test_expanded_development_matrix_is_frozen():
    protocol = json.loads(EXPANDED.read_text(encoding="utf-8"))
    adaptation = json.loads(ADAPTATION.read_text(encoding="utf-8"))
    for city in protocol["city_scenarios"]:
        assert len(active_candidate_keys(protocol, adaptation, city)) == 27
    identities = all_rollout_identities(protocol, adaptation)
    assert len(identities) == 864
    assert len(set(identities)) == 864
    assert all(identity[3] != "phase_pressure" for identity in identities)


def test_validation_and_prospective_seeds_are_disjoint_from_development():
    protocol = json.loads(EXPANDED.read_text(encoding="utf-8"))
    development = set(protocol["expanded_development_grid"]["additional_seeds"])
    development.update(
        protocol["expanded_development_grid"]["original_adaptation_seeds"]
    )
    validation = set(protocol["new_untouched_validation"]["seeds"])
    prospective = set(protocol["prospective_confirmation_reservation"]["seeds"])
    assert len(validation) == 16
    assert development.isdisjoint(validation)
    assert development.isdisjoint(prospective)
    assert validation.isdisjoint(prospective)


def test_expanded_development_splits_evenly_across_six_nodes():
    protocol = json.loads(EXPANDED.read_text(encoding="utf-8"))
    adaptation = json.loads(ADAPTATION.read_text(encoding="utf-8"))
    shards = [
        shard_identities(protocol, adaptation, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    assert [len(shard) for shard in shards] == [144] * 6
    flattened = [identity for shard in shards for identity in shard]
    assert len(flattened) == len(set(flattened)) == 864


def _paired_rows(*, executed_overrides: int = 1, final_delta: float = -0.1):
    rows = []
    for seed in range(10):
        delta = final_delta if seed == 9 else -0.1
        rows.append(
            {
                "city": "test_city",
                "scenario": "test_scenario",
                "seed": seed,
                "candidate": "candidate",
                "method_mean_waiting_time": 100.0 * (1.0 + delta),
                "phase_mean_waiting_time": 100.0,
                "executed_overrides": executed_overrides,
                "method_safety": {
                    "ending_teleports": 0,
                    "collision_incidents": 0,
                },
                "phase_safety": {
                    "ending_teleports": 0,
                    "collision_incidents": 0,
                },
            }
        )
    return rows


def test_ten_seed_selector_enforces_worst_seed_and_intervention_gates():
    protocol = json.loads(EXPANDED.read_text(encoding="utf-8"))
    rule = protocol["expanded_development_grid"]["selection_rule"]
    bootstrap = protocol["expanded_development_grid"]["bootstrap"] | {
        "replicates": 200
    }
    candidate = {
        "key": "candidate",
        "family": "cfcmt_pressure_regularized_direct",
        "blend_weight": 0.25,
        "risk_multiplier": 0.0,
        "min_context_trust": 0.25,
        "cooldown_intervals": 6,
    }
    selected = select_city(
        city="test_city",
        candidates=[candidate],
        paired_rows=_paired_rows(),
        scenarios=["test_scenario"],
        seeds=list(range(10)),
        selection_rule=rule,
        bootstrap=bootstrap,
    )
    assert selected["selected"]["key"] == "candidate"

    worst_seed_failure = select_city(
        city="test_city",
        candidates=[candidate],
        paired_rows=_paired_rows(final_delta=0.06),
        scenarios=["test_scenario"],
        seeds=list(range(10)),
        selection_rule=rule,
        bootstrap=bootstrap,
    )
    assert worst_seed_failure["selected"]["key"] == "phase_pressure"

    intervention_failure = select_city(
        city="test_city",
        candidates=[candidate],
        paired_rows=_paired_rows(executed_overrides=0),
        scenarios=["test_scenario"],
        seeds=list(range(10)),
        selection_rule=rule,
        bootstrap=bootstrap,
    )
    assert intervention_failure["selected"]["key"] == "phase_pressure"
