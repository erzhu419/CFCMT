from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "cf_h2o/config/traffic_signal_tsc_v118_eth_boston_complete_published_confirmation.json"
)


def test_eth_boston_protocol_freezes_complete_published_not_full_day_scope() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    assert protocol["protocol"] == (
        "tsc-v118-eth-boston-complete-published-independent-confirmation-v1"
    )
    assert protocol["scientific_status"] == (
        "frozen_before_boston_static_inventory_microscopic_or_controller_outcomes"
    )
    assert protocol["claim_scope"] == (
        "independent_complete_published_approximately_12h_external_stress_test_not_full_day"
    )
    source = protocol["source"]
    assert source["city_code"] == "BOS"
    assert source["known_before_freeze"] == {
        "minimum_depart_sec": 7201.78,
        "maximum_depart_sec": 50395.74,
        "basis": "v102_streaming_departure_bounds_only",
    }


def test_eth_boston_protocol_keeps_strict_microscopic_admission() -> None:
    admission = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))[
        "admission"
    ]
    assert admission["sequential_fail_closed"] is True
    assert admission["backend"] == "libsumo"
    assert admission["require_all_published_trips"] is True
    assert admission["allow_route_or_time_sampling"] is False
    assert admission["allow_demand_scaling"] is False
    assert admission["ignore_route_errors"] is False
    assert admission["time_to_teleport_sec"] == -1
    assert admission["max_depart_delay_sec"] == -1
    assert admission["require_exact_vehicle_conservation"] is True
    assert admission["require_zero_collisions"] is True
    assert admission["require_zero_starting_and_ending_teleports"] is True


def test_eth_boston_protocol_freezes_information_budgets_and_confirmation_gate() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    deployment = protocol["deployment"]
    assert deployment["full_network_and_complete_published_demand_active"] is True
    assert deployment["controlled_traffic_light_count"] == 4
    assert deployment["adaptation_seeds"] == [5057, 6067]
    assert deployment["target_transition_group_budget"] == 100

    seeds = protocol["fresh_seeds"]
    assert len(seeds) == 32
    assert len(set(seeds)) == 32
    assert all(10_000_000 <= seed < 100_000_000 for seed in seeds)
    assert not set(seeds).intersection(deployment["adaptation_seeds"])

    arms = protocol["information_arms"]
    assert arms["target_only_b100"]["target_transition_label_budget"] == 100
    assert arms["source_selected_b100"]["target_transition_label_budget"] == 100
    assert arms["source_selected_b0"]["target_transition_label_budget"] == 0
    assert arms["h2oplus_dense_b100"]["target_transition_label_budget"] == 100
    assert arms["h2oplus_dense_b0"]["target_transition_label_budget"] == 0

    gate = protocol["confirmation_gate"]
    assert gate["joint_rule"] == "all_applicable_primary_gates_must_pass"
    assert gate[
        "maximum_95pct_upper_relative_delta_source_b100_vs_h2oplus_b100"
    ] == 0.0
    assert gate[
        "maximum_95pct_upper_relative_delta_source_b100_vs_target_only_b100"
    ] == 0.0
    assert gate[
        "maximum_95pct_upper_relative_delta_source_b100_vs_phase_pressure"
    ] == 0.01
    assert gate["no_failed_seed_exclusion"] is True
