from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import scripts.cluster.run_tsc_source_city_component_shard as shard_module
from scripts.cluster.audit_tsc_target_offline_source_fresh_confirmation import (
    _target_anchor_contract,
)
from scripts.cluster.run_tsc_source_city_component_shard import (
    CANDIDATE_PROTOCOL,
    all_identities,
    load_candidate_specs,
    shard_identities,
)


def test_candidate_specs_require_convex_weights(tmp_path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "protocol": CANDIDATE_PROTOCOL,
                "candidates": [
                    {
                        "key": "bad",
                        "weights_by_city": {
                            "los_angeles": {"a": 0.6, "b": 0.5},
                            "jinan": {},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not convex"):
        load_candidate_specs(path)


def test_component_shards_partition_exact_identity_grid() -> None:
    expected = set(all_identities((11, 13), ("target_only", "source_a")))
    observed = set()
    for shard_index in range(3):
        shard = set(
            shard_identities(
                (11, 13),
                ("target_only", "source_a"),
                shard_index=shard_index,
                shard_count=3,
            )
        )
        assert not observed & shard
        observed |= shard

    assert observed == expected


def test_component_shards_can_target_a_strict_scenario_subset() -> None:
    scenarios = ("jinan_3x4_real", "jinan_3x4_real_2500")
    expected = set(
        all_identities((11, 13), ("source_a",), scenarios=scenarios)
    )
    observed = set()
    for shard_index in range(3):
        observed.update(
            shard_identities(
                (11, 13),
                ("source_a",),
                shard_index=shard_index,
                shard_count=3,
                scenarios=scenarios,
            )
        )

    assert observed == expected
    assert {identity[0] for identity in observed} == set(scenarios)


def test_confirmation_candidates_must_share_city_guards(tmp_path) -> None:
    path = tmp_path / "candidates.json"
    common = {
        "los_angeles": None,
        "jinan": {
            "risk_multiplier": 0.5,
            "min_context_trust": 0.0,
            "margin": 0.0,
            "max_relative_rule_gap": 1.0,
        },
    }
    changed = json.loads(json.dumps(common))
    changed["jinan"]["risk_multiplier"] = 0.25
    path.write_text(
        json.dumps(
            {
                "protocol": CANDIDATE_PROTOCOL,
                "candidates": [
                    {
                        "key": "baseline",
                        "weights_by_city": {"los_angeles": {}, "jinan": {}},
                        "guard_by_city": common,
                    },
                    {
                        "key": "source",
                        "weights_by_city": {
                            "los_angeles": {},
                            "jinan": {"atlanta": 1.0},
                        },
                        "guard_by_city": changed,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="share city guards"):
        load_candidate_specs(path)


def test_phase_pressure_baseline_is_explicit_and_unguarded(tmp_path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {
                "protocol": CANDIDATE_PROTOCOL,
                "candidates": [
                    {
                        "key": "pressure_rule",
                        "baseline_policy": "phase_pressure",
                        "weights_by_city": {"los_angeles": {}, "jinan": {}},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    candidates = load_candidate_specs(path)
    assert candidates[0]["baseline_policy"] == "phase_pressure"

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["candidates"][0]["guard_by_city"] = {
        "los_angeles": None,
        "jinan": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline candidate contract"):
        load_candidate_specs(path)


def test_run_one_executes_pressure_baseline_without_source_model(
    tmp_path, monkeypatch
) -> None:
    result_path = tmp_path / "result.json"
    tripinfo_path = tmp_path / "tripinfo.xml"

    def fail_source_rollout(**_kwargs):
        raise AssertionError("pressure baseline must not load the source model")

    def fake_closed_loop(**kwargs):
        Path(kwargs["tripinfo_out"]).write_text("<tripinfos />\n", encoding="utf-8")
        assert kwargs["diagnostic_spec"].source_policy == "phase_pressure"
        return {
            "metrics": {
                "mean_tripinfo_waiting_time": 10.0,
                "mean_tripinfo_duration": 20.0,
                "mean_tripinfo_time_loss": 5.0,
                "system_vehicle_hours": 1.0,
                "completion_ratio": 1.0,
                "collision_incidents": 0,
                "starting_teleports": 0,
                "ending_teleports": 0,
            }
        }

    monkeypatch.setattr(
        shard_module, "run_source_city_component_rollout", fail_source_rollout
    )
    monkeypatch.setattr(
        shard_module, "run_external_closed_loop_rollout", fake_closed_loop
    )
    row = shard_module._run_one(
        {
            "identity": ("jinan_3x4_real", 101, "pressure_rule"),
            "result_path": str(result_path),
            "tripinfo_path": str(tripinfo_path),
            "candidate_by_key": {
                "pressure_rule": {
                    "key": "pressure_rule",
                    "baseline_policy": "phase_pressure",
                    "weights_by_city": {"los_angeles": {}, "jinan": {}},
                }
            },
            "guard_config": None,
            "allowed_seeds": (101,),
            "protocol": "/protocol.json",
            "offline_authorization": "/authorization.json",
            "offline_authorization_sha256": "a" * 64,
            "joint_audit": "/joint.json",
            "joint_audit_sha256": "b" * 64,
            "freeze_root": "/freeze",
            "external_manifest": "/external.json",
            "external_manifest_sha256": "c" * 64,
            "conversion_root": "/conversion",
            "conversion_manifest": "/conversion.json",
            "conversion_manifest_sha256": "d" * 64,
            "conversion_tree_sha256": "e" * 64,
        }
    )

    assert row["baseline_policy"] == "phase_pressure"
    assert row["source_city_weights"] == {}
    assert row["guard_config"] is None
    assert row["target_only_anchor"] is None


def test_v98_fresh_confirmation_seeds_reproduce_and_exclude_v93() -> None:
    root = Path(__file__).resolve().parents[2]
    current = json.loads(
        (
            root
            / "cf_h2o/config/traffic_signal_tsc_v98_offline_source_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    previous = json.loads(
        (
            root
            / "cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    observed = current["confirmation_seeds"]
    excluded = set(previous["development_seeds"] + previous["confirmation_seeds"])
    rng = np.random.default_rng(
        current["confirmation_seed_generation"]["seed"]
    )
    generated = []
    seen = set(excluded)
    while len(generated) < len(observed):
        value = int(rng.integers(1_000_000, 10_000_000))
        if value not in seen:
            seen.add(value)
            generated.append(value)

    assert observed == sorted(generated)
    assert not set(observed) & excluded
    assert {row["key"] for row in load_candidate_specs(
        root
        / "cf_h2o/config/traffic_signal_tsc_v98_offline_source_fresh_confirmation_v1.json"
    )} == {"target_only", "offline_selector"}


def test_v98_strict_target_confirmation_seeds_are_fresh() -> None:
    root = Path(__file__).resolve().parents[2]
    config_root = root / "cf_h2o/config"
    v93 = json.loads(
        (
            config_root
            / "traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    v98_v1 = json.loads(
        (
            config_root
            / "traffic_signal_tsc_v98_offline_source_fresh_confirmation_v1.json"
        ).read_text(encoding="utf-8")
    )
    current_path = (
        config_root
        / "traffic_signal_tsc_v98_strict_target_fresh_confirmation_v2.json"
    )
    current = json.loads(current_path.read_text(encoding="utf-8"))
    observed = current["confirmation_seeds"]
    excluded = set(
        v93["development_seeds"]
        + v93["confirmation_seeds"]
        + v98_v1["confirmation_seeds"]
    )
    rng = np.random.default_rng(current["confirmation_seed_generation"]["seed"])
    generated = []
    seen = set(excluded)
    while len(generated) < len(observed):
        value = int(rng.integers(1_000_000, 10_000_000))
        if value not in seen:
            seen.add(value)
            generated.append(value)

    assert observed == sorted(generated)
    assert not set(observed) & excluded
    assert current["strict_target_only_contract"]["source_rows_consumed"] == 0
    assert {row["key"] for row in load_candidate_specs(current_path)} == {
        "target_only",
        "offline_selector",
    }


def test_v111_three_arm_confirmation_seeds_are_fresh() -> None:
    root = Path(__file__).resolve().parents[2]
    config_root = root / "cf_h2o/config"
    previous_paths = (
        config_root / "traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json",
        config_root / "traffic_signal_tsc_v98_offline_source_fresh_confirmation_v1.json",
        config_root / "traffic_signal_tsc_v98_strict_target_fresh_confirmation_v2.json",
    )
    current_path = (
        config_root
        / "traffic_signal_tsc_v111_target_offline_source_guard_confirmation_v1.json"
    )
    current = json.loads(current_path.read_text(encoding="utf-8"))
    excluded = set()
    for path in previous_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(int(value) for value in payload.get("development_seeds", ()))
        excluded.update(int(value) for value in payload.get("confirmation_seeds", ()))
    rng = np.random.default_rng(current["confirmation_seed_generation"]["seed"])
    generated = []
    seen = set(excluded)
    while len(generated) < len(current["confirmation_seeds"]):
        value = int(rng.integers(1_000_000, 10_000_000))
        if value not in seen:
            seen.add(value)
            generated.append(value)

    assert current["confirmation_seeds"] == sorted(generated)
    assert not set(generated) & excluded
    candidates = load_candidate_specs(current_path)
    assert tuple(row["key"] for row in candidates) == (
        "pressure_rule",
        "target_only_guard",
        "source_guard",
    )


def test_strict_target_confirmation_audits_model_identity() -> None:
    spec = {
        "strict_target_only_contract": {
            "family": "causal_target_only_v2",
            "model_protocol": "cfcmt-strict-target-only-action-model-v1",
            "source_rows_consumed": 0,
            "jinan_model_sha256": "expected",
        }
    }
    launch = {
        "artifact_contract": {
            "jinan_strict_target_only_model": {
                "path": "/remote/model.pkl",
                "sha256": "expected",
            }
        }
    }

    contract, claim = _target_anchor_contract(spec, launch)
    assert contract["source_rows_consumed"] == 0
    assert contract["model_sha256"] == "expected"
    assert "causal_target_only_v2" in claim

    launch["artifact_contract"]["jinan_strict_target_only_model"][
        "sha256"
    ] = "changed"
    with pytest.raises(ValueError, match="contract changed"):
        _target_anchor_contract(spec, launch)
