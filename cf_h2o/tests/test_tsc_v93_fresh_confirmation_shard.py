import copy
import json
from pathlib import Path

import pytest

from scripts.cluster import run_tsc_v93_fresh_confirmation_shard as worker

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "experiments/tsc_v93_exact_reference_20261031/relaunch.json"
CANDIDATE = ROOT / "cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json"


@pytest.fixture
def inputs(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "confirmation_seeds": list(range(100000001, 100000057)),
        "methods": list(worker.METHODS), "city_scenarios": worker.CITY_SCENARIOS,
    }))
    return protocol


def payload(inputs, tmp_path, method, scenario="jinan_3x4_real"):
    launch, _, candidate, seeds = worker.load_inputs(PARENT, inputs, CANDIDATE)
    return {
        "identity": (scenario, seeds[0], method), "allowed_seeds": seeds,
        "artifact_contract": launch["artifact_contract"], "candidate": candidate,
        "rollout_kwargs": worker.rollout_kwargs(launch), "results_root": str(tmp_path / "runs"),
        "stage": "prospective-fresh-seed-confirmation", "protocol_path": str(inputs),
        "parent_launch": str(PARENT), "candidate_spec": str(CANDIDATE),
        "source_root": str(ROOT),
    }


def result(scenario, runtime_policy):
    return {
        "city": "los_angeles" if scenario == "la_1x4" else "jinan",
        "runtime_policy": runtime_policy, "evaluation_horizon_sec": 3600,
        "control_interval_sec": 10, "warmup_sec": 60,
        "metrics": {**{key: 0 for key in worker.METRICS},
                    "mean_tripinfo_waiting_time": 12.5,
                    "guard_audit": {"decisions": 100, "effective_overrides": 2},
                    "reference_policy_audit": {
                        "reference_policy_spec": worker.REFERENCE_SPEC,
                        "decision_requests": 4320, "post_warmup_decision_requests": 4248,
                    }},
    }


def test_fresh_and_smoke_rosters_are_separate_and_shards_cover_every_arm(inputs):
    _, _, _, seeds = worker.load_inputs(PARENT, inputs, CANDIDATE)
    shards = [worker.shard_identities(seeds, shard_index=i, shard_count=5) for i in range(5)]
    identities = [identity for shard in shards for identity in shard]
    assert len(identities) == len(set(identities)) == 896
    assert {seed for _, seed, _ in identities} == set(seeds)
    smoke = worker.shard_identities(seeds, shard_index=0, shard_count=1, smoke=True)
    assert len(smoke) == 8
    assert {(scenario, seed) for scenario, seed, _ in smoke} == set(worker.SMOKE_PAIRS)
    assert not set(seeds).intersection(seed for _, seed, _ in smoke)


def test_previous_seed_or_changed_source_guard_is_rejected(inputs, tmp_path):
    protocol = json.loads(inputs.read_text())
    protocol["confirmation_seeds"][0] = 1020485
    inputs.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="old 64 seeds"):
        worker.load_inputs(PARENT, inputs, CANDIDATE)
    protocol["confirmation_seeds"][0] = 100000001
    inputs.write_text(json.dumps(protocol))
    candidate = json.loads(CANDIDATE.read_text())
    candidate["candidates"][2]["guard_by_city"]["jinan"]["risk_multiplier"] = 0
    changed = tmp_path / "changed_candidate.json"
    changed.write_text(json.dumps(candidate))
    with pytest.raises(ValueError, match="city guards changed"):
        worker.load_inputs(PARENT, inputs, changed)


@pytest.mark.parametrize("scenario", ["la_1x4", "jinan_3x4_real"])
def test_v93_keeps_city_specific_models_weights_and_original_guard(inputs, tmp_path, monkeypatch, scenario):
    received = {}
    def run(**kwargs):
        received.update(kwargs)
        return result(scenario, "frozen_cfcmt_runtime")
    monkeypatch.setattr(worker, "run_source_city_component_rollout", run)
    monkeypatch.setattr(worker, "run_external_closed_loop_rollout", lambda **_: pytest.fail("CFCMT sent to baseline"))
    p = payload(inputs, tmp_path, "frozen_selector", scenario)
    row = worker._run_one(p)
    city = row["city"]
    assert received["candidate_key"] == "frozen_selector"
    assert str(received["main_model_path"]) == p["artifact_contract"][f"{city}_main_model"]["path"]
    assert str(received["component_bundle_path"]) == p["artifact_contract"][f"{city}_component_bundle"]["path"]
    assert "strict_target_only_model_path" not in received
    assert "source_only_guard_path" not in received
    if city == "jinan":
        assert received["source_city_weights"] == {"atlanta": 1.0}
        assert received["guard_config"].enabled
        assert received["guard_config"].risk_multiplier == .5
    else:
        assert received["source_city_weights"] == {}
        assert received["guard_config"] is None
    assert received["rollout_kwargs"]["seed"] == p["identity"][1]
    assert row["metrics"]["mean_tripinfo_waiting_time"] == 12.5
    assert row["guard_audit"]["effective_overrides"] == 2
    with pytest.raises(FileExistsError):
        worker._run_one(p)


@pytest.mark.parametrize("method,source_policy,mode", [
    ("max_pressure", "max_pressure", "sparse"),
    ("phase_pressure", "phase_pressure", "sparse"),
    ("exact_generalized_pressure", "selected_source_prior", "direct"),
])
def test_pressure_arms_do_not_enter_learned_model_path(inputs, tmp_path, monkeypatch, method, source_policy, mode):
    received = {}
    def run(**kwargs):
        received.update(kwargs)
        return result(kwargs["scenario"], source_policy)
    monkeypatch.setattr(worker, "run_external_closed_loop_rollout", run)
    monkeypatch.setattr(worker, "run_source_city_component_rollout", lambda **_: pytest.fail("baseline sent to CFCMT"))
    p = payload(inputs, tmp_path, method)
    row = worker._run_one(p)
    diagnostic = received["diagnostic_spec"]
    assert (diagnostic.source_policy, diagnostic.coordination_mode) == (source_policy, mode)
    assert diagnostic.guard_config is None
    assert "method_runtime_override" not in received
    assert received["expected_sumo_version"] == "1.22.0"
    assert received["seed"] == p["identity"][1]
    assert row["method"] == method


def test_wrong_exact_reference_is_not_written_as_completed(inputs, tmp_path, monkeypatch):
    wrong = copy.deepcopy(result("jinan_3x4_real", "selected_source_prior"))
    wrong["metrics"]["reference_policy_audit"]["reference_policy_spec"]["downstream_occupancy_weight"] = 0
    monkeypatch.setattr(worker, "run_external_closed_loop_rollout", lambda **_: wrong)
    p = payload(inputs, tmp_path, "exact_generalized_pressure")
    with pytest.raises(ValueError, match="frozen V93 generalized pressure"):
        worker._run_one(p)
    assert not list(Path(p["results_root"]).rglob("result.json"))
