import json
from pathlib import Path

from scripts.cluster import run_tsc_v93_max_pressure_shard as worker


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "cf_h2o/config/traffic_signal_tsc_v93_source_city_disjoint_confirmation_v1.json"
LAUNCH = ROOT / "cf_h2o/results/cluster/tsc_v93_source_city_components_20260831/disjoint_confirmation_v1/launch.json"


def test_six_shards_retain_the_complete_frozen_224_rollout_cohort():
    _, seeds = worker.load_cohort(LAUNCH, SPEC)
    shards = [
        worker.shard_identities(seeds, shard_index=index, shard_count=6)
        for index in range(6)
    ]
    identities = [identity for shard in shards for identity in shard]
    assert len(identities) == len(set(identities)) == 224
    assert {seed for _, seed in identities} == set(seeds)
    assert {scenario for scenario, _ in identities} == {
        "la_1x4", "jinan_3x4_real", "jinan_3x4_real_2000", "jinan_3x4_real_2500"
    }


def test_rollout_dispatches_actual_max_pressure_and_keeps_original_inputs(tmp_path, monkeypatch):
    launch, seeds = worker.load_cohort(LAUNCH, SPEC)
    received = {}

    def fake_rollout(**kwargs):
        received.update(kwargs)
        return {
            "city": "los_angeles",
            "runtime_policy": "max_pressure",
            "evaluation_horizon_sec": 3600,
            "control_interval_sec": 10,
            "warmup_sec": 60,
            "metrics": {key: 0 for key in worker.METRICS},
        }

    monkeypatch.setattr(worker, "run_external_closed_loop_rollout", fake_rollout)
    row = worker._run_one({
        "identity": ("la_1x4", seeds[0]),
        "allowed_seeds": seeds,
        "results_root": str(tmp_path),
        "rollout_kwargs": worker.rollout_kwargs(launch),
    })
    diagnostic = received["diagnostic_spec"]
    assert diagnostic.source_policy == "max_pressure"
    assert diagnostic.guard_config is None
    assert diagnostic.allowed_seeds == seeds
    assert received["expected_sumo_version"] == "1.22.0"
    assert str(received["freeze_root"]).endswith("full_freeze_v3")
    assert str(received["conversion_root"]).endswith("full_networks_v8")
    assert "method_runtime_override" not in received
    assert row["runtime_policy"] == "max_pressure"
    assert json.loads(Path(row["result_path"]).read_text())["runtime_policy"] == "max_pressure"
