from types import SimpleNamespace

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite import (
    _collect_worker,
    _source_rule_cache_identity_v3,
    _source_rule_cache_path_v3,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_closed_loop import (
    RUNTIME_MODEL_PROTOCOL,
    StateConditionedSourceUtilityOriginator,
)
from cf_h2o.traffic_signal.dataset_cache import save_mechanism_dataset
from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset


def test_cached_worker_rejects_old_bank_even_with_self_consistent_identity(tmp_path) -> None:
    # The offline bank loader passes an archive's own identity to validation.
    # Matching itself must not make a percentage-scaled-equation bank current.
    old_identity = {
        "version": "v20-policy-consistent-multihorizon-prefix-costs-two-pass-20260830",
        "scenario": "cologne1", "seed": 41242, "collection_shard_index": 0,
    }
    dataset = MechanismDataset(
        feature_names=("service_pressure",), features=np.asarray([[1.0]]),
        context_names=("context",), context=np.asarray([[0.0]]),
        priors={"interval_cost": np.asarray([1.0])},
        targets={"interval_cost": np.asarray([1.0])},
        domains=np.asarray(["cologne1"]),
        metadata={"counterfactual_cache_identity": old_identity},
    )
    cache_path = tmp_path / "old_bank.npz"
    save_mechanism_dataset(cache_path, dataset)

    with pytest.raises(ValueError, match="occupancy.*contract"):
        _collect_worker({
            "scenario": "cologne1", "seed": 41242, "collection_shard_index": 0,
            "cache_identity": old_identity, "cache_path": str(cache_path),
        })


@pytest.mark.parametrize("protocol, error", [
    ("tsc-v150k-state-conditioned-source-runtime-model-v1", "model protocol changed"),
    (RUNTIME_MODEL_PROTOCOL, "occupancy.*contract"),
])
def test_v150l_rejects_old_payload_before_any_controller_use(protocol, error) -> None:
    with pytest.raises(ValueError, match=error):
        StateConditionedSourceUtilityOriginator(
            runtime_payload={"protocol": protocol},
            network_context=SimpleNamespace(),
            arm="rigid_target_only",
        )


def test_new_rule_cache_identity_cannot_hit_old_spillback_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "cf_h2o.eval.traffic_signal_resco_cfcmt_v3_suite._scenario_input_fingerprint",
        lambda path: "existing_input",
    )
    identity = _source_rule_cache_identity_v3(
        sumocfg=tmp_path / "scenario.sumocfg", scenario="cologne1",
        policy_spec=SimpleNamespace(to_dict=lambda: {"key": "phase_spillback_pressure"}),
        seed=41242, duration_sec=300.0, control_interval_sec=30,
        warmup_sec=120.0, sumo_version="1.22.0",
    )
    old_identity = dict(identity)
    old_identity.pop("occupancy_equation_protocol")
    assert _source_rule_cache_path_v3(tmp_path, identity) != _source_rule_cache_path_v3(
        tmp_path, old_identity
    )
