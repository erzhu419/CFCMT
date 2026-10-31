from cf_h2o.eval.traffic_signal_external_long_horizon_target_veto_freeze import (
    _target_model_config,
)


def test_v66_target_model_capacity_matches_predeclared_v65_contract() -> None:
    config = _target_model_config()

    assert config.max_iter == 80
    assert config.max_leaf_nodes == 7
    assert config.min_samples_leaf == 4
    assert config.l2_regularization == 16.0
    assert config.random_state == 20260803
    assert config.candidate_only is False
    assert config.balance_candidate_signs is False
