import numpy as np

from cf_h2o.eval.traffic_signal_transfer_feasibility import (
    ACTION_GRID_DEFAULT,
    _network_specs,
    _sample_batch,
    _sim_next_queue,
    run_experiment,
)


def test_signal_queue_simulator_shapes_are_stable():
    spec = _network_specs()[0]
    batch = _sample_batch(spec, 16, np.random.default_rng(3))
    q_next = _sim_next_queue(batch, 0.65)

    assert batch.q.shape == (16, 4)
    assert batch.arrivals.shape == (16, 4)
    assert batch.downstream.shape == (16, 4)
    assert q_next.shape == (16, 4)
    assert np.all(np.isfinite(q_next))


def test_traffic_signal_feasibility_smoke():
    result = run_experiment(
        samples_per_network=80,
        eval_samples_per_network=40,
        fewshot_samples=12,
        seed=11,
        action_grid=ACTION_GRID_DEFAULT,
    )

    assert result["setting"]["transfer_split"] == "leave_one_network_out"
    assert result["setting"]["zero_shot_target_labels"] == 0
    assert len(result["targets"]) == 5
    policy = result["aggregate"]["policy"]
    assert "spillback_pressure" in policy
    assert "h2oplus_dense_mpc" in policy
    assert "cfcmt_weighted_mpc" in policy
    assert "cfcmt_fewshot_bias_mpc" in policy
    for metrics in policy.values():
        assert np.isfinite(metrics["mean_cost"])
        assert np.isfinite(metrics["mean_regret"])
