import numpy as np

from cf_h2o.eval.traffic_signal_resco_cfcmt_v2_suite import (
    _aggregate_seed_rows,
    paired_hierarchical_bootstrap,
)


def _rows():
    rows = []
    for target_idx, target in enumerate(("a", "b", "c")):
        for seed in (1, 2):
            for policy, offset in (("cfcmt_guard", -1.0), ("baseline", 0.0)):
                rows.append(
                    {
                        "target": target,
                        "seed": seed,
                        "policy": policy,
                        "metrics": {
                            "ok": True,
                            "mean_queue": 10.0 + target_idx + offset,
                            "p90_queue": 12.0 + target_idx + offset,
                            "mean_tripinfo_duration": 30.0,
                            "mean_tripinfo_waiting_time": 5.0,
                            "mean_tripinfo_time_loss": 8.0,
                            "throughput_ratio": 0.9,
                            "guard_audit": {},
                            "phase_execution_audit": {},
                        },
                    }
                )
    return rows


def test_hierarchical_bootstrap_uses_networks_as_the_outer_unit():
    result = paired_hierarchical_bootstrap(
        _rows(),
        reference="cfcmt_guard",
        baseline="baseline",
        samples=500,
        seed=9,
    )
    assert result["network_count"] == 3
    assert result["mean_delta"] == -1.0
    assert result["ci_low"] == -1.0
    assert result["ci_high"] == -1.0
    assert result["win_networks"] == 3


def test_seed_aggregation_first_forms_target_means():
    targets, aggregate = _aggregate_seed_rows(
        _rows(),
        policies=("cfcmt_guard", "baseline"),
    )
    assert len(targets) == 3
    assert aggregate["cfcmt_guard"]["target_count"] == 3
    assert aggregate["cfcmt_guard"]["mean_queue"] == 10.0
    assert aggregate["baseline"]["mean_queue"] == 11.0
