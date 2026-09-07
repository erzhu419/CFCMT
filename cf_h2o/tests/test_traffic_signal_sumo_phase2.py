import shutil

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_sumo_phase2 import (
    _scenario_specs,
    build_scenario,
    run_experiment,
)


def test_build_phase2_multi_intersection_scenario(tmp_path):
    if not shutil.which("netgenerate"):
        pytest.skip("SUMO netgenerate is not installed")
    build = build_scenario(
        spec=_scenario_specs()[0],
        out_root=tmp_path,
        duration_sec=150.0,
        control_interval_sec=15,
        seed=9,
    )

    assert build.net_path.exists()
    assert build.route_path.exists()
    assert build.config_path.exists()
    assert build.vehicles > 0
    assert build.scenario_summary.shape == (8,)
    assert np.all(np.isfinite(build.scenario_summary))


def test_phase2_multi_intersection_smoke_with_oracle(tmp_path):
    if not shutil.which("sumo") or not shutil.which("netgenerate"):
        pytest.skip("SUMO binaries are not installed")
    result = run_experiment(
        out_root=tmp_path,
        duration_sec=120.0,
        collect_duration_sec=120.0,
        fewshot_duration_sec=75.0,
        control_interval_sec=15,
        warmup_sec=30.0,
        seed=13,
        max_scenarios=2,
        policies=("fixed_offset", "sim_generic_mpc", "sim_source_avg_mpc", "spillback_pressure", "cfcmt_global_mpc"),
        oracle_targets=1,
        oracle_horizon=1,
    )

    assert result["setting"]["network"] == "generated_4x4_grid_four_center_tls"
    assert result["setting"]["transfer_split"] == "leave_one_scenario_out"
    assert len(result["targets"]) == 2
    assert "oracle_mpc" in result["targets"][0]["policy_metrics"]
    assert result["targets"][0]["source_transition_count"] > 0
    assert "sim_generic_mpc" in result["aggregate"]
    assert "sim_source_avg_mpc" in result["aggregate"]
    assert np.isfinite(result["aggregate"]["cfcmt_global_mpc"]["mean_cost"])
