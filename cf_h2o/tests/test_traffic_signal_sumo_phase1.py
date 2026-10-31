import shutil

import numpy as np
import pytest

from cf_h2o.eval.traffic_signal_sumo_phase1 import (
    _scenario_specs,
    build_scenario,
    run_experiment,
)


def test_build_phase1_sumo_scenario(tmp_path):
    if not shutil.which("netgenerate"):
        pytest.skip("SUMO netgenerate is not installed")
    build = build_scenario(
        spec=_scenario_specs()[0],
        out_root=tmp_path,
        duration_sec=120.0,
        control_interval_sec=15,
        seed=3,
    )

    assert build.net_path.exists()
    assert build.route_path.exists()
    assert build.config_path.exists()
    assert build.vehicles > 0
    assert build.summary.shape[0] == 8
    assert np.all(np.isfinite(build.summary))


def test_phase1_sumo_smoke(tmp_path):
    if not shutil.which("sumo") or not shutil.which("netgenerate"):
        pytest.skip("SUMO binaries are not installed")
    result = run_experiment(
        out_root=tmp_path,
        duration_sec=150.0,
        collect_duration_sec=150.0,
        fewshot_duration_sec=75.0,
        control_interval_sec=15,
        warmup_sec=30.0,
        seed=5,
        max_scenarios=2,
        policies=("fixed_time", "queue_pressure", "cfcmt_global_mpc"),
    )

    assert result["setting"]["simulator"] == "SUMO/libsumo"
    assert result["setting"]["transfer_split"] == "leave_one_scenario_out"
    assert len(result["targets"]) == 2
    for target in result["targets"]:
        assert target["source_transition_count"] > 0
        assert "cfcmt_global_mpc" in target["policy_metrics"]
    assert np.isfinite(result["aggregate"]["cfcmt_global_mpc"]["mean_cost"])
