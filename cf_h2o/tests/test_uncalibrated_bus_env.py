from __future__ import annotations

import pytest

from cf_h2o.rl.uncalibrated_bus_env import make_uncalibrated_bus_env, run_uncalibrated_bus_env_smoke


def test_uncalibrated_bus_env_uses_raw_h2oplus_data_and_bounded_cache():
    result = run_uncalibrated_bus_env_smoke({"repeat": 3})

    assert result.raw_profile.path.endswith("H2Oplus/bus_h2o")
    assert "calibrated_env" not in result.raw_profile.path
    assert result.raw_profile.timetables == 50
    assert result.raw_profile.state_dim == 15
    assert result.raw_cache_entries == 1
    assert result.cache_growth_after_repeats <= 2
    assert result.traced_growth_bytes < 5_000_000

    assert result.calibrated_profile is not None
    assert result.calibrated_profile.path.endswith("H2Oplus/bus_h2o/calibrated_env")
    assert result.calibrated_profile.timetables != result.raw_profile.timetables
    assert result.calibrated_profile.stations != result.raw_profile.stations


def test_uncalibrated_bus_env_rejects_calibrated_path():
    with pytest.raises(ValueError):
        make_uncalibrated_bus_env(bus_h2o_root="H2Oplus/bus_h2o/calibrated_env")
