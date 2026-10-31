import json
from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_future_randomness_identifiability import (
    CONFIG_RELATIVE,
    PROJECT_ROOT,
    _validate_config,
    build_specs,
)


def test_v150f_frozen_config_and_city_matrix():
    config = json.loads((PROJECT_ROOT / CONFIG_RELATIVE).read_text(encoding="utf-8"))

    _validate_config(config)

    assert tuple(sorted(config["city_scenarios"])) == EXPECTED_CITY_GROUPS
    assert config["snapshot_protocol"]["save_rng"] is False


def test_v150f_specs_use_libsumo_wrapper_and_unique_results():
    config = json.loads((PROJECT_ROOT / CONFIG_RELATIVE).read_text(encoding="utf-8"))
    specs = build_specs(
        snapshot_root=Path("/remote/snapshot"),
        config=config,
        remote_output_root=Path("/remote/results"),
        local_output_root=Path("/local/results"),
    )

    assert len(specs) == 7
    assert len({spec["signature"] for spec in specs}) == 7
    assert len({spec["result_dir"] for spec in specs}) == 7
    assert all("run_cfcmt_sumo122.sh" in spec["cmd"] for spec in specs)
    assert all("traffic_signal_future_randomness_identifiability" in spec["cmd"] for spec in specs)
    assert all("traci" not in spec["cmd"].lower() for spec in specs)
