from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
)
from scripts.cluster.launch_tsc_state_conditioned_source_closed_loop import (
    CONFIG_RELATIVE,
    POLICY_ARMS,
    PROJECT_ROOT,
    _read_json,
    _validate_config,
    build_specs,
    rollout_identities,
)


def _runtime_records() -> dict[str, dict]:
    return {
        city: {
            "model_path": f"/models/{city}/model.pkl",
            "model_sha256": str(index) * 64,
        }
        for index, city in enumerate(EXPECTED_CITY_GROUPS, start=1)
    }


def test_frozen_matrix_covers_all_scenarios_arms_and_seeds() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    _validate_config(config)

    smoke = rollout_identities(config, stage="smoke")
    full = rollout_identities(config, stage="full")

    assert len(smoke) == 16
    assert len(full) == 216
    assert {row[3] for row in full} == set(POLICY_ARMS)
    assert {row[0] for row in full} == set(EXPECTED_CITY_GROUPS)
    assert len({row[1] for row in full}) == 18


def test_specs_keep_tripinfo_and_runtime_models_outside_result_tree() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        conversion_root=Path("/conversion"),
        config=config,
        runtime_records=_runtime_records(),
        stage="smoke",
        remote_output_root=Path("/results"),
        local_output_root=Path("/local"),
        remote_scratch_root=Path("/scratch"),
    )

    assert len(specs) == 16
    for spec in specs:
        assert spec["result_dir"].startswith("/results/")
        assert "--tripinfo /scratch/" in spec["cmd"]
        assert "/models/" not in spec["result_dir"]
        if spec["signature"].endswith("/phase_pressure"):
            assert "--runtime-model" not in spec["cmd"]
        else:
            assert "--runtime-model /models/" in spec["cmd"]
