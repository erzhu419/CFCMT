from copy import deepcopy
from pathlib import Path

from cf_h2o.eval.traffic_signal_dense_pressure_pairwise_source_utility import (
    METHOD_PROTOCOL,
    RESULT_PROTOCOL,
    UTILITY_GATE_PROTOCOL,
    aggregate_results,
)
from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    SOURCE_PRIOR_STRENGTH,
)
from scripts.cluster.launch_tsc_dense_pressure_pairwise_source_utility import (
    CONFIG_RELATIVE,
    PROJECT_ROOT,
    _validate_config,
    build_specs,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import _read_json


def _result(city: str, *, admitted: bool) -> dict:
    rigid = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    source = {
        str(seed): value - (0.02 if admitted else 0.0)
        for seed, value in rigid.items()
    }
    placebo = {
        str(seed): value + (0.01 if admitted else 0.0)
        for seed, value in source.items()
    }
    blind = {
        str(seed): value + (0.015 if admitted else 0.0)
        for seed, value in source.items()
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": 100,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
            "selector": {"admitted": admitted},
        },
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_dense_source_pairwise": {"seed_values": source},
            "selected_dense_matched_placebo": {"seed_values": placebo},
            "selected_dense_source_blind": {"seed_values": blind},
        },
    }


def test_v150o_config_and_aggregate_contract() -> None:
    _validate_config(_read_json(PROJECT_ROOT / CONFIG_RELATIVE))
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert result["source_admission_count"] == 2
    assert result["development_gate"]["decision"] == (
        "authorize_v150o_dense_pairwise_closed_loop_development"
    )


def test_v150o_aggregate_rejects_source_blind_tie() -> None:
    rows = [
        _result(city, admitted=index < 2)
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]
    for index in range(2):
        rows[index]["arm_summaries"]["selected_dense_source_blind"] = deepcopy(
            rows[index]["arm_summaries"]["selected_dense_source_pairwise"]
        )

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is False
    assert result["development_gate"]["checks"][
        "mean_selected_effect_beats_source_blind"
    ] is False


def test_v150o_launcher_forwards_frozen_v150n_evidence_once() -> None:
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/authorization.json"),
        authorization_sha256="a" * 64,
        v150j_rejection_remote=Path("/v150j.json"),
        v150j_rejection_sha256="b" * 64,
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit.json"),
        fit_result_sha256="c" * 64,
        source_cache_root=Path("/source"),
        source_cache_audit_remote=Path("/source-audit.json"),
        source_cache_audit_sha256="d" * 64,
        remote_output_root=Path("/results"),
        local_output_root=Path("/local-results"),
        cache_workers=8,
        runtime_model_root=Path("/runtime-models"),
        extra_target_args=(
            "--v150n-rejection",
            "/v150n.json",
            "--v150n-rejection-sha256",
            "e" * 64,
        ),
    )

    assert len(specs) == len(EXPECTED_CITY_GROUPS)
    for spec in specs:
        command = spec["cmd"]
        assert command.count("--v150n-rejection ") == 1
        assert command.count("--v150n-rejection-sha256 ") == 1
        assert "--cache-workers 8" in command
        assert "--fit-workers 6" in command
