from pathlib import Path

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    SOURCE_SEEDS,
)
from cf_h2o.eval.traffic_signal_pressure_aligned_source_utility import (
    METHOD_PROTOCOL,
    RESULT_PROTOCOL,
    UTILITY_GATE_PROTOCOL,
    aggregate_results,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    SOURCE_PRIOR_STRENGTH,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL,
)
from scripts.cluster.launch_tsc_pressure_aligned_source_utility import (
    CONFIG_RELATIVE,
    MODULE,
    PROJECT_ROOT,
    _validate_config,
    build_specs,
)
from scripts.cluster.launch_tsc_state_conditioned_source_utility import _read_json


def _result(
    city: str, *, admitted: bool, target_effect: float, placebo_effect: float
) -> dict:
    rigid = {str(seed): -0.2 for seed in SOURCE_SEEDS}
    source = {
        str(seed): rigid[str(seed)] + target_effect for seed in SOURCE_SEEDS
    }
    placebo = {
        str(seed): source[str(seed)] - placebo_effect for seed in SOURCE_SEEDS
    }
    return {
        "protocol": RESULT_PROTOCOL,
        "target_city": city,
        "target_budget": 100,
        "method": {
            "protocol": METHOD_PROTOCOL,
            "source_prior_strength": SOURCE_PRIOR_STRENGTH,
            "utility_gate_protocol": UTILITY_GATE_PROTOCOL,
            "selector": {
                "candidate": "per_state_source_mechanism_pool",
                "admitted": admitted,
                "candidate_score_constraint": (
                    PRESSURE_ALIGNED_CANDIDATE_PROTOCOL
                ),
            },
        },
        "source_null_contract": {"summaries_equal": True},
        "arm_summaries": {
            "rigid_target_only": {"seed_values": rigid},
            "selected_source_corrected_rigid": {"seed_values": source},
            "same_candidate_placebo_corrected_rigid": {
                "seed_values": placebo
            },
        },
    }


def test_pressure_aligned_aggregate_preserves_source_identity_comparison() -> None:
    rows = [
        _result(
            city,
            admitted=index < 2,
            target_effect=-0.02 if index < 2 else 0.0,
            placebo_effect=-0.01 if index < 2 else 0.0,
        )
        for index, city in enumerate(EXPECTED_CITY_GROUPS)
    ]

    result = aggregate_results(rows)

    assert result["development_gate"]["passed"] is True
    assert result["source_admission_count"] == 2
    assert result["development_gate"]["decision"] == (
        "authorize_v150m_pressure_aligned_closed_loop_development"
    )


def test_pressure_aligned_launcher_uses_distinct_module_and_model_root() -> None:
    config = _read_json(PROJECT_ROOT / CONFIG_RELATIVE)
    _validate_config(config)
    specs = build_specs(
        snapshot_root=Path("/snapshot"),
        authorization_result_remote=Path("/authorization.json"),
        authorization_sha256="a" * 64,
        v150j_rejection_remote=Path("/v150j.json"),
        v150j_rejection_sha256="b" * 64,
        conversion_root=Path("/conversion"),
        fit_result_remote=Path("/fit.json"),
        fit_result_sha256="c" * 64,
        source_cache_root=Path("/cache"),
        source_cache_audit_remote=Path("/audit.json"),
        source_cache_audit_sha256="d" * 64,
        remote_output_root=Path("/results"),
        local_output_root=Path("/local"),
        cache_workers=8,
        runtime_model_root=Path("/models"),
    )

    assert len(specs) == len(EXPECTED_CITY_GROUPS)
    assert all(f"-m {MODULE}" in spec["cmd"] for spec in specs)
    assert all("--runtime-model-out /models/" in spec["cmd"] for spec in specs)
