"""Nested source utility over the binary PhasePressure-vs-rigid contrast."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from cf_h2o.eval.traffic_signal_pressure_aligned_source_utility import run_cli
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    aggregate_results as _aggregate_results,
    run_target as _run_target,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
)


RESULT_PROTOCOL = "tsc-v150n-pressure-pairwise-source-utility-target-v1"
AGGREGATE_PROTOCOL = "tsc-v150n-pressure-pairwise-source-utility-aggregate-v1"
METHOD_PROTOCOL = "nested-pressure-pairwise-source-mechanism-utility-v1"
UTILITY_GATE_PROTOCOL = (
    "ridge-upper-cost-bound-over-pressure-pairwise-source-proposals-v1"
)
RUNTIME_MODEL_PROTOCOL = "tsc-v150n-pressure-pairwise-source-runtime-model-v1"


def run_target(**kwargs: Any) -> dict[str, Any]:
    return _run_target(
        **kwargs,
        candidate_score_constraint=PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
        result_protocol=RESULT_PROTOCOL,
        method_protocol=METHOD_PROTOCOL,
        utility_gate_protocol=UTILITY_GATE_PROTOCOL,
        runtime_model_protocol=RUNTIME_MODEL_PROTOCOL,
        scientific_status="seven-city-pressure-pairwise-source-development-target",
        claim_boundary=(
            "V150N is a seven-city B100 target-offline adaptation experiment. "
            "Each real-source and matched-placebo mechanism supplies only a "
            "pairwise preference between the target-only rigid action and the "
            "deploy-observable PhasePressure reference before nested utility "
            "fitting. It is not closed-loop or fresh-city confirmation."
        ),
    )


def aggregate_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(dict(row) for row in results)
    if any(
        row.get("method", {}).get("protocol") != METHOD_PROTOCOL
        or row.get("method", {})
        .get("selector", {})
        .get("candidate_score_constraint")
        != PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL
        for row in rows
    ):
        raise ValueError("V150N aggregate requires pressure-pairwise target results")
    return _aggregate_results(
        rows,
        result_protocol=RESULT_PROTOCOL,
        aggregate_protocol=AGGREGATE_PROTOCOL,
        utility_gate_protocol=UTILITY_GATE_PROTOCOL,
        pass_decision="authorize_v150n_pressure_pairwise_closed_loop_development",
        reject_decision="retain_rigid_target_adaptation_and_reject_v150n_source_claim",
        claim_boundary=(
            "V150N tests nested source contribution for the binary "
            "PhasePressure-vs-rigid decision on seven development cities. "
            "Passing does not establish closed-loop or fresh-city efficacy."
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(
        argv,
        target_runner=run_target,
        result_aggregator=aggregate_results,
        version_label="V150N",
    )


if __name__ == "__main__":
    raise SystemExit(main())
