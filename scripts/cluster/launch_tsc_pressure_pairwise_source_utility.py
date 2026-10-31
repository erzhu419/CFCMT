#!/usr/bin/env python3
"""Submit seven V150N pressure-pairwise source-utility targets."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from cf_h2o.eval.traffic_signal_mechanism_parameter_prior_feasibility import (
    EXPECTED_CITY_GROUPS,
    FOLD_COUNT,
    MINIMUM_OOF_GAIN,
)
from cf_h2o.eval.traffic_signal_pressure_pairwise_source_utility import (
    UTILITY_GATE_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_rigid_mechanism_prior_integration import (
    AUTHORIZATION_PROTOCOL,
)
from cf_h2o.eval.traffic_signal_state_conditioned_source_utility import (
    GATE_CONFIG,
    MINIMUM_INTERVENTION_GROUPS,
    MINIMUM_NONDEGRADING_FOLDS,
    SOURCE_PRIOR_STRENGTH,
    TARGET_BUDGET,
)
from cf_h2o.traffic_signal.state_conditioned_source_utility import (
    PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL,
)
from scripts.cluster.launch_tsc_pressure_aligned_source_utility import (
    main as _main,
)


LAUNCH_PROTOCOL = "tsc-v150n-pressure-pairwise-source-utility-launch-v1"
CONFIG_RELATIVE = Path(
    "cf_h2o/config/traffic_signal_tsc_v150n_pressure_pairwise_source_utility.json"
)
MODULE = "cf_h2o.eval.traffic_signal_pressure_pairwise_source_utility"
SIGNATURE_PREFIX = "CFCMT/v150n/pressure-pairwise-source-utility-v1"
RESOURCE_FAMILY = "CFCMT-v150n-pressure-pairwise-source-utility"


def _validate_config(config: Mapping[str, Any]) -> None:
    gate = dict(config.get("development_gate", {}))
    utility = dict(config.get("utility_gate", {}))
    if (
        config.get("protocol")
        != "tsc-v150n-pressure-pairwise-source-utility-v1"
        or config.get("redeveloped_after")
        != "tsc-v150m-pressure-aligned-source-utility-aggregate-v1"
        or config.get("offline_parent")
        != "tsc-v150k-state-conditioned-source-utility-aggregate-v1"
        or config.get("authorization_protocol") != AUTHORIZATION_PROTOCOL
        or tuple(config.get("target_city_groups", ())) != EXPECTED_CITY_GROUPS
        or int(config.get("target_budget", -1)) != TARGET_BUDGET
        or int(config.get("common_evaluation_reserve_budget", -1))
        != TARGET_BUDGET
        or float(config.get("source_prior_strength", -1.0))
        != SOURCE_PRIOR_STRENGTH
        or int(config.get("outer_fold_count", -1)) != FOLD_COUNT
        or int(config.get("inner_fold_count", -1)) != FOLD_COUNT - 1
        or int(config.get("candidate_count_per_target", -1)) != 36
        or config.get("candidate_representation")
        != "single-state-conditioner-v1"
        or config.get("candidate_score_constraint")
        != PRESSURE_PAIRWISE_CANDIDATE_PROTOCOL
        or config.get("constraint_application")
        != "before_nested_utility_record_construction_and_runtime_selection"
        or config.get("selector")
        != "nested_per_state_pressure_pairwise_source_mechanism_utility"
        or utility.get("protocol") != UTILITY_GATE_PROTOCOL
        or float(utility.get("ridge_alpha", -1.0)) != GATE_CONFIG.ridge_alpha
        or float(utility.get("error_quantile", -1.0))
        != GATE_CONFIG.error_quantile
        or int(utility.get("minimum_records", -1)) != GATE_CONFIG.minimum_records
        or float(utility.get("minimum_predicted_gain", -1.0))
        != GATE_CONFIG.minimum_predicted_gain
        or utility.get("source_identity_feature") is not False
        or float(config.get("minimum_target_oof_gain", -1.0))
        != MINIMUM_OOF_GAIN
        or float(config.get("minimum_identity_oof_gain", -1.0))
        != MINIMUM_OOF_GAIN
        or int(config.get("minimum_nondegrading_outer_folds", -1))
        != MINIMUM_NONDEGRADING_FOLDS
        or int(config.get("minimum_crossfitted_intervention_groups", -1))
        != MINIMUM_INTERVENTION_GROUPS
        or config.get("source_null_fallback") != "bitwise_exact_rigid_cfcmt"
        or config.get("matched_placebo_pipeline")
        != "separately_nested_same_capacity_pressure_pairwise_utility_gate"
        or config.get("inference_unit") != "city"
        or int(gate.get("minimum_identity_verified_source_cities", -1)) != 2
        or gate.get("require_no_city_regression") is not True
        or gate.get("require_every_admitted_city_improve_rigid") is not True
        or gate.get("require_every_admitted_city_beat_matched_placebo") is not True
        or gate.get("require_exact_fallback_for_rejected_cities") is not True
    ):
        raise ValueError("V150N frozen configuration changed")


def main(argv: Sequence[str] | None = None) -> int:
    return _main(
        argv,
        config_relative=CONFIG_RELATIVE,
        config_validator=_validate_config,
        module=MODULE,
        signature_prefix=SIGNATURE_PREFIX,
        resource_family=RESOURCE_FAMILY,
        version_label="V150N",
        launch_protocol=LAUNCH_PROTOCOL,
        intent_label="CFCMT-v150n-pressure-pairwise-source-utility-v1",
    )


if __name__ == "__main__":
    raise SystemExit(main())
