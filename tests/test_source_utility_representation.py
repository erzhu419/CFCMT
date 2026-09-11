from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_right_of_way_signature import RESULT_PROTOCOL as SIGNATURE_PROTOCOL
from cf_h2o.eval.traffic_signal_source_utility_representation import (
    BASE_CONTEXT_FEATURES,
    BASE_DYNAMIC_FEATURES,
    V144_PROTOCOL,
    run_source_utility_representation_diagnostic,
)
from cf_h2o.traffic_signal.right_of_way_context import RIGHT_OF_WAY_CONTEXT_FEATURES


CITIES = tuple(f"city_{index}" for index in range(7))


def _v144_results() -> list[dict]:
    context_names = tuple(BASE_CONTEXT_FEATURES)
    feature_names = tuple(BASE_DYNAMIC_FEATURES)
    signatures = {}
    for index, city in enumerate(CITIES):
        signatures[city] = {
            "context_names": list(context_names),
            "context_mean": [float(index + offset) for offset in range(len(context_names))],
            "feature_names": list(feature_names),
            "feature_mean": [float(index) / (offset + 1.0) for offset in range(len(feature_names))],
            "feature_std": [float(index + 1) / (offset + 2.0) for offset in range(len(feature_names))],
        }
    rows = []
    for target_index, target in enumerate(CITIES):
        target_seed = {"1": 0.1, "2": 0.2, "3": 0.3}
        sources = {}
        for source_index, source in enumerate(CITIES):
            if source == target:
                continue
            effect = 0.01 * (abs(source_index - target_index) - 3)
            sources[source] = {
                "seed_values": {
                    seed: value + effect for seed, value in target_seed.items()
                }
            }
        rows.append(
            {
                "protocol": V144_PROTOCOL,
                "target_city": target,
                "city_covariate_signatures": signatures,
                "arm_summaries": {
                    "target_only_causal": {"seed_values": target_seed}
                },
                "source_arm_summaries": sources,
            }
        )
    return rows


def _right_of_way_signature() -> dict:
    return {
        "protocol": SIGNATURE_PROTOCOL,
        "feature_names": list(RIGHT_OF_WAY_CONTEXT_FEATURES),
        "city_signatures": {
            city: {
                "mean": [float(index + 1)] * len(RIGHT_OF_WAY_CONTEXT_FEATURES),
                "std": [float(index + 1) / 10.0]
                * len(RIGHT_OF_WAY_CONTEXT_FEATURES),
            }
            for index, city in enumerate(CITIES)
        },
    }


def test_source_utility_representation_excludes_each_outer_city() -> None:
    result = run_source_utility_representation_diagnostic(
        v144_results=_v144_results(),
        right_of_way_signature=_right_of_way_signature(),
    )

    assert result["pair_count"] == 42
    assert result["information_boundary"]["outer_evaluation_city_labels_used_for_fit"] is False
    for representation in result["representations"].values():
        assert representation["outer_fold_count"] == 7
        assert representation["pair_count"] == 42
        assert np.isfinite(representation["pair_mae"])
        assert all(
            fold["heldout_city_absent_as_source_and_target"]
            and fold["training_pair_count"] == 30
            and fold["evaluation_pair_count"] == 6
            for fold in representation["folds"]
        )
    assert len(result["matched_cyclic_placebos"]) == 6
