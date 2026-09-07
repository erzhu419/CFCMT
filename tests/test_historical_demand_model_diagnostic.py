from __future__ import annotations

import numpy as np

from cf_h2o.eval.traffic_signal_external_historical_demand_model_diagnostic import (
    BASE_VARIANT,
    HISTORICAL_VARIANT,
    feature_matrix,
    feature_names,
)
from cf_h2o.traffic_signal.demand_timeline_context import (
    TLS_TOPOLOGY_FEATURE_NAMES,
)
from cf_h2o.traffic_signal.historical_local_demand_context import (
    HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES,
)
from cf_h2o.traffic_signal.target_action_support import TARGET_SUPPORT_FEATURES


class _Topology:
    def vector_for(self, _tls_id: str) -> np.ndarray:
        return np.arange(len(TLS_TOPOLOGY_FEATURE_NAMES), dtype=float) + 100.0


class _Historical:
    def vector_at(self, _tls_id: str, _time_sec: float) -> np.ndarray:
        return np.arange(len(HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES), dtype=float) + 200.0


def _records():
    return [
        {
            "group_id": "scenario:seed1:1:tls_a",
            "scenario": "scenario",
            "time_sec": 300.0,
            "candidate_features": {
                name: float(index)
                for index, name in enumerate(TARGET_SUPPORT_FEATURES)
            },
        }
    ]


def test_historical_variant_adds_only_prefix_features() -> None:
    common = {
        "records": _records(),
        "profiles": {},
        "temporal": {},
        "topology": {"scenario": _Topology()},
        "historical": {"scenario": _Historical()},
    }
    base = feature_matrix(variant=BASE_VARIANT, **common)
    historical = feature_matrix(variant=HISTORICAL_VARIANT, **common)
    assert historical.shape[1] - base.shape[1] == len(
        HISTORICAL_LOCAL_DEMAND_FEATURE_NAMES
    )
    np.testing.assert_array_equal(historical[:, : base.shape[1]], base)
    np.testing.assert_array_equal(
        historical[0, base.shape[1] :],
        _Historical().vector_at("tls_a", 300.0),
    )


def test_feature_contract_excludes_same_day_global_schedule() -> None:
    assert tuple(feature_names(BASE_VARIANT)) == (
        *TARGET_SUPPORT_FEATURES,
        *TLS_TOPOLOGY_FEATURE_NAMES,
    )
