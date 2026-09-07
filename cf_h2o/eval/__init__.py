"""Evaluation helpers with optional experiment dependencies loaded lazily."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DataEfficiencyPoint",
    "ImperfectSimH2OOnlineResult",
    "LearnedDAGDataEfficiencyPoint",
    "LearnedDAGDataEfficiencyResult",
    "SyntheticAblationResult",
    "SyntheticDataEfficiencyResult",
    "run_imperfect_sim_h2o_online_ablation",
    "run_learned_dag_data_efficiency_ablation",
    "run_synthetic_causal_ablation",
    "run_synthetic_data_efficiency_ablation",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("cf_h2o.eval.synthetic_bus_ablation")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
