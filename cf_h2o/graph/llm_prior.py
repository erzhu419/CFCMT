"""Optional weak LLM prior provider for DAG discovery.

The default backend is ``none`` and returns a neutral prior. Stage 2 never uses
LLM output as ground truth; it can only contribute bounded priors.
"""

from __future__ import annotations

import json
from typing import Any

from cf_h2o.graph.feature_registry import FeatureRegistry


class LLMDAGPriorProvider:
    def __init__(self, backend: str = "none", model: str | None = None):
        self.backend = backend
        self.model = model

    def build_prompt(self, registry: FeatureRegistry, route_schema: dict | None) -> str:
        payload = {
            "feature_nodes": registry.node_names,
            "groups": registry.infer_groups(),
            "route_schema": route_schema or {},
        }
        return (
            "You are assisting a causal mechanism discovery module for bus holding control.\n"
            "You are NOT allowed to decide the final causal graph.\n"
            "Return only JSON.\n"
            "Rules:\n"
            "1. Never allow future-to-past edges.\n"
            "2. Never allow reward@t1 to cause state@t.\n"
            "3. Never allow city_id/source_label/real_sim_label to directly cause policy action.\n"
            "4. Prefer temporal edges from state@t/action@t to state@t1/reward@t1.\n"
            "5. Provide log_prior in [-3, 3], except forbidden edges use -999.\n"
            "6. If uncertain, use log_prior=0.\n"
            f"Input:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
        )

    def propose_prior(self, registry: FeatureRegistry, route_schema: dict | None = None) -> dict[str, Any]:
        """Return a JSON-compatible weak prior.

        No external LLM call is made unless a future backend implements it.
        """

        if self.backend == "none":
            return {
                "edge_priors": [],
                "forbidden_edges": [],
                "suggested_groups": {},
                "warnings": ["llm_backend=none; using neutral priors"],
            }
        return {
            "edge_priors": [],
            "forbidden_edges": [],
            "suggested_groups": {},
            "warnings": [f"llm_backend={self.backend!r} is not implemented; using neutral priors"],
        }

