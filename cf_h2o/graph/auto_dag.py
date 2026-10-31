"""Automatic temporal DAG discovery for Stage 2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cf_h2o.graph.feature_registry import FeatureRegistry, mechanism_specs_from_registry
from cf_h2o.graph.graph_posterior import (
    sample_graph_from_edge_probs,
    save_graph_posterior,
    write_discovered_mechanisms,
    write_edge_report,
)
from cf_h2o.graph.llm_prior import LLMDAGPriorProvider
from cf_h2o.graph.mechanism_discovery import apply_max_parents, ridge_parent_edge_probabilities
from cf_h2o.schemas import GraphPosterior


class AutoDAGDiscoverer:
    """Data-driven temporal graph posterior estimator.

    This first Stage 2 implementation uses bootstrap stability over
    standardized ridge parent scores. It avoids manual DAG inputs and keeps all
    forbidden future-to-past edges at probability zero.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    def fit(
        self,
        dataset,
        registry: FeatureRegistry | None = None,
        route_schema: dict | None = None,
    ) -> GraphPosterior:
        registry = registry or FeatureRegistry.from_transition_dataset(dataset)
        node_names = registry.node_names
        n_nodes = len(node_names)
        values = registry.values_from_dataset(dataset)
        hard_mask = registry.build_temporal_hard_mask().to(device=values.device)

        llm_prior = LLMDAGPriorProvider(
            backend=str(self.config.get("llm_backend", "none")),
            model=self.config.get("llm_model"),
        ).propose_prior(registry, route_schema)
        prior_logits, hard_mask = self._combine_priors(registry, hard_mask, llm_prior)

        bootstrap_runs = int(self.config.get("bootstrap_runs", 20))
        bootstrap_runs = max(1, bootstrap_runs)
        seed = int(self.config.get("seed", 0))
        generator = torch.Generator(device=values.device)
        generator.manual_seed(seed)
        edge_probs_accum = []
        validation_mse = []
        parent_indices_by_child = {
            child_idx: torch.where(hard_mask[:, child_idx])[0]
            for child_idx in registry.child_node_indices()
        }

        for _ in range(bootstrap_runs):
            row_idx = torch.randint(values.shape[0], (values.shape[0],), generator=generator, device=values.device)
            boot_values = values[row_idx]
            edge_probs = torch.zeros(n_nodes, n_nodes, dtype=torch.float32, device=values.device)
            for child_idx, parent_indices in parent_indices_by_child.items():
                if parent_indices.numel() == 0:
                    continue
                parents = boot_values[:, parent_indices]
                child = boot_values[:, child_idx : child_idx + 1]
                probs, metrics = ridge_parent_edge_probabilities(
                    parents,
                    child,
                    l2=float(self.config.get("ridge_l2", 1e-3)),
                    score_threshold=float(self.config.get("score_threshold", 0.08)),
                    score_temperature=float(self.config.get("score_temperature", 0.04)),
                )
                if prior_logits is not None:
                    local_prior = prior_logits[parent_indices, child_idx]
                    logits = torch.logit(probs.clamp(1e-5, 1.0 - 1e-5)) + local_prior
                    probs = torch.sigmoid(logits)
                probs = apply_max_parents(probs, self.config.get("max_parents", 8))
                edge_probs[parent_indices, child_idx] = probs
                validation_mse.append(metrics["mse"])
            edge_probs = torch.where(hard_mask, edge_probs, torch.zeros_like(edge_probs))
            edge_probs_accum.append(edge_probs)

        edge_marginals = torch.stack(edge_probs_accum, dim=0).mean(dim=0)
        edge_marginals = torch.where(hard_mask, edge_marginals, torch.zeros_like(edge_marginals))

        graph_generator = torch.Generator(device=values.device)
        graph_generator.manual_seed(seed + 10_000)
        graphs = [
            sample_graph_from_edge_probs(edge_probs, hard_mask, node_names, registry.node_groups(), generator=graph_generator)
            for edge_probs in edge_probs_accum
        ]
        log_weights = torch.zeros(len(graphs), dtype=torch.float32, device=values.device)
        diagnostics = self._diagnostics(registry, hard_mask, edge_marginals, validation_mse, llm_prior, bootstrap_runs)
        return GraphPosterior(graphs=graphs, log_weights=log_weights, edge_marginals=edge_marginals, diagnostics=diagnostics)

    def save(self, posterior: GraphPosterior, out_dir: str | Path):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        mechanisms = posterior.diagnostics.get("mechanisms", [])
        return {
            "graph_posterior": save_graph_posterior(posterior, out_path),
            "edge_report": write_edge_report(posterior, out_path),
            "discovered_mechanisms": write_discovered_mechanisms(mechanisms, out_path),
        }

    def _combine_priors(self, registry: FeatureRegistry, hard_mask: torch.Tensor, prior: dict[str, Any]):
        prior_logits = torch.zeros_like(hard_mask, dtype=torch.float32)
        node_index = registry.node_index
        for item in prior.get("edge_priors", []):
            src = item.get("src")
            dst = item.get("dst")
            if src not in node_index or dst not in node_index:
                continue
            src_idx = node_index[src]
            dst_idx = node_index[dst]
            log_prior = float(item.get("log_prior", 0.0))
            if log_prior <= -999.0:
                hard_mask[src_idx, dst_idx] = False
            else:
                prior_logits[src_idx, dst_idx] = max(-3.0, min(3.0, log_prior))
        for item in prior.get("forbidden_edges", []):
            src = item.get("src")
            dst = item.get("dst")
            if src in node_index and dst in node_index:
                hard_mask[node_index[src], node_index[dst]] = False
        return prior_logits.to(device=hard_mask.device), hard_mask

    def _diagnostics(
        self,
        registry: FeatureRegistry,
        hard_mask: torch.Tensor,
        edge_marginals: torch.Tensor,
        validation_mse: list[float],
        llm_prior: dict[str, Any],
        bootstrap_runs: int,
    ) -> dict[str, Any]:
        node_names = registry.node_names
        child_indices = registry.child_node_indices()
        features_with_no_stable_parent = []
        for child_idx in child_indices:
            if float(edge_marginals[:, child_idx].max().detach().cpu()) < 0.3:
                features_with_no_stable_parent.append(node_names[child_idx])

        low_confidence_edges = []
        for src_idx, src in enumerate(node_names):
            for dst_idx, dst in enumerate(node_names):
                prob = float(edge_marginals[src_idx, dst_idx].detach().cpu())
                if bool(hard_mask[src_idx, dst_idx]) and 0.3 <= prob <= 0.8:
                    low_confidence_edges.append({"src": src, "dst": dst, "probability": prob})

        mechanisms = mechanism_specs_from_registry(
            registry,
            edge_marginals=edge_marginals.detach().cpu(),
            min_parent_prob=float(self.config.get("mechanism_parent_threshold", 0.3)),
        )
        warnings = list(llm_prior.get("warnings", []))
        if any("domain" in name.lower() or "source" in name.lower() or "city" in name.lower() for name in node_names):
            warnings.append("shortcut-like node names were detected; temporal hard mask prevents direct policy-action leakage")

        return {
            "node_names": node_names,
            "node_groups": registry.infer_groups(),
            "hard_mask": hard_mask.detach().cpu(),
            "bootstrap_runs": bootstrap_runs,
            "mean_validation_mse": float(sum(validation_mse) / max(1, len(validation_mse))),
            "low_confidence_edges": low_confidence_edges,
            "features_with_no_stable_parent": features_with_no_stable_parent,
            "mechanisms": mechanisms,
            "warnings": warnings,
            "config": dict(self.config),
        }

