"""DAG-GFlowNet-style posterior sampler for temporal mechanism graphs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from cf_h2o.graph.feature_registry import FeatureRegistry, mechanism_specs_from_registry
from cf_h2o.graph.graph_posterior import (
    save_graph_posterior,
    write_discovered_mechanisms,
    write_edge_report,
)
from cf_h2o.graph.llm_prior import LLMDAGPriorProvider
from cf_h2o.graph.mechanism_discovery import apply_max_parents, ridge_parent_edge_probabilities
from cf_h2o.schemas import GraphPosterior, GraphSpec


class DAGGFlowNetDiscoverer:
    """Sample DAG posterior graphs with a lightweight trajectory-balance GFlowNet.

    This implementation follows the core DAG-GFlowNet idea: states are partial
    DAGs, actions add one valid edge or stop, and terminal graph probability is
    trained toward a decomposable graph reward. The temporal hard mask keeps the
    CF-H2O graph bipartite from present-time variables to next-time targets,
    while an explicit cycle check keeps the class usable for looser future masks.
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
        values = registry.values_from_dataset(dataset)
        hard_mask = registry.build_temporal_hard_mask().to(device=values.device)
        hard_mask = self._apply_llm_prior(registry, hard_mask, route_schema)

        context = _GFlowContext(
            values=values,
            registry=registry,
            hard_mask=hard_mask,
            max_parents=int(self.config.get("max_parents", 8)),
            l2=float(self.config.get("ridge_l2", 1e-3)),
            complexity_penalty=float(self.config.get("complexity_penalty", 0.5)),
            reward_scale=float(self.config.get("reward_scale", 0.002)),
            min_log_reward=float(self.config.get("min_log_reward", -20.0)),
            max_log_reward=float(self.config.get("max_log_reward", 80.0)),
        )
        model = _TrajectoryBalancePolicy(
            hard_mask.shape[0],
            hard_mask,
            initial_stop_logit=float(self.config.get("initial_stop_logit", -2.0)),
            stop_edge_count_slope=float(self.config.get("stop_edge_count_slope", 0.12)),
        ).to(device=values.device)
        if bool(self.config.get("warm_start_ridge", True)):
            model.initialize_edge_logits(_ridge_warm_start_logits(values, registry, hard_mask, self.config))

        train_metrics = self._train(model, context)
        graphs, log_rewards = self._sample_graphs(model, context)
        if not graphs:
            empty = torch.zeros_like(hard_mask, dtype=torch.float32)
            graphs = [_graph_spec(empty, hard_mask, registry)]
            log_rewards = torch.zeros(1, dtype=torch.float32, device=values.device)

        adjacency_stack = torch.stack([graph.adjacency.to(device=values.device, dtype=torch.float32) for graph in graphs])
        edge_marginals = adjacency_stack.mean(dim=0)
        edge_marginals = torch.where(hard_mask, edge_marginals, torch.zeros_like(edge_marginals))
        log_weights = log_rewards.to(device=values.device, dtype=torch.float32)
        diagnostics = self._diagnostics(registry, hard_mask, edge_marginals, train_metrics, log_rewards, context)
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

    def _apply_llm_prior(self, registry: FeatureRegistry, hard_mask: torch.Tensor, route_schema: dict | None) -> torch.Tensor:
        prior = LLMDAGPriorProvider(
            backend=str(self.config.get("llm_backend", "none")),
            model=self.config.get("llm_model"),
        ).propose_prior(registry, route_schema)
        node_index = registry.node_index
        for item in prior.get("forbidden_edges", []):
            src = item.get("src")
            dst = item.get("dst")
            if src in node_index and dst in node_index:
                hard_mask[node_index[src], node_index[dst]] = False
        for item in prior.get("edge_priors", []):
            if float(item.get("log_prior", 0.0)) > -999.0:
                continue
            src = item.get("src")
            dst = item.get("dst")
            if src in node_index and dst in node_index:
                hard_mask[node_index[src], node_index[dst]] = False
        return hard_mask

    def _train(self, model: "_TrajectoryBalancePolicy", context: "_GFlowContext") -> dict[str, Any]:
        train_steps = int(self.config.get("train_steps", 400))
        lr = float(self.config.get("lr", 5e-2))
        grad_clip = float(self.config.get("grad_clip_norm", 10.0))
        seed = int(self.config.get("seed", 0))
        generator = torch.Generator(device=context.values.device)
        generator.manual_seed(seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []
        terminal_rewards = []
        for _ in range(max(1, train_steps)):
            adjacency, log_pf, log_pb = model.sample_trajectory(context, generator=generator, training=True)
            log_reward = context.log_reward(adjacency).detach()
            loss = (model.log_z + log_pf - log_pb - log_reward).pow(2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            terminal_rewards.append(float(log_reward.detach().cpu()))
        return {
            "train_steps": max(1, train_steps),
            "initial_tb_loss": losses[0],
            "final_tb_loss": losses[-1],
            "mean_tb_loss": float(sum(losses) / max(1, len(losses))),
            "mean_terminal_log_reward": float(sum(terminal_rewards) / max(1, len(terminal_rewards))),
        }

    def _sample_graphs(
        self,
        model: "_TrajectoryBalancePolicy",
        context: "_GFlowContext",
    ) -> tuple[list[GraphSpec], torch.Tensor]:
        num_samples = int(self.config.get("num_samples", 32))
        seed = int(self.config.get("seed", 0)) + 10_000
        generator = torch.Generator(device=context.values.device)
        generator.manual_seed(seed)
        graphs = []
        log_rewards = []
        with torch.no_grad():
            for _ in range(max(1, num_samples)):
                adjacency, _, _ = model.sample_trajectory(context, generator=generator, training=False)
                graphs.append(_graph_spec(adjacency, context.hard_mask, context.registry))
                log_rewards.append(context.log_reward(adjacency))
        return graphs, torch.stack(log_rewards)

    def _diagnostics(
        self,
        registry: FeatureRegistry,
        hard_mask: torch.Tensor,
        edge_marginals: torch.Tensor,
        train_metrics: dict[str, Any],
        log_rewards: torch.Tensor,
        context: "_GFlowContext",
    ) -> dict[str, Any]:
        mechanisms = mechanism_specs_from_registry(
            registry,
            edge_marginals=edge_marginals.detach().cpu(),
            min_parent_prob=float(self.config.get("mechanism_parent_threshold", 0.3)),
        )
        low_confidence_edges = []
        node_names = registry.node_names
        for src_idx, src in enumerate(node_names):
            for dst_idx, dst in enumerate(node_names):
                prob = float(edge_marginals[src_idx, dst_idx].detach().cpu())
                if bool(hard_mask[src_idx, dst_idx]) and 0.3 <= prob <= 0.8:
                    low_confidence_edges.append({"src": src, "dst": dst, "probability": prob})
        return {
            "method": "dag_gflownet_tb",
            "node_names": node_names,
            "node_groups": registry.infer_groups(),
            "hard_mask": hard_mask.detach().cpu(),
            "train": train_metrics,
            "num_samples": int(log_rewards.numel()),
            "sample_log_reward_mean": float(log_rewards.mean().detach().cpu()),
            "sample_log_reward_max": float(log_rewards.max().detach().cpu()),
            "low_confidence_edges": low_confidence_edges,
            "mechanisms": mechanisms,
            "config": dict(self.config),
            "score_cache_size": context.cache_size,
        }


class _TrajectoryBalancePolicy(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        hard_mask: torch.Tensor,
        *,
        initial_stop_logit: float,
        stop_edge_count_slope: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.register_buffer("hard_mask", hard_mask.clone())
        self.edge_logits = nn.Parameter(torch.zeros(self.n_nodes, self.n_nodes))
        self.stop_logit = nn.Parameter(torch.tensor(float(initial_stop_logit)))
        self.stop_edge_count_slope = float(stop_edge_count_slope)
        self.log_z = nn.Parameter(torch.zeros(()))

    def initialize_edge_logits(self, logits: torch.Tensor) -> None:
        with torch.no_grad():
            self.edge_logits.copy_(logits.to(device=self.edge_logits.device, dtype=self.edge_logits.dtype))

    def sample_trajectory(
        self,
        context: "_GFlowContext",
        *,
        generator: torch.Generator,
        training: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adjacency = torch.zeros_like(self.hard_mask, dtype=torch.float32)
        log_pf = self.edge_logits.new_zeros(())
        log_pb = self.edge_logits.new_zeros(())
        max_edges = int(context.max_edges)
        for _ in range(max_edges + 1):
            valid = context.valid_add_mask(adjacency)
            valid_flat = torch.where(valid.reshape(-1))[0]
            if valid_flat.numel() == 0:
                break
            flat_logits = self.edge_logits.reshape(-1)[valid_flat]
            stop_logit = self.stop_logit + adjacency.sum() * self.stop_edge_count_slope
            logits = torch.cat([flat_logits, stop_logit.reshape(1)])
            log_probs = F.log_softmax(logits, dim=0)
            action_idx = _sample_index(log_probs, generator)
            log_pf = log_pf + log_probs[action_idx]
            if int(action_idx) == valid_flat.numel():
                break
            edge_flat = int(valid_flat[action_idx])
            src = edge_flat // self.n_nodes
            dst = edge_flat % self.n_nodes
            adjacency[src, dst] = 1.0
            log_pb = log_pb - torch.log(adjacency.sum().clamp_min(1.0))
        if not training:
            adjacency = adjacency.detach()
            log_pf = log_pf.detach()
            log_pb = log_pb.detach()
        return adjacency, log_pf, log_pb


class _GFlowContext:
    def __init__(
        self,
        *,
        values: torch.Tensor,
        registry: FeatureRegistry,
        hard_mask: torch.Tensor,
        max_parents: int,
        l2: float,
        complexity_penalty: float,
        reward_scale: float,
        min_log_reward: float,
        max_log_reward: float,
    ):
        self.values = values.to(dtype=torch.float32)
        self.registry = registry
        self.hard_mask = hard_mask
        self.max_parents = max(1, int(max_parents))
        self.max_edges = min(int(hard_mask.sum().detach().cpu()), self.max_parents * len(registry.child_node_indices()))
        self.l2 = float(l2)
        self.complexity_penalty = float(complexity_penalty)
        self.reward_scale = float(reward_scale)
        self.min_log_reward = float(min_log_reward)
        self.max_log_reward = float(max_log_reward)
        self.child_indices = registry.child_node_indices()
        self.parent_candidates = {
            child_idx: torch.where(hard_mask[:, child_idx])[0]
            for child_idx in self.child_indices
        }
        self._local_score_cache: dict[tuple[int, tuple[int, ...]], torch.Tensor] = {}

    @property
    def cache_size(self) -> int:
        return len(self._local_score_cache)

    def valid_add_mask(self, adjacency: torch.Tensor) -> torch.Tensor:
        valid = self.hard_mask & (adjacency <= 0.0)
        if self.max_parents > 0:
            parent_counts = adjacency.sum(dim=0)
            valid = valid & (parent_counts.reshape(1, -1) < self.max_parents)
        src_idx, dst_idx = torch.where(valid)
        for src, dst in zip(src_idx.tolist(), dst_idx.tolist()):
            if _would_create_cycle(adjacency, src, dst):
                valid[src, dst] = False
        return valid

    def log_reward(self, adjacency: torch.Tensor) -> torch.Tensor:
        score = self.values.new_zeros(())
        for child_idx in self.child_indices:
            parent_idx = torch.where(adjacency[:, child_idx] > 0.0)[0]
            score = score + self._local_score(child_idx, parent_idx)
        return (score * self.reward_scale).clamp(self.min_log_reward, self.max_log_reward)

    def _local_score(self, child_idx: int, parent_idx: torch.Tensor) -> torch.Tensor:
        key = (int(child_idx), tuple(sorted(int(idx) for idx in parent_idx.detach().cpu().tolist())))
        cached = self._local_score_cache.get(key)
        if cached is not None:
            return cached.to(device=self.values.device)
        y = _standardize(self.values[:, child_idx : child_idx + 1])
        null_mse = y.pow(2).mean().clamp_min(1e-8)
        if len(key[1]) == 0:
            score = self.values.new_zeros(())
        else:
            x = _standardize(self.values[:, list(key[1])])
            coeff = _ridge_coeff(x, y, self.l2)
            mse = F.mse_loss(x @ coeff, y).clamp_min(1e-8)
            improvement = 0.5 * self.values.shape[0] * (torch.log(null_mse) - torch.log(mse))
            penalty = self.complexity_penalty * len(key[1]) * torch.log(self.values.new_tensor(float(self.values.shape[0] + 1)))
            score = improvement - penalty
        self._local_score_cache[key] = score.detach().cpu()
        return score


def _ridge_warm_start_logits(
    values: torch.Tensor,
    registry: FeatureRegistry,
    hard_mask: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    logits = torch.full(hard_mask.shape, -8.0, dtype=torch.float32, device=values.device)
    for child_idx in registry.child_node_indices():
        parent_indices = torch.where(hard_mask[:, child_idx])[0]
        if parent_indices.numel() == 0:
            continue
        probs, _ = ridge_parent_edge_probabilities(
            values[:, parent_indices],
            values[:, child_idx : child_idx + 1],
            l2=float(config.get("ridge_l2", 1e-3)),
            score_threshold=float(config.get("score_threshold", 0.08)),
            score_temperature=float(config.get("score_temperature", 0.04)),
        )
        probs = apply_max_parents(probs, config.get("max_parents", 8)).clamp(1e-4, 1.0 - 1e-4)
        logits[parent_indices, child_idx] = torch.logit(probs).clamp(-6.0, 6.0)
    return torch.where(hard_mask, logits, torch.full_like(logits, -8.0))


def _graph_spec(adjacency: torch.Tensor, hard_mask: torch.Tensor, registry: FeatureRegistry) -> GraphSpec:
    edge_probs = torch.where(hard_mask, adjacency.to(dtype=torch.float32), torch.zeros_like(adjacency, dtype=torch.float32))
    return GraphSpec(
        node_names=registry.node_names,
        adjacency=edge_probs.detach().cpu(),
        hard_mask=hard_mask.detach().cpu(),
        edge_probs=edge_probs.detach().cpu(),
        node_groups=registry.node_groups(),
    )


def _standardize(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.to(dtype=torch.float32)
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (tensor - mean) / std


def _ridge_coeff(x: torch.Tensor, y: torch.Tensor, l2: float) -> torch.Tensor:
    eye = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
    return torch.linalg.solve(x.T @ x / max(1, x.shape[0]) + float(l2) * eye, x.T @ y / max(1, x.shape[0]))


def _sample_index(log_probs: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    probs = torch.softmax(log_probs.detach(), dim=0)
    return torch.multinomial(probs, num_samples=1, generator=generator).reshape(())


def _would_create_cycle(adjacency: torch.Tensor, src: int, dst: int) -> bool:
    if int(src) == int(dst):
        return True
    stack = [int(dst)]
    visited = set()
    adjacency_cpu = adjacency.detach().cpu()
    while stack:
        node = stack.pop()
        if node == int(src):
            return True
        if node in visited:
            continue
        visited.add(node)
        children = torch.where(adjacency_cpu[node] > 0.0)[0].tolist()
        stack.extend(int(child) for child in children)
    return False
