"""Serialization and reporting helpers for graph posterior outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from cf_h2o.schemas import GraphPosterior, GraphSpec


def sample_graph_from_edge_probs(
    edge_probs: torch.Tensor,
    hard_mask: torch.Tensor,
    node_names: list[str],
    node_groups: dict[str, list[int]],
    *,
    generator: torch.Generator | None = None,
) -> GraphSpec:
    random_values = torch.rand(edge_probs.shape, generator=generator, device=edge_probs.device, dtype=edge_probs.dtype)
    adjacency = ((random_values < edge_probs) & hard_mask).to(dtype=edge_probs.dtype)
    return GraphSpec(
        node_names=list(node_names),
        adjacency=adjacency,
        hard_mask=hard_mask.clone(),
        edge_probs=edge_probs.clone(),
        node_groups={key: list(value) for key, value in node_groups.items()},
    )


def graph_posterior_to_dict(posterior: GraphPosterior) -> dict[str, Any]:
    first = posterior.graphs[0] if posterior.graphs else None
    node_names = first.node_names if first is not None else posterior.diagnostics.get("node_names", [])
    hard_mask = first.hard_mask if first is not None else posterior.diagnostics.get("hard_mask")
    if isinstance(hard_mask, torch.Tensor):
        hard_mask_list = hard_mask.cpu().tolist()
    else:
        hard_mask_list = hard_mask or []

    graphs = []
    for idx, graph in enumerate(posterior.graphs):
        edges = _edge_list(graph.adjacency, graph.node_names)
        weight = float(torch.softmax(posterior.log_weights, dim=0)[idx].detach().cpu()) if posterior.log_weights.numel() else 1.0
        graphs.append({"weight": weight, "edges": edges})

    return {
        "version": "0.1",
        "node_names": list(node_names),
        "edge_marginals": posterior.edge_marginals.detach().cpu().tolist(),
        "hard_mask": hard_mask_list,
        "graphs": graphs,
        "diagnostics": _json_safe(posterior.diagnostics),
    }


def save_graph_posterior(posterior: GraphPosterior, out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "graph_posterior.json"
    target.write_text(json.dumps(graph_posterior_to_dict(posterior), indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def write_edge_report(posterior: GraphPosterior, out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "edge_report.md"
    first = posterior.graphs[0] if posterior.graphs else None
    node_names = first.node_names if first is not None else posterior.diagnostics.get("node_names", [])
    hard_mask = first.hard_mask if first is not None else torch.zeros_like(posterior.edge_marginals, dtype=torch.bool)
    edge_probs = posterior.edge_marginals.detach().cpu()
    hard_mask_cpu = hard_mask.detach().cpu()

    high: list[tuple[str, float]] = []
    uncertain: list[tuple[str, float]] = []
    rejected: list[tuple[str, float, str]] = []
    for src_idx, src in enumerate(node_names):
        for dst_idx, dst in enumerate(node_names):
            prob = float(edge_probs[src_idx, dst_idx])
            edge = f"{src} -> {dst}"
            if not bool(hard_mask_cpu[src_idx, dst_idx]):
                if prob > 0.0 or _interesting_forbidden(src, dst):
                    rejected.append((edge, prob, "forbidden by temporal/leakage mask"))
                continue
            if prob > 0.8:
                high.append((edge, prob))
            elif prob >= 0.3:
                uncertain.append((edge, prob))
            else:
                rejected.append((edge, prob, "low posterior probability"))

    lines = [
        "# Edge Report",
        "",
        "## High-confidence edges",
        "",
        "| Edge | Probability | Type | Reason |",
        "|---|---:|---|---|",
    ]
    lines.extend([f"| {edge} | {prob:.3f} | learned | stable across bootstrap |" for edge, prob in sorted(high)])
    if not high:
        lines.append("| none | 0.000 | learned | no p > 0.8 edges |")

    lines.extend(["", "## Uncertain edges", "", "| Edge | Probability | Warning |", "|---|---:|---|"])
    lines.extend([f"| {edge} | {prob:.3f} | keep as soft parent mask |" for edge, prob in sorted(uncertain)])
    if not uncertain:
        lines.append("| none | 0.000 | no 0.3 <= p <= 0.8 edges |")

    lines.extend(["", "## Rejected / forbidden edges", "", "| Edge | Probability | Reason |", "|---|---:|---|"])
    for edge, prob, reason in sorted(rejected)[:200]:
        lines.append(f"| {edge} | {prob:.3f} | {reason} |")

    warnings = posterior.diagnostics.get("warnings", [])
    low_parent = posterior.diagnostics.get("features_with_no_stable_parent", [])
    lines.extend(["", "## Shortcut warnings", ""])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    else:
        lines.append("- none")
    lines.extend(["", "## Features with no stable parent", ""])
    if low_parent:
        lines.extend([f"- {item}" for item in low_parent])
    else:
        lines.append("- none")

    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_discovered_mechanisms(mechanisms: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target = out_path / "discovered_mechanisms.yaml"
    lines = ["mechanisms:"]
    for mechanism in mechanisms:
        lines.append(f"  - name: {mechanism['name']}")
        lines.append("    child_names:")
        for child in mechanism.get("child_names", []):
            lines.append(f"      - {child}")
        lines.append("    parent_names:")
        for parent in mechanism.get("parent_names", []):
            lines.append(f"      - {parent}")
        lines.append(f"    latent_dim: {int(mechanism.get('latent_dim', 1))}")
        lines.append(f"    output_dim: {int(mechanism.get('output_dim', 1))}")
        lines.append(f"    loss_type: {mechanism.get('loss_type', 'mse')}")
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _edge_list(adjacency: torch.Tensor, node_names: list[str]) -> list[list[str]]:
    adjacency_cpu = adjacency.detach().cpu()
    edges: list[list[str]] = []
    for src_idx, src in enumerate(node_names):
        for dst_idx, dst in enumerate(node_names):
            if float(adjacency_cpu[src_idx, dst_idx]) > 0.0:
                edges.append([src, dst])
    return edges


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _interesting_forbidden(src: str, dst: str) -> bool:
    return src.endswith("@t1") or dst.endswith("@t") or src.startswith("reward@")

