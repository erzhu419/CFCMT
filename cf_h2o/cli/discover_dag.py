"""CLI for Stage 2 automatic DAG discovery."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from cf_h2o.graph.auto_dag import AutoDAGDiscoverer
from cf_h2o.graph.dag_gflownet import DAGGFlowNetDiscoverer
from cf_h2o.graph.feature_registry import FeatureRegistry
from cf_h2o.schemas import TransitionBatch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Discover a CF-H2O temporal graph posterior.")
    parser.add_argument("--offline-buffer", type=Path, default=None)
    parser.add_argument("--sim-buffer", type=Path, default=None)
    parser.add_argument("--route-schema", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["ridge_bootstrap", "dag_gflownet"], default="ridge_bootstrap")
    parser.add_argument("--llm-backend", default="none")
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--gflownet-train-steps", type=int, default=400)
    parser.add_argument("--gflownet-samples", type=int, default=32)
    parser.add_argument("--max-parents", type=int, default=8)
    parser.add_argument("--obs-names", default=None, help="Comma-separated observation names.")
    parser.add_argument("--action-names", default=None, help="Comma-separated action names.")
    args = parser.parse_args(argv)

    batches = []
    for path in (args.offline_buffer, args.sim_buffer):
        if path is not None:
            batches.append(_load_transition_batch(path))
    if not batches:
        raise SystemExit("At least one of --offline-buffer or --sim-buffer is required.")

    dataset = _concat_batches(batches)
    obs_names = args.obs_names.split(",") if args.obs_names else None
    action_names = args.action_names.split(",") if args.action_names else None
    registry = FeatureRegistry.from_transition_dataset(dataset, obs_names=obs_names, action_names=action_names)
    route_schema = json.loads(args.route_schema.read_text(encoding="utf-8")) if args.route_schema else None
    config = {
        "llm_backend": args.llm_backend,
        "max_parents": args.max_parents,
    }
    if args.method == "dag_gflownet":
        config.update(
            {
                "train_steps": args.gflownet_train_steps,
                "num_samples": args.gflownet_samples,
            }
        )
        discoverer = DAGGFlowNetDiscoverer(config)
    else:
        config["bootstrap_runs"] = args.bootstrap
        discoverer = AutoDAGDiscoverer(config)
    posterior = discoverer.fit(dataset, registry=registry, route_schema=route_schema)
    outputs = discoverer.save(posterior, args.out_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


def _load_transition_batch(path: Path) -> TransitionBatch:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if isinstance(value, TransitionBatch):
        return value
    if isinstance(value, dict):
        return TransitionBatch(**value)
    raise TypeError(f"Unsupported buffer type in {path}: {type(value)!r}")


def _concat_batches(batches: list[TransitionBatch]) -> TransitionBatch:
    if len(batches) == 1:
        return batches[0]

    def cat(name: str):
        values = [getattr(batch, name) for batch in batches]
        if any(value is None for value in values):
            return None
        return torch.cat(values, dim=0)

    source = []
    for batch in batches:
        if batch.source is not None:
            source.extend(batch.source)

    metadata: dict[str, Any] = {}
    for batch in batches:
        metadata.update(batch.metadata)

    return TransitionBatch(
        observations=cat("observations"),
        actions=cat("actions"),
        rewards=cat("rewards"),
        next_observations=cat("next_observations"),
        dones=cat("dones"),
        z_t=cat("z_t"),
        z_t1=cat("z_t1"),
        domain_id=cat("domain_id"),
        global_time=cat("global_time"),
        source=source or None,
        metadata=metadata,
    )


if __name__ == "__main__":
    raise SystemExit(main())
