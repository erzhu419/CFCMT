from __future__ import annotations

import json

import torch

from cf_h2o.graph.auto_dag import AutoDAGDiscoverer
from cf_h2o.graph.feature_registry import FeatureRegistry
from cf_h2o.graph.validation import edge_auc, forbidden_edge_max_probability
from cf_h2o.schemas import TransitionBatch


OBS_NAMES = ["waiting", "load", "dwell", "travel_time", "headway", "speed"]
ACTION_NAMES = ["holding"]


def _make_synthetic_bus_batch(n_samples: int = 512) -> tuple[TransitionBatch, set[tuple[str, str]]]:
    generator = torch.Generator().manual_seed(11)
    waiting = torch.randn(n_samples, generator=generator)
    load = torch.randn(n_samples, generator=generator)
    dwell = torch.randn(n_samples, generator=generator)
    travel_time = torch.randn(n_samples, generator=generator)
    headway = torch.randn(n_samples, generator=generator)
    speed = torch.randn(n_samples, generator=generator)
    holding = torch.randn(n_samples, generator=generator)
    noise = lambda scale=0.03: scale * torch.randn(n_samples, generator=generator)

    waiting_next = 0.88 * waiting + noise()
    load_next = 0.72 * load + 0.35 * waiting - 0.20 * holding + noise()
    dwell_next = 0.80 * waiting + 0.42 * load + 0.55 * holding + noise()
    travel_next = 0.76 * travel_time - 0.45 * speed + noise()
    headway_next = 0.74 * headway + 0.48 * dwell + 0.50 * travel_time - 0.44 * holding + noise()
    speed_next = 0.90 * speed + noise()
    reward = -0.65 * waiting - 0.55 * headway - 0.25 * dwell + 0.15 * holding + noise()

    observations = torch.stack([waiting, load, dwell, travel_time, headway, speed], dim=1)
    next_observations = torch.stack([waiting_next, load_next, dwell_next, travel_next, headway_next, speed_next], dim=1)
    actions = holding.reshape(-1, 1)
    dones = torch.zeros(n_samples)

    true_edges = {
        ("waiting@t", "waiting@t1"),
        ("load@t", "load@t1"),
        ("waiting@t", "load@t1"),
        ("holding@t", "load@t1"),
        ("waiting@t", "dwell@t1"),
        ("load@t", "dwell@t1"),
        ("holding@t", "dwell@t1"),
        ("travel_time@t", "travel_time@t1"),
        ("speed@t", "travel_time@t1"),
        ("headway@t", "headway@t1"),
        ("dwell@t", "headway@t1"),
        ("travel_time@t", "headway@t1"),
        ("holding@t", "headway@t1"),
        ("speed@t", "speed@t1"),
        ("waiting@t", "reward@t1"),
        ("headway@t", "reward@t1"),
        ("dwell@t", "reward@t1"),
        ("holding@t", "reward@t1"),
    }
    batch = TransitionBatch(
        observations=observations,
        actions=actions,
        rewards=reward,
        next_observations=next_observations,
        dones=dones,
        metadata={"obs_names": OBS_NAMES, "action_names": ACTION_NAMES},
    )
    return batch, true_edges


def test_feature_registry_builds_temporal_nodes_and_forbidden_mask():
    batch, _ = _make_synthetic_bus_batch(32)
    registry = FeatureRegistry.from_transition_dataset(batch)
    node_index = registry.node_index
    hard_mask = registry.build_temporal_hard_mask()

    assert "waiting@t" in node_index
    assert "waiting@t1" in node_index
    assert "holding@t" in node_index
    assert "reward@t1" in node_index
    assert bool(hard_mask[node_index["waiting@t"], node_index["dwell@t1"]])
    assert not bool(hard_mask[node_index["waiting@t1"], node_index["waiting@t"]])
    assert not bool(hard_mask[node_index["reward@t1"], node_index["waiting@t"]])
    assert registry.values_from_dataset(batch).shape == (32, len(registry.node_names))


def test_auto_dag_synthetic_recovers_allowed_edges_and_saves_outputs(tmp_path):
    batch, true_edge_names = _make_synthetic_bus_batch()
    registry = FeatureRegistry.from_transition_dataset(batch)
    discoverer = AutoDAGDiscoverer(
        {
            "bootstrap_runs": 8,
            "max_parents": 8,
            "score_threshold": 0.08,
            "score_temperature": 0.04,
            "seed": 5,
        }
    )

    posterior = discoverer.fit(batch, registry=registry)
    node_index = registry.node_index
    hard_mask = posterior.graphs[0].hard_mask
    true_edges = {(node_index[src], node_index[dst]) for src, dst in true_edge_names}
    auc = edge_auc(posterior.edge_marginals, true_edges, hard_mask)

    assert posterior.edge_marginals.shape == (len(registry.node_names), len(registry.node_names))
    assert len(posterior.graphs) == 8
    assert forbidden_edge_max_probability(posterior.edge_marginals, hard_mask) < 0.05
    assert auc > 0.75

    outputs = discoverer.save(posterior, tmp_path)
    assert outputs["graph_posterior"].exists()
    assert outputs["edge_report"].exists()
    assert outputs["discovered_mechanisms"].exists()
    saved = json.loads(outputs["graph_posterior"].read_text(encoding="utf-8"))
    assert saved["node_names"] == registry.node_names
    assert saved["diagnostics"]["bootstrap_runs"] == 8
