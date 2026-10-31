"""Adapters from copied H2Oplus replay-buffer batches to CF-H2O schemas."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import torch

from cf_h2o.schemas import TransitionBatch


def _as_source_list(source: Optional[str | list[str]], batch_size: int) -> Optional[list[str]]:
    if source is None:
        return None
    if isinstance(source, str):
        return [source] * batch_size
    return list(source)


def transition_batch_from_h2o(
    batch: Mapping[str, Any],
    *,
    source: Optional[str | list[str]] = None,
    domain_id: Optional[torch.Tensor] = None,
    line_id: Optional[list[str]] = None,
    route_id: Optional[list[str]] = None,
    snapshot_t: Optional[list[dict[str, Any]]] = None,
    snapshot_t1: Optional[list[dict[str, Any]]] = None,
) -> TransitionBatch:
    """Convert a H2Oplus sample dict into a `TransitionBatch`.

    Required keys mirror `BusMixedReplayBuffer.sample()`:
    `observations`, `actions`, `rewards`, `next_observations`, `dones`.
    Optional keys include `z_t`, `z_t1`, `_indices`, and metadata. Existing
    tensors are preserved on their current device.
    """

    required = ("observations", "actions", "rewards", "next_observations", "dones")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"H2O batch is missing required keys: {missing}")

    observations = batch["observations"]
    batch_size = int(observations.shape[0])
    metadata = {
        key: value
        for key, value in batch.items()
        if key not in {
            "observations",
            "actions",
            "rewards",
            "next_observations",
            "dones",
            "z_t",
            "z_t1",
        }
    }

    return TransitionBatch(
        observations=observations,
        actions=batch["actions"],
        rewards=batch["rewards"],
        next_observations=batch["next_observations"],
        dones=batch["dones"],
        z_t=batch.get("z_t"),
        z_t1=batch.get("z_t1"),
        domain_id=domain_id if domain_id is not None else batch.get("domain_id"),
        line_id=line_id if line_id is not None else batch.get("line_id"),
        route_id=route_id if route_id is not None else batch.get("route_id"),
        global_time=batch.get("global_time"),
        snapshot_t=snapshot_t if snapshot_t is not None else batch.get("snapshot_t"),
        snapshot_t1=snapshot_t1 if snapshot_t1 is not None else batch.get("snapshot_t1"),
        source=_as_source_list(source if source is not None else batch.get("source"), batch_size),
        metadata=metadata,
    )


def sample_transition_batch(
    replay_buffer: Any,
    batch_size: int,
    *,
    scope: Optional[str] = None,
    source: Optional[str] = None,
) -> TransitionBatch:
    """Sample a H2Oplus replay buffer and return a `TransitionBatch`."""

    return transition_batch_from_h2o(
        replay_buffer.sample(batch_size, scope=scope),
        source=source if source is not None else scope,
    )
