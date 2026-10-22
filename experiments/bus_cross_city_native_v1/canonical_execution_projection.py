"""Execution-action interval for the frozen v7 quarter-load-sixty guard.

The actor represents executed holding, not a proposal to be divided by eight.
The continuous monotone guard's image is [g_s(-1), g_s(+1)].  Its sixty-second
tail floor subsumes the v17 twenty-second floor.  All other states have zero as
their lower endpoint and the v16 headway reserve as their upper endpoint.

Canonical v7 states have five global columns and sixteen fourteen-column
event slots.  Their headway ratios are capped at five by the physical contract;
this projection uses that same canonical representation for model states.
"""

from __future__ import annotations

import torch


def canonical_execution_bounds(states, config):
    """Return normalized execution endpoints for canonical states [B, 229].

State gradients are unnecessary.  The current slot index is scaled by fifteen,
targets by the configured scheduled-headway scale, and demand by positive
capacity and mean target headway.  Thus a positive canonical demand column is
equivalent to a positive observed eligible-arrival rate.
"""
    states = states.detach()
    slots = states[:, 5:].reshape(-1, 16, 14)
    current = (states[:, 2] * 15).round().long()
    event = slots[torch.arange(len(states), device=states.device), current]
    front = event[:, 11] > .5
    back = event[:, 12] > .5
    target_front = event[:, 5] * config.scheduled_headway_scale_seconds
    target_back = event[:, 6] * config.scheduled_headway_scale_seconds
    front_error = (event[:, 3] - 1) * target_front * front
    back_error = (event[:, 4] - 1) * target_back * back
    reserve = (.5 * (back_error - front_error)).clamp(0, config.max_hold_seconds)
    base = torch.full_like(reserve, .125 * config.max_hold_seconds)
    upper_seconds = torch.where(states[:, 0] < .75, torch.maximum(base, reserve), base)
    floor = ((states[:, 0] >= .15) & (states[:, 0] < .75)
             & front & ~back & (event[:, 9] > 0) & (event[:, 8] <= .25))
    lower_seconds = torch.where(floor, config.max_hold_seconds, 0.)
    upper_seconds = torch.where(floor, config.max_hold_seconds, upper_seconds)
    encode = lambda seconds: (2 * seconds / config.max_hold_seconds - 1).unsqueeze(-1)
    return encode(lower_seconds), encode(upper_seconds)


def project_canonical_executed_action(states, desired_actions, config):
    """Clip actions [B, 1] to the interval, retaining action gradients."""
    lower, upper = canonical_execution_bounds(states, config)
    return torch.clamp(desired_actions, min=lower, max=upper)
