"""Checks for diagnostic role isolation and matched bootstrap randomness."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from diagnose_values import ARRAYS, read_roles, read_source_samples, value_arrays


def test_roles_are_explicit_and_source_terminal_anchors_are_kept(tmp_path):
    path = tmp_path / "pool.h5"
    with h5py.File(path, "w") as data:
        data["split_id"] = np.repeat([0, 1, 2], [300, 2, 3])
        data.attrs["config"] = json.dumps(dict(horizon_seconds=24000))
        data.attrs["station_max_by_line"] = json.dumps([30]*12)
        for key in ARRAYS:
            data[key] = np.arange(305, dtype=np.float32)
        data["terminals"][:] = 0
        data["terminals"][[13, 149, 299]] = 1
    roles, _, _ = read_roles(path)
    assert roles["stop"]["observations"].tolist() == [300, 301]
    assert roles["calibrate"]["observations"].tolist() == [302, 303, 304]
    source = read_source_samples(path)
    assert len(source["uniform256"]["row_indices"]) == 256
    assert source["all_terminal_tails"]["row_indices"].tolist() == [13, 149, 299]


def test_matched_draws_do_not_turn_random_bootstrap_into_model_bias():
    class Actor:
        def __call__(self, states, deterministic):
            return torch.zeros((len(states), 1)), None

    class Learner:
        actor = Actor()
        value = staticmethod(lambda states: states[:, 0])
        critic = staticmethod(lambda states, actions: torch.stack((states[:, 0], states[:, 0]+1)))
        target_critic = critic

        def _target(self, batch, backup):
            bootstrap = self.value(batch["next_observations"])
            if backup == "h2oplus":
                bootstrap = bootstrap+torch.rand(len(bootstrap))+float(self.rng.random())
            return batch["rewards"]+batch["discounts"]*(1-batch["terminals"])*bootstrap

    batch = dict(observations=torch.tensor([[0.], [1.]]), actions=torch.zeros(2, 1),
                 next_observations=torch.tensor([[2.], [3.]]), rewards=torch.tensor([-2., -4.]),
                 discounts=torch.ones(2), terminals=torch.tensor([0., 1.]),
                 delta_t_seconds=torch.ones(2))
    model = dict(batch, next_observations=batch["next_observations"]+10)
    learner = Learner()
    actual = value_arrays(learner, batch, 123)
    predicted = value_arrays(learner, model, 123)
    assert np.allclose(actual["iql_target"], [0., -4.])
    assert np.allclose(predicted["mixed_entropy_target"]-actual["mixed_entropy_target"], [10., 0.])
