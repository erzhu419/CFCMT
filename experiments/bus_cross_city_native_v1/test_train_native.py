"""Actual transfer hazards: city IDs, arrival semantics, split roles and 58 stops."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import torch
from torch import nn

from train_native import (
    EventModel, LINE_COLUMNS_PARAMETER, ModelContinuation, Normalizer,
    RESET_TRANSFER_COLUMNS, TRANSFER_MODULES, continuation_trainer,
    event_targets, load_model_roles, prediction_losses, transfer_parameters,
)


class Encoder(nn.Module):
    def forward(self, states):
        return states.new_zeros((len(states), 256))


def raw_pair():
    state = np.zeros((2, 229), np.float32)
    state[:, 1] = 1/16
    state[:, 5] = 1
    state[:, 7] = 1
    return dict(s=state, a=np.zeros((2, 1), np.float32), ns=state.copy(),
        dt=np.ones(2, np.float32), r=-np.ones(2, np.float32))


def test_reachable_station_57_fits_and_decodes_without_clamping():
    model = EventModel(Encoder, 58)
    stations = torch.full((12,), 57.)
    targets = event_targets(raw_pair(), stations, "cpu")
    norm = Normalizer(targets)
    assert targets["station"][0, 0].item() == 57
    output = model(targets["s"], targets["a"])
    assert torch.isfinite(prediction_losses(model, output, targets, norm)).all()
    augmentation = ModelContinuation.__new__(ModelContinuation)
    augmentation.models, augmentation.norm = [model], norm
    augmentation.stations, augmentation.device = stations, "cpu"
    augmentation.p = dict(horizon_seconds=24000)
    augmentation.generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        prediction = torch.zeros_like(output)
        parts = model.unpack(prediction)
        parts["station"].fill_(-1000)
        parts["station"][:, :, 57] = 1000
        decoded, _, _ = augmentation.decode(targets["s"], prediction)
    slots = decoded[:, 5:].reshape(-1, 16, 14)
    assert torch.equal(slots[:, 0, 2], torch.ones(2))


def test_source_transfers_shared_features_and_resets_city_and_rate_columns():
    class Network:
        def __init__(self):
            self.state = {LINE_COLUMNS_PARAMETER: torch.ones(3, 22), "bias": torch.ones(3)}
        def state_dict(self):
            return self.state
        def load_state_dict(self, state):
            self.state = state
    learner = SimpleNamespace(**{name: Network() for name in TRANSFER_MODULES})
    payload = {name: {LINE_COLUMNS_PARAMETER: torch.full((3, 22), 7.), "bias": torch.full((3,), 9.)}
        for name in TRANSFER_MODULES}
    transfer_parameters(learner, payload, reset_line_columns=True)
    for name in TRANSFER_MODULES:
        state = getattr(learner, name).state_dict()
        assert torch.equal(state[LINE_COLUMNS_PARAMETER][:, RESET_TRANSFER_COLUMNS], torch.ones(3, 13))
        assert torch.all(state[LINE_COLUMNS_PARAMETER][:, [12, 13, 14, 15, 16, 17, 18, 20, 21]] == 7)
        assert torch.all(state["bias"] == 9)
        assert torch.all(payload[name][LINE_COLUMNS_PARAMETER] == 7)


def test_world_model_uses_explicit_h5_roles_and_rejects_zero_duration(tmp_path):
    path = tmp_path / "pool.h5"
    with h5py.File(path, "w") as data:
        data["split_id"] = [0, 0, 0, 0, 1, 1, 2, 2]
        data["delta_t_seconds"] = [1, 1, 0, 1, 1, 1, 1, 1]
        data["terminals"] = [0, 0, 0, 1, 0, 1, 0, 0]
        for name in ("observations", "next_observations"):
            data[name] = np.zeros((8, 229), np.float32)
        data["actions"] = np.zeros((8, 1), np.float32)
        data["rewards"] = -np.arange(8, dtype=np.float32)
        data.attrs["station_max_by_line"] = json.dumps([57] * 12)
    raw, rows, _ = load_model_roles(path, 123, 8192)
    assert rows["fit"].tolist() == [0, 1]
    assert rows["stop"].tolist() == [4]
    assert rows["calibrate"].tolist() == [6, 7]
    assert raw["calibrate"]["r"].tolist() == [-6, -7]


def test_model_td_coefficient_is_quarter_of_one_critic_loss():
    class Base:
        def _ensemble_td_loss(self, predictions, targets, weights=None):
            return (predictions - targets).square().mean()
    learner = continuation_trainer(Base)()
    real_prediction = torch.tensor([3.], requires_grad=True)
    model_prediction = torch.tensor([4.], requires_grad=True)
    learner.model_loss = model_prediction.square().mean()
    learner.model_coefficient = .25
    loss = learner._ensemble_td_loss(real_prediction, torch.tensor([0.]))
    loss.backward()
    assert loss.item() == 10.75
    assert real_prediction.grad.item() == 4.5
    assert model_prediction.grad.item() == 2.
