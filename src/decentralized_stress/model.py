from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn

from .data import Records


ModelState = OrderedDict[str, Tensor]


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features).squeeze(-1)


def make_initial_state(input_dim: int, hidden_dim: int, seed: int) -> ModelState:
    torch.manual_seed(seed)
    model = TinyMLP(input_dim, hidden_dim)
    return clone_state(model.state_dict())


def clone_state(state: Mapping[str, Tensor]) -> ModelState:
    return OrderedDict((key, value.detach().cpu().clone()) for key, value in state.items())


def validate_state(state: Mapping[str, Tensor]) -> None:
    if not state:
        raise ValueError("Model state cannot be empty")
    for key, value in state.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"State value {key} is not a tensor")
        if not torch.isfinite(value).all():
            raise ValueError(f"State value {key} contains non-finite values")


def train_local(
    state: Mapping[str, Tensor],
    records: Records,
    *,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[ModelState, float]:
    validate_state(state)
    if len(records) == 0:
        raise ValueError("A peer cannot train on an empty dataset")
    model = TinyMLP(records.feature_count, hidden_dim)
    model.load_state_dict(state, strict=True)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)
    features = torch.from_numpy(records.x.astype(np.float32, copy=False))
    labels = torch.from_numpy(records.y.astype(np.float32, copy=False))
    positives = float(torch.sum(labels))
    negatives = float(len(labels) - positives)
    # The class weight is computed from this peer's own rows only. It prevents the
    # 0.5 decision threshold from degenerating to the majority class without
    # pooling label counts across peers.
    positive_weight = negatives / positives if positives > 0.0 and negatives > 0.0 else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, dtype=torch.float32))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    losses: list[float] = []
    for _ in range(epochs):
        order = torch.randperm(len(records), generator=generator)
        for start in range(0, len(records), batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[indices])
            loss = criterion(logits, labels[indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite local training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    updated = clone_state(model.state_dict())
    delta = OrderedDict((key, updated[key] - state[key]) for key in updated)
    validate_state(delta)
    return delta, float(np.mean(losses))


def apply_delta(state: Mapping[str, Tensor], delta: Mapping[str, Tensor]) -> ModelState:
    if tuple(state) != tuple(delta):
        raise ValueError("State and delta keys differ")
    result = OrderedDict()
    for key in state:
        if state[key].shape != delta[key].shape:
            raise ValueError(f"Shape mismatch for {key}")
        result[key] = state[key] + delta[key]
    validate_state(result)
    return result


def predict(state: Mapping[str, Tensor], records: Records, hidden_dim: int) -> tuple[np.ndarray, np.ndarray]:
    validate_state(state)
    model = TinyMLP(records.feature_count, hidden_dim)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        features = torch.from_numpy(records.x.astype(np.float32, copy=False))
        logits = model(features)
        probabilities = torch.sigmoid(logits)
        labels = torch.from_numpy(records.y.astype(np.float32, copy=False))
        losses = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return probabilities.numpy().astype(np.float64), losses.numpy().astype(np.float64)


def delta_payload_bytes(delta: Mapping[str, Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in delta.values())


def mean_pairwise_state_distance(states: Mapping[int, Mapping[str, Tensor]]) -> float:
    ids = sorted(states)
    if len(ids) < 2:
        return 0.0
    distances: list[float] = []
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            squared = 0.0
            count = 0
            for key in states[left]:
                difference = states[left][key] - states[right][key]
                squared += float(torch.sum(difference * difference))
                count += difference.numel()
            distances.append((squared / max(count, 1)) ** 0.5)
    return float(np.mean(distances))
