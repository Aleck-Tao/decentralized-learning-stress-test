from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from .model import ModelState, validate_state


def _validate_deltas(deltas: Sequence[Mapping[str, Tensor]]) -> tuple[str, ...]:
    if not deltas:
        raise ValueError("At least one delta is required")
    keys = tuple(deltas[0])
    for delta in deltas:
        validate_state(delta)
        if tuple(delta) != keys:
            raise ValueError("Delta keys differ")
        for key in keys:
            if delta[key].shape != deltas[0][key].shape:
                raise ValueError(f"Delta shape mismatch for {key}")
    return keys


def aggregate_deltas(
    deltas: Sequence[Mapping[str, Tensor]],
    method: str,
    *,
    trim_count: int,
) -> ModelState:
    keys = _validate_deltas(deltas)
    result: ModelState = OrderedDict()
    for key in keys:
        stacked = torch.stack([delta[key] for delta in deltas], dim=0)
        if method == "peer_mean":
            value = torch.mean(stacked, dim=0)
        elif method == "coordinate_median":
            sorted_values = torch.sort(stacked, dim=0).values
            midpoint = len(deltas) // 2
            if len(deltas) % 2:
                value = sorted_values[midpoint]
            else:
                value = (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0
        elif method == "trimmed_mean":
            if trim_count < 0 or len(deltas) <= 2 * trim_count:
                raise ValueError("trimmed_mean has insufficient messages for trim_count")
            sorted_values = torch.sort(stacked, dim=0).values
            retained = sorted_values[trim_count : len(deltas) - trim_count]
            value = torch.mean(retained, dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        result[key] = value
    validate_state(result)
    return result

