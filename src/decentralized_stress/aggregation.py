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


def trimming_diagnostics(
    deltas: Sequence[Mapping[str, Tensor]], *, trim_count: int
) -> list[dict[str, float | int]]:
    """Inspect coordinate rejection without changing the aggregation calculation.

    Exact index rejection follows torch.sort's tie order. Fractional rejection
    shares the rejected count equally among equal values, avoiding a sender-ID
    interpretation for a boundary tie. Neither measure labels an update useful.
    """
    keys = _validate_deltas(deltas)
    peers = len(deltas)
    if trim_count < 0 or peers <= 2 * trim_count:
        raise ValueError("trimming diagnostics require a non-empty retained set")
    counts = [dict(coordinates=0, rejected_coordinates=0,
                   ambiguous_boundary_coordinates=0, tied_coordinates=0,
                   tie_averaged_rejected_coordinates=0.0, update_squared_norm=0.0)
              for _ in deltas]
    for key in keys:
        stacked = torch.stack([delta[key].reshape(-1) for delta in deltas])
        indices = torch.sort(stacked, dim=0).indices
        rejected = torch.zeros_like(stacked, dtype=torch.bool)
        if trim_count:
            rejected.scatter_(0, indices[:trim_count], True)
            rejected.scatter_(0, indices[peers - trim_count:], True)
        for sender in range(peers):
            equals = stacked == stacked[sender].unsqueeze(0)
            multiplicity = equals.sum(dim=0)
            removed_equal = (equals & rejected).sum(dim=0)
            fractional = removed_equal.to(torch.float64) / multiplicity
            counts[sender]["coordinates"] += stacked.shape[1]
            counts[sender]["rejected_coordinates"] += int(rejected[sender].sum())
            counts[sender]["tied_coordinates"] += int((multiplicity > 1).sum())
            counts[sender]["ambiguous_boundary_coordinates"] += int(
                ((removed_equal > 0) & (removed_equal < multiplicity)).sum())
            counts[sender]["tie_averaged_rejected_coordinates"] += float(fractional.sum())
            counts[sender]["update_squared_norm"] += float(stacked[sender].to(torch.float64).square().sum())
    for row in counts:
        row["rejection_rate"] = row["rejected_coordinates"] / row["coordinates"]
        row["tie_averaged_rejection_rate"] = row["tie_averaged_rejected_coordinates"] / row["coordinates"]
        row["update_l2_norm"] = row.pop("update_squared_norm") ** 0.5
    return counts

