from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from torch import Tensor

from .model import ModelState, validate_state


def sign_flip(delta: Mapping[str, Tensor], scale: float) -> ModelState:
    if scale <= 0:
        raise ValueError("sign-flip scale must be positive")
    validate_state(delta)
    attacked = OrderedDict((key, -scale * value) for key, value in delta.items())
    validate_state(attacked)
    return attacked

