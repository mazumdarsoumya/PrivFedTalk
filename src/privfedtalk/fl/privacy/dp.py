# /home/vineet/PycharmProjects/PrivFedTalk/src/privfedtalk/fl/privacy/dp.py

from __future__ import annotations

from typing import Optional

import torch

from privfedtalk.fl.utils.adapter_state import (
    AdapterState,
    flatten_state,
    unflatten_state,
    state_l2_norm,
)


def clip_and_noise_adapter_delta(
    delta: AdapterState,
    clip_norm: float,
    noise_multiplier: float,
    generator: Optional[torch.Generator] = None,
) -> AdapterState:
    """
    Client-level DP on adapter deltas:
      1. flatten delta
      2. clip to L2 norm C
      3. add Gaussian noise N(0, (sigma*C)^2 I)
      4. reshape back
    """
    flat, meta = flatten_state(delta)
    if flat.numel() == 0:
        return {}

    norm = torch.norm(flat, p=2)
    if clip_norm is not None and clip_norm > 0:
        scale = min(1.0, float(clip_norm) / (float(norm.item()) + 1e-12))
        flat = flat * scale

    if noise_multiplier is not None and noise_multiplier > 0 and clip_norm is not None and clip_norm > 0:
        std = float(noise_multiplier) * float(clip_norm)
        noise = torch.randn(
            flat.shape,
            device=flat.device,
            dtype=flat.dtype,
            generator=generator,
        ) * std
        flat = flat + noise

    return unflatten_state(flat, meta)


def adapter_delta_norm(delta: AdapterState) -> float:
    return state_l2_norm(delta)