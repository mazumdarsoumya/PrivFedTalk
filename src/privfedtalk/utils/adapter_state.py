#/home/vineet/PycharmProjects/PrivFedTalk/src/privfedtalk/utils/adapter_state.py

from __future__ import annotations

from typing import Dict, List, Tuple, Any

import torch


AdapterState = Dict[str, torch.Tensor]
FlatMeta = List[Tuple[str, int, torch.Size, torch.dtype, torch.device]]


def get_adapter_state(model, trainable_only: bool = True) -> AdapterState:
    """
    Extract only trainable adapter parameters by default.
    This fits LoRA-only FL well because your backbone is frozen.
    """
    state_dict = model.state_dict()
    out: AdapterState = {}

    if trainable_only:
        trainable_names = {name for name, p in model.named_parameters() if p.requires_grad}
        for name in trainable_names:
            if name in state_dict:
                out[name] = state_dict[name].detach().clone()
    else:
        for name, tensor in state_dict.items():
            out[name] = tensor.detach().clone()

    return out


def load_adapter_state(model, adapter_state: AdapterState) -> None:
    current = model.state_dict()
    for k, v in adapter_state.items():
        if k in current:
            current[k] = v.detach().clone()
    model.load_state_dict(current, strict=False)


def clone_state(state: AdapterState) -> AdapterState:
    return {k: v.detach().clone() for k, v in state.items()}


def move_state_to(state: AdapterState, device: torch.device | str) -> AdapterState:
    return {k: v.detach().to(device).clone() for k, v in state.items()}


def subtract_adapter_states(a: AdapterState, b: AdapterState) -> AdapterState:
    keys = sorted(set(a.keys()) & set(b.keys()))
    return {k: a[k] - b[k] for k in keys}


def add_adapter_delta(base: AdapterState, delta: AdapterState, scale: float = 1.0) -> AdapterState:
    out = clone_state(base)
    for k, v in delta.items():
        if k in out:
            out[k] = out[k] + scale * v
        else:
            out[k] = scale * v.detach().clone()
    return out


def scale_state(state: AdapterState, scale: float) -> AdapterState:
    return {k: v * scale for k, v in state.items()}


def zero_like_state(template: AdapterState) -> AdapterState:
    return {k: torch.zeros_like(v) for k, v in template.items()}


def weighted_sum_states(states: List[AdapterState], weights: List[float]) -> AdapterState:
    if len(states) == 0:
        return {}

    out = zero_like_state(states[0])
    for st, w in zip(states, weights):
        for k in out.keys():
            out[k] = out[k] + st[k] * float(w)
    return out


def flatten_state(state: AdapterState) -> Tuple[torch.Tensor, FlatMeta]:
    flats = []
    meta: FlatMeta = []

    for k in sorted(state.keys()):
        v = state[k]
        flats.append(v.reshape(-1))
        meta.append((k, v.numel(), v.shape, v.dtype, v.device))

    if len(flats) == 0:
        return torch.empty(0), []

    return torch.cat(flats, dim=0), meta


def unflatten_state(flat: torch.Tensor, meta: FlatMeta) -> AdapterState:
    out: AdapterState = {}
    pos = 0
    for name, numel, shape, dtype, device in meta:
        chunk = flat[pos : pos + numel].view(shape).to(device=device, dtype=dtype)
        out[name] = chunk.clone()
        pos += numel
    return out


def state_l2_norm(state: AdapterState) -> float:
    flat, _ = flatten_state(state)
    if flat.numel() == 0:
        return 0.0
    return float(torch.norm(flat, p=2).item())