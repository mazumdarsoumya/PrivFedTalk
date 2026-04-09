from __future__ import annotations

from privfedtalk.utils.adapter_state import (
    AdapterState,
    flatten_state,
    unflatten_state,
    state_l2_norm,
)

__all__ = [
    "weighted_sum_states",
    "AdapterState",
    "flatten_state",
    "unflatten_state",
    "state_l2_norm",
]


def weighted_sum_states(states, weights=None):
    """
    Weighted sum over adapter states.

    Supports:
      - list[dict[str, torch.Tensor]]
      - tuple/list of AdapterState-like dicts

    If weights is None, uses uniform averaging.
    """
    import torch

    if states is None or len(states) == 0:
        raise ValueError("states must be a non-empty list")

    if weights is None:
        weights = [1.0 / len(states)] * len(states)

    if len(weights) != len(states):
        raise ValueError("weights and states must have same length")

    ref = states[0]
    out = {}

    for k in ref.keys():
        base = ref[k]
        if not torch.is_tensor(base):
            raise TypeError(f"State entry '{k}' is not a tensor")

        acc = torch.zeros_like(base)
        for s, w in zip(states, weights):
            acc = acc + s[k].to(base.device) * float(w)
        out[k] = acc

    return out



# =========================
# Federated compatibility helpers
# =========================

def _iter_adapter_named_parameters(model):
    picked = []
    for name, param in model.named_parameters():
        if not torch.is_tensor(param):
            continue
        lname = name.lower()
        if param.requires_grad or ("lora" in lname) or ("adapter" in lname):
            picked.append((name, param))

    if not picked:
        for name, param in model.named_parameters():
            if torch.is_tensor(param):
                picked.append((name, param))
    return picked


def get_adapter_state(model, detach=True, clone=True, to_cpu=False):
    state = {}
    for name, param in _iter_adapter_named_parameters(model):
        t = param
        if detach:
            t = t.detach()
        if clone:
            t = t.clone()
        if to_cpu:
            t = t.cpu()
        state[name] = t
    return state


def set_adapter_state(model, state, strict=False):
    params = dict(model.named_parameters())
    loaded = 0
    missing = []
    shape_mismatch = []

    with torch.no_grad():
        for name, tensor in state.items():
            if name not in params:
                missing.append(name)
                continue

            p = params[name]
            if tuple(p.shape) != tuple(tensor.shape):
                shape_mismatch.append((name, tuple(p.shape), tuple(tensor.shape)))
                continue

            p.copy_(tensor.to(device=p.device, dtype=p.dtype))
            loaded += 1

    if strict and (missing or shape_mismatch):
        raise RuntimeError(
            f"set_adapter_state failed | missing={missing} | shape_mismatch={shape_mismatch}"
        )

    return {
        "loaded": loaded,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
    }


def clone_state(state, to_cpu=False):
    out = {}
    for k, v in state.items():
        t = v.detach().clone()
        if to_cpu:
            t = t.cpu()
        out[k] = t
    return out


def subtract_states(a, b):
    keys = sorted(set(a.keys()) & set(b.keys()))
    return {k: a[k] - b[k].to(device=a[k].device, dtype=a[k].dtype) for k in keys}


def add_states(a, b):
    keys = sorted(set(a.keys()) & set(b.keys()))
    return {k: a[k] + b[k].to(device=a[k].device, dtype=a[k].dtype) for k in keys}


def scale_state(state, scale):
    s = float(scale)
    return {k: v * s for k, v in state.items()}


def zeros_like_state(state):
    return {k: torch.zeros_like(v) for k, v in state.items()}


def weighted_sum_states(states, weights=None):
    if states is None or len(states) == 0:
        raise ValueError("states must be a non-empty list")

    if weights is None:
        weights = [1.0 / len(states)] * len(states)

    if len(weights) != len(states):
        raise ValueError("weights and states must have same length")

    ref = states[0]
    out = {}
    for k in ref.keys():
        acc = torch.zeros_like(ref[k])
        for s, w in zip(states, weights):
            acc = acc + s[k].to(device=ref[k].device, dtype=ref[k].dtype) * float(w)
        out[k] = acc
    return out


try:
    __all__
except NameError:
    __all__ = []

for _name in [
    "_iter_adapter_named_parameters",
    "get_adapter_state",
    "set_adapter_state",
    "clone_state",
    "subtract_states",
    "add_states",
    "scale_state",
    "zeros_like_state",
    "weighted_sum_states",
]:
    if _name not in __all__:
        __all__.append(_name)



def load_adapter_state(model_or_path, maybe_path=None, strict=False, map_location="cpu"):
    """
    Compatibility loader.

    Supported:
      1) load_adapter_state(path)
         -> returns adapter state dict
      2) load_adapter_state(model, path)
         -> loads into model and returns status dict
    """
    if maybe_path is None:
        path = model_or_path
        obj = torch.load(path, map_location=map_location)

        if isinstance(obj, dict):
            if "adapter_state" in obj and isinstance(obj["adapter_state"], dict):
                return obj["adapter_state"]
            if "model" in obj and isinstance(obj["model"], dict):
                return obj["model"]
            if all(torch.is_tensor(v) for v in obj.values()):
                return obj

        raise RuntimeError(f"Unsupported adapter checkpoint format: {path}")

    model = model_or_path
    path = maybe_path
    state = load_adapter_state(path, map_location=map_location)
    return set_adapter_state(model, state, strict=strict)


def save_adapter_state(model_or_state, path):
    """
    Compatibility saver.

    Supported:
      1) save_adapter_state(model, path)
      2) save_adapter_state(state_dict, path)
    """
    if isinstance(model_or_state, dict):
        state = clone_state(model_or_state, to_cpu=True)
    else:
        state = get_adapter_state(model_or_state, detach=True, clone=True, to_cpu=True)

    payload = {"adapter_state": state}
    torch.save(payload, path)
    return path


if "load_adapter_state" not in __all__:
    __all__.append("load_adapter_state")

if "save_adapter_state" not in __all__:
    __all__.append("save_adapter_state")



# =========================
# Compatibility algebra API
# =========================

def _state_to_cpu_clone(state):
    return {k: v.detach().cpu().clone() for k, v in state.items()}

def _state_to_like_device(state, ref_state):
    out = {}
    for k, v in state.items():
        if k in ref_state and torch.is_tensor(ref_state[k]):
            out[k] = v.to(ref_state[k].device, dtype=ref_state[k].dtype)
        else:
            out[k] = v
    return out

def add_adapter_states(a, b, alpha: float = 1.0):
    keys = sorted(set(a.keys()) & set(b.keys()))
    return {k: a[k] + alpha * b[k].to(a[k].device, dtype=a[k].dtype) for k in keys}

def subtract_adapter_states(a, b):
    keys = sorted(set(a.keys()) & set(b.keys()))
    return {k: a[k] - b[k].to(a[k].device, dtype=a[k].dtype) for k in keys}

def scale_adapter_state(state, scale: float):
    return {k: v * scale for k, v in state.items()}

def average_adapter_states(states):
    if not states:
        return {}
    keys = sorted(states[0].keys())
    out = {}
    n = float(len(states))
    for k in keys:
        acc = None
        for s in states:
            v = s[k]
            acc = v.clone() if acc is None else acc + v.to(acc.device, dtype=acc.dtype)
        out[k] = acc / n
    return out

def weighted_sum_states(states, weights=None):
    if not states:
        return {}
    if weights is None:
        weights = [1.0] * len(states)
    if len(states) != len(weights):
        raise ValueError("states and weights must have same length")
    keys = sorted(states[0].keys())
    out = {}
    for k in keys:
        acc = None
        for s, w in zip(states, weights):
            v = s[k]
            term = v * float(w)
            acc = term.clone() if acc is None else acc + term.to(acc.device, dtype=acc.dtype)
        out[k] = acc
    return out

def average_adapter_deltas(deltas, weights=None):
    if not deltas:
        return {}
    if weights is None:
        weights = [1.0] * len(deltas)
    sw = float(sum(weights))
    if sw <= 0:
        raise ValueError("sum of weights must be > 0")
    summed = weighted_sum_states(deltas, weights)
    return {k: v / sw for k, v in summed.items()}

def adapter_delta(base_state, new_state):
    return subtract_adapter_states(new_state, base_state)

def apply_adapter_delta(base_state, delta, scale: float = 1.0):
    return add_adapter_states(base_state, delta, alpha=scale)

def load_adapter_state(model_or_path, maybe_path=None, strict=False, map_location="cpu"):
    """
    Supported:
      load_adapter_state(path) -> state dict
      load_adapter_state(model, path) -> load into model
    """
    if maybe_path is None:
        path = model_or_path
        obj = torch.load(path, map_location=map_location)

        if isinstance(obj, dict):
            if "adapter_state" in obj and isinstance(obj["adapter_state"], dict):
                return obj["adapter_state"]
            if "model" in obj and isinstance(obj["model"], dict):
                return obj["model"]
            if all(torch.is_tensor(v) for v in obj.values()):
                return obj

        raise RuntimeError(f"Unsupported adapter checkpoint format: {path}")

    model = model_or_path
    path = maybe_path
    state = load_adapter_state(path, map_location=map_location)
    return set_adapter_state(model, state, strict=strict)

def save_adapter_state(model_or_state, path):
    if isinstance(model_or_state, dict):
        state = clone_state(model_or_state, to_cpu=True) if "clone_state" in globals() else _state_to_cpu_clone(model_or_state)
    else:
        state = get_adapter_state(model_or_state, detach=True, clone=True, to_cpu=True)
    torch.save({"adapter_state": state}, path)
    return path


if "add_adapter_states" not in __all__:
    __all__.append("add_adapter_states")

if "subtract_adapter_states" not in __all__:
    __all__.append("subtract_adapter_states")

if "scale_adapter_state" not in __all__:
    __all__.append("scale_adapter_state")

if "average_adapter_states" not in __all__:
    __all__.append("average_adapter_states")

if "average_adapter_deltas" not in __all__:
    __all__.append("average_adapter_deltas")

if "adapter_delta" not in __all__:
    __all__.append("adapter_delta")

if "apply_adapter_delta" not in __all__:
    __all__.append("apply_adapter_delta")



def add_adapter_delta(base_state, delta, scale: float = 1.0):
    return apply_adapter_delta(base_state, delta, scale=scale)

def sub_adapter_states(a, b):
    return subtract_adapter_states(a, b)

def sub_adapter_delta(a, b):
    return subtract_adapter_states(a, b)


if "add_adapter_delta" not in __all__:
    __all__.append("add_adapter_delta")

if "sub_adapter_states" not in __all__:
    __all__.append("sub_adapter_states")

if "sub_adapter_delta" not in __all__:
    __all__.append("sub_adapter_delta")



def move_state_to(state, device):
    import torch
    if isinstance(device, str):
        device = torch.device(device)
    out = {}
    for k, v in state.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


if "move_state_to" not in __all__:
    __all__.append("move_state_to")



# ---- compatibility override for federated trainer ----
def _compat_is_adapter_name(name: str) -> bool:
    n = name.lower()
    return (
        "lora" in n
        or "adapter" in n
        or "ia3" in n
        or "prefix" in n
        or "prompt" in n
    )

def get_adapter_state(model, trainable_only: bool = False):
    state = {}
    for name, p in model.named_parameters():
        if trainable_only:
            take = bool(p.requires_grad)
        else:
            take = _compat_is_adapter_name(name) or bool(p.requires_grad)
        if take:
            state[name] = p.detach().clone()
    return state


# ==== PrivFedTalk federated compatibility overrides ====
import torch
from pathlib import Path
from typing import Dict

AdapterState = Dict[str, torch.Tensor]

def _compat_is_adapter_name(name: str) -> bool:
    n = name.lower()
    return (
        "lora" in n
        or "adapter" in n
        or "ia3" in n
        or "prefix" in n
        or "prompt" in n
    )

def get_adapter_state(model, trainable_only: bool = False) -> AdapterState:
    state: AdapterState = {}
    for name, p in model.named_parameters():
        take = bool(p.requires_grad) if trainable_only else (_compat_is_adapter_name(name) or bool(p.requires_grad))
        if take:
            state[name] = p.detach().clone()
    return state

def move_state_to(state: AdapterState, device) -> AdapterState:
    device = torch.device(device)
    return {k: v.detach().to(device) for k, v in state.items()}

def subtract_adapter_states(a: AdapterState, b: AdapterState) -> AdapterState:
    out: AdapterState = {}
    for k, v in a.items():
        if k in b:
            out[k] = v.detach() - b[k].detach().to(v.device)
        else:
            out[k] = v.detach().clone()
    return out

def add_adapter_delta(base: AdapterState, delta: AdapterState, scale: float = 1.0) -> AdapterState:
    out: AdapterState = {}
    keys = set(base.keys()) | set(delta.keys())
    for k in keys:
        if k in base and k in delta:
            out[k] = base[k].detach() + float(scale) * delta[k].detach().to(base[k].device)
        elif k in base:
            out[k] = base[k].detach().clone()
        else:
            out[k] = float(scale) * delta[k].detach().clone()
    return out

def weighted_sum_states(states, weights) -> AdapterState:
    out: AdapterState = {}
    for st, w in zip(states, weights):
        w = float(w)
        for k, v in st.items():
            if k not in out:
                out[k] = w * v.detach().clone()
            else:
                out[k] += w * v.detach()
    return out

def state_l2_norm(state: AdapterState) -> float:
    total = 0.0
    for v in state.values():
        total += float((v.detach().float() ** 2).sum().item())
    return total ** 0.5

def adapter_delta_norm(state: AdapterState) -> float:
    return state_l2_norm(state)

def flatten_state(state: AdapterState):
    keys = sorted(state.keys())
    if not keys:
        return torch.empty(0), []
    flat = torch.cat([state[k].reshape(-1).float().cpu() for k in keys], dim=0)
    shapes = [(k, tuple(state[k].shape), state[k].dtype) for k in keys]
    return flat, shapes

def unflatten_state(flat: torch.Tensor, shapes):
    out: AdapterState = {}
    offset = 0
    for k, shape, dtype in shapes:
        numel = 1
        for s in shape:
            numel *= s
        out[k] = flat[offset:offset+numel].view(*shape).to(dtype=dtype)
        offset += numel
    return out

def load_adapter_state(arg1, arg2=None, map_location="cpu"):
    """
    Overloaded behavior:
      - load_adapter_state(path) -> returns adapter state dict
      - load_adapter_state(model, state_dict) -> loads state into model in-place
    """
    # file-loading mode
    if isinstance(arg1, (str, Path)) and arg2 is None:
        obj = torch.load(str(arg1), map_location=map_location)
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        return obj

    # model-loading mode
    model = arg1
    state = arg2
    if state is None:
        raise ValueError("load_adapter_state(model, state) requires a state dict")

    named = dict(model.named_parameters())
    loaded = 0
    for k, v in state.items():
        if k in named:
            with torch.no_grad():
                named[k].copy_(v.to(device=named[k].device, dtype=named[k].dtype))
            loaded += 1
    return loaded

