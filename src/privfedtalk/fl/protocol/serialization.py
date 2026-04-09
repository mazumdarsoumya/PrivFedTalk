import torch
from typing import Dict

def add_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    keys = set(a.keys()).union(b.keys())
    for k in keys:
        if k in a and k in b:
            out[k] = a[k] + b[k]
        elif k in a:
            out[k] = a[k].clone()
        else:
            out[k] = b[k].clone()
    return out

def scale_state(a: Dict[str, torch.Tensor], s: float) -> Dict[str, torch.Tensor]:
    return {k: v * s for k, v in a.items()}
