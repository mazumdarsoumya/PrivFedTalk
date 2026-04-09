import torch
from typing import Dict

def clip_and_add_noise(delta: Dict[str, torch.Tensor], clip_norm: float, noise_mult: float, seed: int = 0) -> Dict[str, torch.Tensor]:
    params = [v.reshape(-1) for v in delta.values() if torch.is_tensor(v)]
    if not params:
        return delta
    vec = torch.cat(params)
    norm = torch.norm(vec, p=2)
    scale = min(1.0, clip_norm / (norm.item() + 1e-12))

    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for k, v in delta.items():
        if not torch.is_tensor(v):
            continue
        vv = v * scale
        if noise_mult > 0:
            vv = vv + torch.randn(vv.shape, generator=g).to(vv.device) * (noise_mult * clip_norm)
        out[k] = vv
    return out
