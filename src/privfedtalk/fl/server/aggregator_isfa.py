import math
from typing import List, Dict
import torch
from privfedtalk.fl.protocol.messages import ClientUpdate

def aggregate_isfa(updates: List[ClientUpdate], gamma: float = 2.0) -> Dict[str, torch.Tensor]:
    if not updates:
        return {}
    weights = [u.num_samples * math.exp(gamma * float(u.score)) for u in updates]
    s = sum(weights) + 1e-12
    weights = [w / s for w in weights]
    agg: Dict[str, torch.Tensor] = {}
    for u, w in zip(updates, weights):
        for k, v in u.delta.items():
            if k not in agg:
                agg[k] = v.detach().clone() * w
            else:
                agg[k] += v.detach().clone() * w
    return agg
