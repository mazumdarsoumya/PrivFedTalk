from typing import List, Dict
import torch
from privfedtalk.fl.protocol.messages import ClientUpdate

def aggregate_fedavg(updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
    total = sum(u.num_samples for u in updates) if updates else 1
    agg = {}
    for u in updates:
        w = u.num_samples / total
        for k, v in u.delta.items():
            if k not in agg:
                agg[k] = v.detach().clone() * w
            else:
                agg[k] += v.detach().clone() * w
    return agg
