import torch
from typing import Dict, List, Sequence
from privfedtalk.fl.protocol.serialization import add_state

def secure_mask_updates(
    deltas: List[Dict[str, torch.Tensor]],
    client_ids: Sequence[int],
    round_seed: int,
    scale: float = 1e-3,
) -> List[Dict[str, torch.Tensor]]:
    """
    Pairwise-canceling masks (simulation-grade):
    ensures SUM(masked) == SUM(original) across participating clients.
    """
    if not deltas:
        return deltas

    masked = [{k: v.clone() for k, v in d.items()} for d in deltas]

    for i in range(len(client_ids)):
        for j in range(i + 1, len(client_ids)):
            a = int(client_ids[i])
            b = int(client_ids[j])
            seed = (round_seed * 1000003 + a * 9176 + b * 1337) & 0xFFFFFFFF
            g = torch.Generator(device="cpu").manual_seed(seed)

            shared = set(masked[i].keys()).intersection(masked[j].keys())
            for k in shared:
                va = masked[i][k]
                m = (torch.randn(va.shape, generator=g) * scale).to(va.device)
                masked[i][k] = masked[i][k] + m
                masked[j][k] = masked[j][k] - m

    return masked

# -------------------------------------------------------------------
# Legacy helpers (kept for backward compatibility with old tests)
# -------------------------------------------------------------------
def mask_update(delta: Dict[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = {}
    for k, v in delta.items():
        if not torch.is_tensor(v):
            continue
        mask = torch.randn(v.shape, generator=g) * 0.001
        out[k] = v + mask.to(v.device)
    return out

def unmask_aggregate(masked_updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    agg = {}
    for u in masked_updates:
        agg = add_state(agg, u) if agg else {k: v.clone() for k, v in u.items()}
    return agg
