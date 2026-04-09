import torch.nn.functional as F
def identity_similarity(emb_a, emb_b) -> float:
    emb_a = F.normalize(emb_a, dim=-1)
    emb_b = F.normalize(emb_b, dim=-1)
    return float((emb_a * emb_b).sum(dim=-1).mean().item())
