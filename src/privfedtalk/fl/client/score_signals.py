def compute_identity_score(id_ref, id_pred) -> float:
    sim = (id_ref * id_pred).sum(dim=-1).mean().item()
    return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))

def compute_temporal_stability(video) -> float:
    if video.size(1) < 2:
        return 1.0
    dif = (video[:, 1:] - video[:, :-1]).abs().mean().item()
    return float(max(0.0, min(1.0, 1.0 - min(1.0, dif * 10.0))))
