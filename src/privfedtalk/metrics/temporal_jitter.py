def temporal_jitter(video) -> float:
    if video.size(1)<2: return 0.0
    return float((video[:,1:]-video[:,:-1]).abs().mean().item())
