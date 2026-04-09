import torch.nn.functional as F

def sync_score(audio_emb, video_emb) -> float:
    audio_emb=F.normalize(audio_emb,dim=-1); video_emb=F.normalize(video_emb,dim=-1)
    return float((audio_emb*video_emb).sum(dim=-1).mean().item())
