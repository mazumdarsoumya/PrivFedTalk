import torch.nn.functional as F
def simple_sync_loss(audio_emb, video_emb):
    audio_emb = F.normalize(audio_emb, dim=-1)
    video_emb = F.normalize(video_emb, dim=-1)
    sim = (audio_emb * video_emb).sum(dim=-1)
    return (1.0 - sim).mean()
