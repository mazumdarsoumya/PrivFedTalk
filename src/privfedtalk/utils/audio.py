import torch
def pad_or_trim(audio: torch.Tensor, length: int):
    if audio.numel()>=length: return audio[:length]
    out=torch.zeros(length, device=audio.device, dtype=audio.dtype)
    out[:audio.numel()] = audio
    return out
