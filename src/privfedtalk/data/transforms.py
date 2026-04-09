import torch

def normalize_video(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0

def denormalize_video(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.0) / 2.0
