import torch

def make_scaler(enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=enabled and torch.cuda.is_available())
