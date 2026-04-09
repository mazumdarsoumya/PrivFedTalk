import torch

def resolve_device(device='auto'):
    if device=='auto': device='cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)
