import os, torch
from privfedtalk.utils.io import ensure_dir

def save_checkpoint(path: str, payload: dict):
    ensure_dir(os.path.dirname(path)); torch.save(payload, path)

def load_checkpoint(path: str, map_location='cpu'):
    return torch.load(path, map_location=map_location)
