# RAVDESS dataset loader stub (complete placeholder).
from typing import Dict, Any
from torch.utils.data import Dataset

class RAVDESSDataset(Dataset):
    def __init__(self, cfg: Dict[str, Any], root: str):
        self.cfg = cfg
        self.root = root
        raise NotImplementedError("RAVDESS loader not implemented in this reference build. Use synthetic or implement here.")

    def __len__(self):
        return 0

    def __getitem__(self, idx: int):
        raise NotImplementedError
