import torch
from torch.utils.data import Dataset
from typing import Dict, Any

class SyntheticTalkingHeadDataset(Dataset):
    """Synthetic (video, audio, ref) for end-to-end runnable pipeline."""
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.T = cfg["data"]["video_frames"]
        self.H = cfg["data"]["video_size"]
        self.W = cfg["data"]["video_size"]
        self.L = cfg["data"]["audio_len"]
        self.N = cfg["data"]["num_clients"] * cfg["data"]["samples_per_client"]
        self.gen = torch.Generator().manual_seed(cfg["seed"])

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        speaker_bin = idx % max(1, self.cfg["data"]["num_clients"])
        base = (speaker_bin / max(1, self.cfg["data"]["num_clients"]-1)) if self.cfg["data"]["num_clients"] > 1 else 0.5

        ref = torch.rand(3, self.H, self.W, generator=self.gen) * 0.3 + base * 0.7
        video = ref.unsqueeze(0).repeat(self.T, 1, 1, 1)
        noise = 0.05 * torch.randn(self.T, 3, self.H, self.W, generator=self.gen)
        ramp = torch.linspace(0, 1, steps=self.T).view(self.T, 1, 1, 1)
        video = torch.clamp(video + noise + 0.1 * ramp, 0.0, 1.0)

        audio = torch.randn(self.L, generator=self.gen) * 0.2 + base * 0.1

        return {"video": video, "audio": audio, "ref": ref}
