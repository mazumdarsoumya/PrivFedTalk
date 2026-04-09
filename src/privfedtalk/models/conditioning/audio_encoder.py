import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyAudioEncoder(nn.Module):
    def __init__(self, in_len: int, cond_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 9, 2, 4), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 9, 2, 4), nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 9, 2, 4), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, cond_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return F.normalize(x, dim=-1)
