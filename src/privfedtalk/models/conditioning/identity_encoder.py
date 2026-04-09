import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyIdentityEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, id_dim: int = 128, base_channels: int = 32):
        super().__init__()
        ch = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, ch, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch*2, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch*2, ch*4, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(ch*4, id_dim)

    def forward(self, ref: torch.Tensor) -> torch.Tensor:
        x = self.net(ref).flatten(1)
        x = self.fc(x)
        return F.normalize(x, dim=-1)
