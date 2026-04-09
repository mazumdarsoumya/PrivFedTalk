import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyPerceptual(nn.Module):
    """Torchvision-free tiny perceptual feature extractor (demo-friendly)."""
    def __init__(self, in_ch: int = 3, base: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(base, base, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(base * 2, base * 2, 3, 2, 1), nn.ReLU(True),
        )
        for p in self.net.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def perceptual_l1(perceptual: TinyPerceptual, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(perceptual(x_hat), perceptual(x))
