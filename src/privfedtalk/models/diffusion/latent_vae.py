import torch
import torch.nn as nn

class TinyLatentVAE(nn.Module):
    # Minimal VAE-like bottleneck (demo).
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 32):
        super().__init__()
        ch = base_channels
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, ch, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch*2, latent_dim, 3, 1, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_dim, ch*2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch*2, ch, 4, 2, 1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, in_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = video.shape
        flat = video.view(B*T, C, H, W)
        z = self.encode(flat)
        _, L, H2, W2 = z.shape
        return z.view(B, T, L, H2, W2)

    def decode_video(self, lat: torch.Tensor) -> torch.Tensor:
        B, T, L, H2, W2 = lat.shape
        flat = lat.view(B*T, L, H2, W2)
        x = self.decode(flat)
        _, C, H, W = x.shape
        return x.view(B, T, C, H, W)
