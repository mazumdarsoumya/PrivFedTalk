import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device)) * torch.arange(0, half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class CrossAttn(nn.Module):
    def __init__(self, channels: int, cond_dim: int, id_dim: int):
        super().__init__()
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(cond_dim + id_dim, channels, bias=False)
        self.to_v = nn.Linear(cond_dim + id_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels, bias=False)

        # Names for LoRA targeting
        self.attn_q = self.to_q
        self.attn_k = self.to_k
        self.attn_v = self.to_v
        self.attn_out = self.to_out

    def forward(self, x: torch.Tensor, cond: torch.Tensor, ident: torch.Tensor):
        """
        IMPORTANT: Use attn_q/k/v/out so LoRA injection affects forward pass.
        """
        B, N, C = x.shape
        ctx = torch.cat([cond, ident], dim=-1).unsqueeze(1)  # (B,1,D)

        q = self.attn_q(x)
        k = self.attn_k(ctx)
        v = self.attn_v(ctx)

        attn = torch.softmax((q * k).sum(dim=-1, keepdim=True) / (C ** 0.5), dim=1)  # (B,N,1)
        out = attn * v
        return self.attn_out(out)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, 1, 1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).view(t_emb.size(0), -1, 1, 1, 1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet3D(nn.Module):
    def __init__(self, latent_dim: int, base_channels: int, cond_dim: int, id_dim: int):
        super().__init__()
        self.time_dim = 128
        self.time_emb = TimeEmbedding(self.time_dim)

        self.in_conv = nn.Conv3d(latent_dim, base_channels, 3, 1, 1)
        self.down1 = ResBlock3D(base_channels, base_channels, self.time_dim)
        self.down2 = ResBlock3D(base_channels, base_channels * 2, self.time_dim)
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.mid = ResBlock3D(base_channels * 2, base_channels * 2, self.time_dim)
        self.attn = CrossAttn(channels=base_channels * 2, cond_dim=cond_dim, id_dim=id_dim)
        self.up = nn.Upsample(scale_factor=(1, 2, 2), mode="nearest")
        self.up1 = ResBlock3D(base_channels * 2, base_channels, self.time_dim)
        self.out = nn.Conv3d(base_channels, latent_dim, 3, 1, 1)

    def forward(self, zt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, ident: torch.Tensor):
        z = rearrange(zt, "b t c h w -> b c t h w")
        t_emb = self.time_emb(t)

        h0 = self.in_conv(z)
        h1 = self.down1(h0, t_emb)
        h2 = self.pool(h1)
        h2 = self.down2(h2, t_emb)

        hm = self.mid(h2, t_emb)
        B, C, TT, H, W = hm.shape
        tokens = rearrange(hm, "b c t h w -> b (t h w) c")
        tokens = tokens + self.attn(tokens, cond=cond, ident=ident)
        hm = rearrange(tokens, "b (t h w) c -> b c t h w", t=TT, h=H, w=W)

        hu = self.up(hm)
        hu = self.up1(hu, t_emb)
        out = self.out(hu + h1)
        return rearrange(out, "b c t h w -> b t c h w")
