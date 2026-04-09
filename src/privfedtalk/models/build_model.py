from __future__ import annotations

from typing import Dict, Any
import torch

from privfedtalk.models.diffusion.scheduler import DiffusionScheduler
from privfedtalk.models.diffusion.unet3d import TinyUNet3D
from privfedtalk.models.diffusion.latent_vae import TinyLatentVAE
from privfedtalk.models.conditioning.audio_encoder import TinyAudioEncoder
from privfedtalk.models.conditioning.identity_encoder import TinyIdentityEncoder
from privfedtalk.models.diffusion.gaussian_noise import GaussianNoiseSampler
from privfedtalk.models.adapters.inject import inject_lora
from privfedtalk.models.adapters.freeze import unfreeze_lora_only


class PrivFedTalkModel(torch.nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mcfg = cfg["model"]
        self.cfg = cfg

        self.vae = TinyLatentVAE(
            in_channels=3,
            latent_dim=mcfg["latent_dim"],
            base_channels=mcfg["base_channels"],
        )
        self.audio_enc = TinyAudioEncoder(
            in_len=cfg["data"]["audio_len"],
            cond_dim=mcfg["cond_dim"],
        )
        self.id_enc = TinyIdentityEncoder(
            in_channels=3,
            id_dim=mcfg["id_dim"],
            base_channels=mcfg["base_channels"],
        )

        self.unet = TinyUNet3D(
            latent_dim=mcfg["latent_dim"],
            base_channels=mcfg["base_channels"],
            cond_dim=mcfg["cond_dim"],
            id_dim=mcfg["id_dim"],
        )

        inject_lora(self.unet, mcfg["lora"])
        unfreeze_lora_only(self)

        self.scheduler = DiffusionScheduler(
            timesteps=mcfg["timesteps"],
            beta_start=mcfg["beta_start"],
            beta_end=mcfg["beta_end"],
        )
        self.noise = GaussianNoiseSampler()

    def to(self, *args, **kwargs):
        mod = super().to(*args, **kwargs)

        device = None
        if len(args) > 0:
            a0 = args[0]
            if isinstance(a0, (str, torch.device)):
                device = torch.device(a0)
            elif torch.is_tensor(a0):
                device = a0.device
        if device is None and "device" in kwargs:
            device = torch.device(kwargs["device"])

        if device is not None:
            self.scheduler.to(device)

        return mod

    def forward(self, video, audio, ref):
        B, T, C, H, W = video.shape
        z0 = self.vae.encode_video(video)
        cond = self.audio_enc(audio) if audio is not None else None
        ident = self.id_enc(ref)

        t = self.scheduler.sample_timesteps(B, device=video.device)
        eps = self.noise.sample_like(z0)
        zt = self.scheduler.q_sample(z0, t, eps)
        eps_pred = self.unet(zt, t, cond=cond, ident=ident)

        return {
            "z0": z0,
            "zt": zt,
            "t": t,
            "eps": eps,
            "eps_pred": eps_pred,
            "cond": cond,
            "ident": ident,
        }

    @torch.no_grad()
    def sample(self, audio, ref, num_frames: int, height: int, width: int):
        device = ref.device if audio is None else audio.device
        cond = self.audio_enc(audio) if audio is not None else None
        ident = self.id_enc(ref)

        H = height // 4
        W = width // 4
        lat = torch.randn(
            ref.shape[0],
            num_frames,
            self.cfg["model"]["latent_dim"],
            H,
            W,
            device=device,
        )

        for i in reversed(range(self.scheduler.timesteps)):
            t = torch.full((ref.shape[0],), i, device=device, dtype=torch.long)
            eps_pred = self.unet(lat, t, cond=cond, ident=ident)
            lat = self.scheduler.p_sample(lat, t, eps_pred)

        return self.vae.decode_video(lat)


def build_model(cfg: Dict[str, Any]) -> PrivFedTalkModel:
    return PrivFedTalkModel(cfg)
