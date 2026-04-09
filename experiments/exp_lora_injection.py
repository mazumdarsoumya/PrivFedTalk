import torch
from privfedtalk.models.diffusion.unet3d import TinyUNet3D
from privfedtalk.models.adapters.inject import inject_lora

def test_lora_injection():
    unet = TinyUNet3D(latent_dim=8, base_channels=16, cond_dim=64, id_dim=64)
    n = inject_lora(unet, {"r":4, "alpha":8, "dropout":0.0, "target_modules":["attn_q","attn_k","attn_v","attn_out"]})
    assert n > 0
