import torch
from privfedtalk.fl.privacy.dp_clip_noise import clip_and_add_noise

def test_dp_clip_noise():
    delta = {"w": torch.ones(100)}
    out = clip_and_add_noise(delta, clip_norm=1.0, noise_mult=0.1, seed=0)
    assert out["w"].shape == delta["w"].shape
