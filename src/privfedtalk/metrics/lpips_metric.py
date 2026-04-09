import torch
from privfedtalk.losses.perceptual_loss import TinyPerceptual
@torch.no_grad()
def lpips_like(x, y, device="cpu") -> float:
    net = TinyPerceptual().to(device).eval()
    fx, fy = net(x.to(device)), net(y.to(device))
    return float((fx - fy).abs().mean().item())
